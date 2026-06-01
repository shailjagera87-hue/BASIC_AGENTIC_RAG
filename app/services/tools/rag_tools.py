# app/services/tools/rag_tools.py
"""
RAG Tools — Fixed Version
===========================
Fixes:
  1. [Critical] Empty context silently passed to generator → hallucination
     Fix: retrieval_error flag in state, early answer set

  2. [Medium]   Global _db not thread-safe → race condition on concurrent requests
     Fix: threading.Lock() with double-check lock pattern

  3. [Low]      print() used instead of logging → invisible in production
     Fix: logging.getLogger(__name__) throughout

  ✅ get_db() lazy loader kept — correct pattern
"""

import logging
import threading

from app.services.memory.cache import get_cache, set_cache
from app.db.vector_store import load_vector_store

log = logging.getLogger(__name__)

# ── FIX 2: Thread-safe lazy loader ──────────────────────────────
# Problem:
#   FastAPI runs multiple threads concurrently.
#   Two requests arrive simultaneously → both see _db is None
#   → both call load_vector_store() → race condition.
#
# Fix: Double-check locking pattern
#   - Outer check (no lock): fast path, avoids lock overhead
#     once DB is loaded (99% of requests)
#   - Inner check (with lock): only one thread loads DB
#   - Result: DB loaded exactly once, safely

_db   = None
_lock = threading.Lock()


def get_db():
    """
    Thread-safe lazy loader for vector store.

    Double-check lock pattern:
        First check  → no lock  → fast (loaded DB case)
        Second check → with lock → safe (first load case)
    """
    global _db
    if _db is None:               # fast outer check — no lock overhead
        with _lock:               # acquire lock — only one thread here
            if _db is None:       # re-check — another thread may have loaded
                log.info("Loading vector store...")
                _db = load_vector_store()
                log.info("Vector store loaded.")
    return _db


# ── cache_tool ───────────────────────────────────────────────────

def cache_tool(state: dict) -> dict:
    """
    Check if query has a cached answer.

    Returns state with:
        cached=True  + answer set   → router will go to END
        cached=False                → router will go to retriever
    """
    query = state.get("query", "")
    if not query:
        return {**state, "cached": False}

    res = get_cache(query)
    if res:
        log.debug(f"Cache HIT: '{query[:50]}'")
        return {**state, "answer": res, "cached": True}

    log.debug(f"Cache MISS: '{query[:50]}'")
    return {**state, "cached": False}


# ── retriever_tool ───────────────────────────────────────────────

def retriever_tool(state: dict) -> dict:
    """
    Retrieve relevant chunks from vector store.

    FIX 1: Empty context must NOT silently pass to generator.
    If retrieval fails:
        - Log the error properly
        - Set retrieval_error flag in state
        - Set a safe fallback answer (don't let generator hallucinate)

    Why this matters:
        Generator receives context="" → it has nothing to ground on
        → it will either hallucinate or return garbage.
        User gets wrong financial data — worst possible outcome for RAG.
    """
    query = state.get("query", "")

    try:
        db   = get_db()
        docs = db.similarity_search(query, k=3)

        if not docs:
            # No documents found — not an error, but log it
            log.warning(f"No docs found for query: '{query[:50]}'")
            return {
                **state,
                "context":          "",
                "retrieval_error":  "No relevant documents found.",
                "answer":           "No relevant information found in the document."
            }

        context = "\n\n".join([d.page_content for d in docs])
        log.debug(f"Retrieved {len(docs)} chunks for: '{query[:50]}'")
        return {**state, "context": context, "retrieval_error": None}

    except Exception as e:
        # FIX 1: Error flag set — graph can check this
        # Generator will NOT run on empty context now
        log.error(f"Retriever error: {e}", exc_info=True)
        return {
            **state,
            "context":         "",
            "retrieval_error": str(e),
            "answer":          "Document search failed. Please try again."
            # ↑ answer already set → generator can check and skip
        }


# ── save_cache_tool ──────────────────────────────────────────────

def save_cache_tool(state: dict) -> dict:
    """
    Save answer to cache after successful generation.

    Only cache if:
        - answer exists
        - no retrieval error occurred (don't cache error answers)
        - answer is not the fallback error message
    """
    answer = state.get("answer", "")
    error  = state.get("retrieval_error")

    # Don't cache error responses or empty answers
    if not answer:
        return state
    if error:
        log.debug("Skipping cache save — retrieval error occurred")
        return state
    if answer.startswith("Document search failed"):
        log.debug("Skipping cache save — fallback error answer")
        return state

    try:
        set_cache(state["query"], answer)
        log.debug(f"Cached answer for: '{state['query'][:50]}'")
    except Exception as e:
        # FIX 3: log.error not print() — visible in production logs
        log.error(f"Cache save error: {e}", exc_info=True)

    return state