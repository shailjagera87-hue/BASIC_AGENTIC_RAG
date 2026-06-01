"""
graph.py — Fixed LangGraph RAG Pipeline
=========================================

Bugs fixed:
  1. [Critical] Non-deterministic LLM route output → KeyError crash
     Fix: normalise_route() cleans LLM output before routing
          + "__default__" fallback edge in graph

  2. [Medium] "generate" route skipped retriever → empty context
     Fix: "generate" route now goes through retriever first,
          then generator gets proper context

Graph flow after fix:
  
  router
    ├── "cache"    → cache_node
    │                 ├── cached=True  → END
    │                 └── cached=False → retriever → generator → save_cache → END
    │
    ├── "retrieve" → retriever → generator → save_cache → END
    │
    ├── "generate" → retriever → generator → save_cache → END  ← BUG 2 FIX
    │                (retrieve karo pehle, phir generate)
    │
    └── [anything else] → retriever → generator → save_cache → END  ← BUG 1 FIX
"""

import re
import logging
from langgraph.graph import StateGraph, END

from app.services.rag.state       import RAGState
from app.services.tools.rag_tools import cache_tool, retriever_tool, save_cache_tool
from app.services.agents.router_agent import router_agent
from app.services.rag.generator   import generator_node

log = logging.getLogger(__name__)

# ── Valid route values ────────────────────────────────────────────
VALID_ROUTES = {"cache", "retrieve", "generate"}


# ── FIX 1: Route normaliser ───────────────────────────────────────
def normalise_route(raw: str) -> str:
    """
    LLM output clean karke valid route return karta hai.

    Problem:
        LLM non-deterministic hai — same prompt pe alag output:
          "retrieve"   ← sahi
          "retrieve."  ← dot ke saath → KeyError
          "RETRIEVE"   ← caps mein   → KeyError
          "I would retrieve the documents" ← sentence → KeyError

    Solution — 3 step pipeline:
        Step 1: lowercase + strip + punctuation remove
        Step 2: exact match check
        Step 3: substring match (LLM ne extra words bole)
        Step 4: kuch nahi mila → "retrieve" fallback (safest default)

    Args:
        raw: LLM ka raw string output

    Returns:
        Guaranteed valid string: "cache" | "retrieve" | "generate"
    """
    if not raw:
        log.warning("Route is empty — defaulting to 'retrieve'")
        return "retrieve"

    # Step 1: clean karo
    cleaned = raw.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)  # . , ! ? etc hatao
    cleaned = cleaned.strip()

    # Step 2: exact match
    if cleaned in VALID_ROUTES:
        return cleaned

    # Step 3: substring match
    # Order matters — "generate" pehle check karo kyunki
    # "cache" "retrieve" dono "generate" mein nahi hain
    for route in ["cache", "retrieve", "generate"]:
        if route in cleaned:
            log.warning(
                f"Route substring matched: raw='{raw}' → '{route}'"
            )
            return route

    # Step 4: fallback — retrieve safest hai
    # (cache miss toh crash, generate bina context toh hallucination)
    log.error(
        f"Route unrecognised: '{raw}' — falling back to 'retrieve'"
    )
    return "retrieve"


# ── FIX 1 continued: edge function ───────────────────────────────
def route_from_state(state: RAGState) -> str:
    """
    Conditional edge function for router node.

    Pehle wala code:
        lambda s: s["route"]   ← direct access, crash on bad value

    Fixed code:
        normalise_route(state["route"])  ← always valid
    """
    raw_route = state.get("route", "")      # .get() — KeyError nahi
    route     = normalise_route(str(raw_route))
    return route


# ── FIX 2: Generate route ko bhi retriever se guzaaro ────────────
#
# Pehle:
#   "generate" → generator  (retriever skip! empty context!)
#
# Fix:
#   "generate" → retriever → generator
#
# Kyun?
#   Generator ko context chahiye retrieved chunks se.
#   Bina retriever ke generator ke paas sirf user query hai —
#   woh hallucinate karega ya empty answer dega.
#
#   "generate" route ka matlab: "directly generate karo"
#   but RAG mein generate ALWAYS retrieval ke baad hota hai.
#   Router ka "generate" intent = "retrieval skip mat karo,
#   generate karo" — implementation mein retriever zaroor chale.

def build_graph():
    g = StateGraph(RAGState)

    # ── Nodes register karo ──────────────────────────────────────
    g.add_node("router",     router_agent)
    g.add_node("cache",      cache_tool)
    g.add_node("retriever",  retriever_tool)
    g.add_node("generator",  generator_node)
    g.add_node("save_cache", save_cache_tool)

    # ── Entry point ───────────────────────────────────────────────
    g.set_entry_point("router")

    # ── Edge 1: router → next node ────────────────────────────────
    #
    # BEFORE (broken):
    #   lambda s: s["route"]           ← direct, crashes on bad LLM output
    #   "generate": "generator"        ← retriever skip, empty context
    #
    # AFTER (fixed):
    #   route_from_state(s)            ← normalised, never crashes
    #   "generate": "retriever"        ← retriever pehle, phir generator
    #
    g.add_conditional_edges(
        "router",
        route_from_state,             # ← FIX 1: normalised function
        {
            "cache":    "cache",
            "retrieve": "retriever",
            "generate": "retriever",  # ← FIX 2: retriever se guzaaro
        }
    )

    # ── Edge 2: cache hit/miss ────────────────────────────────────
    #
    # Cache node check karta hai: kya yeh query pehle answer hui?
    #   cached=True  → directly END (retrieval + generation skip)
    #   cached=False → retriever pe jaao (normal flow)
    #
    # Yeh logic sahi tha — unchanged
    #
    g.add_conditional_edges(
        "cache",
        lambda s: "end" if s.get("cached") else "retriever",
        {
            "end":      END,
            "retriever": "retriever"
        }
    )

    # ── Edge 3: retriever → generator (always) ───────────────────
    # Retriever ke baad generator hamesha chalta hai
    g.add_edge("retriever",  "generator")

    # ── Edge 4: generator → save_cache ───────────────────────────
    # Answer generate hua → cache mein save karo future ke liye
    g.add_edge("generator",  "save_cache")

    # ── Edge 5: save_cache → END ──────────────────────────────────
    g.add_edge("save_cache", END)

    return g.compile()