# app/services/memory/cache.py
"""
CACHE SERVICE — Fixed Version
================================
Bugs fixed:
  1. [Critical] In-memory dict → data loss on restart
     Fix: disk-based shelve (no deps) + optional Redis

  2. [Medium]   No TTL → stale financial data served forever
     Fix: TTL per entry (default 24h, configurable)

  3. [Medium]   MD5 → collision-prone hash
     Fix: SHA-256 (collision-resistant)

  4. [Low]      No type check → None stored as valid cache entry
     Fix: explicit None guard in set_cache()

Two backends available:
  CacheBackend.MEMORY  → development only (original behaviour)
  CacheBackend.DISK    → shelve (no extra deps, survives restart)
  CacheBackend.REDIS   → production (set REDIS_URL env var)

Usage:
    from app.services.memory.cache import get_cache, set_cache

    val = get_cache("What is total expenditure?")
    if val is None:
        val = call_llm(...)
        set_cache("What is total expenditure?", val)
"""

import os
import time
import json
import hashlib
import shelve
import logging
from enum import Enum
from typing import Optional, Any

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────
DEFAULT_TTL_SECONDS = int(os.getenv("CACHE_TTL", 86400))   # 24 hours
CACHE_BACKEND       = os.getenv("CACHE_BACKEND", "disk")   # memory|disk|redis
DISK_CACHE_PATH     = os.getenv("DISK_CACHE_PATH", "/tmp/finance_rag_cache")
REDIS_URL           = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ── FIX 3: SHA-256 replaces MD5 ──────────────────────────────────
def _make_key(query: str) -> str:
    """
    SHA-256 hash of the query string.

    Why SHA-256 over MD5:
        MD5  → 2^64 operations to find collision (broken since 2004)
        SHA-256 → 2^128 operations (practically collision-free)

    For cache keys in financial RAG:
        "total expenditure" and "total revenue" must NEVER
        produce the same key — wrong cached answer = wrong financial data.

    Returns first 32 chars of hex digest (128 bits — more than enough
    for cache key uniqueness, saves storage vs full 64 chars).
    """
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:32]


# ── Cache entry wrapper ──────────────────────────────────────────
def _make_entry(value: Any, ttl: int) -> dict:
    """Wrap value with expiry timestamp."""
    return {
        "value":      value,
        "expires_at": time.time() + ttl,
        "created_at": time.time(),
    }


def _is_expired(entry: dict) -> bool:
    """True if TTL has passed."""
    return time.time() > entry.get("expires_at", 0)


# ── Backend: Memory (dev only) ───────────────────────────────────
_MEM_CACHE: dict = {}


def _mem_get(key: str) -> Optional[Any]:
    entry = _MEM_CACHE.get(key)
    if entry is None:
        return None
    if _is_expired(entry):
        del _MEM_CACHE[key]     # cleanup expired entry
        return None
    return entry["value"]


def _mem_set(key: str, value: Any, ttl: int) -> None:
    _MEM_CACHE[key] = _make_entry(value, ttl)


def _mem_delete(key: str) -> None:
    _MEM_CACHE.pop(key, None)


# ── FIX 1 option A: Disk backend (shelve — no extra deps) ────────
# shelve = Python built-in persistent dict backed by file
# Survives process restart. Good for single-server deployments.
# NOT suitable for multi-worker (use Redis then).

def _disk_get(key: str) -> Optional[Any]:
    try:
        with shelve.open(DISK_CACHE_PATH) as db:
            raw = db.get(key)
        if raw is None:
            return None
        entry = json.loads(raw)
        if _is_expired(entry):
            _disk_delete(key)
            return None
        return entry["value"]
    except Exception as e:
        log.error(f"Cache disk get error: {e}")
        return None


def _disk_set(key: str, value: Any, ttl: int) -> None:
    try:
        with shelve.open(DISK_CACHE_PATH) as db:
            db[key] = json.dumps(_make_entry(value, ttl), default=str)
    except Exception as e:
        log.error(f"Cache disk set error: {e}")


def _disk_delete(key: str) -> None:
    try:
        with shelve.open(DISK_CACHE_PATH) as db:
            db.pop(key, None)
    except Exception as e:
        log.error(f"Cache disk delete error: {e}")


# ── FIX 1 option B: Redis backend (production) ───────────────────
# Redis = in-memory + persistent + multi-worker safe
# Install: pip install redis
# Set env: REDIS_URL=redis://localhost:6379/0

def _redis_get(key: str) -> Optional[Any]:
    try:
        import redis
        r   = redis.from_url(REDIS_URL, decode_responses=True)
        raw = r.get(key)
        if raw is None:
            return None
        entry = json.loads(raw)
        # Redis TTL handles expiry — but we double-check our timestamp
        if _is_expired(entry):
            r.delete(key)
            return None
        return entry["value"]
    except ImportError:
        log.warning("redis not installed — falling back to disk cache")
        return _disk_get(key)
    except Exception as e:
        log.error(f"Redis get error: {e} — falling back to disk")
        return _disk_get(key)


def _redis_set(key: str, value: Any, ttl: int) -> None:
    try:
        import redis
        r   = redis.from_url(REDIS_URL, decode_responses=True)
        raw = json.dumps(_make_entry(value, ttl), default=str)
        r.setex(key, ttl, raw)   # Redis native TTL — auto-expires
    except ImportError:
        log.warning("redis not installed — falling back to disk cache")
        _disk_set(key, value, ttl)
    except Exception as e:
        log.error(f"Redis set error: {e} — falling back to disk")
        _disk_set(key, value, ttl)


def _redis_delete(key: str) -> None:
    try:
        import redis
        redis.from_url(REDIS_URL).delete(key)
    except Exception as e:
        log.error(f"Redis delete error: {e}")


# ── Backend router ───────────────────────────────────────────────
_BACKENDS = {
    "memory": (_mem_get, _mem_set, _mem_delete),
    "disk":   (_disk_get, _disk_set, _disk_delete),
    "redis":  (_redis_get, _redis_set, _redis_delete),
}


def _get_backend():
    backend = CACHE_BACKEND.lower()
    if backend not in _BACKENDS:
        log.warning(f"Unknown backend '{backend}' — using disk")
        backend = "disk"
    return _BACKENDS[backend]


# ── Public API ───────────────────────────────────────────────────

def get_cache(query: str) -> Optional[Any]:
    """
    Retrieve cached answer for query.

    Returns:
        Cached value if found and not expired.
        None if cache miss or expired.

    Usage in graph node:
        cached = get_cache(state["query"])
        if cached:
            return {**state, "answer": cached, "cached": True}
    """
    if not query or not query.strip():
        return None

    key       = _make_key(query)
    get_fn, _, _ = _get_backend()
    value     = get_fn(key)

    if value is not None:
        log.debug(f"Cache HIT: '{query[:50]}'")
    else:
        log.debug(f"Cache MISS: '{query[:50]}'")

    return value


def set_cache(
    query: str,
    value: Any,
    ttl: int = DEFAULT_TTL_SECONDS,
) -> None:
    """
    Store answer in cache with TTL.

    Args:
        query: user's question (used as cache key)
        value: answer to cache (any JSON-serializable type)
        ttl:   seconds until expiry (default: 24h from env var)

    FIX 4: None guard — None values are NOT cached.
    Reason: caller uses `if get_cache(q) is None` to detect cache miss.
            Storing None would make every lookup look like a hit
            but return nothing — silent bug, very hard to debug.

    Usage in graph node:
        set_cache(state["query"], state["answer"])
    """
    # FIX 4: guard against None / empty value
    if value is None:
        log.warning(f"set_cache called with None value — skipping")
        return
    if not query or not query.strip():
        log.warning("set_cache called with empty query — skipping")
        return

    key          = _make_key(query)
    _, set_fn, _ = _get_backend()
    set_fn(key, value, ttl)
    log.debug(f"Cache SET: '{query[:50]}' | TTL: {ttl}s")


def invalidate_cache(query: str) -> None:
    """
    Remove a specific query from cache.
    Call this when a document is updated/re-uploaded.

    Usage:
        # User uploads new version of document
        invalidate_cache("What is the total expenditure?")
    """
    if not query:
        return
    key             = _make_key(query)
    _, _, delete_fn = _get_backend()
    delete_fn(key)
    log.info(f"Cache INVALIDATED: '{query[:50]}'")


def invalidate_all() -> None:
    """
    Clear entire cache.
    Call after bulk document re-upload.
    """
    backend = CACHE_BACKEND.lower()
    if backend == "memory":
        _MEM_CACHE.clear()
    elif backend == "disk":
        import os, glob
        for f in glob.glob(f"{DISK_CACHE_PATH}*"):
            os.remove(f)
    elif backend == "redis":
        try:
            import redis
            redis.from_url(REDIS_URL).flushdb()
        except Exception as e:
            log.error(f"Redis flush error: {e}")
    log.info("Cache CLEARED (all entries)")


def cache_stats() -> dict:
    """
    Return cache stats for monitoring / debugging.
    Useful for /health endpoint or admin dashboard.
    """
    backend = CACHE_BACKEND.lower()
    stats   = {"backend": backend, "ttl_default": DEFAULT_TTL_SECONDS}

    if backend == "memory":
        # Count non-expired entries
        active = sum(
            1 for e in _MEM_CACHE.values()
            if not _is_expired(e)
        )
        stats["total_entries"] = len(_MEM_CACHE)
        stats["active_entries"] = active

    elif backend == "redis":
        try:
            import redis
            r = redis.from_url(REDIS_URL)
            stats["redis_keys"] = r.dbsize()
            stats["redis_ping"] = r.ping()
        except Exception as e:
            stats["redis_error"] = str(e)

    return stats