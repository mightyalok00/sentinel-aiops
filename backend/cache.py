"""Redis and In-Memory Hybrid Caching Layer.

Provides high-performance caching for:
1. Case telemetry queries (reducing parquet disk reads).
2. Model scoring results for identical telemetry windows.
3. IP-based rate limiting counters.

Design:
Attempts connection to Redis via `REDIS_URL` (default: redis://localhost:6379/0).
If Redis is unreachable or uninstalled, transparently falls back to an in-memory
TTL cache with zero service disruption.
"""

import json
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("sentinel.cache")

_REDIS_CLIENT: Optional[Any] = None
_REDIS_INITIALIZED = False

# Fallback in-memory storage: key -> (value, expiry_timestamp)
_MEMORY_CACHE: dict[str, tuple[str, float]] = {}
_RATE_LIMIT_STORE: dict[str, list[float]] = {}


def _get_redis() -> Optional[Any]:
    """Initializes and returns a lazy Redis connection singleton."""
    global _REDIS_CLIENT, _REDIS_INITIALIZED
    if _REDIS_INITIALIZED:
        return _REDIS_CLIENT

    _REDIS_INITIALIZED = True
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis
        client = redis.Redis.from_url(
            redis_url,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True
        )
        client.ping()
        _REDIS_CLIENT = client
        logger.info(f"Connected to Redis cache at {redis_url}")
    except Exception as exc:
        logger.warning(f"Redis unavailable ({exc}); using in-memory TTL cache fallback.")
        _REDIS_CLIENT = None

    return _REDIS_CLIENT


def cache_get(key: str) -> Optional[Any]:
    """Retrieves a deserialized JSON value from Redis or in-memory fallback.

    Args:
        key: Unique cache key string.

    Returns:
        Optional[Any]: Deserialized object or None if missing/expired.
    """
    client = _get_redis()
    if client:
        try:
            val = client.get(key)
            return json.loads(val) if val else None
        except Exception:
            pass

    # In-memory fallback
    entry = _MEMORY_CACHE.get(key)
    if entry:
        val_str, expiry = entry
        if time.time() < expiry:
            return json.loads(val_str)
        else:
            del _MEMORY_CACHE[key]
    return None


def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> None:
    """Serializes and stores a value in Redis or in-memory fallback with TTL.

    Args:
        key: Unique cache key string.
        value: JSON-serializable object to store.
        ttl_seconds: Expiration lifetime in seconds (default: 300s).
    """
    client = _get_redis()
    try:
        val_str = json.dumps(value)
    except Exception as exc:
        logger.error(f"Failed to serialize value for key {key}: {exc}")
        return

    if client:
        try:
            client.setex(key, ttl_seconds, val_str)
            return
        except Exception:
            pass

    # In-memory fallback
    _MEMORY_CACHE[key] = (val_str, time.time() + ttl_seconds)


def check_rate_limit(client_id: str, max_requests: int = 120, window_seconds: int = 60) -> bool:
    """Evaluates whether a client has exceeded their request quota within a sliding window.

    Args:
        client_id: Unique identifier for the client (e.g. IP address or API key).
        max_requests: Maximum allowable requests within the time window.
        window_seconds: Duration of the sliding rate-limit window in seconds.

    Returns:
        bool: True if request is allowed, False if rate limit is exceeded.
    """
    now = time.time()
    client = _get_redis()

    if client:
        try:
            key = f"rate_limit:{client_id}"
            pipe = client.pipeline()
            pipe.zadd(key, {str(now): now})
            pipe.zremrangebyscore(key, 0, now - window_seconds)
            pipe.zcard(key)
            pipe.expire(key, window_seconds)
            _, _, count, _ = pipe.execute()
            return count <= max_requests
        except Exception:
            pass

    # In-memory sliding window rate limiting
    timestamps = _RATE_LIMIT_STORE.setdefault(client_id, [])
    # Evict timestamps older than window
    cutoff = now - window_seconds
    _RATE_LIMIT_STORE[client_id] = [t for t in timestamps if t > cutoff]

    if len(_RATE_LIMIT_STORE[client_id]) >= max_requests:
        return False

    _RATE_LIMIT_STORE[client_id].append(now)
    return True

