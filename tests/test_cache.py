"""Unit Tests for Hybrid Redis and In-Memory Caching and Rate Limiting Layer."""

import time
from backend.cache import cache_get, cache_set, check_rate_limit


def test_memory_cache_set_and_get():
    """Verifies that cache_set stores values and cache_get retrieves deserialized objects."""
    key = "test_key_123"
    data = {"metric": "cpu", "value": 99.4}
    cache_set(key, data, ttl_seconds=10)

    retrieved = cache_get(key)
    assert retrieved == data


def test_memory_cache_ttl_expiration():
    """Verifies that cached keys expire after the configured TTL."""
    key = "test_expire_key"
    cache_set(key, {"expired": True}, ttl_seconds=1)
    time.sleep(1.1)
    assert cache_get(key) is None


def test_rate_limit_sliding_window():
    """Verifies that clients exceeding quota are blocked by check_rate_limit."""
    client_id = "test_client_rate_limit"
    # Allow 5 requests in a 10-second window
    for _ in range(5):
        assert check_rate_limit(client_id, max_requests=5, window_seconds=10) is True

    # 6th request must be rejected
    assert check_rate_limit(client_id, max_requests=5, window_seconds=10) is False

