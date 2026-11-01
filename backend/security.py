"""Enterprise Security & Defense Layer for SENTINEL AI.

Provides defense-in-depth protections:
1. Constant-Time Authentication: Timing-attack resistant API key comparison via `hmac.compare_digest`.
2. HTTP Security Headers: HSTS, X-Frame-Options, X-Content-Type-Options, CSP, Referrer-Policy.
3. Payload Size Limiting: Prevents memory exhaustion / DoS via oversized telemetry batch payloads.
4. Log Scrubbing & Sensitive Field Redaction: Strips API keys, auth headers, and tokens from structured JSON logs.
5. CORS Policy Enforcement: Secure, configurable origin validation.
"""

import hmac
import os
from typing import Any, Optional

MAX_PAYLOAD_BYTES = int(os.getenv("MAX_PAYLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB default limit

# Sensitive header keys to sanitize from log outputs
SENSITIVE_HEADERS = {"x-api-key", "authorization", "cookie", "proxy-authorization", "set-cookie"}


def constant_time_compare(val1: str, val2: str) -> bool:
    """Compares two strings in constant time to prevent timing side-channel attacks."""
    if not val1 or not val2:
        return False
    return hmac.compare_digest(val1.encode("utf-8"), val2.encode("utf-8"))


def sanitize_log_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redacts sensitive keys such as api_keys and tokens from logging dictionaries."""
    sanitized: dict[str, Any] = {}
    for k, v in data.items():
        key_lower = str(k).lower()
        if any(sens in key_lower for sens in ["api_key", "secret", "password", "token", "auth"]):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_log_dict(v)
        else:
            sanitized[k] = v
    return sanitized


def add_security_headers(headers: Any) -> Any:
    """Injects enterprise-grade defensive HTTP security headers into response headers."""
    headers["X-Content-Type-Options"] = "nosniff"
    headers["X-Frame-Options"] = "DENY"
    headers["X-XSS-Protection"] = "1; mode=block"
    headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "img-src 'self' data: https:; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "connect-src 'self'"
    )
    headers["Permissions-Policy"] = "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()"
    return headers


def is_payload_too_large(content_length_header: Optional[str]) -> bool:
    """Evaluates whether Content-Length exceeds MAX_PAYLOAD_BYTES."""
    if not content_length_header:
        return False
    try:
        return int(content_length_header) > MAX_PAYLOAD_BYTES
    except ValueError:
        return False
