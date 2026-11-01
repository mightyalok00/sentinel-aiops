"""Enterprise Security Tests for SENTINEL AI.

Validates:
1. HTTP Security Headers (OWASP recommendations: HSTS, X-Frame-Options, CSP, nosniff, Permissions-Policy).
2. Constant-time timing-attack mitigation for API key auth.
3. Payload limit calculations (rejection of requests > 10MB).
4. Sensitive token and credential log scrubbing.
5. Strict path traversal prevention on filesystem endpoints.
6. RBAC privilege boundaries.
"""

import pytest
from fastapi import HTTPException

from backend.app import _validate_case_id, require_role
from backend.security import (
    MAX_PAYLOAD_BYTES,
    add_security_headers,
    constant_time_compare,
    is_payload_too_large,
    sanitize_log_dict,
)


def test_security_headers_injection():
    """Verifies that all OWASP defense-in-depth headers are injected into HTTP response dictionaries."""
    headers = {}
    add_security_headers(headers)
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "max-age=31536000" in headers.get("Strict-Transport-Security", "")
    assert "default-src 'self'" in headers.get("Content-Security-Policy", "")
    assert "geolocation=()" in headers.get("Permissions-Policy", "")


def test_constant_time_compare():
    """Verifies constant-time string comparison behaves correctly."""
    assert constant_time_compare("super-secret-key-123", "super-secret-key-123") is True
    assert constant_time_compare("super-secret-key-123", "wrong-key") is False
    assert constant_time_compare("", "key") is False
    assert constant_time_compare("key", "") is False


def test_sanitize_log_dict():
    """Verifies that sensitive credentials are recursively redacted from logs."""
    raw_log = {
        "event": "auth_attempt",
        "api_key": "sentinel-live-admin-token-xyz",
        "nested": {
            "auth_token": "secret123",
            "safe_metric": 42.0
        },
        "client_ip": "10.0.0.1"
    }

    sanitized = sanitize_log_dict(raw_log)
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["auth_token"] == "[REDACTED]"
    assert sanitized["nested"]["safe_metric"] == 42.0
    assert sanitized["client_ip"] == "10.0.0.1"


def test_payload_limit_evaluator():
    """Verifies that payloads exceeding MAX_PAYLOAD_BYTES return True for oversized check."""
    assert is_payload_too_large(None) is False
    assert is_payload_too_large("invalid_int") is False
    assert is_payload_too_large(str(MAX_PAYLOAD_BYTES - 100)) is False
    assert is_payload_too_large(str(MAX_PAYLOAD_BYTES)) is False
    assert is_payload_too_large(str(MAX_PAYLOAD_BYTES + 1)) is True
    assert is_payload_too_large(str(MAX_PAYLOAD_BYTES * 2)) is True


def test_path_traversal_prevention():
    """Verifies that directory traversal attacks fail with 400 Bad Request."""
    malicious_ids = [
        "../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "../rcaeval",
        "incident/../../etc",
        "case;rm -rf /",
        "case*glob"
    ]
    for bad_id in malicious_ids:
        with pytest.raises(HTTPException) as exc_info:
            _validate_case_id(bad_id)
        assert exc_info.value.status_code in [400, 404]


def test_rbac_constant_time_enforcement(monkeypatch):
    """Verifies RBAC enforcement with multiple configured keys and constant-time checking."""
    monkeypatch.setattr("backend.app.API_KEYS_ROLES", {
        "viewer-key-999": "viewer",
        "operator-key-888": "operator",
        "admin-key-777": "admin"
    })

    # Test viewer key trying operator endpoint
    checker_op = require_role("operator")
    with pytest.raises(HTTPException) as exc_op:
        checker_op(x_api_key="viewer-key-999")
    assert exc_op.value.status_code == 403

    # Test operator key on operator endpoint
    assert checker_op(x_api_key="operator-key-888") == "operator"

    # Test invalid key
    with pytest.raises(HTTPException) as exc_unauth:
        checker_op(x_api_key="completely-fake-key")
    assert exc_unauth.value.status_code == 401
