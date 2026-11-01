"""Tests for Role-Based Access Control (RBAC) and permission boundaries."""

import pytest
from fastapi import HTTPException

from backend.app import require_role


def test_rbac_default_allow_when_no_keys_configured(monkeypatch):
    """When API_KEYS_ROLES is empty, defaults to admin role for seamless local dev."""
    monkeypatch.setattr("backend.app.API_KEYS_ROLES", {})
    checker = require_role("operator")
    role = checker(None)
    assert role == "admin"


def test_rbac_rejects_missing_key_when_keys_configured(monkeypatch):
    """When API keys are configured, missing header must raise 401 Unauthorized."""
    monkeypatch.setattr("backend.app.API_KEYS_ROLES", {"key-1": "viewer", "key-2": "admin"})
    checker = require_role("viewer")
    with pytest.raises(HTTPException) as exc:
        checker(None)
    assert exc.value.status_code == 401


def test_rbac_rejects_insufficient_privilege(monkeypatch):
    """A viewer key attempting to access an operator endpoint must receive 403 Forbidden."""
    monkeypatch.setattr("backend.app.API_KEYS_ROLES", {"viewer-key": "viewer", "admin-key": "admin"})
    checker = require_role("operator")
    with pytest.raises(HTTPException) as exc:
        checker("viewer-key")
    assert exc.value.status_code == 403


def test_rbac_permits_equal_or_higher_privilege(monkeypatch):
    """Admin key should satisfy viewer and operator requirements."""
    monkeypatch.setattr("backend.app.API_KEYS_ROLES", {"admin-key": "admin"})
    checker_op = require_role("operator")
    assert checker_op("admin-key") == "admin"
    checker_view = require_role("viewer")
    assert checker_view("admin-key") == "admin"
