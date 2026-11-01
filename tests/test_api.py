"""Unit and Integration Tests for Sentinel AI REST Endpoints and Security Controls."""

import pytest
from fastapi import HTTPException

from backend.app import (
    PostMortemRequest,
    ScoreRequest,
    _validate_case_id,
    generate_postmortem,
    get_case_telemetry,
    get_models,
    health,
    list_cases,
    score,
    summary,
    verify_api_key,
)
from backend.scoring import score_rows


def test_health():
    """Verifies that the /api/v1/health endpoint returns an 'healthy' status and service identity."""
    res = health()
    assert res["status"] == "healthy"
    assert res["service"] == "SENTINEL-AI"
    assert "version" in res
    assert "security" in res


def test_summary():
    """Verifies dataset discovery reporting and pre-trained model readiness status."""
    res = summary()
    assert "dataset_root" in res
    assert "discovered_cases" in res
    assert "models" in res
    assert res["models"]["isolation_forest"] is True
    assert res["models"]["lstm_autoencoder"] is True


def test_model_benchmarks():
    """Verifies that model benchmark reports are properly exposed."""
    res = get_models()
    assert res["status"] == "success"
    summary_data = res["benchmark_summary"]
    assert "isolation_forest" in summary_data
    assert "lstm_autoencoder" in summary_data
    assert summary_data["lstm_autoencoder"]["test_metrics"]["f1"] >= 0.90


def test_cases_listing():
    """Verifies that list_cases discovers benchmark cases."""
    res = list_cases(limit=10)
    assert "cases" in res
    assert len(res["cases"]) > 0
    assert "id" in res["cases"][0]


def test_case_telemetry_valid():
    """Verifies that fetching an existing case returns telemetry rows."""
    res = list_cases(limit=5)
    cases = res.get("cases", [])
    if cases:
        valid_case_id = cases[0]["id"]
        case_data = get_case_telemetry(valid_case_id)
        assert case_data["case_id"] == valid_case_id
        assert "telemetry" in case_data
        assert len(case_data["telemetry"]) > 0


def test_path_traversal_attack_dot_dot_fails():
    """Ensures directory traversal attacks with '..' are rejected with HTTP 400."""
    with pytest.raises(HTTPException) as exc_info:
        _validate_case_id("../../../etc/passwd")
    assert exc_info.value.status_code == 400


def test_path_traversal_attack_special_chars_fails():
    """Ensures invalid case_id characters are rejected with HTTP 400."""
    with pytest.raises(HTTPException) as exc_info:
        _validate_case_id("invalid;rm -rf /")
    assert exc_info.value.status_code == 400


def test_case_not_found_returns_404():
    """Ensures non-existent but syntactically valid case names return 404."""
    with pytest.raises(HTTPException) as exc_info:
        _validate_case_id("non_existent_case_9999")
    assert exc_info.value.status_code == 404


def test_scoring_isolation_forest():
    """Tests batch scoring and Root Cause Analysis using the Isolation Forest artifact."""
    req = ScoreRequest(rows=[{"adservice_cpu": 0.85, "cartservice_cpu": 0.2}], model="iforest")
    res = score(req, role="operator")
    assert "scores" in res
    assert "root_cause_analysis" in res
    assert "suspect_services" in res["root_cause_analysis"]
    assert len(res["root_cause_analysis"]["suspect_services"]) > 0


def test_scoring_lstm():
    """Tests temporal sequence reconstruction and anomaly flag evaluation via LSTM Autoencoder."""
    req = ScoreRequest(rows=[{"adservice_cpu": 0.85, "cartservice_cpu": 0.2}], model="lstm")
    res = score(req, role="operator")
    assert "scores" in res
    assert "root_cause_analysis" in res
    assert "anomaly_detected" in res
    assert isinstance(res["anomaly_detected"], bool)
    assert "attribution_share_pct" in res["root_cause_analysis"]["suspect_services"][0]
    assert "calibrated_probability_pct" in res["root_cause_analysis"]["suspect_services"][0]


def test_postmortem_generation():
    """Tests automated SRE post-mortem markdown report synthesis."""
    req = PostMortemRequest(
        case_id="re1ob_adservice_cpu_1",
        service="adservice",
        metric="cpu_usage",
        model="LSTM Autoencoder",
        confidence=88.5,
        score=0.92,
        threshold=0.018
    )
    res = generate_postmortem(req, role="operator")
    assert "report_markdown" in res
    assert "# [CRITICAL ALERT] SRE INCIDENT POST-MORTEM REPORT" in res["report_markdown"]
    assert "adservice" in res["report_markdown"]


def test_scoring_empty_fails():
    """Ensures that passing an empty list of rows raises an exception."""
    with pytest.raises((ValueError, IndexError)):
        score_rows([])


def test_api_key_auth_verification():
    """Verifies that API key verification succeeds when unconfigured."""
    # When SENTINEL_API_KEY is not set or empty, verify_api_key must pass
    role = verify_api_key(None)
    assert role in {"admin", "viewer"}
