"""Unit and Integration Tests for Sentinel AI REST Endpoints and Model Scoring."""

import pytest
from backend.app import health, summary, score, ScoreRequest, generate_postmortem, PostMortemRequest
from backend.scoring import score_rows


def test_health():
    """Verifies that the /api/v1/health endpoint returns an 'ok' status and service identity."""
    res = health()
    assert res["status"] == "ok"
    assert res["service"] == "sentinel-ai"
    assert "version" in res


def test_summary():
    """Verifies dataset discovery reporting and pre-trained model readiness status."""
    res = summary()
    assert res["dataset"] == "RCAEval"
    assert "available_models" in res
    assert res["available_models"]["isolation_forest"] is True
    assert res["available_models"]["lstm_autoencoder"] is True


def test_scoring_isolation_forest():
    """Tests batch scoring and Root Cause Analysis using the Isolation Forest artifact."""
    req = ScoreRequest(rows=[{"adservice_cpu": 0.85, "cartservice_cpu": 0.2}], model="iforest")
    res = score(req)
    assert "scores" in res
    assert "root_cause_analysis" in res
    assert "suspect_services" in res["root_cause_analysis"]
    assert len(res["root_cause_analysis"]["suspect_services"]) > 0


def test_scoring_lstm():
    """Tests temporal sequence reconstruction and anomaly flag evaluation via LSTM Autoencoder."""
    req = ScoreRequest(rows=[{"adservice_cpu": 0.85, "cartservice_cpu": 0.2}], model="lstm")
    res = score(req)
    assert "scores" in res
    assert "root_cause_analysis" in res
    assert "anomaly_detected" in res
    assert isinstance(res["anomaly_detected"], bool)


def test_postmortem_generation():
    """Tests automated SRE post-mortem markdown report synthesis."""
    req = PostMortemRequest(
        case_id="re1ob_adservice_cpu_1",
        model="LSTM Autoencoder",
        overall_score=0.92,
        suspect_services=[{"service": "adservice", "confidence_pct": 88.5}]
    )
    res = generate_postmortem(req)
    assert "report_markdown" in res
    assert "# [CRITICAL ALERT] SRE INCIDENT POST-MORTEM REPORT" in res["report_markdown"]
    assert "adservice" in res["report_markdown"]


def test_scoring_empty_fails():
    """Ensures that passing an empty list of rows raises an exception."""
    with pytest.raises(Exception):
        score_rows([])
