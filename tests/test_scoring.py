"""Unit Tests for Telemetry Anomaly Scoring and Windowing Logic."""

import numpy as np
import pytest

from backend.scoring import WINDOW_SIZE, _aggregate_root_cause, score_rows


def test_sliding_window_enforcement_short_sequence():
    """Verifies that sequences shorter than 30 are padded to 30 timesteps matching training."""
    short_input = [{"adservice_cpu": 0.5, "cartservice_cpu": 0.1}] * 5
    result = score_rows(short_input, model_type="lstm")
    assert len(result["scores"]) == 5
    assert result["window_size_used"] == WINDOW_SIZE
    assert "root_cause_analysis" in result


def test_sliding_window_enforcement_long_sequence():
    """Verifies that sequences longer than 30 generate sliding windows of size 30 with 1:1 score mapping."""
    long_input = [{"adservice_cpu": 0.5 + (i * 0.01), "cartservice_cpu": 0.2} for i in range(45)]
    result = score_rows(long_input, model_type="lstm")
    assert len(result["scores"]) == 45
    assert result["window_size_used"] == WINDOW_SIZE


def test_attribution_share_pct_sums_to_100():
    """Ensures attribution_share_pct normalizes properly across microservices to ~100%."""
    features = ["adservice_cpu", "adservice_mem", "cartservice_cpu", "paymentservice_latency"]
    deviations = np.array([10.0, 5.0, 15.0, 20.0])
    rca = _aggregate_root_cause(features, deviations)

    shares = [s["attribution_share_pct"] for s in rca["suspect_services"]]
    total_share = sum(shares)
    assert 99.0 <= total_share <= 101.0
    assert rca["suspect_services"][0]["service"] == "paymentservice"


def test_missing_feature_imputation():
    """Verifies that missing feature columns are imputed without crashing the model."""
    partial_input = [{"unknown_custom_metric": 42.0, "adservice_cpu": 0.9}]
    result = score_rows(partial_input, model_type="iforest")
    assert len(result["scores"]) == 1
    assert "suspect_services" in result["root_cause_analysis"]


def test_fallback_when_invalid_rows():
    """Ensures empty rows raise a ValueError."""
    with pytest.raises(ValueError):
        score_rows([])

