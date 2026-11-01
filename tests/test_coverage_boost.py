"""Coverage boost tests for ml_engine data pipeline, drift edge cases, cache, and multimodal."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.cache import cache_get, cache_set, check_rate_limit, clear_cache
from ml_engine.data import (
    discover_cases,
    download_dataset,
    inspect_dataset,
    prepare_dataset,
    read_metrics,
)
from ml_engine.drift import compute_psi, monitor_telemetry_drift
from ml_engine.multimodal import analyze_logs, analyze_traces, fuse_multimodal_evidence


def test_download_dataset_invalid_suite(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid suite"):
        download_dataset(suite="invalid_suite", output_dir=tmp_path)


def test_discover_cases_nonexistent():
    with pytest.raises(FileNotFoundError):
        discover_cases(Path("non_existent_path_xyz_123"))


def test_read_metrics_missing_time(tmp_path: Path):
    case = tmp_path / "case_bad"
    case.mkdir()
    (case / "metrics.json").write_text(json.dumps([{"not_time": 1, "cpu": 50}]), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required 'time' column"):
        read_metrics(case)


def test_read_metrics_missing_file(tmp_path: Path):
    case = tmp_path / "case_empty"
    case.mkdir()
    with pytest.raises(FileNotFoundError):
        read_metrics(case)


def test_inspect_dataset(tmp_path: Path):
    c1 = tmp_path / "re1ob_case1"
    c1.mkdir()
    (c1 / "metrics.json").write_text(json.dumps([
        {"time": 1, "cpu": 10},
        {"time": 2, "cpu": 15}
    ]), encoding="utf-8")
    (c1 / "logs.csv").write_text("timestamp,message\n1,error", encoding="utf-8")

    c_bad = tmp_path / "re1ob_case_bad"
    c_bad.mkdir()
    (c_bad / "metrics.json").write_text(json.dumps([{"no_time": 1}]), encoding="utf-8")

    info = inspect_dataset(tmp_path)
    assert info["case_count"] == 1
    assert len(info["errors"]) == 1


def test_prepare_dataset_validation(tmp_path: Path):
    out = tmp_path / "proc"
    with pytest.raises(ValueError, match="Invalid window parameters"):
        prepare_dataset(tmp_path, out, window_size=1, stride=1)
    with pytest.raises(ValueError, match="At least 3"):
        prepare_dataset(tmp_path, out, window_size=5, stride=2, pattern="*")


def test_compute_psi_edge_cases():
    # Identical distributions should have near zero PSI
    arr = np.random.normal(0, 1, 100)
    assert compute_psi(arr, arr) < 0.05

    # Small or empty arrays
    assert compute_psi(np.array([1.0]), np.array([2.0])) >= 0.0


def test_monitor_telemetry_drift_fallback():
    frame = pd.DataFrame({"cpu": [10.0, 12.0, 11.0], "mem": [20.0, 22.0, 21.0]})
    # Without model artifacts, monitor_telemetry_drift should return fallback payload
    res = monitor_telemetry_drift(frame)
    assert "status" in res or "overall_psi" in res or "drift_detected" in res

    # Empty payload
    empty_res = monitor_telemetry_drift(pd.DataFrame())
    assert empty_res["drift_detected"] is False


def test_multimodal_analysis_missing_files(tmp_path: Path):
    empty_case = tmp_path / "empty_case"
    empty_case.mkdir()

    log_res = analyze_logs(empty_case)
    assert log_res["has_logs"] is False
    assert log_res["total_logs"] == 0
    assert log_res["error_rate_pct"] == 0.0

    trace_res = analyze_traces(empty_case)
    assert trace_res["has_traces"] is False
    assert trace_res["total_spans"] == 0


def test_multimodal_analysis_with_data(tmp_path: Path):
    case = tmp_path / "case_with_data"
    case.mkdir()

    # Create dummy logs parquet
    logs_df = pd.DataFrame({
        "message": [
            "2026-05-01 ERROR Connection refused to checkoutservice",
            "2026-05-01 INFO Handled GET /products",
            "CrashLoopBackOff in pod frontend-xyz"
        ],
        "service": ["checkoutservice", "frontend", "frontend"]
    })
    logs_df.to_parquet(case / "logs.parquet")

    # Create dummy traces parquet
    traces_df = pd.DataFrame({
        "service_name": ["frontend", "frontend", "cart"],
        "duration": [50.0, 800.0, 10.0],
        "status_code": ["OK", "ERROR", "OK"]
    })
    traces_df.to_parquet(case / "traces.parquet")

    log_res = analyze_logs(case)
    assert log_res["has_logs"] is True
    assert log_res["total_logs"] == 3
    assert log_res["error_logs_count"] >= 1

    trace_res = analyze_traces(case)
    assert trace_res["has_traces"] is True
    assert trace_res["total_spans"] == 3

    dummy_rca = {"ranked_services": [{"service": "frontend", "attribution_share_pct": 80.0}]}
    fused = fuse_multimodal_evidence(case, metric_score=0.9, metric_rca=dummy_rca)
    assert fused["fused_incident_score"] > 0
    assert "multimodal_suspect_services" in fused
    assert fused["telemetry_coverage"]["logs"] is True


def test_cache_memory_ttl_and_rate_limit():
    clear_cache()
    assert cache_get("missing_key") is None
    cache_set("test_key", {"data": 123}, ttl_seconds=1)
    cached = cache_get("test_key")
    assert cached == {"data": 123}

    # Test rate limiter
    for _ in range(5):
        assert check_rate_limit("1.2.3.4", max_requests=10, window_seconds=60) is True
    # Exceed limit
    assert check_rate_limit("1.2.3.4", max_requests=5, window_seconds=60) is False
