"""Unit Tests for Multimodal Log, Trace, and Cross-Signal Fusion Processing."""

from pathlib import Path

from ml_engine.multimodal import ERROR_PATTERN, analyze_logs, analyze_traces, fuse_multimodal_evidence


def test_regex_error_pattern_matching():
    """Verifies that ERROR_PATTERN matches typical microservice log error signatures."""
    assert ERROR_PATTERN.search("Connection timeout to checkoutservice:50051") is not None
    assert ERROR_PATTERN.search("HTTP 500 Internal Server Error") is not None
    assert ERROR_PATTERN.search("FATAL: OutOfMemoryError killed process") is not None
    assert ERROR_PATTERN.search("Normal health check 200 OK") is None


def test_analyze_logs_empty_or_missing(tmp_path: Path):
    """Verifies that missing log files return safe defaults with has_logs=False."""
    result = analyze_logs(tmp_path)
    assert result["has_logs"] is False
    assert result["log_anomaly_score"] == 0.0
    assert result["error_logs_count"] == 0


def test_analyze_traces_empty_or_missing(tmp_path: Path):
    """Verifies that missing trace files return safe defaults with has_traces=False."""
    result = analyze_traces(tmp_path)
    assert result["has_traces"] is False
    assert result["trace_anomaly_score"] == 0.0
    assert result["total_spans"] == 0


def test_fuse_multimodal_evidence_weights(tmp_path: Path):
    """Verifies that fuse_multimodal_evidence correctly weights signals when only metrics exist."""
    fake_metric_rca = {
        "suspect_services": [
            {"service": "checkoutservice", "attribution_share_pct": 70.0},
            {"service": "frontend", "attribution_share_pct": 30.0}
        ]
    }
    fused = fuse_multimodal_evidence(tmp_path, metric_score=0.80, metric_rca=fake_metric_rca)
    assert fused["fused_incident_score"] == 0.80
    assert len(fused["multimodal_suspect_services"]) >= 1
    assert fused["multimodal_suspect_services"][0]["service"] == "checkoutservice"

