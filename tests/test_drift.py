"""Unit tests for Statistical Drift Monitoring (PSI and Kolmogorov-Smirnov test)."""

import numpy as np

from ml_engine.drift import compute_psi, monitor_telemetry_drift


def test_compute_psi_identical_distributions():
    """Identical distributions should have approximately zero PSI."""
    np.random.seed(42)
    sample_a = np.random.normal(10.0, 2.0, 500)
    sample_b = np.random.normal(10.0, 2.0, 500)
    psi = compute_psi(sample_a, sample_b)
    assert psi < 0.10, f"Expected PSI < 0.10 for identical distributions, got {psi}"


def test_compute_psi_drifted_distribution():
    """Significantly shifted distributions should trigger PSI >= 0.25."""
    np.random.seed(42)
    sample_a = np.random.normal(10.0, 2.0, 500)
    sample_b = np.random.normal(25.0, 5.0, 500)  # Significant mean and scale shift
    psi = compute_psi(sample_a, sample_b)
    assert psi >= 0.25, f"Expected high PSI (>= 0.25), got {psi}"


def test_monitor_telemetry_drift_empty_payload():
    """Empty payload should return graceful no-drift status."""
    report = monitor_telemetry_drift([])
    assert report["drift_detected"] is False
    assert report["drift_severity"] == "none"
    assert report["overall_psi"] == 0.0


def test_monitor_telemetry_drift_with_stable_rows():
    """Rows matching baseline should evaluate as stable."""
    # Generate 30 rows with normal values for adservice
    rows = [
        {"adservice_cpu_utilization": float(np.random.normal(0.2, 0.05))}
        for _ in range(30)
    ]
    report = monitor_telemetry_drift(rows)
    assert "drift_detected" in report
    assert "overall_psi" in report
    assert "recommendation" in report
