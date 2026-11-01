"""Deep unit tests for backend/postmortem.py."""

from backend.postmortem import generate_incident_postmortem


def test_generate_incident_postmortem_formatting():
    """Verifies markdown structure of the generated SRE post-mortem report."""
    incident = {
        "case_id": "test-inc-42",
        "service": "frontend",
        "metric": "cpu_utilization",
        "model": "LSTM Autoencoder",
        "confidence": 92.5,
        "score": 0.08,
        "threshold": 0.02,
        "suspect_services": [
            {"service": "frontend", "attribution_share_pct": 92.5, "calibrated_probability_pct": 89.0}
        ],
        "top_contributing_metrics": [
            {"metric": "frontend_cpu_utilization", "service": "frontend", "anomaly_contribution": 0.08}
        ],
        "logs_summary": {
            "has_logs": True,
            "error_logs_count": 14,
            "error_rate_pct": 12.5,
            "service_error_counts": {"frontend": 14},
            "sample_errors": [{"service": "frontend", "message": "Connection refused to backend"}]
        },
        "traces_summary": {
            "has_traces": True,
            "p95_duration_ms": 250.0,
            "p99_duration_ms": 450.0,
            "slow_services": [
                {"service": "frontend", "slow_spans": 5, "avg_duration_ms": 280.0, "max_duration_ms": 500.0}
            ]
        }
    }

    report = generate_incident_postmortem(incident)
    assert "# [CRITICAL ALERT] SRE INCIDENT POST-MORTEM REPORT" in report
    assert "test-inc-42" in report
    assert "SEV-1 (Critical Telemetry Anomaly)" in report
    assert "frontend" in report
    assert "Immediate Remediation & SRE Runbook Actions" in report
    assert "kubectl rollout restart deployment/frontend" in report
    assert "Connection refused to backend" in report
    assert "250.0ms" in report


def test_generate_incident_postmortem_fallback_defaults():
    """Verifies graceful fallbacks when log/trace summaries are empty or missing."""
    incident = {
        "case_id": "live-case-101",
        "service": "cartservice",
        "confidence": 88.0,
        "overall_score": 0.065
    }

    report = generate_incident_postmortem(incident)
    assert "live-case-101" in report
    assert "cartservice" in report
    assert "88.0%" in report
    assert "kubectl describe pods -l app=cartservice" in report
    assert "No discrete error logs in window" in report
