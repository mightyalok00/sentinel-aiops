"""Tests for Prometheus /metrics telemetry exporter."""

from backend.app import prometheus_metrics


def test_prometheus_metrics_format():
    """Verifies that Prometheus metrics endpoint returns standard OpenMetrics text format."""
    response = prometheus_metrics()
    assert response.status_code == 200
    assert response.media_type.startswith("text/plain")

    body = response.body.decode("utf-8")
    assert "# HELP sentinel_requests_total" in body
    assert "# TYPE sentinel_requests_total counter" in body
    assert "sentinel_anomalies_detected_total{model=\"lstm\"}" in body
    assert "sentinel_inference_latency_seconds_sum" in body
    assert "sentinel_system_info{version=\"2.7.0\"" in body
