"""Tests for OpenTelemetry (OTel) metrics and traces ingestion endpoints."""

from backend.app import OTelMetricsPayload, OTelTracesPayload, ingest_otel_metrics, ingest_otel_traces


def test_otel_metrics_ingestion():
    """Verifies parsing of standard OTLP JSON metrics into model scoring and RCA."""
    otlp_payload = OTelMetricsPayload(
        resourceMetrics=[
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "cartservice"}}
                    ]
                },
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": "cpu_utilization",
                                "gauge": {
                                    "dataPoints": [
                                        {"timeUnixNano": 1700000001000, "asDouble": 0.95},
                                        {"timeUnixNano": 1700000002000, "asDouble": 0.98}
                                    ]
                                }
                            }
                        ]
                    }
                ]
            }
        ],
        model="lstm"
    )

    res = ingest_otel_metrics(otlp_payload, role="operator")
    assert res["status"] == "ingested_and_scored"
    assert res["data_points_processed"] == 2
    assert "overall_score" in res
    assert "root_cause_analysis" in res


def test_otel_traces_ingestion():
    """Verifies distributed traces span latency profiling and bottleneck localization."""
    trace_payload = OTelTracesPayload(
        resourceSpans=[
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "checkoutservice"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {"durationNano": 120_000_000, "status": {"code": 0}},  # 120ms
                            {"durationNano": 850_000_000, "status": {"code": 2}},  # 850ms error
                            {"durationNano": 250_000_000, "status": {"code": 0}}   # 250ms
                        ]
                    }
                ]
            }
        ]
    )

    res = ingest_otel_traces(trace_payload, role="operator")
    assert res["status"] == "traces_analyzed"
    assert res["spans_processed"] == 3
    assert res["error_spans"] == 1
    assert res["p95_latency_ms"] > 0
    assert res["primary_bottleneck"]["service"] == "checkoutservice"
