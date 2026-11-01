"""Multimodal Telemetry Fusion Engine for Incident Detection and Triage.

This module unifies multiple observability signals:
1. Container Error Log Mining: Identifies error bursts, exception stack traces,
   and HTTP 5xx responses using regular expression pattern matching and container attribution.
2. Distributed Trace Span Profiling: Evaluates RPC durations across the microservice call
   graph, calculating P95/P99 latency percentiles and identifying slow parent/child spans.
3. Multi-Signal Fusion Scoring: Formulates an integrated incident severity score:
       S_fused = w_m * S_metric + w_l * S_log + w_t * S_trace
   and cross-validates microservice attribution across all telemetry modalities.
"""

from pathlib import Path
import re
from typing import Any
import pandas as pd

# Regular expression matching critical error signatures in microservice container logs
ERROR_PATTERN = re.compile(
    r"(?:error|exception|fail|timeout|timed out|refused|crash|fatal|killed|unhandled|500|502|503|504|oom|panic)",
    re.IGNORECASE
)


def analyze_logs(case_dir: Path, max_samples: int = 5) -> dict[str, Any]:
    """Analyzes container logs from `logs.parquet` for error bursts and service attribution.

    Filters container logs for critical exception patterns, aggregates counts per
    microservice container, and computes a normalized log anomaly score in [0.0, 1.0].

    Args:
        case_dir: Directory path containing incident telemetry.
        max_samples: Maximum number of representative error log messages to return.

    Returns:
        dict[str, Any]: Dictionary containing:
            - has_logs: Boolean indicating presence of log file.
            - log_anomaly_score: Normalized severity score based on error volume and ratio.
            - total_logs: Count of log lines in the evaluation window.
            - error_logs_count: Total detected error events.
            - error_rate_pct: Percentage of log messages that matched error patterns.
            - service_error_counts: Frequency mapping per container/service.
            - sample_errors: List of sample error log dictionaries.
    """
    log_file = case_dir / "logs.parquet"
    if not log_file.exists():
        return {
            "has_logs": False,
            "log_anomaly_score": 0.0,
            "total_logs": 0,
            "error_logs_count": 0,
            "error_rate_pct": 0.0,
            "service_error_counts": {},
            "sample_errors": []
        }

    try:
        df = pd.read_parquet(log_file)
        if df.empty or "message" not in df.columns:
            return {
                "has_logs": True,
                "log_anomaly_score": 0.0,
                "total_logs": 0,
                "error_logs_count": 0,
                "error_rate_pct": 0.0,
                "service_error_counts": {},
                "sample_errors": []
            }

        df["message_str"] = df["message"].astype(str)
        is_error = df["message_str"].str.contains(ERROR_PATTERN)
        error_df = df[is_error]

        service_col = "container_name" if "container_name" in df.columns else "service"
        service_error_counts: dict[str, int] = {}
        if service_col in error_df.columns:
            service_error_counts = error_df[service_col].value_counts().to_dict()

        total = len(df)
        err_count = len(error_df)
        err_ratio = err_count / max(total, 1)

        # Log anomaly score scales with raw error count and percentage
        log_anomaly_score = float(min(1.0, (err_count / 20.0) + (err_ratio * 2.0)))

        samples: list[dict[str, str]] = []
        for _, row in error_df.head(max_samples).iterrows():
            samples.append({
                "service": str(row.get(service_col, "unknown")),
                "timestamp": str(row.get("timestamp", "")),
                "message": str(row.get("message_str", ""))[:180]
            })

        return {
            "has_logs": True,
            "log_anomaly_score": round(log_anomaly_score, 4),
            "total_logs": total,
            "error_logs_count": err_count,
            "error_rate_pct": round(err_ratio * 100, 2),
            "service_error_counts": {str(k): int(v) for k, v in service_error_counts.items()},
            "sample_errors": samples
        }
    except Exception as exc:
        return {"has_logs": True, "error": str(exc), "log_anomaly_score": 0.0}


def analyze_traces(case_dir: Path, max_samples: int = 4) -> dict[str, Any]:
    """Analyzes distributed trace spans from `traces.parquet` for RPC latency degradation.

    Computes P95 and P99 latency percentiles across RPC spans and isolates microservices
    experiencing execution bottlenecks.

    Args:
        case_dir: Directory path containing incident telemetry.
        max_samples: Maximum number of bottleneck services to report.

    Returns:
        dict[str, Any]: Dictionary containing:
            - has_traces: Boolean indicating presence of trace spans.
            - trace_anomaly_score: Normalized latency degradation score.
            - total_spans: Total distributed trace spans evaluated.
            - p95_duration_ms: 95th percentile span latency in milliseconds.
            - p99_duration_ms: 99th percentile span latency in milliseconds.
            - slow_services: List of services ranked by average span execution duration.
    """
    trace_file = case_dir / "traces.parquet"
    if not trace_file.exists():
        return {
            "has_traces": False,
            "trace_anomaly_score": 0.0,
            "total_spans": 0,
            "slow_services": []
        }

    try:
        df = pd.read_parquet(trace_file)
        if df.empty or "duration" not in df.columns:
            return {
                "has_traces": True,
                "trace_anomaly_score": 0.0,
                "total_spans": 0,
                "slow_services": []
            }

        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
        p95 = float(df["duration"].quantile(0.95))
        p99 = float(df["duration"].quantile(0.99))

        service_col = "serviceName" if "serviceName" in df.columns else "service"
        slow_df = df[df["duration"] >= p95]

        slow_services: list[dict[str, Any]] = []
        if service_col in slow_df.columns:
            grouped = slow_df.groupby(service_col)["duration"].agg(["count", "mean", "max"]).reset_index()
            grouped = grouped.sort_values(by="mean", ascending=False)
            for _, r in grouped.head(max_samples).iterrows():
                slow_services.append({
                    "service": str(r[service_col]),
                    "slow_spans": int(r["count"]),
                    "avg_duration_ms": round(float(r["mean"]) / 1000.0, 1),
                    "max_duration_ms": round(float(r["max"]) / 1000.0, 1)
                })

        # Duration in RCAEval spans is recorded in microseconds (100,000us = 100ms)
        trace_anomaly_score = float(min(1.0, (p99 / 1000.0) / 2000.0)) if p99 > 500000 else 0.1

        return {
            "has_traces": True,
            "trace_anomaly_score": round(trace_anomaly_score, 4),
            "total_spans": len(df),
            "p95_duration_ms": round(p95 / 1000.0, 1),
            "p99_duration_ms": round(p99 / 1000.0, 1),
            "slow_services": slow_services
        }
    except Exception as exc:
        return {"has_traces": True, "error": str(exc), "trace_anomaly_score": 0.0}


def fuse_multimodal_evidence(
    case_dir: Path,
    metric_score: float,
    metric_rca: dict[str, Any]
) -> dict[str, Any]:
    """Combines metrics, container logs, and distributed traces into a unified triage assessment.

    Weights multi-signal inputs:
    - 50% Metric Reconstruction Loss
    - 30% Container Error Log Frequency
    - 20% Distributed Trace RPC Latency

    Cross-validates suspect microservices across signals to produce robust attribution confidence.

    Args:
        case_dir: Incident case directory containing telemetry files.
        metric_score: Anomaly score produced by the primary time-series model.
        metric_rca: Root Cause Analysis dictionary from metric scoring.

    Returns:
        dict[str, Any]: Fused incident score, telemetry modality coverage,
        multimodal ranked suspect services, and discrete signal summaries.
    """
    logs_info = analyze_logs(case_dir)
    traces_info = analyze_traces(case_dir)

    log_score = logs_info.get("log_anomaly_score", 0.0)
    trace_score = traces_info.get("trace_anomaly_score", 0.0)

    has_logs = logs_info.get("has_logs", False)
    has_traces = traces_info.get("has_traces", False)

    # Weighted multimodal fusion formula
    if has_logs and has_traces:
        fused_score = 0.50 * metric_score + 0.30 * log_score + 0.20 * trace_score
    elif has_logs:
        fused_score = 0.65 * metric_score + 0.35 * log_score
    else:
        fused_score = metric_score

    # Multi-signal service attribution aggregation
    fused_attribution: dict[str, float] = {}

    # 1. Metric contribution (50% weight)
    for item in metric_rca.get("suspect_services", []):
        srv = item["service"]
        share = item.get("attribution_share_pct", item.get("confidence_pct", 50.0))
        fused_attribution[srv] = fused_attribution.get(srv, 0.0) + (share * 0.5)

    # 2. Log error burst contribution (35% weight)
    service_errs = logs_info.get("service_error_counts", {})
    total_errs = sum(service_errs.values()) or 1
    for srv, count in service_errs.items():
        fused_attribution[srv] = fused_attribution.get(srv, 0.0) + ((count / total_errs) * 35.0)

    # 3. Trace latency bottleneck contribution (15% weight)
    for srv_info in traces_info.get("slow_services", []):
        srv = srv_info["service"]
        fused_attribution[srv] = fused_attribution.get(srv, 0.0) + 15.0

    # Normalize fused attribution share into percentages
    total_conf = sum(fused_attribution.values()) or 1.0
    ranked_fused_services = sorted(
        [
            {
                "service": srv,
                "attribution_share_pct": round((val / total_conf) * 100, 1),
                "confidence_pct": round((val / total_conf) * 100, 1),  # Backward compatibility alias
                "has_metric_anomaly": any(x["service"] == srv for x in metric_rca.get("suspect_services", [])),
                "log_errors": service_errs.get(srv, 0)
            }
            for srv, val in fused_attribution.items()
        ],
        key=lambda x: x["attribution_share_pct"],
        reverse=True
    )

    return {
        "fused_incident_score": round(float(fused_score), 4),
        "telemetry_coverage": {
            "metrics": True,
            "logs": has_logs,
            "traces": has_traces
        },
        "multimodal_suspect_services": ranked_fused_services[:3],
        "logs_summary": logs_info,
        "traces_summary": traces_info
    }
