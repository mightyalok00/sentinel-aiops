"""FastAPI Application for SENTINEL AI.

Production-grade AIOps platform providing:
- RESTful telemetry anomaly scoring and Root Cause Analysis (PyTorch LSTM + Isolation Forest)
- OpenTelemetry (OTel) Metrics & Distributed Traces ingestion endpoints
- Prometheus `/metrics` exporter in standard OpenMetrics format
- Statistical Distribution Drift Monitoring (Population Stability Index + KS test)
- Multimodal evidence fusion (Metrics + Container Logs + Distributed Traces)
- Automated SRE Incident Post-Mortem report generation with Kubernetes runbooks
- Role-Based Access Control (RBAC: viewer, operator, admin)
- Structured JSON logging and token-bucket sliding-window rate limiting
- Secure path traversal validation on benchmark incident directories
"""

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd  # type: ignore # pyrefly: ignore
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.cache import cache_get, cache_set, check_rate_limit
from backend.postmortem import generate_incident_postmortem
from backend.scoring import score_rows
from backend.security import (
    MAX_PAYLOAD_BYTES,
    add_security_headers,
    constant_time_compare,
    is_payload_too_large,
    sanitize_log_dict,
)
from ml_engine.drift import monitor_telemetry_drift
from ml_engine.multimodal import analyze_logs, analyze_traces, fuse_multimodal_evidence

ROOT = Path(__file__).resolve().parents[1]
CASE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")

# Role definitions & access levels
ROLE_LEVELS: dict[str, int] = {
    "viewer": 1,
    "operator": 2,
    "admin": 3
}

# Optional configured API Keys mapping: key -> role
SENTINEL_API_KEY = os.getenv("SENTINEL_API_KEY", "")
SENTINEL_API_KEYS_RAW = os.getenv("SENTINEL_API_KEYS", "")
API_KEYS_ROLES: dict[str, str] = {}

if SENTINEL_API_KEYS_RAW:
    try:
        API_KEYS_ROLES = json.loads(SENTINEL_API_KEYS_RAW)
    except Exception:
        pass
elif SENTINEL_API_KEY:
    API_KEYS_ROLES[SENTINEL_API_KEY] = "admin"

# Prometheus metrics state counters
_METRICS_STATE: dict[str, Any] = {
    "requests_total": {},  # (endpoint, status) -> count
    "anomalies_detected": {"lstm": 0, "iforest": 0},
    "inference_latencies": {"lstm": [], "iforest": []},
    "drift_checks": {"stable": 0, "moderate": 0, "critical": 0},
    "cache_hits": 0,
    "cache_misses": 0
}

logger = logging.getLogger("sentinel.app")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(
    title="SENTINEL AI",
    version="2.7.0",
    description="Enterprise AIOps Multimodal Telemetry Anomaly Scoring, Root Cause Analysis & Autonomous Incident Triage Platform.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[orig.strip() for orig in os.getenv("ALLOWED_ORIGINS", "*").split(",") if orig.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Security, Path Traversal Prevention & Role-Based Access Control (RBAC)
# -----------------------------------------------------------------------------
def _validate_case_id(case_id: str) -> Path:
    """Validates case_id against regex patterns and verifies strict containment within data/rcaeval.

    Prevents directory traversal attacks (e.g. `../../etc/passwd`).

    Args:
        case_id: Directory name of the benchmark incident.

    Returns:
        Path: Resolved, verified filesystem Path object.

    Raises:
        HTTPException: 400 Bad Request on path traversal or invalid format; 404 if not found.
    """
    if not CASE_ID_PATTERN.match(case_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid case_id syntax. Only alphanumeric characters, hyphens, and underscores are allowed."
        )

    data_dir = (ROOT / "data" / "rcaeval").resolve()
    case_dir = (data_dir / case_id).resolve()

    # Enforce strict path containment
    if not str(case_dir).startswith(str(data_dir)):
        raise HTTPException(
            status_code=400,
            detail="Security violation: Path traversal outside dataset directory is prohibited."
        )

    if not case_dir.exists() or not case_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Incident case '{case_id}' not found.")

    return case_dir


def require_role(min_role: str = "viewer"):
    """Dependency factory enforcing Role-Based Access Control (RBAC).

    Roles:
    - viewer: Read-only telemetry, cases, summary, and dashboard.
    - operator: Triage incidents, run scoring, and generate post-mortems.
    - admin: High-privilege model retraining, configuration, and diagnostics.
    """
    def role_checker(x_api_key: Optional[str] = Header(None)) -> str:
        # If no API keys are configured, default to admin for development
        if not API_KEYS_ROLES:
            return "admin"

        assigned_role = None
        if x_api_key:
            for valid_key, role in API_KEYS_ROLES.items():
                if constant_time_compare(x_api_key, valid_key):
                    assigned_role = role
                    break

        if not assigned_role:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Missing or invalid 'X-API-Key' header."
            )

        if ROLE_LEVELS.get(assigned_role, 0) < ROLE_LEVELS.get(min_role, 1):
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: Insufficient privileges. Role '{min_role}' or higher is required."
            )

        return assigned_role

    return role_checker


# Backward-compatible alias
verify_api_key = require_role("viewer")


# -----------------------------------------------------------------------------
# Structured JSON Logging & Rate Limiting Middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def operational_middleware(request: Request, call_next: Any) -> Response:
    """Enforces token-bucket rate limits, payload size caps, security headers, and structured JSON telemetry."""
    # 1. Payload size limit
    content_length = request.headers.get("content-length")
    if is_payload_too_large(content_length):
        return JSONResponse(
            status_code=413,
            content={"detail": f"Payload too large. Maximum permitted request body is {MAX_PAYLOAD_BYTES // (1024 * 1024)} MB."}
        )

    client_ip = request.client.host if request.client else "127.0.0.1"
    path = request.url.path
    method = request.method
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    # 2. Rate limiting on compute-intensive inference endpoints
    if path.startswith("/api/v1/score") or path.startswith("/api/v1/postmortem") or path.startswith("/api/v1/otel"):
        if not check_rate_limit(client_ip, max_requests=120, window_seconds=60):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded: Maximum 120 requests per minute permitted."}
            )

    response: Response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    status_code = response.status_code

    # 3. Defensive OWASP security headers
    add_security_headers(response.headers)

    # 4. Record Prometheus request counter
    key = (path, str(status_code))
    _METRICS_STATE["requests_total"][key] = _METRICS_STATE["requests_total"].get(key, 0) + 1

    # 5. Structured JSON log entry (sanitizing any sensitive data)
    if not path.startswith("/static") and path != "/metrics":
        log_payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "client_ip": client_ip
        }
        logger.info(json.dumps(sanitize_log_dict(log_payload)))

    return response


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    """Schema for incoming telemetry anomaly scoring requests."""
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=10000, description="List of metric dictionaries across time steps.")
    model: str = Field(default="lstm", max_length=50, description="Inference engine: 'lstm' (LSTM Autoencoder) or 'iforest' (Isolation Forest).")
    threshold: float = Field(default=0.0, ge=0.0, le=1000.0, description="Optional custom anomaly threshold override; 0.0 uses calibrated value.")
    check_drift: bool = Field(default=True, description="Whether to evaluate statistical feature distribution drift.")


class DriftRequest(BaseModel):
    """Schema for telemetry distribution drift monitoring requests."""
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=10000, description="List of metric dictionaries to evaluate against baseline.")


class PostMortemRequest(BaseModel):
    """Schema for incident post-mortem generation requests."""
    case_id: str = Field(default="incident-live", max_length=100, description="Benchmark case ID or live incident tracking number.")
    service: str = Field(default="adservice", max_length=100, description="Suspect primary faulty microservice.")
    metric: str = Field(default="cpu_usage", max_length=100, description="Primary anomalous metric symptom.")
    model: str = Field(default="PyTorch LSTM Autoencoder", max_length=100, description="Model identifying the incident.")
    confidence: float = Field(default=95.2, ge=0.0, le=100.0, description="Attribution share percentage or confidence.")
    score: float = Field(default=0.045, ge=0.0, description="Overall telemetry reconstruction error or anomaly score.")
    threshold: float = Field(default=0.018, ge=0.0, description="Anomaly decision threshold.")


class OTelMetricsPayload(BaseModel):
    """Schema for standard OpenTelemetry (OTLP) JSON metric payloads."""
    resourceMetrics: list[dict[str, Any]] = Field(..., description="OTLP ResourceMetrics collection.")
    model: str = Field(default="lstm", description="Inference engine: 'lstm' or 'iforest'.")


class OTelTracesPayload(BaseModel):
    """Schema for standard OpenTelemetry (OTLP) JSON trace payloads."""
    resourceSpans: list[dict[str, Any]] = Field(..., description="OTLP ResourceSpans collection.")


# -----------------------------------------------------------------------------
# Prometheus Exporter Endpoint
# -----------------------------------------------------------------------------
@app.get("/metrics", tags=["Observability & Monitoring"])
def prometheus_metrics() -> Response:
    """Exposes platform telemetry in standard OpenMetrics / Prometheus text format."""
    lines: list[str] = [
        "# HELP sentinel_requests_total Total HTTP requests handled by endpoint and status.",
        "# TYPE sentinel_requests_total counter"
    ]
    for (endpoint, status), count in _METRICS_STATE["requests_total"].items():
        lines.append(f'sentinel_requests_total{{endpoint="{endpoint}",status="{status}"}} {count}')

    lines.extend([
        "# HELP sentinel_anomalies_detected_total Total anomalies flagged by ML model engine.",
        "# TYPE sentinel_anomalies_detected_total counter",
        f'sentinel_anomalies_detected_total{{model="lstm"}} {_METRICS_STATE["anomalies_detected"]["lstm"]}',
        f'sentinel_anomalies_detected_total{{model="iforest"}} {_METRICS_STATE["anomalies_detected"]["iforest"]}',
        "# HELP sentinel_inference_latency_seconds_sum Total inference latency sum.",
        "# TYPE sentinel_inference_latency_seconds_sum counter",
        f'sentinel_inference_latency_seconds_sum{{model="lstm"}} {sum(_METRICS_STATE["inference_latencies"]["lstm"]):.4f}',
        f'sentinel_inference_latency_seconds_sum{{model="iforest"}} {sum(_METRICS_STATE["inference_latencies"]["iforest"]):.4f}',
        "# HELP sentinel_inference_latency_seconds_count Total inference invocations count.",
        "# TYPE sentinel_inference_latency_seconds_count counter",
        f'sentinel_inference_latency_seconds_count{{model="lstm"}} {len(_METRICS_STATE["inference_latencies"]["lstm"])}',
        f'sentinel_inference_latency_seconds_count{{model="iforest"}} {len(_METRICS_STATE["inference_latencies"]["iforest"])}',
        "# HELP sentinel_drift_checks_total Total telemetry drift evaluations by severity.",
        "# TYPE sentinel_drift_checks_total counter",
        f'sentinel_drift_checks_total{{severity="stable"}} {_METRICS_STATE["drift_checks"]["stable"]}',
        f'sentinel_drift_checks_total{{severity="moderate"}} {_METRICS_STATE["drift_checks"]["moderate"]}',
        f'sentinel_drift_checks_total{{severity="critical"}} {_METRICS_STATE["drift_checks"]["critical"]}',
        "# HELP sentinel_system_info Metadata and version information for Sentinel AI.",
        "# TYPE sentinel_system_info gauge",
        'sentinel_system_info{version="2.7.0",framework="FastAPI",ml_backend="PyTorch"} 1'
    ])

    body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/v1/health", tags=["Platform Diagnostics"])
def health() -> dict[str, Any]:
    """Returns platform operational health status and active engine metadata."""
    return {
        "status": "healthy",
        "service": "SENTINEL-AI",
        "version": "2.7.0",
        "security": {
            "rate_limiting": "enabled (120 req/min)",
            "rbac": "enabled",
            "path_traversal_validation": "enforced"
        }
    }


@app.get("/api/v1/summary", tags=["Platform Diagnostics"])
def summary() -> dict[str, Any]:
    """Returns dataset discovery metrics and artifact readiness indicators."""
    data_dir = ROOT / "data" / "rcaeval"
    discovered = len(list(data_dir.glob("re*"))) if data_dir.exists() else 0
    return {
        "dataset_root": str(data_dir),
        "discovered_cases": discovered,
        "models": {
            "isolation_forest": (ROOT / "artifacts" / "isolation_forest.joblib").exists(),
            "lstm_autoencoder": (ROOT / "artifacts" / "lstm_autoencoder.pt").exists(),
        }
    }


@app.get("/api/v1/models", tags=["Machine Learning"])
def get_models() -> dict[str, Any]:
    """Returns evaluation metrics comparing the pre-trained LSTM Autoencoder and Isolation Forest."""
    results: dict[str, Any] = {}
    iforest_eval = ROOT / "artifacts" / "evaluation.json"
    lstm_eval = ROOT / "artifacts" / "lstm_evaluation.json"
    rca_eval = ROOT / "artifacts" / "rca_benchmark_results.json"

    if iforest_eval.exists():
        results["isolation_forest"] = json.loads(iforest_eval.read_text(encoding="utf-8"))
    if lstm_eval.exists():
        results["lstm_autoencoder"] = json.loads(lstm_eval.read_text(encoding="utf-8"))
    if rca_eval.exists():
        results["rca_localization_benchmark"] = json.loads(rca_eval.read_text(encoding="utf-8"))

    return {
        "status": "success",
        "benchmark_summary": results
    }


@app.get("/api/v1/cases", tags=["Multimodal AIOps"])
def list_cases(
    limit: int = Query(50, ge=1, le=200),
    modality: str = Query("all", description="Filter by modality: 'all', 'multimodal', or 'metrics_only'")
) -> dict[str, Any]:
    """Discovers available RCAEval benchmark incident cases."""
    data_dir = ROOT / "data" / "rcaeval"
    if not data_dir.exists():
        return {"cases": [], "total": 0}

    discovered = sorted([p for p in data_dir.glob("re*") if p.is_dir()])
    cases = []
    for d in discovered:
        has_metrics = (d / "metrics.parquet").exists() or (d / "metrics.json").exists()
        has_logs = (d / "log.txt").exists()
        has_traces = (d / "trace.json").exists()
        is_multi = has_logs or has_traces

        if modality == "multimodal" and not is_multi:
            continue
        if modality == "metrics_only" and is_multi:
            continue

        if has_metrics:
            cases.append({
                "id": d.name,
                "has_metrics": True,
                "has_logs": has_logs,
                "has_traces": has_traces,
                "suite": "RE2 (Multimodal)" if is_multi else "RE1 (Metrics-only)"
            })
            if len(cases) >= limit:
                break

    return {"cases": cases, "total": len(cases)}


@app.get("/api/v1/cases/{case_id}", tags=["Multimodal AIOps"])
def get_case_telemetry(case_id: str) -> dict[str, Any]:
    """Retrieves full telemetry, log analysis, and trace statistics for an incident case."""
    case_dir = _validate_case_id(case_id)

    cache_key = f"case_telemetry_{case_id}"
    cached = cache_get(cache_key)
    if cached:
        _METRICS_STATE["cache_hits"] += 1
        return cached
    _METRICS_STATE["cache_misses"] += 1

    parquet = case_dir / "metrics.parquet"
    json_file = case_dir / "metrics.json"

    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif json_file.exists():
        df = pd.DataFrame(json.loads(json_file.read_text(encoding="utf-8")))
    else:
        raise HTTPException(status_code=404, detail="No telemetry metrics file found in case directory.")

    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")

    if len(df) > 200:
        df = df.iloc[::2].copy()

    records = df.to_dict(orient="records")

    inject_time = None
    inj_file = case_dir / "inject_time.txt"
    if inj_file.exists():
        try:
            inject_time = float(inj_file.read_text(encoding="utf-8").strip())
        except ValueError:
            pass

    logs_data = analyze_logs(case_dir)
    traces_data = analyze_traces(case_dir)

    result = {
        "case_id": case_id,
        "inject_time": inject_time,
        "row_count": len(records),
        "telemetry": records,
        "logs_summary": logs_data,
        "traces_summary": traces_data,
        "is_multimodal": logs_data.get("has_logs", False) or traces_data.get("has_traces", False)
    }

    cache_set(cache_key, result, ttl_seconds=600)
    return result


@app.post("/api/v1/score", tags=["Machine Learning"])
def score(request: ScoreRequest, role: str = Depends(require_role("operator"))) -> dict[str, Any]:
    """Evaluates telemetry rows against the active model and performs Root Cause Attribution.

    Enforces 30-timestep sliding windowing matching model training dynamics.
    Evaluates distribution drift against training baseline.
    """
    start_t = time.perf_counter()
    try:
        result = score_rows(request.rows, model_type=request.model, custom_threshold=request.threshold)
        duration_s = time.perf_counter() - start_t

        threshold = result["threshold"]
        flags = [bool(s >= threshold) for s in result["scores"]]
        is_anomaly = any(flags)

        # Track Prometheus metrics
        m_key = "lstm" if "lstm" in request.model.lower() else "iforest"
        _METRICS_STATE["inference_latencies"][m_key].append(duration_s)
        if is_anomaly:
            _METRICS_STATE["anomalies_detected"][m_key] += 1

        # Evaluate statistical telemetry drift
        drift_report = None
        if request.check_drift and len(request.rows) >= 10:
            drift_report = monitor_telemetry_drift(request.rows)
            sev = drift_report.get("drift_severity", "stable")
            sev_key = sev if sev in _METRICS_STATE["drift_checks"] else "stable"
            _METRICS_STATE["drift_checks"][sev_key] += 1

        return {
            **result,
            "anomaly_flags": flags,
            "anomaly_detected": is_anomaly,
            "drift_analysis": drift_report
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/drift", tags=["Machine Learning"])
def drift_check(request: DriftRequest, role: str = Depends(require_role("operator"))) -> dict[str, Any]:
    """Performs statistical distribution drift analysis on incoming telemetry using PSI and KS-tests."""
    report = monitor_telemetry_drift(request.rows)
    sev = report.get("drift_severity", "stable")
    sev_key = sev if sev in _METRICS_STATE["drift_checks"] else "stable"
    _METRICS_STATE["drift_checks"][sev_key] += 1
    return report


@app.post("/api/v1/otel/v1/metrics", tags=["OpenTelemetry Ingestion"])
def ingest_otel_metrics(payload: OTelMetricsPayload, role: str = Depends(require_role("operator"))) -> dict[str, Any]:
    """Ingests standard OpenTelemetry (OTLP) JSON metric payloads and triggers real-time anomaly scoring."""
    flattened_rows: list[dict[str, Any]] = []

    for rm in payload.resourceMetrics:
        res = rm.get("resource", {})
        attrs = {a.get("key"): a.get("value", {}).get("stringValue", "") for a in res.get("attributes", [])}
        service_name = attrs.get("service.name", "unknown_service")

        for sm in rm.get("scopeMetrics", []):
            for m in sm.get("metrics", []):
                metric_name = m.get("name", "metric")
                feature_name = f"{service_name}_{metric_name}"

                # Extract datapoints from gauge, sum, or histogram
                dps = (
                    m.get("gauge", {}).get("dataPoints")
                    or m.get("sum", {}).get("dataPoints")
                    or []
                )

                for dp in dps:
                    val = dp.get("asDouble") or dp.get("asInt") or 0.0
                    ts = dp.get("timeUnixNano", 0)
                    flattened_rows.append({"time": ts, feature_name: float(val)})

    if not flattened_rows:
        return {"status": "received", "data_points": 0, "anomaly_detected": False}

    # Group into sequential temporal records
    df = pd.DataFrame(flattened_rows).groupby("time", as_index=False).first()
    rows = df.drop(columns=["time"], errors="ignore").to_dict(orient="records")

    score_res = score_rows(rows, model_type=payload.model)
    threshold = score_res["threshold"]
    is_anomaly = any(s >= threshold for s in score_res["scores"])

    return {
        "status": "ingested_and_scored",
        "data_points_processed": len(rows),
        "overall_score": score_res["overall_score"],
        "threshold": threshold,
        "anomaly_detected": is_anomaly,
        "root_cause_analysis": score_res["root_cause_analysis"]
    }


@app.post("/api/v1/otel/v1/traces", tags=["OpenTelemetry Ingestion"])
def ingest_otel_traces(payload: OTelTracesPayload, role: str = Depends(require_role("operator"))) -> dict[str, Any]:
    """Ingests standard OpenTelemetry (OTLP) distributed traces and identifies RPC latency bottlenecks."""
    span_durations: list[float] = []
    error_count = 0
    service_latencies: dict[str, list[float]] = {}

    for rs in payload.resourceSpans:
        res = rs.get("resource", {})
        attrs = {a.get("key"): a.get("value", {}).get("stringValue", "") for a in res.get("attributes", [])}
        service_name = attrs.get("service.name", "unknown_service")

        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                duration_ms = float(span.get("durationNano", 0)) / 1_000_000.0
                span_durations.append(duration_ms)

                if service_name not in service_latencies:
                    service_latencies[service_name] = []
                service_latencies[service_name].append(duration_ms)

                # Check error status (StatusCode.ERROR is 2)
                if span.get("status", {}).get("code") == 2:
                    error_count += 1

    total_spans = len(span_durations)
    if total_spans == 0:
        return {"status": "received", "spans_processed": 0}

    p95 = float(np.percentile(span_durations, 95))
    p99 = float(np.percentile(span_durations, 99))
    error_rate_pct = round((error_count / total_spans) * 100, 2)

    bottlenecks = [
        {
            "service": s,
            "mean_latency_ms": round(float(np.mean(lats)), 2),
            "p95_latency_ms": round(float(np.percentile(lats, 95)), 2)
        }
        for s, lats in service_latencies.items()
    ]
    bottlenecks.sort(key=lambda x: x["p95_latency_ms"], reverse=True)

    return {
        "status": "traces_analyzed",
        "spans_processed": total_spans,
        "error_spans": error_count,
        "error_rate_pct": error_rate_pct,
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "primary_bottleneck": bottlenecks[0] if bottlenecks else None,
        "service_breakdown": bottlenecks[:5]
    }


@app.post("/api/v1/triage-multimodal/{case_id}", tags=["Multimodal AIOps"])
def triage_multimodal(case_id: str, model: str = "lstm", role: str = Depends(require_role("operator"))) -> dict[str, Any]:
    """Executes holistic multimodal incident triage across metrics, logs, and traces."""
    case_dir = _validate_case_id(case_id)

    parquet = case_dir / "metrics.parquet"
    if not parquet.exists():
        raise HTTPException(status_code=404, detail="metrics.parquet missing from incident case.")

    df = pd.read_parquet(parquet)
    if len(df) > 200:
        df = df.iloc[::2].copy()
    records = df.to_dict(orient="records")

    metric_res = score_rows(records, model_type=model)
    fused_res = fuse_multimodal_evidence(case_dir, metric_res["overall_score"], metric_res["root_cause_analysis"])

    return {
        "case_id": case_id,
        "metric_score": metric_res["overall_score"],
        "metric_threshold": metric_res["threshold"],
        "anomaly_detected": bool(metric_res["overall_score"] >= metric_res["threshold"]),
        "multimodal_analysis": fused_res
    }


@app.post("/api/v1/postmortem", tags=["Autonomous SRE"])
def generate_postmortem(req: PostMortemRequest, role: str = Depends(require_role("operator"))) -> dict[str, str]:
    """Generates an executive-ready, technical SRE incident post-mortem in Markdown."""
    md_report = generate_incident_postmortem(req.model_dump())
    return {
        "report_markdown": md_report
    }


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard() -> str:
    """Renders the single-page Sentinel AI Observability Intelligence web dashboard."""
    return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
