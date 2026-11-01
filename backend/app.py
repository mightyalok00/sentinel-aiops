"""FastAPI Application for SENTINEL AI.

Provides RESTful endpoints for:
- Health and platform diagnostics
- Telemetry case discovery and streaming replay (with Redis caching and path traversal validation)
- Model evaluation benchmarks
- Real-time and batch anomaly scoring (with rate limiting and API key authentication)
- Multimodal evidence fusion (Metrics + Container Logs + Distributed Traces)
- Automated SRE Incident Post-Mortem generation
"""

import os
from pathlib import Path
import re
from typing import Any, Optional
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.scoring import score_rows
from backend.postmortem import generate_incident_postmortem
from backend.cache import cache_get, cache_set, check_rate_limit
from ml_engine.multimodal import analyze_logs, analyze_traces, fuse_multimodal_evidence

ROOT = Path(__file__).resolve().parents[1]
CASE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")
SENTINEL_API_KEY = os.getenv("SENTINEL_API_KEY", "")

app = FastAPI(
    title="SENTINEL AI",
    version="2.6.0",
    description="AIOps Multimodal Telemetry Anomaly Scoring, Root Cause Analysis & Autonomous Incident Triage Platform.",
    docs_url="/docs",
    redoc_url="/redoc"
)


# -----------------------------------------------------------------------------
# Security & Path Traversal Prevention
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


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Authenticates API requests via X-API-Key header when SENTINEL_API_KEY is configured."""
    if SENTINEL_API_KEY and x_api_key != SENTINEL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid 'X-API-Key' header."
        )


# -----------------------------------------------------------------------------
# Rate Limiting Middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def rate_limiting_middleware(request: Request, call_next):
    """Enforces token-bucket rate limits per client IP on compute-intensive scoring endpoints."""
    client_ip = request.client.host if request.client else "127.0.0.1"
    path = request.url.path

    if path.startswith("/api/v1/score") or path.startswith("/api/v1/postmortem"):
        if not check_rate_limit(client_ip, max_requests=120, window_seconds=60):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded: Maximum 120 requests per minute permitted."}
            )

    response = await call_next(request)
    return response


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    """Schema for incoming telemetry anomaly scoring requests."""
    rows: list[dict[str, Any]] = Field(..., min_length=1, description="List of metric dictionaries across time steps.")
    model: str = Field(default="lstm", description="Inference engine: 'lstm' (LSTM Autoencoder) or 'iforest' (Isolation Forest).")
    threshold: float = Field(default=0.0, description="Optional custom anomaly threshold override; 0.0 uses calibrated value.")


class PostMortemRequest(BaseModel):
    """Schema for incident post-mortem generation requests."""
    case_id: str = Field(default="incident-live", description="Benchmark case ID or live incident tracking number.")
    model: str = Field(default="LSTM Autoencoder", description="Inference model used during evaluation.")
    overall_score: float = Field(default=0.85, description="Composite anomaly severity score.")
    fault_type: str = Field(default="resource_saturation / latency_spike", description="Nature of the operational fault.")
    suspect_services: list[dict[str, Any]] = Field(default=[], description="Ranked microservices with attribution share percentages.")
    top_contributing_metrics: list[dict[str, Any]] = Field(default=[], description="Features exhibiting largest deviations.")
    logs_summary: dict[str, Any] = Field(default={}, description="Container error log burst summary.")
    traces_summary: dict[str, Any] = Field(default={}, description="Distributed trace latency percentiles.")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/v1/health", tags=["System"])
def health() -> dict[str, str]:
    """Returns system status, service name, active version, and security mode."""
    return {
        "status": "ok",
        "service": "sentinel-ai",
        "version": "2.6.0",
        "auth_enabled": bool(SENTINEL_API_KEY)
    }


@app.get("/api/v1/summary", tags=["Observability"])
def summary() -> dict[str, Any]:
    """Returns dataset directory metrics, case discovery counts, and model readiness status."""
    folder = ROOT / "data" / "rcaeval"
    count = len(list(folder.glob("**/metrics.parquet"))) if folder.exists() else 0
    artifacts_dir = ROOT / "artifacts"
    has_iforest = (artifacts_dir / "isolation_forest.joblib").exists()
    has_lstm = (artifacts_dir / "lstm_autoencoder.pt").exists()

    return {
        "dataset": "RCAEval",
        "metric_case_files_found": count,
        "data_directory": str(folder),
        "available_models": {
            "isolation_forest": has_iforest,
            "lstm_autoencoder": has_lstm
        },
        "multimodal_support": {
            "metrics": True,
            "logs": True,
            "traces": True
        }
    }


@app.get("/api/v1/models", tags=["Machine Learning"])
def get_model_benchmarks() -> dict[str, Any]:
    """Returns side-by-side held-out test evaluation benchmarks for all trained models."""
    artifacts_dir = ROOT / "artifacts"
    results = {}
    iforest_eval = artifacts_dir / "evaluation.json"
    if iforest_eval.exists():
        results["isolation_forest"] = json.loads(iforest_eval.read_text(encoding="utf-8"))
    lstm_eval = artifacts_dir / "lstm_evaluation.json"
    if lstm_eval.exists():
        results["lstm_autoencoder"] = json.loads(lstm_eval.read_text(encoding="utf-8"))
    return results


@app.get("/api/v1/cases", tags=["Observability"])
def list_sample_cases(limit: int = Query(35, ge=1, le=100)) -> list[dict[str, Any]]:
    """Discovers and catalogs available benchmark incident cases."""
    data_dir = ROOT / "data" / "rcaeval"
    if not data_dir.exists():
        return []

    cache_key = f"cases_list:{limit}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    all_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and (p / "metrics.parquet").exists())
    re1_dirs = [p for p in all_dirs if p.name.startswith("re1")][:15]
    re2_dirs = [p for p in all_dirs if p.name.startswith("re2")][:20]
    selected = re2_dirs + re1_dirs

    cases = []
    for p in selected[:limit]:
        inject_file = p / "inject_time.txt"
        inject_time = float(inject_file.read_text(encoding="utf-8").strip()) if inject_file.exists() else None
        parts = p.name.split("_")
        suite = parts[0]
        service = parts[1] if len(parts) > 1 else "unknown"
        fault = parts[2] if len(parts) > 2 else "unknown"
        has_logs = (p / "logs.parquet").exists()
        has_traces = (p / "traces.parquet").exists()
        cases.append({
            "case_id": p.name,
            "suite": suite,
            "target_service": service,
            "fault_type": fault,
            "inject_time": inject_time,
            "has_logs": has_logs,
            "has_traces": has_traces,
            "is_multimodal": has_logs and has_traces
        })

    cache_set(cache_key, cases, ttl_seconds=600)
    return cases


@app.get("/api/v1/cases/{case_id}", tags=["Observability"])
def get_case_telemetry(case_id: str) -> dict[str, Any]:
    """Fetches metric time-series data and multi-signal telemetry for an incident case with path security."""
    case_dir = _validate_case_id(case_id)

    cache_key = f"case_telemetry:{case_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    parquet = case_dir / "metrics.parquet"
    if not parquet.exists():
        raise HTTPException(status_code=404, detail="metrics.parquet missing from incident case.")

    df = pd.read_parquet(parquet)
    inject_file = case_dir / "inject_time.txt"
    inject_time = float(inject_file.read_text(encoding="utf-8").strip()) if inject_file.exists() else None

    if len(df) > 200:
        df = df.iloc[::2].copy()

    records = df.to_dict(orient="records")
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
def score(request: ScoreRequest, _auth: None = Depends(verify_api_key)) -> dict[str, Any]:
    """Evaluates telemetry rows against the active model and performs Root Cause Attribution.

    Enforces 30-timestep sliding windowing matching model training dynamics.
    """
    try:
        result = score_rows(request.rows, model_type=request.model, custom_threshold=request.threshold)
        threshold = result["threshold"]
        flags = [bool(s >= threshold) for s in result["scores"]]
        return {
            **result,
            "anomaly_flags": flags,
            "anomaly_detected": any(flags)
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/triage-multimodal/{case_id}", tags=["Multimodal AIOps"])
def triage_multimodal(case_id: str, model: str = "lstm", _auth: None = Depends(verify_api_key)) -> dict[str, Any]:
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
def generate_postmortem(req: PostMortemRequest, _auth: None = Depends(verify_api_key)) -> dict[str, str]:
    """Generates an executive-ready, technical SRE incident post-mortem in Markdown."""
    md_report = generate_incident_postmortem(req.model_dump())
    return {
        "report_markdown": md_report
    }


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard() -> str:
    """Renders the single-page Sentinel AI Observability Intelligence web dashboard."""
    return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
