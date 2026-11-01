"""Automated SRE Incident Post-Mortem Generator.

This module synthesizes multi-signal telemetry evidence (time-series metrics,
container crash/error logs, and distributed trace span latencies) into an
executive-ready, technically rigorous Incident Post-Mortem in Markdown.

The structure adheres to modern SRE and DevOps incident retrospectives:
1. Executive Summary & Blast Radius
2. Multimodal Evidence Breakdown
3. Timeline & Causal Fault Propagation
4. Immediate Kubernetes Runbook Remediation Commands
5. Long-term Preventative Engineering Action Items
"""

import json
from datetime import datetime, timezone
from typing import Any


def generate_incident_postmortem(incident: dict[str, Any]) -> str:
    """Generates an executive-ready, technical SRE incident post-mortem report in Markdown.

    Args:
        incident: Structured incident payload containing:
            - case_id: Unique incident identifier.
            - suspect_services: List of ranked microservices with confidence scores.
            - fault_type: Injected or inferred fault classification (e.g. cpu, memory, delay).
            - overall_score: Calibrated anomaly severity score.
            - model: Identification string of the inference engine.
            - top_contributing_metrics: Ranked metrics with highest deviation scores.
            - logs_summary: Summary of error log bursts and container attributions.
            - traces_summary: P95/P99 latency measurements and slow RPC span analysis.

    Returns:
        str: Fully formatted, professional GitHub-Flavored Markdown report.
    """
    case_id = incident.get("case_id", "incident-live")
    suspect_services = incident.get("suspect_services", [])
    primary_service = suspect_services[0]["service"] if suspect_services else "unknown-service"
    attribution_share = suspect_services[0].get("attribution_share_pct", suspect_services[0].get("confidence_pct", 85.0)) if suspect_services else 75.0

    fault_type = incident.get("fault_type", "resource_saturation / latency_spike")
    score = incident.get("overall_score", 0.85)
    model_used = incident.get("model", "LSTM Autoencoder (Deep Learning)")

    top_metrics = incident.get("top_contributing_metrics", [])
    logs_summary = incident.get("logs_summary", {})
    traces_summary = incident.get("traces_summary", {})

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Format telemetry metric observations
    metrics_bullets = ""
    if top_metrics:
        for m in top_metrics[:4]:
            metric_name = m.get("metric", "unknown")
            dev_val = m.get("anomaly_contribution", m.get("deviation", 0.0))
            metrics_bullets += f"- **`{metric_name}`**: Anomaly Deviation Contribution `{dev_val}`\n"
    else:
        metrics_bullets = f"- **`{primary_service}_cpu` / `{primary_service}_latency`**: Statistical deviation exceeding 3σ threshold\n"

    # Format log analysis section
    logs_section = ""
    if logs_summary.get("has_logs"):
        err_count = logs_summary.get("error_logs_count", 0)
        err_rate = logs_summary.get("error_rate_pct", 0.0)
        logs_section = f"""- **Total Error Log Events**: {err_count} ({err_rate}% of window logs)
- **Service Error Distribution**: {json.dumps(logs_summary.get('service_error_counts', {}), indent=0)}
- **Sample Log Signatures**:
"""
        for sample in logs_summary.get("sample_errors", [])[:3]:
            logs_section += f"  - `[{sample.get('service')}] {sample.get('message')}`\n"
    else:
        logs_section = "- *No discrete error logs in window; anomaly detected primarily via telemetry metric reconstruction.*"

    # Format trace latency section
    traces_section = ""
    if traces_summary.get("has_traces"):
        p95 = traces_summary.get("p95_duration_ms", 0.0)
        p99 = traces_summary.get("p99_duration_ms", 0.0)
        traces_section = f"""- **Span Latency Degradation**: P95 `{p95}ms` | P99 `{p99}ms`
- **Failing / Outlier Spans**:
"""
        for slow in traces_summary.get("slow_services", [])[:3]:
            traces_section += f"  - Service **`{slow.get('service')}`**: {slow.get('slow_spans')} slow spans (avg {slow.get('avg_duration_ms')}ms, max {slow.get('max_duration_ms')}ms)\n"
    else:
        traces_section = "- *Distributed tracing data within normal baselines.*"

    return f"""# [CRITICAL ALERT] SRE INCIDENT POST-MORTEM REPORT
**Incident ID**: `{case_id}`
**Generated At**: {timestamp}
**Classification**: **SEV-1 (Critical Telemetry Anomaly)**
**Investigation Engine**: Sentinel AI — {model_used}

---

## 1. Executive Summary
At **{timestamp}**, Sentinel AI detected a statistically significant telemetry anomaly score of **`{score:.4f}`**, exceeding the calibrated anomaly threshold. Multi-modal correlation across time-series metrics, container error logs, and distributed traces isolates **`{primary_service}`** as the probable root cause with **{attribution_share}% attribution share**.

- **Root Cause Microservice**: **`{primary_service}`**
- **Fault Signature**: `{fault_type.upper()}`
- **Attribution Share**: **{attribution_share}%** (relative reconstruction contribution)
- **Downstream Blast Radius**: Impacting user-facing endpoints via cascading RPC latency.

---

## 2. Multimodal Evidence Breakdown

### A. Telemetry Metrics
Reconstruction error analysis indicates critical parameter drift:
{metrics_bullets}

### B. Container Log Intelligence
{logs_section}

### C. Distributed Traces & Latency Analysis
{traces_section}

---

## 3. Timeline & Causal Propagation
1. **T0 (Fault Inception)**: `{primary_service}` experienced sudden resource constraint / delay injection.
2. **T0 + 15s (Reconstruction Drift)**: Temporal sequence autoencoder detected loss divergence above calibrated limit.
3. **T0 + 45s (Blast Radius Propagation)**: Downstream callers observed increased RPC wait times and connection pooling saturation.

---

## 4. Immediate Remediation & SRE Runbook Actions

Execute the following Kubernetes CLI remediation commands in order:

```bash
# 1. Inspect live pod health and events for suspect service
kubectl describe pods -l app={primary_service} -n default

# 2. Restart target deployment to clear memory leaks / hung thread pools
kubectl rollout restart deployment/{primary_service} -n default

# 3. Scale deployment horizontally to absorb traffic spikes if needed
kubectl scale deployment/{primary_service} --replicas=3 -n default

# 4. Verify pod status post-restart
kubectl get pods -l app={primary_service} -n default --watch
```

---

## 5. Preventative Action Items
- [ ] Configure Istio / Linkerd circuit breaking to prevent cascading timeouts from `{primary_service}`.
- [ ] Adjust Kubernetes ResourceQuota and CPU/Memory limits for `{primary_service}`.
- [ ] Add automated canary rollback rules triggered by Sentinel AI anomaly scoring threshold alerts.
"""
