# 🛡️ SENTINEL AI — Multimodal AIOps & Root Cause Intelligence Platform

[![CI Pipeline](https://github.com/mightyalok00/sentinel-aiops/actions/workflows/ci.yml/badge.svg)](https://github.com/mightyalok00/sentinel-aiops/actions)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.7.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA)-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker%20Compose-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-60%2B%20Passed-brightgreen.svg)](https://github.com/mightyalok00/sentinel-aiops)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Evidence-driven multimodal AIOps platform fusing time-series metrics, container logs, and distributed traces for real-time anomaly scoring, microservice fault localization, and automated SRE incident post-mortems.**

---

## 🌟 Key Features

- **Multimodal Telemetry Fusion (RE1 & RE2)**:
  - **Metrics**: Point-wise reconstruction loss and z-score standard deviations across microservice metrics.
  - **Container Logs**: Regex-driven error mining for crash loops, connection timeouts, and HTTP 5xx spikes.
  - **Distributed Traces**: RPC P95/P99 latency degradation and span bottleneck tracking.
- **Dual Model Architectures**:
  - **PyTorch LSTM Autoencoder**: Temporal sequence reconstruction with CUDA GPU acceleration (**91.1% Test F1**, **3.8% False Alarm Rate**).
  - **Isolation Forest Baseline**: Statistical summary feature space ($[\mu, \sigma, \min, \max]$) with median imputation and robust scaling (**73.2% Test F1**).
- **Training-Consistent Inference Windows**: Enforces fixed 30-timestep sliding windows during inference with median padding for short bursts, eliminating score distribution drift between training and production serving.
- **Real-World OpenTelemetry & Prometheus Ingestion**:
  - **OTel Metrics**: Ingests standard OpenTelemetry (OTLP) JSON payloads (`/api/v1/otel/v1/metrics`).
  - **OTel Traces**: Parses distributed RPC spans and localizes latency bottlenecks (`/api/v1/otel/v1/traces`).
  - **Prometheus Exporter**: Scrapeable `/metrics` endpoint in standard OpenMetrics text format.
- **Statistical Drift Monitoring**: Real-time distribution shift detection using **Population Stability Index (PSI)** and **2-Sample Kolmogorov-Smirnov (KS) tests** ($p < 0.01$) to alert on concept and traffic drift.
- **Calibrated Root Cause Attribution (RCA)**: Decomposes reconstruction errors into `attribution_share_pct` (relative error share) and `calibrated_probability_pct` (temperature-scaled Softmax confidence).
- **Empirical RCA Localization Benchmark**: Evaluated against ground-truth injection targets with **80.0% Hit@3** and **0.722 MRR**.
- **Enterprise Security & Reliability**:
  - **Role-Based Access Control (RBAC)**: Enforces `viewer`, `operator`, and `admin` permission boundaries with constant-time HMAC key verification.
  - **Defensive HTTP Headers**: Comprehensive OWASP security headers (HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy).
  - **Payload Size Limiting**: 10MB maximum request size mitigation preventing DoS memory exhaustion.
  - **Sensitive Log Scrubbing**: Recursive credential and token sanitization across structured JSON log entries.
  - **Path Traversal Protection**: Regex validation and strict directory containment checks preventing `../` attacks.
  - **Rate Limiting**: Sliding-window rate limiting (120 req/min per IP) via Redis with in-memory fallback.
  - **Container Hardening**: Dockerfile configured to run as non-root user `appuser` (UID 1000).
- **Cryptographic Reproducibility**: SHA-256 dataset checksums manifest (`data/checksums.json`) with automated integrity verification CLI.
- **Autonomous SRE Post-Mortems**: One-click generation of technical Markdown incident reports complete with Kubernetes CLI runbook commands (`kubectl rollout restart`, `kubectl scale`).

---

## 📊 Model Benchmark Results (Held-Out Test Set: 11,522 Windows)

### Anomaly Detection Performance

| Model Architecture | Precision | Recall | F1-Score | PR-AUC | False Alarm Rate (FAR) | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch LSTM Autoencoder** (CUDA) | **95.6%** | **87.0%** | **91.1%** | **91.4%** | **3.8%** | **Primary Engine** |
| **Isolation Forest Baseline** | 77.8% | 69.1% | 73.2% | 75.4% | 19.1% | Classical ML Baseline |

### Root Cause Localization Accuracy (Empirical Benchmark)

| Metric | Benchmark Score | Description |
| :--- | :--- | :--- |
| **Hit@1 (Top-1 Accuracy)** | **66.7%** | True faulty microservice ranked #1 suspect |
| **Hit@3 (Top-3 Accuracy)** | **80.0%** | True faulty microservice within top 3 suspects |
| **Hit@5 (Top-5 Accuracy)** | **80.0%** | True faulty microservice within top 5 suspects |
| **Mean Reciprocal Rank (MRR)** | **0.722** | Average reciprocal rank ($\frac{1}{\text{rank}}$) |
| **CPU Pressure Accuracy** | **100.0% Hit@1 (MRR 1.0)** | Instantaneous localization on CPU saturation |
| **Memory Pressure Accuracy** | **100.0% Hit@1 (MRR 1.0)** | Memory leak attribution |
| **Disk I/O Pressure Accuracy** | **100.0% Hit@1 (MRR 1.0)** | Storage bottleneck attribution |

---

## 🔬 Data & Evaluation Methodology

1. **Proxy Ground-Truth Labels**:
   - Labels in RCAEval are synthesized based on **fault-injection timestamps** ($t \ge t_{\text{inject}}$).
   - In real-world telemetry, physical failure manifestation (e.g., slow memory leak saturation or cascading connection timeouts) may lag slightly behind fault injection. Proxy labels provide a rigorous, reproducible benchmark for comparison, though they differ from human-adjudicated operational tickets.
2. **Leak-Free Split Strategy**:
   - Rather than shuffling sliding windows across the full dataset (which risks placing adjacent windows from the same incident into both train and test), **Sentinel AI splits by unique incident folder first**.
   - Train (60%), Validation (20%), and Held-out Test (20%) contain completely distinct incident instances.
3. **Cryptographic Integrity**:
   - Every benchmark file is hashed with SHA-256 and cataloged in `data/checksums.json`, verifiable via `python -m ml_engine.cli verify-checksums`.

---

## 🚀 Quick Start

### Option A: Run with Docker Compose (Recommended)

```bash
docker compose up -d --build
```

- **Dashboard**: [Local Dashboard](http://localhost:8000/)
- **Swagger API Docs**: [Interactive Swagger UI](http://localhost:8000/docs)
- **Prometheus Metrics**: [OpenMetrics Exporter](http://localhost:8000/metrics)

### Option B: Local Python Development

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Window & prepare dataset (pattern: re1ob* Online Boutique)
python -m ml_engine.cli prepare --data-dir data\rcaeval --output-dir data\processed --pattern "re1ob*"

# 3. Train models
python -m ml_engine.cli train-iforest --data-dir data\processed --output-dir artifacts
python -m ml_engine.cli train-lstm --data-dir data\processed --output-file artifacts\lstm_autoencoder.pt --epochs 5

# 4. Run Empirical RCA Benchmark
python -m ml_engine.cli evaluate-rca --data-dir data\rcaeval --model lstm --output artifacts\rca_benchmark_results.json

# 5. Launch FastAPI server & Dashboard
uvicorn backend.app:app --reload --port 8000
```

---

## 📡 API Endpoints

| Method | Endpoint | Role | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Public | Health check and platform security status. |
| `GET` | `/api/v1/summary` | Public | Dataset discovery counts and model artifact readiness. |
| `GET` | `/api/v1/models` | Public | Benchmark metrics and RCA localization accuracy. |
| `GET` | `/api/v1/cases` | Public | Discovers available RE1 and RE2 multimodal benchmark cases. |
| `GET` | `/api/v1/cases/{case_id}` | Public | Telemetry time-series, log summaries, and trace stats. |
| `POST` | `/api/v1/score` | Operator | Scores telemetry batches with 30-step sliding windows and drift check. |
| `POST` | `/api/v1/drift` | Operator | Evaluates feature distribution drift via PSI & KS-tests. |
| `POST` | `/api/v1/otel/v1/metrics` | Operator | Ingests OpenTelemetry (OTLP) JSON metrics. |
| `POST` | `/api/v1/otel/v1/traces` | Operator | Ingests OpenTelemetry (OTLP) distributed traces. |
| `POST` | `/api/v1/triage-multimodal/{case_id}` | Operator | Unified cross-modal incident triage (metrics + logs + traces). |
| `POST` | `/api/v1/postmortem` | Operator | Generates an SRE incident post-mortem with Kubernetes runbooks. |
| `GET` | `/metrics` | Public | Prometheus OpenMetrics telemetry exporter. |

---

## 🏗️ Project Architecture

```text
SENTINEL-AI/
├── .github/
│   └── workflows/ci.yml        # Matrix CI testing Python 3.11 & 3.12 (ruff + pytest)
├── backend/
│   ├── app.py                  # FastAPI server with RBAC, OTel, Prometheus, logging & security
│   ├── scoring.py              # 30-step sliding window scoring & calibrated RCA
│   ├── security.py             # Constant-time auth, OWASP headers, payload limit & log redaction
│   ├── cache.py                # Hybrid Redis + in-memory TTL cache & rate limiting
│   └── postmortem.py           # Automated SRE post-mortem report generator
├── ml_engine/
│   ├── data.py                 # Dataset discovery & leak-free window partitioner
│   ├── lstm_autoencoder.py     # PyTorch LSTM temporal autoencoder (CUDA/CPU)
│   ├── train.py                # Isolation Forest trainer & statistical pooling
│   ├── multimodal.py           # Container log error mining & trace span profiler
│   ├── drift.py                # Population Stability Index (PSI) & KS drift monitor
│   ├── rca_evaluator.py        # Empirical Hit@K and MRR benchmark evaluator
│   ├── checksums.py            # SHA-256 reproducibility verification engine
│   └── cli.py                  # Developer CLI for pipeline execution
├── frontend/
│   └── index.html              # Dark-mode dashboard with HTML5 canvas & incident replayer
├── artifacts/                  # Model weights, evaluations, and RCA benchmark results
├── data/
│   ├── rcaeval/                # RCAEval dataset junction (RE1 & RE2 benchmark cases)
│   ├── processed/              # Windowed train.npz, val.npz, and test.npz partitions
│   └── checksums.json          # Cryptographic SHA-256 verification manifest
├── tests/                      # 60+ unit, integration, and security tests
├── pyproject.toml              # Ruff, mypy, and pytest configurations
├── Dockerfile                  # Hardened non-root container image
├── docker-compose.yml          # Multi-container orchestration (FastAPI + Redis)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Comprehensive documentation
```

---

## 📜 Attribution & License

- **Dataset Attribution**: RCAEval benchmark dataset ([GitHub: phamquiluan/RCAEval](https://github.com/phamquiluan/RCAEval)).
- **License**: MIT License.
