# 🛡️ SENTINEL AI — Multimodal AIOps & Root Cause Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.5.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA)-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker%20Compose-Ready-2496ED?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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
- **Leak-Free Partitioning**: Incidents are split at the case-directory level prior to sliding window generation, guaranteeing zero cross-incident data leakage.
- **Root Cause Attribution (RCA)**: Decomposes reconstruction errors across individual metrics and aggregates them by microservice topology to output ranked suspect culprits with confidence percentages.
- **Autonomous SRE Post-Mortems**: One-click generation of technical Markdown incident reports complete with Kubernetes CLI runbook commands (`kubectl rollout restart`, `kubectl scale`).
- **Interactive Observability Dashboard**: Single-page dark-mode interface with live HTML5 Canvas telemetry timeline, fault injection markers, and incident replayers.
- **Production-Ready Dockerization**: Multi-container stack configured with FastAPI and Redis 7 Alpine via Docker Compose.

---

## 📊 Model Benchmark Results (Held-Out Test Set: 11,522 Windows)

| Model Architecture | Precision | Recall | F1-Score | PR-AUC | False Alarm Rate (FAR) | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch LSTM Autoencoder** (CUDA) | **95.6%** | **87.0%** | **91.1%** | **91.4%** | **3.8%** | **Primary Engine** |
| **Isolation Forest Baseline** | 77.8% | 69.1% | 73.2% | 75.4% | 19.1% | Classical ML Baseline |

---

## 🚀 Quick Start

### Option A: Run with Docker Compose (Recommended)
```bash
docker compose up -d --build
```
* **Dashboard**: [http://localhost:8000/](http://localhost:8000/)
* **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option B: Local Python Development
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Window & prepare dataset (pattern: re1ob* Online Boutique)
python -m ml_engine.cli prepare --data-dir data\rcaeval --output-dir data\processed --pattern "re1ob*"

# 3. Train models
python -m ml_engine.cli train-iforest --data-dir data\processed --output-dir artifacts
python -m ml_engine.cli train-lstm --data-dir data\processed --output-file artifacts\lstm_autoencoder.pt --epochs 5

# 4. View side-by-side benchmark comparison
python -m ml_engine.cli evaluate --data-dir data\processed --artifacts-dir artifacts

# 5. Launch FastAPI server & Dashboard
uvicorn backend.app:app --reload --port 8000
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Health check and active engine version. |
| `GET` | `/api/v1/summary` | Dataset discovery counts and model artifact readiness. |
| `GET` | `/api/v1/models` | Side-by-side benchmark evaluation metrics. |
| `GET` | `/api/v1/cases` | Discovers available RE1 and RE2 multimodal benchmark cases. |
| `GET` | `/api/v1/cases/{case_id}` | Returns telemetry time-series, log summaries, and trace stats. |
| `POST` | `/api/v1/score` | Scores telemetry batches with model selection (`lstm` or `iforest`). |
| `POST` | `/api/v1/triage-multimodal/{case_id}` | Runs unified cross-modal incident triage. |
| `POST` | `/api/v1/postmortem` | Generates an SRE incident post-mortem with Kubernetes runbooks. |

---

## 🏗️ Project Architecture

```
SENTINEL-AI/
├── backend/
│   ├── app.py                  # FastAPI server with OpenAPI routes & dashboard
│   ├── scoring.py              # In-memory artifact caching & RCA attribution engine
│   └── postmortem.py           # Automated SRE post-mortem report generator
├── ml_engine/
│   ├── data.py                 # Dataset discovery & leak-free window partitioner
│   ├── lstm_autoencoder.py     # PyTorch LSTM temporal autoencoder (CUDA/CPU)
│   ├── train.py                # Isolation Forest trainer & statistical pooling
│   ├── multimodal.py           # Container log error mining & trace span profiler
│   └── cli.py                  # Command-line interface for pipeline execution
├── frontend/
│   └── index.html              # Dark-mode dashboard with HTML5 canvas & incident replayer
├── artifacts/                  # Persisted model weights and benchmark evaluation jsons
├── data/
│   ├── rcaeval/                # RCAEval dataset junction (RE1 & RE2 benchmark cases)
│   └── processed/              # Windowed train.npz, val.npz, and test.npz partitions
├── tests/
│   ├── test_api.py             # REST API & model scoring tests
│   └── test_data.py            # Data pipeline & discovery tests
├── Dockerfile                  # Container definition for production deployment
├── docker-compose.yml          # Multi-container orchestration (FastAPI + Redis)
├── requirements.txt            # Python dependencies
└── README.md                   # Comprehensive project documentation
```

---

## 📜 Attribution & License
* **Dataset Attribution**: RCAEval benchmark dataset ([GitHub: phamquiluan/RCAEval](https://github.com/phamquiluan/RCAEval)).
* **License**: MIT License.
