# 🛡️ FedShield
### Privacy-Preserving Network Intrusion Detection using Federated Learning, Online Learning & Explainable AI

[![CI/CD](https://github.com/fedshield-team/fedshield/actions/workflows/ci.yml/badge.svg)](https://github.com/fedshield-team/fedshield/actions)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)
![Flower](https://img.shields.io/badge/Flower-FL-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

> **"In a world where data is the most valuable asset, FedShield protects both the network and the data it carries."**

---

## 🔥 The Problem

Traditional Intrusion Detection Systems (IDS) often centralize sensitive network traffic data. FedShield is designed to keep training data at the edge while sharing model weights for federated coordination.

FedShield combines intrusion detection with federated and online learning without requiring raw training traffic to be centralized.

---

## 🏗️ Architecture

```
Edge Node 1 (Hospital)  ──┐
Edge Node 2 (Bank)      ──┼──► FedAvg Aggregation Server ──► Global Model ──► XAI Dashboard
Edge Node 3 (Enterprise)──┘              │
                                          ▼
                              LLM Incident Reports (SHAP-explained)
                                          │
                                          ▼
                       Online Retraining (rule-labeled, F1-guarded)

Raw data NEVER leaves. Only model weights travel.
```

Nodes are orchestrated locally via **Docker Compose**, using the [Flower](https://flower.ai/) framework for real federated coordination (`server/flower_server.py` + `nodes/flower_client.py`) — each node trains independently on its own traffic split and only shares model weights, never raw data.

The aggregation logic (`server/lambda_aggregator.py`) is written to run as an **AWS Lambda function** for serverless, auto-scaling deployment — the FedAvg implementation is cloud-ready and has been validated with local invoke tests, but the project is currently demonstrated end-to-end via Docker orchestration rather than a live AWS deployment.

---

## ✨ Key Features

### 1. Federated Learning Core
Three simulated edge nodes (hospital, bank, enterprise) train a shared `MultiClassIDS` model with 41 input features and five output classes (`Normal`, `DoS`, `Probe`, `R2L`, `U2R`) on local data splits and aggregate via FedAvg — coordinated through Flower and Docker Compose.

### 2. LLM-Powered Incident Reports
When an intrusion is flagged, FedShield generates a natural-language incident report (via Groq) explaining *why* the traffic was flagged, backed by **SHAP** per-packet feature attributions — turning a raw model score into something an analyst can actually act on.

### 3. Online Incremental Retraining
The system doesn't stay static after deployment:
- New traffic is **rule-confirmed labeled** (no self-training feedback loop, so the model can't reinforce its own mistakes)
- A **replay buffer** mixes in past examples during retraining to prevent catastrophic forgetting of earlier attack patterns
- An **F1 regression guard** evaluates each retrained model against a held-out set before accepting it — a retrain is only promoted to production if it doesn't regress performance. In testing, this correctly accepted a genuine improvement (F1 0.8449 → 0.8691) while being able to reject a retrain that made things worse.

### 4. Explainable, Auditable Dashboard
Real-time React dashboard with Prometheus metrics, JWT-authenticated API access, and a SQLite audit trail of every incident and retraining event.

---

## 📊 Results

| Metric | Centralized Baseline | FedShield (Federated) |
|--------|---------------------|----------------------|
| Binary F1 | 0.9947 | 0.9946 |
| Multi-Class Macro F1 | 0.79 | **0.81** ✅ |
| Privacy | ❌ Data centralized | ✅ Data never shared |
| Data handling | Centralized training data | Federated coordination keeps raw training data at each node |
| Scalability | ❌ Single point | ✅ Distributed |
| DoS Detection | - | F1: 1.00 |
| Probe Detection | - | F1: 0.98 |

**Federated learning BEATS centralized on multi-class — with full privacy.**

### Online Retraining Validation
| Round | Trigger | F1 Before | F1 After | Outcome |
|-------|---------|-----------|----------|---------|
| Test run | Rule-confirmed new traffic | 0.8449 | 0.8691 | ✅ Accepted (regression guard passed) |

---

## 🔍 SHAP Explainability — Top Attack Indicators

| Rank | Feature | SHAP Score | What it means |
|------|---------|------------|---------------|
| 1 | dst_host_serror_rate | 0.0624 | SYN error rate — primary DDoS indicator |
| 2 | logged_in | 0.0530 | Attackers probe without authentication |
| 3 | same_srv_rate | 0.0404 | Port scanners hit same service repeatedly |
| 4 | srv_serror_rate | 0.0350 | Service-level SYN errors confirm flood |
| 5 | protocol_type | 0.0341 | Attack traffic skews specific protocols |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11+ for CI and local development
- Git

### Run with Docker (recommended)
```bash
git clone https://github.com/fedshield-team/fedshield.git
cd fedshield
copy .env.example .env  # PowerShell: Copy-Item .env.example .env
# Set the required values in .env privately before starting Compose.
python download_data.py
python preprocess_multiclass.py
docker-compose up --build
```

Compose requires `JWT_SECRET_KEY`, `FEDSHIELD_PASSWORD`, and `ANALYST_PASSWORD`. `FEDSHIELD_USERNAME` and `ANALYST_USERNAME` may use the documented defaults; Groq and AbuseIPDB keys are required only for their corresponding integrations. Keep all secrets private and do not commit `.env`.

React dashboard available at: **http://localhost:3000**
The API is available through the dashboard proxy at **http://localhost:3000**. The Compose API service is internal and is not published directly to the host.
Compose persists SQLite state and runtime model data through the mounted `./models` volume. A standalone container needs its own persistent volume if SQLite state must survive replacement. Live packet capture is disabled by default for container/cloud deployment; enable `FEDSHIELD_START_CAPTURE=1` only for an appropriate edge/host deployment.

The same Compose setup mounts the real `./data` directory into API, Flower server, and Flower clients. It must contain the preprocessed multiclass arrays (`X_train_mc.npy`, `y_train_mc.npy`, `X_test_mc.npy`, and `y_test_mc.npy`) for SHAP, online retraining, federated training, and evaluation; these generated datasets are intentionally not added to Git. Deploy web/API separately from edge capture: capture requires packet access on the sensor host, and the existing Windows `netsh` firewall action must run on that host rather than in a Linux container.

### Run locally
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python download_data.py
python preprocess_multiclass.py
python train_baseline.py
python federated_noniid.py
python explain.py
# Build the existing React dashboard from web/ (or use Docker Compose)
cd web && npm ci && npm run build
```

Run the backend tests with `python -m pytest -q`.

### Run real Flower FL (4 terminals)
```bash
# Terminal 1
python server/flower_server.py

# Terminal 2, 3, 4
python nodes/flower_client.py 1
python nodes/flower_client.py 2
python nodes/flower_client.py 3
```

### Test the serverless aggregation logic locally
The FedAvg aggregator is written for AWS Lambda deployment. To validate it without deploying to AWS:
```bash
python server/lambda_aggregator.py
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| ML | PyTorch CPU wheel | MultiClassIDS neural network training and inference |
| FL | Flower (flwr) | Federated learning coordination |
| XAI | SHAP | Explainable AI / per-packet feature attribution |
| LLM | Groq | Natural-language incident report generation |
| Online Learning | Custom (rule labels + replay buffer + F1 guard) | Safe incremental retraining |
| Cloud-ready | AWS Lambda (aggregator code) | Serverless aggregation — architected, locally validated |
| Orchestration | Docker Compose | Multi-node local deployment |
| API | FastAPI | Incident reports, retraining, model serving |
| CI/CD | GitHub Actions | Automated testing |
| Dashboard | React + Recharts | Real-time visualization |
| Dataset | NSL-KDD / CICIDS2017 | 125,973 network traffic samples |

---

## 📁 Project Structure
```
fedshield/
├── data/                          # NSL-KDD dataset
├── models/                        # Saved models, history, versioning
│   ├── federated_noniid_model.pth
│   ├── federated_noniid_history.json
│   └── model_version.json
├── nodes/                         # Edge node implementations
│   ├── node.py                    # FedNode class
│   └── flower_client.py           # Flower FL client
├── server/                        # Aggregation server
│   ├── aggregator.py              # FedAvg implementation
│   ├── flower_server.py           # Flower FL server
│   └── lambda_aggregator.py       # AWS Lambda-ready FedAvg (cloud-ready, locally tested)
├── api/                           # FastAPI backend
│   ├── main.py
│   └── incident_reports_endpoints.py
├── web/                            # Active React dashboard
│   ├── src/
│   └── nginx.conf
├── model.py                       # MultiClassIDS neural net (41 features, 5 classes)
├── preprocess_multiclass.py       # Multiclass data preprocessing
├── train_baseline.py              # Centralized baseline
├── federated_noniid.py            # Multiclass federated training
├── explain.py                     # SHAP explainability
├── llm_incident_report.py         # LLM-generated incident reports (Groq + SHAP)
├── online_retrain.py              # Online incremental retraining + F1 regression guard
├── live_capture.py                # Live traffic capture
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔒 Data Handling

- Raw network traffic is intended to remain at the edge node.
- Federated coordination transmits model weights rather than raw training records.
- These are architectural properties of this project, not a compliance certification.

---

## ☁️ Cloud Deployment Path

The system is architected for cloud deployment beyond the local demo:
- `server/lambda_aggregator.py` is written as a standard AWS Lambda handler and has been tested locally against real model weight shapes
- Deployment would package it as a container image on AWS Lambda, with edge nodes running on EC2 and results in RDS instead of SQLite
- This repo demonstrates federated training, aggregation, online retraining, and explainable incident reporting via Docker-orchestrated local nodes.

## Standalone Inference Image

The standalone backend image includes the real active inference artifacts:

- `models/federated_noniid_model.pth`
- `models/scaler_multiclass.pkl`
- `models/encoders_multiclass.pkl`
- `models/artifact_manifest.json`
- `models/model_version.json`

Datasets, databases, logs, caches, secrets, and frontend dependencies are not packaged into the image. Live packet capture is environment-dependent and should be run as an edge/host capability; the API inference runtime does not require packet capture.

---

## 🎯 Deployment Targets

| Sector | Use Case | Compliance |
|--------|---------|-----------|
| Healthcare | Threat detection across hospital branches | Evaluate applicable controls |
| Banking | Fraud and intrusion detection | Evaluate applicable controls |
| Government | Network protection | Evaluate applicable controls |
| Telecom | Multi-tenant network monitoring | Evaluate applicable controls |
| Enterprise | Branch office security | Evaluate applicable controls |

---

## 👥 Team

| Name | Roll No |
|------|---------|
| B. Siri | 23R11A6255 |
| M. R. Meghana | 23R11A6278 |
| P. Hathiram | 23R11A6281 |

**Guide:** Mrs. M. Yellamma, Assistant Professor, CSE – Cyber Security
**Institution:** Geethanjali College of Engineering and Technology

---

## 📄 License
MIT License — see [LICENSE](LICENSE) for details.
