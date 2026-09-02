# 🛡️ FedShield
### Privacy-Preserving Network Intrusion Detection using Federated Learning, Online Learning & Explainable AI

[![CI/CD](https://github.com/fedshield-team/fedshield/actions/workflows/ci.yml/badge.svg)](https://github.com/fedshield-team/fedshield/actions)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)
![Flower](https://img.shields.io/badge/Flower-FL-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

> **"In a world where data is the most valuable asset, FedShield protects both the network and the data it carries."**

---

## 🔥 The Problem

Traditional Intrusion Detection Systems (IDS) require centralizing sensitive network traffic data — violating GDPR, HIPAA, and PCI-DSS compliance. Hospitals, banks, and enterprises **cannot legally share raw traffic data**.

FedShield eliminates this trade-off: **security AND privacy, simultaneously** — and keeps learning from live traffic without ever needing to re-centralize it.

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
Three simulated edge nodes (hospital, bank, enterprise) train a shared `IntrusionDetector` model on local data splits and aggregate via FedAvg — coordinated through Flower and Docker Compose, with no raw traffic ever leaving a node.

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
| Compliance | ❌ GDPR violation | ✅ GDPR compliant |
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
- Python 3.12+
- Git

### Run with Docker (recommended)
```bash
git clone https://github.com/fedshield-team/fedshield.git
cd fedshield
python download_data.py
python preprocess.py
docker-compose up --build
```

React dashboard available at: **http://localhost:3000**
API available at: **http://localhost:8000**

### Run locally
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python download_data.py
python preprocess.py
python train_baseline.py
python federated_train.py
python explain.py
# Build the existing React dashboard from web/ (or use Docker Compose)
cd web && npm ci && npm run build
```

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
| ML | PyTorch | Neural network training |
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
├── model.py                       # IntrusionDetector neural net
├── preprocess.py                  # Data preprocessing
├── train_baseline.py              # Centralized baseline
├── federated_train.py             # Federated training
├── explain.py                     # SHAP explainability
├── llm_incident_report.py         # LLM-generated incident reports (Groq + SHAP)
├── online_retrain.py              # Online incremental retraining + F1 regression guard
├── live_capture.py                # Live traffic capture
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔒 Privacy Guarantees

- ✅ Raw network traffic **never leaves** the edge node
- ✅ Only model weights (mathematical parameters) are transmitted
- ✅ GDPR, HIPAA, PCI-DSS compliant architecture
- ✅ No single point of failure

---

## ☁️ Cloud Deployment Path

The system is architected for cloud deployment beyond the local demo:
- `server/lambda_aggregator.py` is written as a standard AWS Lambda handler and has been tested locally against real model weight shapes
- Deployment would package it as a container image on AWS Lambda, with edge nodes running on EC2 and results in RDS instead of SQLite
- This repo demonstrates the full working system — federated training, aggregation, online retraining, and explainable incident reporting — via Docker-orchestrated local nodes, which validates the same logic that would run in that cloud deployment

---

## 🎯 Deployment Targets

| Sector | Use Case | Compliance |
|--------|---------|-----------|
| Healthcare | Threat detection across hospital branches | HIPAA |
| Banking | Fraud and intrusion detection | PCI-DSS |
| Government | Classified network protection | Zero data exposure |
| Telecom | Multi-tenant network monitoring | GDPR |
| Enterprise | Branch office security | ISO 27001 |

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