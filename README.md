# 🛡️ FedShield
### Privacy-Preserving Network Intrusion Detection using Federated Learning & Cloud Auto-Scaling

[![CI/CD](https://github.com/fedshield-team/fedshield/actions/workflows/ci.yml/badge.svg)](https://github.com/fedshield-team/fedshield/actions)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)
![Flower](https://img.shields.io/badge/Flower-FL-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

> **"In a world where data is the most valuable asset, FedShield protects both the network and the data it carries."**

---

## 🔥 The Problem

Traditional Intrusion Detection Systems (IDS) require centralizing sensitive network traffic data — violating GDPR, HIPAA, and PCI-DSS compliance. Hospitals, banks, and enterprises **cannot legally share raw traffic data**.

FedShield eliminates this trade-off: **security AND privacy, simultaneously**.

---

## 🏗️ Architecture
Edge Node 1 (Hospital)  ──┐

Edge Node 2 (Bank)      ──┼──► AWS Lambda (FedAvg) ──► Global Model ──► XAI Dashboard

Edge Node 3 (Enterprise)──┘

↑

Raw data NEVER leaves. Only model weights travel.

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

Dashboard available at: **http://localhost:8501**

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
streamlit run dashboard/app.py
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

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| ML | PyTorch | Neural network training |
| FL | Flower (flwr) | Federated learning coordination |
| XAI | SHAP | Explainable AI |
| Cloud | AWS Lambda | Serverless aggregation + auto-scaling |
| DevOps | Docker | Containerization |
| CI/CD | GitHub Actions | Automated testing + deployment |
| Dashboard | Streamlit + Plotly | Real-time visualization |
| Dataset | NSL-KDD / CICIDS2017 | 125,973 network traffic samples |

---

## 📁 Project Structure
fedshield/

├── data/               # NSL-KDD dataset

├── models/             # Saved models + results

├── nodes/              # Edge node implementations

│   ├── node.py         # FedNode class

│   └── flower_client.py # Flower FL client

├── server/             # Aggregation server

│   ├── aggregator.py   # FedAvg implementation

│   └── flower_server.py # Flower FL server

├── dashboard/          # Streamlit UI

│   └── app.py

├── model.py            # IntrusionDetector neural net

├── preprocess.py       # Data preprocessing

├── train_baseline.py   # Centralized baseline

├── federated_train.py  # Federated training

├── explain.py          # SHAP explainability

├── Dockerfile

├── docker-compose.yml

└── requirements.txt

---

## 🔒 Privacy Guarantees

- ✅ Raw network traffic **never leaves** the edge node
- ✅ Only model weights (mathematical parameters) are transmitted
- ✅ GDPR, HIPAA, PCI-DSS compliant architecture
- ✅ No single point of failure

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