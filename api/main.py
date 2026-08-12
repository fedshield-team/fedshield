"""
FedShield API — FastAPI backend with JWT Authentication + Prometheus Metrics
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess
)

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY         = "fedshield-jwt-secret-key-change-in-production-2025"
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "models", "fedshield_logs.db")

# ── Prometheus metrics ─────────────────────────────────────────────────────────
PACKETS_TOTAL = Gauge(
    "fedshield_packets_total",
    "Total packets processed by live capture"
)
ATTACKS_TOTAL = Gauge(
    "fedshield_attacks_total",
    "Total attack packets detected"
)
BLOCKED_TOTAL = Gauge(
    "fedshield_blocked_total",
    "Total IPs auto-blocked by firewall"
)
ATTACK_RATE = Gauge(
    "fedshield_attack_rate",
    "Current attack rate as fraction of total traffic"
)
NORMAL_TOTAL = Gauge(
    "fedshield_normal_total",
    "Total normal packets"
)
ATTACK_BY_TYPE = Gauge(
    "fedshield_attacks_by_type",
    "Attack count broken down by prediction type",
    ["prediction"]
)
AVG_CONFIDENCE = Gauge(
    "fedshield_avg_confidence",
    "Average model confidence score across all detections"
)
API_REQUESTS = Counter(
    "fedshield_api_requests_total",
    "Total API requests",
    ["endpoint", "method"]
)
MODEL_F1_SCORE = Gauge(
    "fedshield_model_f1_score",
    "Current model macro F1 score",
    ["model_type"]
)

# Set static model scores from training
MODEL_F1_SCORE.labels(model_type="federated_noniid").set(0.84)
MODEL_F1_SCORE.labels(model_type="federated_iid").set(0.81)
MODEL_F1_SCORE.labels(model_type="centralized").set(0.79)
MODEL_F1_SCORE.labels(model_type="binary_federated").set(0.9946)

# ── Auth ───────────────────────────────────────────────────────────────────────
pwd_context   = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

USERS = {
    "fedshield": {
        "username":        "fedshield",
        "hashed_password": pwd_context.hash("shield2025"),
        "role":            "admin",
    },
    "analyst": {
        "username":        "analyst",
        "hashed_password": pwd_context.hash("analyst2025"),
        "role":            "readonly",
    }
}

class Token(BaseModel):
    access_token: str
    token_type:   str
    expires_in:   int
    role:         str

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username or username not in USERS:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "role": USERS[username]["role"]}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FedShield API",
    description="Privacy-preserving IDS — JWT secured + Prometheus metrics",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from .incident_reports_endpoints import router as incident_router
app.include_router(incident_router, dependencies=[Depends(verify_token)])

def get_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None

def update_prometheus_metrics():
    """Pull latest stats from SQLite and update all Prometheus gauges."""
    try:
        conn = get_db()
        if not conn:
            return

        total   = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        attacks = conn.execute("SELECT COUNT(*) FROM detections WHERE tag='ATTACK'").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM detections WHERE blocked=1").fetchone()[0]
        normal  = total - attacks

        conf    = conn.execute("SELECT AVG(confidence) FROM detections").fetchone()[0] or 0

        breakdown = conn.execute(
            "SELECT prediction, COUNT(*) FROM detections GROUP BY prediction"
        ).fetchall()

        conn.close()

        PACKETS_TOTAL.set(total)
        ATTACKS_TOTAL.set(attacks)
        BLOCKED_TOTAL.set(blocked)
        NORMAL_TOTAL.set(normal)
        ATTACK_RATE.set(round(attacks / max(total, 1), 4))
        AVG_CONFIDENCE.set(round(conf, 4))

        for row in breakdown:
            ATTACK_BY_TYPE.labels(prediction=row[0]).set(row[1])

    except Exception as e:
        pass

# ── Public endpoints ───────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
def health():
    return {
        "status":  "ok",
        "db":      os.path.exists(DB_PATH),
        "auth":    "JWT HS256",
        "metrics": "Prometheus /metrics",
        "version": "2.1.0"
    }

@app.get("/metrics", tags=["Observability"])
def metrics():
    """
    Prometheus metrics endpoint.
    Scrape with: prometheus.yml → scrape_configs → targets: ['localhost:8000']
    """
    update_prometheus_metrics()
    API_REQUESTS.labels(endpoint="/metrics", method="GET").inc()
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# ── Auth endpoints ─────────────────────────────────────────────────────────────
@app.post("/auth/token", response_model=Token, tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(form.username)
    if not user or not pwd_context.verify(form.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_token({"sub": form.username, "role": user["role"]})
    API_REQUESTS.labels(endpoint="/auth/token", method="POST").inc()
    return {
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   TOKEN_EXPIRE_HOURS * 3600,
        "role":         user["role"],
    }

@app.get("/auth/me", tags=["Auth"])
def get_me(current_user: dict = Depends(verify_token)):
    return current_user

# ── Protected API endpoints ────────────────────────────────────────────────────
@app.get("/api/stats", tags=["SOC"])
def stats(_: dict = Depends(verify_token)):
    API_REQUESTS.labels(endpoint="/api/stats", method="GET").inc()
    try:
        conn    = get_db()
        total   = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        attacks = conn.execute("SELECT COUNT(*) FROM detections WHERE tag='ATTACK'").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM detections WHERE blocked=1").fetchone()[0]
        conn.close()
        return {"total": total, "attacks": attacks, "blocked": blocked, "normal": total - attacks}
    except Exception:
        return {"total": 0, "attacks": 0, "blocked": 0, "normal": 0}

@app.get("/api/feed", tags=["SOC"])
def feed(limit: int = 50, _: dict = Depends(verify_token)):
    API_REQUESTS.labels(endpoint="/api/feed", method="GET").inc()
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT timestamp, src, dst, proto, prediction, confidence, tag, blocked
            FROM detections ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

@app.get("/api/breakdown", tags=["SOC"])
def breakdown(_: dict = Depends(verify_token)):
    API_REQUESTS.labels(endpoint="/api/breakdown", method="GET").inc()
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT prediction, COUNT(*) as count
            FROM detections GROUP BY prediction ORDER BY count DESC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

@app.get("/api/timeline", tags=["SOC"])
def timeline(_: dict = Depends(verify_token)):
    API_REQUESTS.labels(endpoint="/api/timeline", method="GET").inc()
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT timestamp, tag, COUNT(*) as count
            FROM detections GROUP BY timestamp, tag
            ORDER BY rowid DESC LIMIT 120
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

@app.get("/api/blocked", tags=["SOC"])
def blocked_ips(_: dict = Depends(verify_token)):
    API_REQUESTS.labels(endpoint="/api/blocked", method="GET").inc()
    try:
        conn = get_db()
        rows = conn.execute("""
            SELECT src, dst, timestamp, prediction, confidence
            FROM detections WHERE blocked=1 ORDER BY id DESC LIMIT 20
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

@app.get("/api/training", tags=["ML"])
def training(_: dict = Depends(verify_token)):
    API_REQUESTS.labels(endpoint="/api/training", method="GET").inc()
    result = {}
    files = {
        "baseline":   "models/baseline_history.json",
        "federated":  "models/federated_history.json",
        "multiclass": "models/multiclass_history.json",
        "iid":        "models/federated_multiclass_history.json",
        "noniid":     "models/federated_noniid_history.json",
    }
    for key, path in files.items():
        full = os.path.join(BASE, path)
        try:
            with open(full) as f:
                result[key] = json.load(f)
        except Exception:
            result[key] = []
    return result

@app.get("/api/shap", tags=["ML"])
def shap(_: dict = Depends(verify_token)):
    API_REQUESTS.labels(endpoint="/api/shap", method="GET").inc()
    try:
        with open(os.path.join(BASE, "models", "shap_results.json")) as f:
            return json.load(f)
    except Exception:
        return {"feature_importance": []}

@app.get("/api/drift", tags=["ML"])
def drift(_: dict = Depends(verify_token)):
    API_REQUESTS.labels(endpoint="/api/drift", method="GET").inc()
    try:
        with open(os.path.join(BASE, "models", "drift_log.json")) as f:
            logs = json.load(f)
            return logs[-1] if logs else {}
    except Exception:
        return {}