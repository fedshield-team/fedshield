"""
FedShield API - FastAPI backend with JWT Authentication + Prometheus Metrics
"""

import sqlite3
import json
import ipaddress
import logging
import os
from dotenv import load_dotenv
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
load_dotenv()

from fedshield_runtime import runtime
from model import MULTICLASS_CLASS_NAMES


logger = logging.getLogger("fedshield.api")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable is required")

ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "models", "fedshield_logs.db")
WEB_DIST = Path(BASE) / "web" / "dist"

capture_module = None
capture_thread = None
capture_start_lock = threading.Lock()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Prometheus metrics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Model scores from the recorded training artifacts
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

MODEL_F1_ARTIFACTS = {
    "federated_noniid": ("federated_noniid_history.json", "macro_f1"),
    "federated_iid": ("federated_multiclass_history.json", "macro_f1"),
    "centralized": ("multiclass_history.json", "macro_f1"),
    "binary_federated": ("federated_history.json", "f1"),
}

for model_type, (filename, metric) in MODEL_F1_ARTIFACTS.items():
    try:
        with open(os.path.join(BASE, "models", filename), encoding="utf-8") as file:
            history = json.load(file)
        if history:
            MODEL_F1_SCORE.labels(model_type=model_type).set(history[-1][metric])
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        logger.warning("Model score artifact unavailable for %s", model_type)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Authentication
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pwd_context = CryptContext(
    schemes=["sha256_crypt"],
    deprecated="auto"
)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/token"
)


# Credentials are supplied through environment variables.
# Passwords are never stored in source code.

FEDSHIELD_USERNAME = os.getenv(
    "FEDSHIELD_USERNAME",
    "fedshield"
)

FEDSHIELD_PASSWORD = os.getenv(
    "FEDSHIELD_PASSWORD"
)

ANALYST_USERNAME = os.getenv(
    "ANALYST_USERNAME",
    "analyst"
)

ANALYST_PASSWORD = os.getenv(
    "ANALYST_PASSWORD"
)


if not FEDSHIELD_PASSWORD:
    raise RuntimeError(
        "FEDSHIELD_PASSWORD environment variable is required"
    )

if not ANALYST_PASSWORD:
    raise RuntimeError(
        "ANALYST_PASSWORD environment variable is required"
    )


USERS = {
    FEDSHIELD_USERNAME: {
        "username": FEDSHIELD_USERNAME,
        "hashed_password": pwd_context.hash(
            FEDSHIELD_PASSWORD
        ),
        "role": "admin",
    },
    ANALYST_USERNAME: {
        "username": ANALYST_USERNAME,
        "hashed_password": pwd_context.hash(
            ANALYST_PASSWORD
        ),
        "role": "readonly",
    },
}


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    role: str


def create_token(data: dict) -> str:
    payload = data.copy()

    payload["exp"] = (
        datetime.utcnow()
        + timedelta(hours=TOKEN_EXPIRE_HOURS)
    )

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def verify_token(
    token: str = Depends(oauth2_scheme)
) -> dict:

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")

        if not username or username not in USERS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={
                    "WWW-Authenticate": "Bearer"
                },
            )

        return {
            "username": username,
            "role": USERS[username]["role"],
        }

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )


def require_admin(current_user: dict = Depends(verify_token)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return current_user


def start_live_capture():
    """Start the existing packet capture loop once, without blocking the API."""
    global capture_module, capture_thread
    with capture_start_lock:
        if capture_thread is not None and capture_thread.is_alive():
            return

        try:
            import live_capture
        except Exception as exc:
            print(f"[WARN] Live capture unavailable: {exc}")
            return

        capture_module = live_capture
        capture_thread = threading.Thread(
            target=live_capture.start_capture,
            name="fedshield-live-capture",
            daemon=True,
        )
        capture_thread.start()


def live_capture_status():
    return {
        "enabled": capture_module is not None,
        "running": bool(
            capture_module is not None
            and getattr(capture_module, "capture_running", False)
        ),
        "thread_alive": bool(
            capture_thread is not None and capture_thread.is_alive()
        ),
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# FastAPI application
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app = FastAPI(
    title="FedShield API",
    description=(
        "Privacy-preserving IDS â€” "
        "JWT secured + Prometheus metrics"
    ),
    version="2.1.0",
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CORS
#
# Local React/Vite development:
#     http://localhost:3000
#
# Production:
# React and FastAPI are served through the same Nginx origin,
# so CORS is not required between them.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Incident Reports
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from .incident_reports_endpoints import router as incident_router

app.include_router(
    incident_router,
    dependencies=[Depends(verify_token)]
)


@app.on_event("startup")
def initialize_runtime():
    """Initialize shared stores and optionally attach real packet capture."""
    from llm_incident_report import init_incident_reports_table
    from online_retrain import init_retrain_buffer

    init_incident_reports_table()
    init_retrain_buffer()
    if os.getenv("FEDSHIELD_START_CAPTURE", "0").lower() not in {
        "0", "false", "no"
    }:
        start_live_capture()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Database
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    except Exception:
        return None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Prometheus metric updater
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def update_prometheus_metrics():
    """
    Pull the latest statistics from SQLite and update
    all Prometheus gauges.
    """

    try:
        conn = get_db()

        if not conn:
            return

        total = conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]

        attacks = conn.execute(
            "SELECT COUNT(*) FROM detections "
            "WHERE tag='ATTACK'"
        ).fetchone()[0]

        blocked = conn.execute(
            "SELECT COUNT(*) FROM detections "
            "WHERE blocked=1"
        ).fetchone()[0]

        normal = total - attacks

        confidence = conn.execute(
            "SELECT AVG(confidence) FROM detections"
        ).fetchone()[0] or 0

        breakdown = conn.execute(
            """
            SELECT prediction, COUNT(*)
            FROM detections
            GROUP BY prediction
            """
        ).fetchall()

        conn.close()

        PACKETS_TOTAL.set(total)
        ATTACKS_TOTAL.set(attacks)
        BLOCKED_TOTAL.set(blocked)
        NORMAL_TOTAL.set(normal)

        ATTACK_RATE.set(
            round(
                attacks / max(total, 1),
                4
            )
        )

        AVG_CONFIDENCE.set(
            round(confidence, 4)
        )

        for row in breakdown:
            ATTACK_BY_TYPE.labels(
                prediction=row[0]
            ).set(row[1])

    except Exception:
        pass


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Public endpoints
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/api/health",
    tags=["System"]
)
def health():

    return {
        "status": "ok",
        "db": os.path.exists(DB_PATH),
        "auth": "JWT HS256",
        "metrics": "Prometheus /metrics",
        "version": "2.1.0",
    }


def _read_training_history(filename):
    """Read one real training artifact without exposing filesystem details."""
    path = os.path.join(BASE, "models", filename)
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _configured_federated_node_count():
    """Count the Flower client services declared by the existing compose file."""
    compose_path = os.path.join(BASE, "docker-compose.yml")
    try:
        with open(compose_path, encoding="utf-8") as file:
            return len(re.findall(r"^\s{2}node\d+:", file.read(), re.MULTILINE))
    except OSError:
        return None


@app.get(
    "/api/public-summary",
    tags=["System"],
)
def public_summary():
    """Expose non-sensitive real experiment values needed before login."""
    result = {
        "binary_federated_f1": None,
        "multiclass_noniid_f1": None,
        "configured_node_count": _configured_federated_node_count(),
        "errors": {},
    }

    sources = {
        "binary_federated_f1": ("federated_history.json", "f1"),
        "multiclass_noniid_f1": (
            "federated_noniid_history.json",
            "macro_f1",
        ),
    }
    for key, (filename, metric) in sources.items():
        try:
            history = _read_training_history(filename)
            if history:
                result[key] = history[-1].get(metric)
            else:
                result["errors"][key] = "Training history unavailable"
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            logger.exception("Public training summary unavailable for %s", key)
            result["errors"][key] = "Training history unavailable"

    if not result["errors"]:
        result.pop("errors")
    return result


@app.get(
    "/metrics",
    tags=["Observability"]
)
def metrics():

    update_prometheus_metrics()

    API_REQUESTS.labels(
        endpoint="/metrics",
        method="GET"
    ).inc()

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Authentication endpoints
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post(
    "/auth/token",
    response_model=Token,
    tags=["Auth"]
)
def login(
    form: OAuth2PasswordRequestForm = Depends()
):

    user = USERS.get(form.username)

    if (
        not user
        or not pwd_context.verify(
            form.password,
            user["hashed_password"]
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    token = create_token(
        {
            "sub": form.username,
            "role": user["role"],
        }
    )

    API_REQUESTS.labels(
        endpoint="/auth/token",
        method="POST"
    ).inc()

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": TOKEN_EXPIRE_HOURS * 3600,
        "role": user["role"],
    }


@app.get(
    "/auth/me",
    tags=["Auth"]
)
def get_me(
    current_user: dict = Depends(verify_token)
):

    return current_user


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SOC â€” Statistics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/api/stats",
    tags=["SOC"]
)
def stats(
    _: dict = Depends(verify_token)
):

    API_REQUESTS.labels(
        endpoint="/api/stats",
        method="GET"
    ).inc()

    try:
        conn = get_db()

        if not conn:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Statistics data is unavailable",
            )

        total = conn.execute(
            "SELECT COUNT(*) FROM detections"
        ).fetchone()[0]

        attacks = conn.execute(
            "SELECT COUNT(*) FROM detections "
            "WHERE tag='ATTACK'"
        ).fetchone()[0]

        blocked = conn.execute(
            "SELECT COUNT(*) FROM detections "
            "WHERE blocked=1"
        ).fetchone()[0]

        conn.close()

        return {
            "total": total,
            "attacks": attacks,
            "blocked": blocked,
            "normal": total - attacks,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to load SOC statistics")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Statistics data is unavailable",
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SOC â€” Live feed
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/api/feed",
    tags=["SOC"]
)
def feed(
    limit: int = 50,
    _: dict = Depends(verify_token)
):

    API_REQUESTS.labels(
        endpoint="/api/feed",
        method="GET"
    ).inc()

    try:
        conn = get_db()

        if not conn:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Detection feed is unavailable",
            )

        rows = conn.execute(
            """
            SELECT
                timestamp,
                src,
                dst,
                proto,
                prediction,
                confidence,
                tag,
                blocked
            FROM detections
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        conn.close()

        return [
            dict(row)
            for row in rows
        ]

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to load detection feed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection feed is unavailable",
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SOC â€” Attack breakdown
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/api/breakdown",
    tags=["SOC"]
)
def breakdown(
    _: dict = Depends(verify_token)
):

    API_REQUESTS.labels(
        endpoint="/api/breakdown",
        method="GET"
    ).inc()

    try:
        conn = get_db()

        if not conn:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Attack breakdown is unavailable",
            )

        rows = conn.execute(
            """
            SELECT
                prediction,
                COUNT(*) AS count
            FROM detections
            GROUP BY prediction
            ORDER BY count DESC
            """
        ).fetchall()

        conn.close()

        return [
            dict(row)
            for row in rows
        ]

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to load attack breakdown")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Attack breakdown is unavailable",
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SOC â€” Timeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/api/timeline",
    tags=["SOC"]
)
def timeline(
    _: dict = Depends(verify_token)
):

    API_REQUESTS.labels(
        endpoint="/api/timeline",
        method="GET"
    ).inc()

    try:
        conn = get_db()

        if not conn:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Attack timeline is unavailable",
            )

        rows = conn.execute(
            """
            SELECT
                timestamp,
                tag,
                COUNT(*) AS count
            FROM detections
            GROUP BY timestamp, tag
            ORDER BY rowid DESC
            LIMIT 120
            """
        ).fetchall()

        conn.close()

        return [
            dict(row)
            for row in rows
        ]

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to load attack timeline")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Attack timeline is unavailable",
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SOC â€” Blocked IPs
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/api/blocked",
    tags=["SOC"]
)
def blocked_ips(
    _: dict = Depends(verify_token)
):

    API_REQUESTS.labels(
        endpoint="/api/blocked",
        method="GET"
    ).inc()

    try:
        conn = get_db()

        if not conn:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Blocked IP data is unavailable",
            )

        rows = conn.execute(
            """
            SELECT
                src,
                dst,
                timestamp,
                prediction,
                confidence
            FROM detections
            WHERE blocked=1
            ORDER BY id DESC
            LIMIT 20
            """
        ).fetchall()

        conn.close()

        return [
            dict(row)
            for row in rows
        ]

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to load blocked IP data")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Blocked IP data is unavailable",
        )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ML â€” Training history
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/api/training",
    tags=["ML"]
)
def training(
    _: dict = Depends(verify_token)
):

    API_REQUESTS.labels(
        endpoint="/api/training",
        method="GET"
    ).inc()

    files = {
        "baseline": "models/baseline_history.json",
        "federated": "models/federated_history.json",
        "multiclass": "models/multiclass_history.json",
        "iid": "models/federated_multiclass_history.json",
        "noniid": "models/federated_noniid_history.json",
    }
    result = {}
    errors = {}

    for key, path in files.items():
        try:
            result[key] = _read_training_history(path.removeprefix("models/"))
        except (OSError, ValueError, TypeError):
            logger.exception("Training history unavailable for %s", key)
            result[key] = []
            errors[key] = "Training history unavailable"

    if errors:
        result["errors"] = errors

    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ML â€” SHAP feature importance
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/api/shap",
    tags=["ML"]
)
def shap(
    _: dict = Depends(verify_token)
):

    API_REQUESTS.labels(
        endpoint="/api/shap",
        method="GET"
    ).inc()

    try:

        path = os.path.join(
            BASE,
            "models",
            "shap_results.json"
        )

        with open(
            path,
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (OSError, ValueError, TypeError):
        logger.exception("SHAP results unavailable")
        return {
            "feature_importance": [],
            "error": "SHAP results unavailable",
        }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ML â€” Data drift
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get(
    "/api/drift",
    tags=["ML"]
)
def drift(
    _: dict = Depends(verify_token)
):

    API_REQUESTS.labels(
        endpoint="/api/drift",
        method="GET"
    ).inc()

    try:
        from drift_detection import (
            DRIFT_CONFIG,
            analyze_training_history,
            detect_drift,
            get_live_metrics,
        )

        metrics = get_live_metrics(
            DRIFT_CONFIG["window_minutes"]
        )
        history = analyze_training_history()
        alerts = detect_drift(metrics, history)

        if "error" in metrics or "error" in history:
            drift_status = "service_unavailable"
        elif metrics["total"] < DRIFT_CONFIG["min_samples"]:
            drift_status = "insufficient_data"
        elif any(
            alert.get("level") == "CRITICAL"
            for alert in alerts
        ):
            drift_status = "drift_detected"
        else:
            drift_status = "no_drift"

        return {
            "status": drift_status,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "metrics": metrics,
            "history": history,
            "alerts": alerts,
            "thresholds": DRIFT_CONFIG,
        }

    except Exception:
        logger.exception("Drift analysis unavailable")
        return {
            "status": "service_unavailable",
            "error": "Drift analysis unavailable",
        }


@app.get(
    "/api/threat-intel/{ip_address}",
    tags=["Security"],
)
def threat_intelligence(
    ip_address: str,
    _: dict = Depends(verify_token),
):
    """Enrich an IP only after it appears in a real attack/blocked event."""
    API_REQUESTS.labels(
        endpoint="/api/threat-intel",
        method="GET",
    ).inc()

    try:
        normalized_ip = str(
            ipaddress.ip_address(ip_address.strip())
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid IP address",
        ) from exc

    conn = get_db()
    if conn is None:
        raise HTTPException(
            status_code=503,
            detail="Detection store unavailable",
        )

    try:
        detected = conn.execute(
            """
            SELECT 1
            FROM detections
            WHERE src=?
            AND (tag='ATTACK' OR blocked=1)
            LIMIT 1
            """,
            (normalized_ip,),
        ).fetchone()
    except sqlite3.Error as exc:
        logger.exception("Threat intelligence lookup failed")
        raise HTTPException(
            status_code=503,
            detail="Detection store unavailable",
        ) from exc
    finally:
        conn.close()

    if detected is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Threat intelligence is limited to IPs from "
                "recorded attack or blocked detections"
            ),
        )

    from threat_intel import check_ip

    result = check_ip(normalized_ip)
    result["source"] = "recorded_detection"
    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ML â€” Shared runtime and operational controls
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class InferenceRequest(BaseModel):
    features: list[float]
    scaled: bool = False
    explain: bool = True


@app.get(
    "/api/runtime",
    tags=["ML"],
)
def runtime_status(_: dict = Depends(verify_token)):
    API_REQUESTS.labels(
        endpoint="/api/runtime",
        method="GET",
    ).inc()
    result = runtime.status()
    result["capture"] = live_capture_status()
    return result


@app.post(
    "/api/inference",
    tags=["ML"],
)
def inference(
    body: InferenceRequest,
    _: dict = Depends(verify_token),
):
    API_REQUESTS.labels(
        endpoint="/api/inference",
        method="POST",
    ).inc()
    return runtime.predict(
        body.features,
        scaled=body.scaled,
        explain=body.explain,
    )


@app.post(
    "/api/live/start",
    tags=["Live detection"],
)
def start_live(_: dict = Depends(require_admin)):
    start_live_capture()
    return live_capture_status()


@app.post(
    "/api/retrain",
    tags=["ML"],
)
def retrain(_: dict = Depends(require_admin)):
    from online_retrain import run_incremental_retrain

    result = run_incremental_retrain()
    if result.get("accepted"):
        runtime.reload(force=True)
    result["runtime"] = runtime.status()
    return result


if WEB_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=WEB_DIST / "assets"),
        name="web-assets",
    )


@app.get("/{full_path:path}", include_in_schema=False)
def serve_react(full_path: str):
    """Serve the existing React SPA without adding a second frontend."""
    index_path = WEB_DIST / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="React dashboard has not been built yet",
        )
    return FileResponse(index_path)