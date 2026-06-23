"""
FedShield API — FastAPI backend with JWT Authentication
Secured endpoints requiring Bearer token authentication.

Auth flow:
    POST /auth/token  →  returns JWT access token (24h expiry)
    GET  /api/*       →  requires Authorization: Bearer <token>

Default credentials (change in production):
    username: fedshield
    password: shield2025
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY  = "fedshield-jwt-secret-key-change-in-production-2025"
ALGORITHM   = "HS256"
TOKEN_EXPIRE_HOURS = 24

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "models", "fedshield_logs.db")

# ── Users (in production: store in DB with hashed passwords) ───────────────────
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

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

# ── JWT helpers ────────────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    role: str

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
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
    description="Privacy-preserving IDS — secured with JWT authentication",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None

# ── Auth endpoints (public) ────────────────────────────────────────────────────
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
    return {
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   TOKEN_EXPIRE_HOURS * 3600,
        "role":         user["role"],
    }

@app.get("/auth/me", tags=["Auth"])
def get_me(current_user: dict = Depends(verify_token)):
    return current_user

# ── Health (public) ────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
def health():
    return {"status": "ok", "db": os.path.exists(DB_PATH), "auth": "JWT HS256"}

# ── Protected endpoints ────────────────────────────────────────────────────────
@app.get("/api/stats", tags=["SOC"])
def stats(_: dict = Depends(verify_token)):
    try:
        conn = get_db()
        total   = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        attacks = conn.execute("SELECT COUNT(*) FROM detections WHERE tag='ATTACK'").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM detections WHERE blocked=1").fetchone()[0]
        conn.close()
        return {"total": total, "attacks": attacks, "blocked": blocked, "normal": total - attacks}
    except Exception:
        return {"total": 0, "attacks": 0, "blocked": 0, "normal": 0}

@app.get("/api/feed", tags=["SOC"])
def feed(limit: int = 50, _: dict = Depends(verify_token)):
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
    try:
        with open(os.path.join(BASE, "models", "shap_results.json")) as f:
            return json.load(f)
    except Exception:
        return {"feature_importance": []}

@app.get("/api/drift", tags=["ML"])
def drift(_: dict = Depends(verify_token)):
    try:
        with open(os.path.join(BASE, "models", "drift_log.json")) as f:
            logs = json.load(f)
            return logs[-1] if logs else {}
    except Exception:
        return {}