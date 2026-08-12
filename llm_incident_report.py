"""
FedShield — LLM-Powered Incident Reports (Groq version)
Generates a natural-language explanation for each detected attack,
using flow stats + SHAP top features + model prediction as context.
Reports are cached in SQLite so we never re-call the API for the
same incident twice.

Requires: pip install groq
Requires: GROQ_API_KEY environment variable set (free key from console.groq.com,
no credit card needed).

Lives at the repo root (same folder as live_capture.py, model.py) so that
live_capture.py's "from llm_incident_report import ..." keeps working.
"""

import os
import sqlite3
from datetime import datetime

from groq import Groq

# Same absolute-path resolution as api/main.py, so both always agree on the
# DB file location regardless of which folder the process is launched from.
BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "models", "fedshield_logs.db")

# gpt-oss-120b is Groq's current recommended general-purpose model
# (replaced llama-3.3-70b-versatile, which Groq deprecated in June 2026).
# Free tier: 30 requests/min, 1000 requests/day, 8000 tokens/min, 200000 tokens/day.
MODEL = "openai/gpt-oss-120b"

client = Groq()  # reads GROQ_API_KEY from env automatically


def init_incident_reports_table():
    """Call this once at startup (e.g. alongside your other DB init code)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incident_reports (
            incident_id     TEXT PRIMARY KEY,
            predicted_class TEXT,
            confidence      REAL,
            report_text     TEXT,
            created_at      TEXT
        )
    """)
    conn.commit()
    conn.close()


def _build_prompt(flow_stats: dict, prediction: str, confidence: float, shap_features: list) -> str:
    """
    flow_stats: dict of raw/derived flow features, e.g.
        {"src_ip": "...", "dst_ip": "...", "dst_port": 445,
         "protocol": "TCP", "duration_ms": 120, "packet_count": 5000,
         "byte_count": 2_400_000, "flags": "SYN,SYN,SYN..."}
    prediction: one of "Normal", "DoS", "Probe", "R2L", "U2R"
    confidence: model's softmax/sigmoid confidence, 0-1
    shap_features: list of (feature_name, shap_value) tuples, top-N by |value|
    """
    shap_lines = "\n".join(
        f"  - {name}: {value:+.4f}" for name, value in shap_features
    )
    flow_lines = "\n".join(f"  - {k}: {v}" for k, v in flow_stats.items())

    return f"""You are a security analyst assistant embedded in a network intrusion detection system (FedShield). A machine learning model just flagged a network flow. Write a short incident explanation for a SOC analyst.

MODEL PREDICTION: {prediction} (confidence: {confidence:.1%})

FLOW STATISTICS:
{flow_lines}

TOP CONTRIBUTING FEATURES (SHAP values, positive = pushed toward this classification):
{shap_lines}

Write a concise incident report with exactly these sections:
1. **What happened** (1-2 sentences, plain language, no jargon dump)
2. **Why the model flagged it** (reference the SHAP features in plain terms)
3. **Suggested action** (1-2 concrete next steps for the analyst)

Keep the whole thing under 120 words. Do not repeat raw numbers already shown above verbatim; interpret them."""


def generate_incident_report(incident_id: str, flow_stats: dict, prediction: str,
                              confidence: float, shap_features: list) -> str:
    """
    Generates (or retrieves cached) incident report.
    Returns the report text.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT report_text FROM incident_reports WHERE incident_id = ?",
        (incident_id,)
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return row[0]

    # Not cached — call the API
    prompt = _build_prompt(flow_stats, prediction, confidence, shap_features)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=700,  # gpt-oss burns some of this on internal reasoning first
            reasoning_effort="low",  # keep reasoning brief so more budget reaches the actual answer
            messages=[{"role": "user", "content": prompt}]
        )
        report_text = response.choices[0].message.content
        if not report_text:
            # Reasoning still ate the whole budget — surface what we can instead of silently failing
            reasoning_snippet = getattr(response.choices[0].message, "reasoning", "") or ""
            report_text = (
                f"[Model returned empty content, likely truncated by reasoning tokens. "
                f"Reasoning trace: {reasoning_snippet[:200]}]"
            )
    except Exception as e:
        report_text = (
            f"[LLM report generation failed: {e}] "
            f"Prediction: {prediction} ({confidence:.1%} confidence). "
            f"Top feature: {shap_features[0][0] if shap_features else 'N/A'}."
        )

    conn.execute(
        "INSERT OR REPLACE INTO incident_reports "
        "(incident_id, predicted_class, confidence, report_text, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (incident_id, prediction, confidence, report_text, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return report_text


def get_cached_report(incident_id: str):
    """Fetch a cached report without generating one. Returns None if not found."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT report_text, predicted_class, confidence, created_at "
        "FROM incident_reports WHERE incident_id = ?",
        (incident_id,)
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "report_text": row[0],
        "predicted_class": row[1],
        "confidence": row[2],
        "created_at": row[3]
    }


if __name__ == "__main__":
    # Quick manual test
    init_incident_reports_table()
    test_report = generate_incident_report(
        incident_id="test-001",
        flow_stats={
            "src_ip": "192.168.1.45", "dst_ip": "10.0.0.12",
            "dst_port": 445, "protocol": "TCP",
            "duration_ms": 80, "packet_count": 8200, "byte_count": 1_950_000
        },
        prediction="DoS",
        confidence=0.94,
        shap_features=[
            ("dst_host_serror_rate", 0.42),
            ("count", 0.31),
            ("srv_count", 0.18)
        ]
    )
    print(test_report)