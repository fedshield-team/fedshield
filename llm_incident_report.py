"""FedShield — Groq-powered incident reports with SQLite caching."""

import os
import sqlite3

from datetime import datetime, timezone

from groq import Groq


BASE = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.join(
    BASE,
    "models",
    "fedshield_logs.db"
)

MODEL = "openai/gpt-oss-120b"


def _connect():

    os.makedirs(
        os.path.dirname(DB_PATH),
        exist_ok=True
    )

    return sqlite3.connect(
        DB_PATH,
        timeout=10
    )


def _client():

    if not os.getenv(
        "GROQ_API_KEY"
    ):

        raise RuntimeError(
            "GROQ_API_KEY is not configured"
        )

    return Groq(
        timeout=8.0
    )


def init_incident_reports_table():

    with _connect() as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_reports (
                incident_id TEXT PRIMARY KEY,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                report_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def _build_prompt(
    flow_stats,
    prediction,
    confidence,
    shap_features
):

    shap_lines = "\n".join(
        f"  - {name}: {value:+.4f}"
        for name, value
        in shap_features
    )

    if not shap_lines:
        shap_lines = (
            "  - unavailable"
        )

    flow_lines = "\n".join(
        f"  - {key}: {value}"
        for key, value
        in flow_stats.items()
    )

    return f"""
You are a security analyst assistant embedded
in a network intrusion detection system (FedShield).

Write a short incident explanation for a SOC analyst.

MODEL PREDICTION:
{prediction}

MODEL CONFIDENCE:
{confidence:.1%}

FLOW STATISTICS:
{flow_lines}

TOP CONTRIBUTING FEATURES:
{shap_lines}

Use exactly these sections:

1. **What happened**
   1-2 sentences.

2. **Why the model flagged it**
   Explain the strongest SHAP features in plain language.

3. **Suggested action**
   Give 1-2 concrete defensive next steps.

Keep the whole report under 120 words.

Do not invent facts that are not present
in the supplied data.
"""


def generate_incident_report(
    incident_id,
    flow_stats,
    prediction,
    confidence,
    shap_features
):

    init_incident_reports_table()

    with _connect() as conn:

        row = conn.execute(
            """
            SELECT report_text
            FROM incident_reports
            WHERE incident_id=?
            """,
            (incident_id,)
        ).fetchone()

    if row:

        return row[0]

    try:

        response = (
            _client()
            .chat
            .completions
            .create(
                model=MODEL,
                max_tokens=700,
                reasoning_effort="low",
                messages=[
                    {
                        "role": "user",
                        "content": _build_prompt(
                            flow_stats,
                            prediction,
                            confidence,
                            shap_features
                        )
                    }
                ]
            )
        )

        report_text = (
            response
            .choices[0]
            .message
            .content
            or ""
        ).strip()

        if not report_text:
            print(
                "[IncidentReport] Groq returned no report content."
            )
            return None

    except Exception as e:

        print(
            "[IncidentReport] Report generation unavailable: "
            f"{e}"
        )
        return None

    with _connect() as conn:

        conn.execute(
            """
            INSERT OR REPLACE INTO incident_reports
            (
                incident_id,
                predicted_class,
                confidence,
                report_text,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                incident_id,
                prediction,
                float(confidence),
                report_text,
                datetime.now(
                    timezone.utc
                ).isoformat()
            )
        )

    return report_text


def get_cached_report(
    incident_id
):

    init_incident_reports_table()

    with _connect() as conn:

        row = conn.execute(
            """
            SELECT
                report_text,
                predicted_class,
                confidence,
                created_at
            FROM incident_reports
            WHERE incident_id=?
            """,
            (incident_id,)
        ).fetchone()

    if not row:
        return None

    return {
        "report_text": row[0],
        "predicted_class": row[1],
        "confidence": row[2],
        "created_at": row[3]
    }


if __name__ == "__main__":

    init_incident_reports_table()

    print(
        "Incident report table ready."
    )