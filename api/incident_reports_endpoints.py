"""
FedShield API — Incident Reports Router

Lives inside api/, next to main.py.

Add to api/main.py, right after the app.add_middleware(...) block:

    from incident_reports_endpoints import router as incident_router
    app.include_router(incident_router, dependencies=[Depends(verify_token)])

Passing dependencies=[Depends(verify_token)] to include_router applies the
same JWT auth as every other protected endpoint in main.py, WITHOUT this
file needing to import anything from main.py — avoids a circular import
between this file and main.py.
"""

import os
import sqlite3

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model import MULTICLASS_CLASS_NAMES
from llm_incident_report import generate_incident_report, get_cached_report

router = APIRouter(prefix="/incidents", tags=["incident-reports"])
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "models", "fedshield_logs.db")


class FlowStats(BaseModel):
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str
    duration_ms: float
    packet_count: int
    byte_count: int


class IncidentReportRequest(BaseModel):
    incident_id: str
    flow_stats: FlowStats
    prediction: str
    confidence: float
    shap_features: list[tuple[str, float]]


def _detection_exists(incident_id: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH, timeout=5) as conn:
            return conn.execute(
                """
                SELECT 1
                FROM detections
                WHERE incident_id=?
                LIMIT 1
                """,
                (incident_id,),
            ).fetchone() is not None
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Detection store unavailable: {exc}",
        ) from exc


@router.get("/{incident_id}/report")
def read_incident_report(incident_id: str):
    """Fetch a cached report only — does not call the LLM. Fast path for the dashboard."""
    if not _detection_exists(incident_id):
        raise HTTPException(
            status_code=404,
            detail="No detection exists for this incident",
        )
    cached = get_cached_report(incident_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="No report generated yet for this incident")
    return cached


@router.post("/{incident_id}/report")
def create_incident_report(incident_id: str, body: IncidentReportRequest):
    """Generate (or retrieve cached) report. Call this from live_capture.py right after detection,
    or lazily from the dashboard the first time an analyst opens an incident."""
    if incident_id != body.incident_id:
        raise HTTPException(
            status_code=400,
            detail="Incident ID in the path must match the request body",
        )

    if not _detection_exists(incident_id):
        raise HTTPException(
            status_code=404,
            detail="No detection exists for this incident",
        )

    if not 0 <= body.confidence <= 1:
        raise HTTPException(
            status_code=422,
            detail="Confidence must be between 0 and 1",
        )

    if body.prediction not in MULTICLASS_CLASS_NAMES:
        raise HTTPException(
            status_code=422,
            detail="Prediction is not a supported multiclass class",
        )

    report_text = generate_incident_report(
        incident_id=incident_id,
        flow_stats=body.flow_stats.model_dump(),
        prediction=body.prediction,
        confidence=body.confidence,
        shap_features=body.shap_features
    )

    if report_text is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Incident report generation is unavailable; "
                "the underlying detection remains recorded"
            ),
        )

    return {"incident_id": incident_id, "report_text": report_text}