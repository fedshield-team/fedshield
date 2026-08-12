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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_incident_report import generate_incident_report, get_cached_report

router = APIRouter(prefix="/incidents", tags=["incident-reports"])


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


@router.get("/{incident_id}/report")
def read_incident_report(incident_id: str):
    """Fetch a cached report only — does not call the LLM. Fast path for the dashboard."""
    cached = get_cached_report(incident_id)
    if cached is None:
        raise HTTPException(status_code=404, detail="No report generated yet for this incident")
    return cached


@router.post("/{incident_id}/report")
def create_incident_report(incident_id: str, body: IncidentReportRequest):
    """Generate (or retrieve cached) report. Call this from live_capture.py right after detection,
    or lazily from the dashboard the first time an analyst opens an incident."""
    report_text = generate_incident_report(
        incident_id=incident_id,
        flow_stats=body.flow_stats.dict(),
        prediction=body.prediction,
        confidence=body.confidence,
        shap_features=body.shap_features
    )
    return {"incident_id": incident_id, "report_text": report_text}