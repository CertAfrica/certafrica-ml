from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from app.models.schemas import AnalysisResponse
from app.services.analysis_service import AnalysisService

router = APIRouter()
service = AnalysisService()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_certificate(scan_id: str = Form(...), certificate: UploadFile = File(...)) -> AnalysisResponse:
    return await service.analyze(scan_id=scan_id, certificate=certificate)
