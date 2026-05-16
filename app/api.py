from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas import (
    AnalysisResponse, TrainingStatsResponse, TrainingUploadResponse,
)
from app.services.analyzer import AnalysisService
from app.training.trainer import Trainer

router = APIRouter()
analyzer = AnalysisService()
trainer = Trainer(embedder=analyzer.embedder, index=analyzer.visual_index)


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    scan_id: str = Form(...),
    certificate: UploadFile = File(...),
) -> AnalysisResponse:
    return await analyzer.analyze(scan_id=scan_id, certificate=certificate)


@router.post("/train/upload", response_model=TrainingUploadResponse)
async def train_upload(
    cert_type: str = Form(...),
    certificate: UploadFile = File(...),
    institution: Optional[str] = Form(None),
) -> TrainingUploadResponse:
    """
    Upload an authentic certificate as a training sample.

    What happens:
    - OCR text is added to the text classifier corpus
    - Image is embedded and added to the visual similarity index
    - Text classifier auto-retrains after 10 new samples
    """
    return await trainer.ingest(
        certificate=certificate,
        cert_type=cert_type,
        institution=institution,
    )


@router.post("/train/retrain")
def train_retrain(force: bool = False) -> dict:
    """Force a text classifier retraining run."""
    result = trainer.retrain()
    analyzer.classifier.reload_model()
    return result


@router.get("/train/stats", response_model=TrainingStatsResponse)
def train_stats() -> TrainingStatsResponse:
    return trainer.stats()