from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from app.ai.models import AIInterpretationRequest, AIInterpretationResponse
from app.ai.services.ai_layer_service import AILayerService

router = APIRouter(prefix="/ai", tags=["ai-layer"])
service = AILayerService()


@router.get("/health")
def ai_health() -> Dict[str, str]:
    return {"status": "ok", "layer": "certafrica-ai"}


@router.post("/interpret", response_model=AIInterpretationResponse)
def interpret_document(payload: AIInterpretationRequest) -> AIInterpretationResponse:
    return service.interpret(payload.model_dump())


@router.post("/train/sample")
def train_sample() -> Dict[str, Any]:
    samples = [
        {"label": "university_degree", "text": "University of Lagos Bachelor of Science awarded"},
        {"label": "waec_certificate", "text": "WAEC West African Examinations Council result"},
        {"label": "nysc_certificate", "text": "NYSC National Youth Service Corps discharge certificate"},
    ]
    model_path = service.type_classifier.train(samples)
    return {"status": "trained", "modelPath": str(model_path)}
