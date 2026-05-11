from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from app.ai.models import CertificateTypePrediction, DocumentTypePrediction, InstitutionPrediction


class OCRBlock(BaseModel):
    text: str = ""
    confidence: float = 0.0
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0


class OCRResult(BaseModel):
    raw_text: str = ""
    extracted_fields: Dict[str, str] = Field(default_factory=dict)
    blocks: List[OCRBlock] = Field(default_factory=list)
    average_confidence: float = 0.0
    language: str = "eng"


class AnomalyResult(BaseModel):
    overall_score: float = 0.0
    findings: List[str] = Field(default_factory=list)
    suspicious_regions: List[Dict[str, Any]] = Field(default_factory=list)
    heatmap_notes: List[str] = Field(default_factory=list)


class VerificationArtifact(BaseModel):
    path: str
    url: str
    mime_type: str


class AnalysisResponse(BaseModel):
    scanId: str
    trustScore: int
    verificationStatus: str
    documentType: Optional[DocumentTypePrediction] = None
    ocrResult: OCRResult
    anomalyResult: AnomalyResult
    aiCertificateType: Optional[CertificateTypePrediction] = None
    aiInstitution: Optional[InstitutionPrediction] = None
    refinedTrustScore: Optional[int] = None
    riskLevel: Optional[str] = None
    confidenceLevel: Optional[str] = None
    verificationSummary: Optional[str] = None
    detailedExplanation: Optional[str] = None
    recommendedAction: Optional[str] = None
    cleanedOCRText: str = ""
    aiReasoning: List[str] = Field(default_factory=list)
    aiMetadata: Dict[str, Any] = Field(default_factory=dict)
    heatmapImageUrl: str
    reportUrl: str
    processingTime: str
    analysisTimestamp: datetime
    artifactPaths: Dict[str, str] = Field(default_factory=dict)
