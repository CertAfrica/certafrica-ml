from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CertificateTypePrediction(BaseModel):
    label: str = "unknown_document"
    display_name: str = "Unknown Document"
    confidence: float = 0.0
    matched_keywords: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class InstitutionPrediction(BaseModel):
    name: str = "Unknown Institution"
    canonical_name: str = "Unknown Institution"
    confidence: float = 0.0
    matched_text: str = ""
    evidence: List[str] = Field(default_factory=list)


class DocumentTypePrediction(BaseModel):
    label: str = "not_document"
    display_name: str = "Not a Document"
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)


class AIInterpretationRequest(BaseModel):
    scanId: str
    trustScore: int = 0
    verificationStatus: str = "inconclusive"
    ocrResult: Dict[str, Any] = Field(default_factory=dict)
    anomalyResult: Dict[str, Any] = Field(default_factory=dict)
    heatmapNotes: List[str] = Field(default_factory=list)


class AIInterpretationResponse(BaseModel):
    scanId: str
    documentType: DocumentTypePrediction
    certificateType: CertificateTypePrediction
    institution: InstitutionPrediction
    refinedTrustScore: int
    riskLevel: str
    confidenceLevel: str
    verificationSummary: str
    detailedExplanation: str
    recommendedAction: str
    anomalyNotes: List[str] = Field(default_factory=list)
    cleanedOCRText: str = ""
    reasoning: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
