from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from app.ai.models import CertificateTypePrediction, DocumentTypePrediction, InstitutionPrediction


class VerificationPayload(BaseModel):
    trustScore: int
    refinedTrustScore: int
    riskLevel: str
    confidenceLevel: str
    status: str
    recommendedAction: str


class DocumentTypeView(BaseModel):
    label: str = "unknown_document"
    displayName: str = "Unknown Document"
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)


class CertificateTypeView(BaseModel):
    label: str = "unknown_document"
    displayName: str = "Unknown Document"
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)


class InstitutionView(BaseModel):
    name: str = "Unknown Institution"
    confidence: float = 0.0
    evidence: List[str] = Field(default_factory=list)


class DocumentAnalysisPayload(BaseModel):
    documentType: DocumentTypeView
    certificateType: CertificateTypeView
    institution: InstitutionView


class OCRAnalysisPayload(BaseModel):
    confidence: float = 0.0
    language: str = "eng"
    rawText: str = ""
    extractedFields: Dict[str, str] = Field(default_factory=dict)


class RegionCoordinates(BaseModel):
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0


class SuspiciousRegionView(BaseModel):
    coordinates: RegionCoordinates
    severity: float = 0.0
    category: str = ""


class FindingCategories(BaseModel):
    possibleTampering: int = 0
    irregularSpacing: int = 0
    moderateAnomaly: int = 0


class FindingSummary(BaseModel):
    totalRegions: int = 0
    categories: FindingCategories = Field(default_factory=FindingCategories)


class AnomalyDetectionPayload(BaseModel):
    overallScore: float = 0.0
    findingSummary: FindingSummary
    suspiciousRegions: List[SuspiciousRegionView] = Field(default_factory=list)


class LayoutAnalysisPayload(BaseModel):
    hasSignatureArea: bool = False
    hasTableStructure: bool = False
    hasHeaderArea: bool = False
    hasFooterArea: bool = False
    blockCount: int = 0


class ArtifactsPayload(BaseModel):
    uploadUrl: str
    heatmapUrl: str
    reportUrl: str
    previewUrl: str


class AnalysisResponse(BaseModel):
    scanId: str
    analysisTimestamp: datetime
    processingTime: str

    verification: VerificationPayload
    documentAnalysis: DocumentAnalysisPayload
    ocrAnalysis: OCRAnalysisPayload
    anomalyDetection: AnomalyDetectionPayload
    layoutAnalysis: LayoutAnalysisPayload
    reasoning: List[str] = Field(default_factory=list)
    artifacts: ArtifactsPayload
    aiOutput: Dict[str, Any] = Field(default_factory=dict)


# Backward-compatible models still used across the codebase
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
