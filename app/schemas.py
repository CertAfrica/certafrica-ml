from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OCRBlock(BaseModel):
    text: str = ""
    confidence: float = 0.0
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0


class OCRResult(BaseModel):
    raw_text: str = ""
    cleaned_text: str = ""
    extracted_fields: Dict[str, str] = Field(default_factory=dict)
    blocks: List[OCRBlock] = Field(default_factory=list)
    average_confidence: float = 0.0
    language: str = "en"


class SuspiciousRegion(BaseModel):
    left: int
    top: int
    width: int
    height: int
    severity: float
    category: str
    area: str


class AnomalyResult(BaseModel):
    overall_score: float = 0.0
    suspicious_regions: List[SuspiciousRegion] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)


class LayoutHints(BaseModel):
    has_signature_area: bool = False
    has_table_structure: bool = False
    has_header_area: bool = False
    has_footer_area: bool = False
    block_count: int = 0


class Classification(BaseModel):
    document_type: str = "unknown_document"
    document_type_display: str = "Unknown Document"
    document_type_confidence: float = 0.0
    certificate_type: str = "unknown_document"
    certificate_type_display: str = "Unknown Document"
    certificate_type_confidence: float = 0.0
    institution: str = "Unknown Institution"
    institution_confidence: float = 0.0
    institution_evidence: List[str] = Field(default_factory=list)
    classification_evidence: List[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    ocr: float
    institution: float
    classification: float
    anomaly: float
    layout: float
    visual: Optional[float] = None  # null when no references for predicted type


class Verdict(BaseModel):
    trust_score: int
    risk_level: str
    confidence: str
    recommended_action: str
    breakdown: ScoreBreakdown
    reasoning: List[str] = Field(default_factory=list)


class RegionCategory(BaseModel):
    category: str
    area: str
    count: int


class AnomalySummary(BaseModel):
    score: float
    total_regions: int
    high_severity_regions: int
    by_category: List[RegionCategory]
    top_regions: List[SuspiciousRegion]


class DocumentSummary(BaseModel):
    type: str
    type_display: str
    type_confidence: float
    certificate_type: str
    certificate_type_display: str
    certificate_type_confidence: float
    institution: str
    institution_confidence: float


class OCRSummary(BaseModel):
    confidence: float
    language: str
    text: str
    fields: Dict[str, str]


class VisualMatchView(BaseModel):
    sample_id: str
    label: str
    institution: str
    similarity: float


class VisualSimilarityView(BaseModel):
    score: Optional[float]
    best_similarity: float
    reference_count: int
    matches: List[VisualMatchView] = Field(default_factory=list)
    note: str = ""


class Artifacts(BaseModel):
    upload_url: str
    heatmap_url: str
    report_url: str
    preview_url: str


class AnalysisResponse(BaseModel):
    scan_id: str
    timestamp: datetime
    processing_time: str
    verdict: Verdict
    document: DocumentSummary
    ocr: OCRSummary
    anomalies: AnomalySummary
    layout: LayoutHints
    visual: VisualSimilarityView
    explanation: str
    artifacts: Artifacts


class TrainingUploadResponse(BaseModel):
    sample_id: str
    label: str
    extracted_text_preview: str
    total_samples_for_label: int
    retrained: bool
    visual_indexed: bool


class TrainingStatsResponse(BaseModel):
    samples_per_label: Dict[str, int]
    total_samples: int
    visual_references_per_label: Dict[str, int]
    total_visual_references: int
    last_trained_at: Optional[datetime] = None
    model_exists: bool