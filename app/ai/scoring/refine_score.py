from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RefinedScoreResult:
    refined_trust_score: int
    risk_level: str
    confidence_level: str
    reasoning: List[str]
    adjustments: Dict[str, int]


class TrustScoreRefiner:
    def refine(
        self,
        ml_payload: Dict[str, Any],
        certificate_type_confidence: float,
        institution_confidence: float,
        document_type_confidence: float = 0.0,
    ) -> RefinedScoreResult:
        base_score = int(ml_payload.get("trustScore", 0) or 0)
        anomaly = ml_payload.get("anomalyResult", {}) or {}
        ocr = ml_payload.get("ocrResult", {}) or {}
        layout_hints = ml_payload.get("layoutHints", {}) or {}
        preprocessing = ml_payload.get("preprocessingMetadata", {}) or {}

        findings = anomaly.get("findings", []) or []
        suspicious_regions = anomaly.get("suspicious_regions", []) or []
        ocr_confidence = float(ocr.get("average_confidence", 0.0) or 0.0)
        raw_text = str(ocr.get("raw_text", "") or "")
        text_lower = raw_text.lower()
        anomaly_score = float(anomaly.get("overall_score", 0.0) or 0.0)
        max_region_severity = max((float(r.get("severity", 0.0) or 0.0) for r in suspicious_regions), default=0.0)
        word_count = len(raw_text.split())

        adjustments: Dict[str, int] = {}
        reasoning: List[str] = []

        institution_bonus = 0
        if institution_confidence >= 0.85:
            institution_bonus = 14
            reasoning.append("institution confidently recognized")
        elif institution_confidence >= 0.65:
            institution_bonus = 7
            reasoning.append("institution partially recognized")
        adjustments["institution_bonus"] = institution_bonus

        type_bonus = 0
        if certificate_type_confidence >= 0.85:
            type_bonus = 11
            reasoning.append("certificate type confidently recognized")
        elif certificate_type_confidence >= 0.6:
            type_bonus = 5
            reasoning.append("certificate type partially recognized")
        if document_type_confidence >= 0.8:
            type_bonus += 5
            reasoning.append("document type strongly supported by OCR text")
        adjustments["type_bonus"] = type_bonus

        layout_bonus = 0
        if layout_hints.get("has_signature_area"):
            layout_bonus += 4
            reasoning.append("signature layout detected")
        if layout_hints.get("has_table_structure"):
            layout_bonus += 4
            reasoning.append("table-like layout detected")
        if layout_hints.get("has_header_area"):
            layout_bonus += 2
        adjustments["layout_bonus"] = layout_bonus

        anomaly_penalty = 0
        if len(suspicious_regions) >= 6:
            anomaly_penalty -= 24
            reasoning.append("multiple suspicious regions detected")
        elif len(suspicious_regions) >= 3:
            anomaly_penalty -= 14
            reasoning.append("several suspicious regions detected")
        elif len(suspicious_regions) >= 1:
            anomaly_penalty -= 6
            reasoning.append("at least one suspicious region detected")

        if any(term in " ".join(findings).lower() for term in ["duplicated", "copied", "copy", "manipulation", "tamper"]):
            anomaly_penalty -= 18
            reasoning.append("duplicate or manipulation indicators present")
        if anomaly_score >= 70:
            anomaly_penalty -= 10
            reasoning.append("high anomaly score")
        adjustments["anomaly_penalty"] = anomaly_penalty

        ocr_penalty = 0
        if ocr_confidence < 0.45:
            ocr_penalty -= 18
            reasoning.append("OCR confidence is low")
        elif ocr_confidence < 0.65:
            ocr_penalty -= 8
            reasoning.append("OCR confidence is moderate")
        if word_count < 12:
            ocr_penalty -= 4
        adjustments["ocr_penalty"] = ocr_penalty

        text_bonus = 0
        if word_count >= 20:
            text_bonus = 4
            reasoning.append("enough readable text for interpretation")
        elif word_count >= 12:
            text_bonus = 2
        adjustments["text_bonus"] = text_bonus

        refined = base_score + institution_bonus + type_bonus + layout_bonus + anomaly_penalty + ocr_penalty + text_bonus
        refined = max(0, min(100, refined))

        if refined >= 88 and anomaly_score < 20 and max_region_severity < 60 and len(suspicious_regions) <= 1:
            risk_level = "safe"
        elif refined >= 72:
            risk_level = "low"
        elif refined >= 50:
            risk_level = "moderate"
        elif refined >= 30:
            risk_level = "high"
        else:
            risk_level = "critical"

        confidence_basis = (certificate_type_confidence + institution_confidence + ocr_confidence) / 3.0
        if min(certificate_type_confidence, institution_confidence, ocr_confidence) >= 0.72 or confidence_basis >= 0.78:
            confidence_level = "high"
        elif confidence_basis >= 0.52:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        if preprocessing:
            reasoning.append("preprocessing context used")

        return RefinedScoreResult(
            refined_trust_score=refined,
            risk_level=risk_level,
            confidence_level=confidence_level,
            reasoning=reasoning or ["analysis completed with available evidence"],
            adjustments=adjustments,
        )
