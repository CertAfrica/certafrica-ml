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
    def refine(self, ml_payload: Dict[str, Any], certificate_type_confidence: float, institution_confidence: float) -> RefinedScoreResult:
        base_score = int(ml_payload.get("trustScore", 0) or 0)
        anomaly = ml_payload.get("anomalyResult", {}) or {}
        ocr = ml_payload.get("ocrResult", {}) or {}
        findings = anomaly.get("findings", []) or []
        suspicious_regions = anomaly.get("suspicious_regions", []) or []
        ocr_confidence = float(ocr.get("average_confidence", 0.0) or 0.0)
        raw_text = str(ocr.get("raw_text", "") or "")
        text_lower = raw_text.lower()

        adjustments: Dict[str, int] = {}
        reasoning: List[str] = []

        institution_bonus = 0
        if institution_confidence >= 0.8:
            institution_bonus = 15
            reasoning.append("institution confidently recognized")
        elif institution_confidence >= 0.55:
            institution_bonus = 8
            reasoning.append("institution partially recognized")
        adjustments["institution_bonus"] = institution_bonus

        type_bonus = 0
        if certificate_type_confidence >= 0.8:
            type_bonus = 10
            reasoning.append("certificate type recognized")
        elif certificate_type_confidence >= 0.55:
            type_bonus = 5
            reasoning.append("certificate type partially recognized")
        adjustments["type_bonus"] = type_bonus

        layout_bonus = 0
        if any(term in " ".join(findings).lower() for term in ["signature", "seal", "layout", "footer", "header"]):
            layout_bonus = 10
            reasoning.append("document layout appears structurally consistent")
        elif any(term in text_lower for term in ["registrar", "vice chancellor", "dean", "principal", "director"]):
            layout_bonus = 8
            reasoning.append("formal signatory structure detected in OCR text")
        adjustments["layout_bonus"] = layout_bonus

        anomaly_penalty = 0
        if len(suspicious_regions) >= 6:
            anomaly_penalty -= 25
            reasoning.append("multiple suspicious regions detected")
        elif len(suspicious_regions) >= 3:
            anomaly_penalty -= 14
            reasoning.append("several suspicious regions detected")
        if any("duplicated" in str(f).lower() or "copied" in str(f).lower() for f in findings):
            anomaly_penalty -= 30
            reasoning.append("duplicated regions detected")
        if any("compression" in str(f).lower() or "manipulation" in str(f).lower() for f in findings):
            anomaly_penalty -= 18
            reasoning.append("image manipulation artifacts indicated")
        if any("ocr confidence" in str(f).lower() for f in findings):
            anomaly_penalty -= 6
            reasoning.append("OCR quality is affecting interpretation")
        adjustments["anomaly_penalty"] = anomaly_penalty

        ocr_penalty = 0
        if ocr_confidence < 0.45:
            ocr_penalty -= 20
            reasoning.append("OCR confidence is below threshold")
        elif ocr_confidence < 0.65:
            ocr_penalty -= 8
            reasoning.append("OCR confidence is moderate")
        adjustments["ocr_penalty"] = ocr_penalty

        text_bonus = 0
        if len(raw_text.split()) >= 20:
            text_bonus = 5
            reasoning.append("document contains enough readable text for interpretation")
        elif len(raw_text.split()) >= 12:
            text_bonus = 3
            reasoning.append("document contains a moderate amount of readable text")
        adjustments["text_bonus"] = text_bonus

        refined = max(0, min(100, base_score + institution_bonus + type_bonus + layout_bonus + anomaly_penalty + ocr_penalty + text_bonus))

        if refined >= 85:
            risk_level = "low"
            confidence_level = "high"
        elif refined >= 65:
            risk_level = "medium"
            confidence_level = "medium"
        elif refined >= 40:
            risk_level = "high"
            confidence_level = "medium"
        else:
            risk_level = "critical"
            confidence_level = "low"

        return RefinedScoreResult(
            refined_trust_score=refined,
            risk_level=risk_level,
            confidence_level=confidence_level,
            reasoning=reasoning or ["analysis completed with available evidence"],
            adjustments=adjustments,
        )
