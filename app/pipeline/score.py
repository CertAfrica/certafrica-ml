from __future__ import annotations

from typing import List, Optional

from app.pipeline.visual import VisualSimilarity
from app.schemas import (
    AnomalyResult, Classification, LayoutHints, OCRResult,
    ScoreBreakdown, Verdict,
)


# Default weights. Must sum to 1.0.
_WEIGHTS_WITH_VISUAL = {
    "ocr": 0.25,
    "institution": 0.20,
    "classification": 0.15,
    "anomaly": 0.20,
    "layout": 0.05,
    "visual": 0.15,
}

# Fallback when visual is unavailable. The 0.15 visual weight gets
# redistributed: 0.10 to OCR, 0.05 to classification.
_WEIGHTS_WITHOUT_VISUAL = {
    "ocr": 0.35,
    "institution": 0.20,
    "classification": 0.20,
    "anomaly": 0.20,
    "layout": 0.05,
}


class TrustScorer:
    def score(self, ocr: OCRResult, anomaly: AnomalyResult,
              classification: Classification, layout: LayoutHints,
              visual: VisualSimilarity) -> Verdict:
        ocr_score = self._score_ocr(ocr)
        inst_score = self._score_institution(classification)
        class_score = self._score_classification(classification)
        anomaly_score = self._score_anomaly(anomaly)
        layout_score = self._score_layout(layout)

        breakdown = ScoreBreakdown(
            ocr=round(ocr_score, 1),
            institution=round(inst_score, 1),
            classification=round(class_score, 1),
            anomaly=round(anomaly_score, 1),
            layout=round(layout_score, 1),
            visual=visual.score,
        )

        if visual.score is not None:
            w = _WEIGHTS_WITH_VISUAL
            final = (
                w["ocr"] * ocr_score
                + w["institution"] * inst_score
                + w["classification"] * class_score
                + w["anomaly"] * anomaly_score
                + w["layout"] * layout_score
                + w["visual"] * visual.score
            )
        else:
            w = _WEIGHTS_WITHOUT_VISUAL
            final = (
                w["ocr"] * ocr_score
                + w["institution"] * inst_score
                + w["classification"] * class_score
                + w["anomaly"] * anomaly_score
                + w["layout"] * layout_score
            )

        final_int = int(max(0, min(100, round(final))))

        risk_level, action = self._risk_band(final_int, anomaly.overall_score)
        confidence = self._confidence_band(
            ocr.average_confidence, classification, inst_score, visual,
        )
        reasoning = self._reasoning(breakdown, classification, anomaly, visual)

        return Verdict(
            trust_score=final_int,
            risk_level=risk_level,
            confidence=confidence,
            recommended_action=action,
            breakdown=breakdown,
            reasoning=reasoning,
        )

    def _score_ocr(self, ocr: OCRResult) -> float:
        base = ocr.average_confidence * 100
        word_count = len(ocr.raw_text.split())
        if word_count < 10:
            base *= 0.7
        elif word_count >= 30:
            base = min(100, base * 1.05)
        field_bonus = min(10, len(ocr.extracted_fields) * 2)
        return max(0, min(100, base + field_bonus))

    def _score_institution(self, c: Classification) -> float:
        conf = c.institution_confidence
        if conf >= 0.85:
            return 90.0 + (conf - 0.85) * 60
        if conf >= 0.65:
            return 60.0 + (conf - 0.65) * 150
        if conf >= 0.4:
            return 30.0 + (conf - 0.4) * 120
        return conf * 75

    def _score_classification(self, c: Classification) -> float:
        conf = max(c.document_type_confidence, c.certificate_type_confidence)
        if c.document_type == "unknown_document" and c.certificate_type == "unknown_document":
            return 25.0
        return max(20, min(100, conf * 100))

    def _score_anomaly(self, anomaly: AnomalyResult) -> float:
        base = 100 - anomaly.overall_score
        high_sev = sum(1 for r in anomaly.suspicious_regions if r.severity >= 85)
        if high_sev >= 5:
            base -= 20
        elif high_sev >= 2:
            base -= 10
        elif high_sev >= 1:
            base -= 5
        return max(0, min(100, base))

    def _score_layout(self, layout: LayoutHints) -> float:
        score = 60.0
        if layout.has_header_area:
            score += 10
        if layout.has_signature_area:
            score += 15
        if layout.has_footer_area:
            score += 5
        if layout.has_table_structure:
            score += 10
        if layout.block_count < 5:
            score -= 20
        return max(0, min(100, score))

    def _risk_band(self, score: int, anomaly_score: float) -> tuple[str, str]:
        if score >= 85 and anomaly_score < 25:
            return "safe", "Accept; keep on file for audit."
        if score >= 70:
            return "low", "Routine review is sufficient."
        if score >= 50:
            return "moderate", "Manual verification recommended."
        if score >= 30:
            return "high", "Escalate for fraud review before acceptance."
        return "critical", "Request a higher-quality scan or secondary verification source."

    def _confidence_band(self, ocr_conf: float, c: Classification,
                         inst_score: float, visual: VisualSimilarity) -> str:
        signals = [
            ocr_conf * 100,
            c.document_type_confidence * 100,
            inst_score,
        ]
        if visual.score is not None:
            signals.append(visual.score)

        avg = sum(signals) / len(signals)
        if avg >= 75 and min(signals) >= 50:
            return "high"
        if avg >= 50:
            return "medium"
        return "low"

    def _reasoning(self, b: ScoreBreakdown, c: Classification,
                   anomaly: AnomalyResult, visual: VisualSimilarity) -> List[str]:
        reasons: List[str] = []

        if b.ocr < 50:
            reasons.append(f"OCR confidence is low ({b.ocr:.0f}/100)")
        elif b.ocr >= 80:
            reasons.append(f"OCR is clean ({b.ocr:.0f}/100)")

        if c.institution != "Unknown Institution" and b.institution >= 80:
            reasons.append(f"Institution recognized: {c.institution}")
        elif b.institution < 40:
            reasons.append("Institution not confidently identified")

        if c.document_type != "unknown_document":
            reasons.append(f"Document type: {c.document_type_display}")

        if visual.score is not None:
            if visual.score >= 80:
                reasons.append(
                    f"Visually matches known {c.document_type_display} samples "
                    f"({int(visual.best_similarity * 100)}% similarity)"
                )
            elif visual.score < 40:
                reasons.append(
                    f"Visual match weak against known samples "
                    f"({int(visual.best_similarity * 100)}% best similarity)"
                )
        elif visual.note:
            reasons.append(f"Visual check skipped: {visual.note}")

        high_sev = sum(1 for r in anomaly.suspicious_regions if r.severity >= 85)
        if high_sev >= 3:
            reasons.append(f"{high_sev} high-severity suspicious regions")
        elif high_sev == 0 and len(anomaly.suspicious_regions) > 0:
            reasons.append("Suspicious regions present but low severity")

        return reasons or ["Analysis completed with limited evidence"]