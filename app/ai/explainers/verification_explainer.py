from __future__ import annotations

from typing import Any, Dict, List, Tuple


class VerificationExplainer:
    def summarize(self, refined_score: int, document_type: str, certificate_type: str, institution: str, reasoning: List[str], anomaly_notes: List[str]) -> Tuple[str, str, str]:
        if refined_score >= 85:
            summary = "likely authentic"
            action = "No urgent escalation required; keep the record for audit purposes."
        elif refined_score >= 65:
            summary = "suspicious"
            action = "Manual institutional verification is recommended."
        elif refined_score >= 40:
            summary = "highly suspicious"
            action = "Escalate for manual fraud review before acceptance."
        else:
            summary = "inconclusive"
            action = "Request a higher-quality scan or a secondary verification source."

        reason_bits = reasoning[:4] if reasoning else ["limited evidence available"]
        explanation_parts = [
            f"This document is classified as {summary}.",
            f"Document type: {document_type}.",
            f"Detected document type: {certificate_type}.",
            f"Likely institution: {institution}.",
            "Reasons: " + "; ".join(reason_bits) + ".",
        ]
        if anomaly_notes:
            explanation_parts.append("Anomaly notes: " + "; ".join(anomaly_notes[:3]) + ".")

        return summary, " ".join(explanation_parts), action

    def build_anomaly_notes(self, suspicious_regions: List[Dict[str, Any]], ocr_result: Dict[str, Any]) -> List[str]:
        notes: List[str] = []
        blocks = ocr_result.get("blocks", []) or []
        raw_text = str(ocr_result.get("raw_text", "") or "").lower()

        for region in suspicious_regions:
            severity = float(region.get("severity", 0.0) or 0.0)
            left = float(region.get("left", 0.0) or 0.0)
            top = float(region.get("top", 0.0) or 0.0)
            if severity >= 90:
                label = "Highly suspicious activity"
            elif severity >= 70:
                label = "Possible manipulation"
            else:
                label = "Moderate anomaly"

            area = self._infer_area(left, top, blocks, raw_text)
            notes.append(f"{label} detected near the {area}.")

        if not notes and blocks:
            notes.append("OCR layout looks mostly consistent with a formal certificate.")
        if not notes:
            notes.append("No readable regions were available for detailed anomaly interpretation.")
        return notes

    def _infer_area(self, left: float, top: float, blocks: List[Dict[str, Any]], raw_text: str) -> str:
        if top < 180:
            return "header area"
        if top > 700:
            return "footer area"
        if left > 650:
            return "signature area"
        if left < 180:
            return "left margin"
        if "registrar" in raw_text or "signature" in raw_text:
            return "registrar signature area"
        if "seal" in raw_text or "stamp" in raw_text:
            return "seal area"
        return "certificate body"
