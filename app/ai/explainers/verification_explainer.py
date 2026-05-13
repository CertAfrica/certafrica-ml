from __future__ import annotations

from typing import Any, Dict, List, Tuple


class VerificationExplainer:
    def summarize(
        self,
        refined_score: int,
        document_type: str,
        certificate_type: str,
        institution: str,
        reasoning: List[str],
        anomaly_notes: List[str],
    ) -> Tuple[str, str, str]:
        if refined_score >= 88:
            summary = "safe"
            action = "No urgent escalation required; keep the record for audit purposes."
        elif refined_score >= 72:
            summary = "low"
            action = "Routine review is sufficient."
        elif refined_score >= 50:
            summary = "moderate"
            action = "Manual verification is recommended."
        elif refined_score >= 30:
            summary = "high"
            action = "Escalate for fraud review before acceptance."
        else:
            summary = "critical"
            action = "Request a higher-quality scan or a secondary verification source."

        reason_bits = reasoning[:4] if reasoning else ["limited evidence available"]
        explanation_parts = [
            f"This document is classified as {summary} risk.",
            f"Document type: {document_type}.",
            f"Detected certificate type: {certificate_type}.",
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

            area = self._infer_area(left, top, blocks, raw_text)
            if severity >= 90:
                prefix = "Possible tampering"
            elif severity >= 70:
                prefix = "Irregular text spacing"
            else:
                prefix = "Moderate anomaly"

            if severity >= 85 and any(term in raw_text for term in ["registration number", "matric", "candidate number", "serial number"]):
                notes.append("Duplicated region detected near registration number.")
            else:
                notes.append(f"{prefix} near the {area}.")

        if not notes and blocks:
            notes.append("OCR layout looks mostly consistent with a formal certificate.")
        if not notes:
            notes.append("No readable regions were available for detailed anomaly interpretation.")
        return notes

    def _infer_area(self, left: float, top: float, blocks: List[Dict[str, Any]], raw_text: str) -> str:
        if top < 180:
            return "certificate header"
        if top > 700:
            return "footer area"
        if left > 650:
            return "signature area"
        if "registration number" in raw_text or "matric" in raw_text or "candidate number" in raw_text:
            return "registration area"
        if "seal" in raw_text or "stamp" in raw_text:
            return "seal area"
        if "signature" in raw_text:
            return "signature area"
        return "certificate body"
