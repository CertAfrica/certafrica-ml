from __future__ import annotations

from typing import Any, Dict


def demo_ml_payload() -> Dict[str, Any]:
    return {
        "scanId": "demo-scan-001",
        "trustScore": 58,
        "verificationStatus": "suspicious",
        "ocrResult": {
            "raw_text": "OF LAGOS\nBachelor of Science\nThis is to certify that John Doe\nCertificate No: UNILAG/2024/001\nRegistrar Signature",
            "extracted_fields": {
                "name": "John Doe",
                "certificate_number": "UNILAG/2024/001"
            },
            "blocks": [
                {"text": "OF LAGOS", "confidence": 0.71, "left": 90, "top": 70, "width": 180, "height": 40},
                {"text": "Registrar Signature", "confidence": 0.88, "left": 680, "top": 760, "width": 160, "height": 40}
            ],
            "average_confidence": 0.79,
        },
        "anomalyResult": {
            "overall_score": 42,
            "findings": ["possible duplicated regions detected", "compression inconsistency hotspots detected"],
            "suspicious_regions": [
                {"left": 680, "top": 760, "width": 160, "height": 40, "severity": 82},
                {"left": 90, "top": 70, "width": 180, "height": 40, "severity": 64}
            ],
            "heatmap_notes": ["Red = suspicious area", "Yellow = moderate anomaly"]
        },
        "heatmapNotes": ["Red = suspicious area", "Yellow = moderate anomaly"]
    }
