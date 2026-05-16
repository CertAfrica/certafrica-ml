"""
Anomaly notes and human-readable explanations.

The old explainer emitted one note per suspicious region, producing 24 notes
where 17 said "Irregular text spacing near the certificate body". This version
classifies each region into (category, area) and groups them.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from app.schemas import (
    AnomalyResult, AnomalySummary, Classification, RegionCategory,
    SuspiciousRegion, Verdict,
)


class Explainer:
    def summarize_anomalies(self, anomaly: AnomalyResult,
                            image_height: int = 800) -> AnomalySummary:
        regions = anomaly.suspicious_regions
        if not regions:
            return AnomalySummary(
                score=anomaly.overall_score,
                total_regions=0,
                high_severity_regions=0,
                by_category=[],
                top_regions=[],
            )

        counts: Counter[Tuple[str, str]] = Counter()
        for region in regions:
            counts[(region.category, region.area)] += 1

        by_category = [
            RegionCategory(category=category, area=area, count=count)
            for (category, area), count in counts.most_common()
        ]

        top_regions = sorted(regions, key=lambda r: r.severity, reverse=True)[:5]
        high_severity = sum(1 for r in regions if r.severity >= 85)

        return AnomalySummary(
            score=anomaly.overall_score,
            total_regions=len(regions),
            high_severity_regions=high_severity,
            by_category=by_category,
            top_regions=top_regions,
        )

    def classify_region(self, severity: float, top: int, left: int,
                        image_height: int = 800, image_width: int = 600,
                        raw_text: str = "") -> Tuple[str, str]:
        if severity >= 90:
            category = "Possible tampering"
        elif severity >= 70:
            category = "Irregular text spacing"
        else:
            category = "Moderate anomaly"

        # Position-based area assignment.
        relative_top = top / max(image_height, 1)
        if relative_top < 0.25:
            area = "header"
        elif relative_top > 0.85:
            area = "footer"
        elif left > image_width * 0.7 and relative_top > 0.6:
            area = "signature area"
        else:
            area = "body"

        return category, area

    def explain(self, verdict: Verdict, classification: Classification,
                anomaly_summary: AnomalySummary) -> str:
        parts: List[str] = []

        parts.append(
            f"Trust score {verdict.trust_score}/100 ({verdict.risk_level} risk, "
            f"{verdict.confidence} confidence)."
        )

        if classification.document_type != "unknown_document":
            parts.append(f"Identified as {classification.document_type_display}.")
        else:
            parts.append("Document type could not be identified from the OCR text.")

        if classification.institution != "Unknown Institution":
            parts.append(
                f"Institution: {classification.institution} "
                f"({int(classification.institution_confidence * 100)}% match)."
            )

        if anomaly_summary.high_severity_regions > 0:
            parts.append(
                f"{anomaly_summary.high_severity_regions} high-severity region(s) detected; "
                f"{anomaly_summary.total_regions} total flagged areas."
            )

        parts.append(verdict.recommended_action)

        return " ".join(parts)
