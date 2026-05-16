"""
Pipeline orchestrator. Each stage runs exactly once. Results flow forward.

Previous code ran OCR + layout derivation + document type detection twice
(once in AnalysisService, once again in AILayerService.interpret). This
orchestrator ensures each step has a single execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from app.pipeline.classify import UnifiedClassifier
from app.pipeline.explain import Explainer
from app.pipeline.score import TrustScorer
from app.pipeline.visual import VisualSimilarity
from app.schemas import (
    AnomalyResult, AnomalySummary, Classification, LayoutHints,
    OCRResult, Verdict,
)


@dataclass
class PipelineResult:
    ocr: OCRResult
    anomaly: AnomalyResult
    layout: LayoutHints
    classification: Classification
    verdict: Verdict
    anomaly_summary: AnomalySummary
    explanation: str


class Pipeline:
    """Single-pass certificate analysis pipeline."""

    def __init__(self, classifier: UnifiedClassifier) -> None:
        self.classifier = classifier
        self.scorer = TrustScorer()
        self.explainer = Explainer()

    def run(self, ocr: OCRResult, anomaly: AnomalyResult,
            image_shape: tuple[int, int]) -> PipelineResult:
        """
        Args:
            ocr: OCR result (already extracted from image)
            anomaly: Anomaly detection result
            image_shape: (height, width) of the preprocessed image
        """
        height, width = image_shape

        # Reclassify anomaly regions with proper area assignment based on image shape.
        for region in anomaly.suspicious_regions:
            category, area = self.explainer.classify_region(
                severity=region.severity,
                top=region.top,
                left=region.left,
                image_height=height,
                image_width=width,
                raw_text=ocr.raw_text,
            )
            region.category = category
            region.area = area

        # Layout hints from OCR blocks. Single computation.
        layout = self._derive_layout(ocr, image_height=height, image_width=width)

        # Classification: document type + certificate type + institution. One pass.
        classification = self.classifier.classify(
            ocr_text=ocr.cleaned_text or ocr.raw_text,
            fields=ocr.extracted_fields,
            layout=layout,
        )

        # Score using all signals.
        verdict = self.scorer.score(
            ocr,
            anomaly,
            classification,
            layout,
            visual=VisualSimilarity(
                score=None,
                best_similarity=0.0,
                reference_count=0,
                note="visual signal computed downstream",
            ),
        )

        # Summarize anomalies with dedup.
        anomaly_summary = self.explainer.summarize_anomalies(anomaly, image_height=height)

        # Final explanation.
        explanation = self.explainer.explain(verdict, classification, anomaly_summary)

        return PipelineResult(
            ocr=ocr,
            anomaly=anomaly,
            layout=layout,
            classification=classification,
            verdict=verdict,
            anomaly_summary=anomaly_summary,
            explanation=explanation,
        )

    def _derive_layout(self, ocr: OCRResult, image_height: int,
                       image_width: int) -> LayoutHints:
        blocks = ocr.blocks
        if not blocks:
            return LayoutHints(block_count=0)

        tops = [b.top for b in blocks]
        lefts = [b.left for b in blocks]

        header_threshold = image_height * 0.2
        footer_threshold = image_height * 0.85
        right_threshold = image_width * 0.7

        return LayoutHints(
            has_signature_area=any(
                t > image_height * 0.7 and l > right_threshold
                for t, l in zip(tops, lefts)
            ),
            has_table_structure=len(blocks) >= 10,
            has_header_area=any(t < header_threshold for t in tops),
            has_footer_area=any(t > footer_threshold for t in tops),
            block_count=len(blocks),
        )
