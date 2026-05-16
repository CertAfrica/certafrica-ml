from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from fastapi import HTTPException, UploadFile, status

from app.config import settings
from app.knowledge.store import KnowledgeBaseStore
from app.outputs.heatmap import HeatmapGenerator
from app.outputs.report import ReportGenerator
from app.pipeline.anomaly import AnomalyDetector
from app.pipeline.classify import UnifiedClassifier
from app.pipeline.ocr import OCRExtractor
from app.pipeline.pipeline import Pipeline
from app.pipeline.preprocess import ImagePreprocessor
from app.pipeline.visual import VisualEmbedder, VisualIndex, VisualSimilarityScorer
from app.schemas import (
    AnalysisResponse, Artifacts, DocumentSummary, OCRSummary,
    VisualMatchView, VisualSimilarityView,
)
from app.storage.s3 import SupabaseS3Storage
import logging

logger = logging.getLogger(__name__)
from app.utils.image import read_image, save_image
from app.utils.validation import validate_upload


class AnalysisService:
    def __init__(self) -> None:
        seed_path = Path(__file__).parent.parent / "knowledge" / "seed.json"
        kb_path = settings.storage_dir / "knowledge_base.json"
        model_path = settings.storage_dir / "classifier.joblib"
        self.kb_store = KnowledgeBaseStore(kb_path, seed_path)
        kb = self.kb_store.load()

        self.storage = SupabaseS3Storage()
        self.preprocessor = ImagePreprocessor()
        self.ocr = OCRExtractor()
        self.anomaly_detector = AnomalyDetector()
        self.classifier = UnifiedClassifier(kb, model_path)

        # Visual pipeline - shared with trainer
        self.embedder = VisualEmbedder()
        self.visual_index = VisualIndex(settings.storage_dir / "training")
        self.visual_scorer = VisualSimilarityScorer()

        self.pipeline = Pipeline(classifier=self.classifier)
        self.heatmap = HeatmapGenerator()
        self.report = ReportGenerator()

    async def analyze(self, scan_id: str) -> AnalysisResponse:
        start = perf_counter()

        with tempfile.TemporaryDirectory(prefix=f"certafrica-{scan_id}-") as work_dir:
            work = Path(work_dir)
            try:
                certificate_key = self.storage.find_scan_certificate_key(scan_id)
                if not certificate_key:
                    logger.error({"scan_id": scan_id}, 'Certificate not found in storage for scan')
                    raise FileNotFoundError(f"No certificate found for scan {scan_id}")

                logger.info({"scan_id": scan_id, "cert_key": certificate_key}, 'Found certificate key; downloading')
                upload_path = work / Path(certificate_key).name
                self.storage.download_to_path(certificate_key, upload_path)
                logger.info({"scan_id": scan_id, "local_path": str(upload_path)}, 'Downloaded certificate to local path')
            except FileNotFoundError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

            image = read_image(upload_path)
            preprocessed = self.preprocessor.preprocess(image)

            # Stage 1: OCR
            ocr_result = self.ocr.extract(preprocessed["rgb"])

            # Stage 2: Anomaly detection
            anomaly_result = self.anomaly_detector.detect(
                preprocessed["rgb"], preprocessed["gray"], ocr_result,
            )

            # Stage 3-7: Pipeline runs classification, visual matching,
            # scoring, summarization
            result = self.pipeline.run(
                ocr=ocr_result,
                anomaly=anomaly_result,
                image_shape=preprocessed["rgb"].shape[:2],
            )

            visual_embedding = self.embedder.embed(preprocessed["rgb"])
            visual_result = self.visual_scorer.score(
                visual_embedding,
                result.classification.document_type,
                self.visual_index,
            )

            suspicious_regions = [region.model_dump() for region in result.anomaly.suspicious_regions]

            # Artifacts
            heatmap_path = work / "heatmap.png"
            preview_path = work / "preview.png"
            report_path = work / "report.pdf"
            save_image(self.heatmap.build_overlay(preprocessed["rgb"], suspicious_regions), heatmap_path)
            save_image(preprocessed["rgb"], preview_path)
            self.report.generate(
                output_path=report_path,
                scan_id=scan_id,
                trust_score=result.verdict.trust_score,
                verification_status=result.verdict.risk_level,
                ocr_result=result.ocr,
                anomaly_result=result.anomaly,
                certificate_image_path=preview_path,
                heatmap_image_path=heatmap_path,
                created_at=datetime.now(timezone.utc),
                ai_context={
                    "documentType": {
                        "display_name": result.classification.document_type_display,
                    },
                    "certificateType": {
                        "display_name": result.classification.certificate_type_display,
                    },
                    "institution": {
                        "canonical_name": result.classification.institution,
                    },
                    "refinedTrustScore": result.verdict.trust_score,
                    "riskLevel": result.verdict.risk_level,
                    "confidenceLevel": result.verdict.confidence,
                    "verificationSummary": result.explanation,
                    "detailedExplanation": result.explanation,
                    "recommendedAction": result.verdict.recommended_action,
                    "reasoning": result.verdict.reasoning,
                    "anomalyNotes": result.anomaly.findings,
                    "metadata": {
                        "visual": asdict(visual_result),
                    },
                },
            )

            # upload artifacts with logging
            logger.info({"scan_id": scan_id, "upload_path": str(upload_path)}, 'Uploading certificate back to storage')
            upload_url = self.storage.upload_file(upload_path, self.storage.object_key(scan_id, upload_path.name))
            logger.info({"scan_id": scan_id, "upload_url": upload_url}, 'Uploaded certificate')

            logger.info({"scan_id": scan_id, "heatmap_path": str(heatmap_path)}, 'Uploading heatmap')
            heatmap_url = self.storage.upload_file(heatmap_path, self.storage.object_key(scan_id, heatmap_path.name))
            logger.info({"scan_id": scan_id, "heatmap_url": heatmap_url}, 'Uploaded heatmap')

            logger.info({"scan_id": scan_id, "preview_path": str(preview_path)}, 'Uploading preview')
            preview_url = self.storage.upload_file(preview_path, self.storage.object_key(scan_id, preview_path.name))
            logger.info({"scan_id": scan_id, "preview_url": preview_url}, 'Uploaded preview')

            logger.info({"scan_id": scan_id, "report_path": str(report_path)}, 'Uploading report')
            report_url = self.storage.upload_file(report_path, self.storage.object_key(scan_id, report_path.name), content_type="application/pdf")
            logger.info({"scan_id": scan_id, "report_url": report_url}, 'Uploaded report')

            elapsed = perf_counter() - start

            return AnalysisResponse(
                scan_id=scan_id,
                timestamp=datetime.now(timezone.utc),
                processing_time=f"{elapsed:.2f}s",
                verdict=result.verdict,
                document=DocumentSummary(
                    type=result.classification.document_type,
                    type_display=result.classification.document_type_display,
                    type_confidence=result.classification.document_type_confidence,
                    certificate_type=result.classification.certificate_type,
                    certificate_type_display=result.classification.certificate_type_display,
                    certificate_type_confidence=result.classification.certificate_type_confidence,
                    institution=result.classification.institution,
                    institution_confidence=result.classification.institution_confidence,
                ),
                ocr=OCRSummary(
                    confidence=result.ocr.average_confidence,
                    language=result.ocr.language,
                    text=result.ocr.cleaned_text or result.ocr.raw_text,
                    fields=result.ocr.extracted_fields,
                ),
                anomalies=result.anomaly_summary,
                layout=result.layout,
                visual=VisualSimilarityView(
                    score=visual_result.score,
                    best_similarity=visual_result.best_similarity,
                    reference_count=visual_result.reference_count,
                    matches=[
                        VisualMatchView(
                            sample_id=m.sample_id,
                            label=m.label,
                            institution=m.institution,
                            similarity=round(m.similarity, 3),
                        )
                        for m in visual_result.matches[:3]
                    ],
                    note=visual_result.note,
                ),
                explanation=result.explanation,
                artifacts=Artifacts(
                    upload_url=upload_url,
                    heatmap_url=heatmap_url,
                    report_url=report_url,
                    preview_url=preview_url,
                ),
            )

    async def _save_upload(self, upload: UploadFile, target: Path,
                           max_bytes: int) -> None:
        written = 0
        upload.file.seek(0)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as buf:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    buf.close()
                    target.unlink(missing_ok=True)
                    raise ValueError("Upload exceeds configured size limit.")
                buf.write(chunk)
        await upload.close()