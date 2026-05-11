from __future__ import annotations

from datetime import datetime, timezone
import tempfile
from pathlib import Path
from time import perf_counter

from fastapi import HTTPException, UploadFile, status

from app.config import settings
from app.ai.services.ai_layer_service import AILayerService
from app.detection.anomaly_detector import OCRExtractor, AnomalyDetector
from app.heatmaps.heatmap_generator import HeatmapGenerator
from app.models.schemas import AnalysisResponse
from app.preprocessing.image_preprocessor import ImagePreprocessor
from app.reports.report_generator import ReportGenerator
from app.storage.s3_storage import SupabaseS3Storage
from app.utils.file_validation import validate_upload
from app.utils.image_io import read_image_from_upload, save_numpy_image


class AnalysisService:
    def __init__(self) -> None:
        self.storage = SupabaseS3Storage()
        self.preprocessor = ImagePreprocessor()
        self.ocr = OCRExtractor()
        self.detector = AnomalyDetector()
        self.heatmap_generator = HeatmapGenerator()
        self.report_generator = ReportGenerator()
        self.ai_layer = AILayerService()

    async def analyze(self, scan_id: str, certificate: UploadFile) -> AnalysisResponse:
        start = perf_counter()
        validate_upload(certificate, settings.max_upload_mb * 1024 * 1024)
        with tempfile.TemporaryDirectory(prefix=f"certafrica-{scan_id}-") as work_dir:
            work_path = Path(work_dir)
            suffix = Path(certificate.filename or "certificate").suffix.lower() or ".bin"
            upload_path = work_path / f"input{suffix}"

            try:
                await self._save_upload_to_path(certificate, upload_path, settings.max_upload_mb * 1024 * 1024)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc

            original_image = read_image_from_upload(upload_path)
            preprocessed = self.preprocessor.preprocess(original_image)
            ocr_result = self.ocr.extract(preprocessed["rgb"], language=settings.ocr_lang)
            anomaly_result = self.detector.detect(preprocessed["rgb"], preprocessed["gray"], ocr_result)
            trust_score = self._compute_trust_score(ocr_result.average_confidence, anomaly_result.overall_score, len(anomaly_result.suspicious_regions), len(ocr_result.extracted_fields))
            verification_status = self._status_from_score(trust_score, anomaly_result.overall_score)
            ai_interpretation = self.ai_layer.interpret(
                {
                    "scanId": scan_id,
                    "trustScore": trust_score,
                    "verificationStatus": verification_status,
                    "ocrResult": ocr_result.model_dump(),
                    "anomalyResult": anomaly_result.model_dump(),
                    "heatmapNotes": anomaly_result.heatmap_notes,
                }
            )

            region_notes = ai_interpretation.anomalyNotes or []
            enriched_regions = []
            for index, region in enumerate(anomaly_result.suspicious_regions):
                enriched_region = dict(region)
                if region_notes:
                    note = region_notes[index] if index < len(region_notes) else region_notes[-1]
                    enriched_region["ai_notes"] = note
                enriched_regions.append(enriched_region)

            merged_findings = region_notes if region_notes else anomaly_result.findings
            merged_raw_text = ai_interpretation.cleanedOCRText or ocr_result.raw_text
            merged_status = ai_interpretation.verificationSummary or verification_status
            merged_trust_score = ai_interpretation.refinedTrustScore or trust_score

            heatmap_image = self.heatmap_generator.build_overlay(preprocessed["rgb"], anomaly_result.suspicious_regions)
            heatmap_path = work_path / "heatmap.png"
            save_numpy_image(heatmap_image, heatmap_path)

            certificate_preview_path = work_path / "preview.png"
            save_numpy_image(preprocessed["rgb"], certificate_preview_path)

            report_path = work_path / "verification-report.pdf"
            created_at = datetime.now(timezone.utc)
            report_ocr_result = ocr_result.model_copy(update={"raw_text": merged_raw_text})
            report_anomaly_result = anomaly_result.model_copy(update={"findings": merged_findings, "suspicious_regions": enriched_regions})
            self.report_generator.generate(
                output_path=report_path,
                scan_id=scan_id,
                trust_score=merged_trust_score,
                verification_status=merged_status,
                ocr_result=report_ocr_result,
                anomaly_result=report_anomaly_result,
                certificate_image_path=certificate_preview_path,
                heatmap_image_path=heatmap_path,
                created_at=created_at,
                ai_context=ai_interpretation.model_dump(),
            )

            upload_key = self.storage.object_key(scan_id, upload_path.name)
            heatmap_key = self.storage.object_key(scan_id, heatmap_path.name)
            report_key = self.storage.object_key(scan_id, report_path.name)
            preview_key = self.storage.object_key(scan_id, certificate_preview_path.name)

            upload_url = self.storage.upload_file(upload_path, upload_key)
            heatmap_url = self.storage.upload_file(heatmap_path, heatmap_key)
            report_url = self.storage.upload_file(report_path, report_key, content_type="application/pdf")
            preview_url = self.storage.upload_file(certificate_preview_path, preview_key)

            elapsed = perf_counter() - start
            return AnalysisResponse(
                scanId=scan_id,
                trustScore=merged_trust_score,
                verificationStatus=merged_status,
                documentType=ai_interpretation.documentType,
                ocrResult=ocr_result.model_copy(update={"raw_text": merged_raw_text}),
                anomalyResult=anomaly_result.model_copy(update={"findings": merged_findings, "suspicious_regions": enriched_regions}),
                aiCertificateType=ai_interpretation.certificateType,
                aiInstitution=ai_interpretation.institution,
                refinedTrustScore=ai_interpretation.refinedTrustScore,
                riskLevel=ai_interpretation.riskLevel,
                confidenceLevel=ai_interpretation.confidenceLevel,
                verificationSummary=ai_interpretation.verificationSummary,
                detailedExplanation=ai_interpretation.detailedExplanation,
                recommendedAction=ai_interpretation.recommendedAction,
                cleanedOCRText=ai_interpretation.cleanedOCRText,
                aiReasoning=ai_interpretation.reasoning,
                aiMetadata={**ai_interpretation.metadata, "documentType": ai_interpretation.documentType.model_dump()},
                heatmapImageUrl=heatmap_url,
                reportUrl=report_url,
                processingTime=f"{elapsed:.2f}s",
                analysisTimestamp=created_at,
                artifactPaths={
                    "upload": upload_url,
                    "heatmap": heatmap_url,
                    "report": report_url,
                    "preview": preview_url,
                },
            )

    def _compute_trust_score(self, ocr_confidence: float, anomaly_score: float, suspicious_regions: int, extracted_fields: int) -> int:
        score = 100.0
        
        # Anomaly score impact (reduced from 0.55 to 0.45)
        score -= anomaly_score * 0.45
        
        # OCR confidence impact (only penalize if very low)
        if ocr_confidence < 0.5:
            score -= (0.5 - ocr_confidence) * 50
        
        # Suspicious regions (reduced penalty per region)
        score -= min(15.0, suspicious_regions * 1.2)
        
        # Only penalize if NO fields extracted (not "fewer than 2")
        if extracted_fields == 0:
            score -= 12.0
        
        # Bonus for good OCR confidence
        if ocr_confidence > 0.85:
            score += 5.0
        
        return int(max(0, min(100, round(score))))

    def _status_from_score(self, trust_score: int, anomaly_score: float) -> str:
        if trust_score >= 88 and anomaly_score < 20:
            return "authentic"
        if trust_score >= 65:
            return "likely_authentic"
        if trust_score >= 35:
            return "suspicious"
        if trust_score > 0:
            return "forged"
        return "inconclusive"

    async def _save_upload_to_path(self, upload: UploadFile, target_path: Path, max_bytes: int) -> None:
        written = 0
        upload.file.seek(0)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as buffer:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    buffer.close()
                    target_path.unlink(missing_ok=True)
                    raise ValueError("Upload exceeds the configured size limit.")
                buffer.write(chunk)
        await upload.close()
