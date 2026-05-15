from __future__ import annotations

from datetime import datetime, timezone
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from fastapi import HTTPException, UploadFile, status

from app.config import settings
from app.ai.services.ai_layer_service import AILayerService
from app.detection.anomaly_detector import OCRExtractor, AnomalyDetector
from app.heatmaps.heatmap_generator import HeatmapGenerator
from app.models.schemas import (
    AnalysisResponse,
    AnomalyDetectionPayload,
    ArtifactsPayload,
    CertificateTypeView,
    DocumentAnalysisPayload,
    DocumentTypeView,
    FindingCategories,
    FindingSummary,
    InstitutionView,
    LayoutAnalysisPayload,
    OCRAnalysisPayload,
    RegionCoordinates,
    SuspiciousRegionView,
    VerificationPayload,
)
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
        max_bytes = settings.max_upload_mb * 1024 * 1024
        validate_upload(certificate, max_bytes)

        with tempfile.TemporaryDirectory(prefix=f"certafrica-{scan_id}-") as work_dir:
            work_path = Path(work_dir)
            suffix = Path(certificate.filename or "certificate").suffix.lower() or ".bin"
            upload_path = work_path / f"input{suffix}"

            try:
                await self._save_upload_to_path(certificate, upload_path, max_bytes)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc

            original_image = read_image_from_upload(upload_path)
            preprocessed = self.preprocessor.preprocess(original_image)
            preprocessing_metadata = {
                "originalShape": list(getattr(original_image, "shape", []) or []),
                "rgbShape": list(getattr(preprocessed["rgb"], "shape", []) or []),
                "grayShape": list(getattr(preprocessed["gray"], "shape", []) or []),
            }

            ocr_result = self.ocr.extract(preprocessed["rgb"], language=settings.ocr_lang)
            anomaly_result = self.detector.detect(preprocessed["rgb"], preprocessed["gray"], ocr_result)
            trust_score = self._compute_trust_score(
                ocr_result.average_confidence,
                anomaly_result.overall_score,
                len(anomaly_result.suspicious_regions),
                len(ocr_result.extracted_fields),
                anomaly_result.findings,
            )
            verification_status = self._status_from_score(trust_score, anomaly_result.overall_score)

            ai_interpretation = self.ai_layer.interpret(
                {
                    "scanId": scan_id,
                    "trustScore": trust_score,
                    "verificationStatus": verification_status,
                    "ocrResult": ocr_result.model_dump(),
                    "anomalyResult": anomaly_result.model_dump(),
                    "heatmapNotes": anomaly_result.heatmap_notes,
                    "preprocessingMetadata": preprocessing_metadata,
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

            # keep status aligned to requested response style
            merged_status = (ai_interpretation.riskLevel or ai_interpretation.verificationSummary or verification_status).lower()
            merged_trust_score = ai_interpretation.refinedTrustScore or trust_score

            heatmap_image = self.heatmap_generator.build_overlay(preprocessed["rgb"], anomaly_result.suspicious_regions)
            heatmap_path = work_path / "heatmap.png"
            save_numpy_image(heatmap_image, heatmap_path)

            certificate_preview_path = work_path / "preview.png"
            save_numpy_image(preprocessed["rgb"], certificate_preview_path)

            report_path = work_path / "verification-report.pdf"
            created_at = datetime.now(timezone.utc)

            normalized_ocr_result = ocr_result.model_copy(update={"raw_text": merged_raw_text})
            normalized_anomaly_result = anomaly_result.model_copy(update={"findings": merged_findings, "suspicious_regions": enriched_regions})

            ai_context = ai_interpretation.model_dump()
            ai_context["metadata"] = {
                **(ai_context.get("metadata") or {}),
                "preprocessingMetadata": preprocessing_metadata,
            }

            self.report_generator.generate(
                output_path=report_path,
                scan_id=scan_id,
                trust_score=ai_interpretation.refinedTrustScore,
                verification_status=merged_status,
                ocr_result=normalized_ocr_result,
                anomaly_result=normalized_anomaly_result,
                certificate_image_path=certificate_preview_path,
                heatmap_image_path=heatmap_path,
                created_at=created_at,
                ai_context=ai_context,
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
            layout_hints = (ai_interpretation.metadata or {}).get("layoutHints", {}) if ai_interpretation.metadata else {}
            finding_summary, suspicious_regions = self._build_anomaly_payload(enriched_regions)

            return AnalysisResponse(
                scanId=scan_id,
                analysisTimestamp=created_at,
                processingTime=f"{elapsed:.2f}s",
                verification=VerificationPayload(
                    trustScore=trust_score,
                    refinedTrustScore=merged_trust_score,
                    riskLevel=ai_interpretation.riskLevel or "critical",
                    confidenceLevel=ai_interpretation.confidenceLevel or "low",
                    status=merged_status,
                    recommendedAction=ai_interpretation.recommendedAction or "Manual review recommended.",
                ),
                documentAnalysis=DocumentAnalysisPayload(
                    documentType=DocumentTypeView(
                        label=ai_interpretation.documentType.label if ai_interpretation.documentType else "unknown_document",
                        displayName=ai_interpretation.documentType.display_name if ai_interpretation.documentType else "Unknown Document",
                        confidence=ai_interpretation.documentType.confidence if ai_interpretation.documentType else 0.0,
                        evidence=ai_interpretation.documentType.evidence if ai_interpretation.documentType else ["insufficient document signals detected"],
                    ),
                    certificateType=CertificateTypeView(
                        label=ai_interpretation.certificateType.label if ai_interpretation.certificateType else "unknown_document",
                        displayName=ai_interpretation.certificateType.display_name if ai_interpretation.certificateType else "Unknown Document",
                        confidence=ai_interpretation.certificateType.confidence if ai_interpretation.certificateType else 0.0,
                        evidence=ai_interpretation.certificateType.evidence if ai_interpretation.certificateType else [],
                    ),
                    institution=InstitutionView(
                        name=ai_interpretation.institution.canonical_name if ai_interpretation.institution else "Unknown Institution",
                        confidence=ai_interpretation.institution.confidence if ai_interpretation.institution else 0.0,
                        evidence=ai_interpretation.institution.evidence if ai_interpretation.institution else [],
                    ),
                ),
                ocrAnalysis=OCRAnalysisPayload(
                    confidence=ocr_result.average_confidence,
                    language=ocr_result.language,
                    rawText=merged_raw_text,
                    extractedFields=ocr_result.extracted_fields,
                ),
                anomalyDetection=AnomalyDetectionPayload(
                    overallScore=anomaly_result.overall_score,
                    findingSummary=finding_summary,
                    suspiciousRegions=suspicious_regions,
                ),
                layoutAnalysis=LayoutAnalysisPayload(
                    hasSignatureArea=bool(layout_hints.get("has_signature_area", False)),
                    hasTableStructure=bool(layout_hints.get("has_table_structure", False)),
                    hasHeaderArea=bool(layout_hints.get("has_header_area", False)),
                    hasFooterArea=bool(layout_hints.get("has_footer_area", False)),
                    blockCount=int(layout_hints.get("blocks_count", len(ocr_result.blocks))),
                ),
                reasoning=ai_interpretation.reasoning or [],
                artifacts=ArtifactsPayload(
                    uploadUrl=upload_url,
                    heatmapUrl=heatmap_url,
                    reportUrl=report_url,
                    previewUrl=preview_url,
                ),
                aiOutput=ai_interpretation.model_dump(),
            )

    def _build_anomaly_payload(self, regions: List[Dict[str, Any]]) -> tuple[FindingSummary, List[SuspiciousRegionView]]:
        categories = FindingCategories()
        output_regions: List[SuspiciousRegionView] = []

        # keep high-severity only for cleaner frontend output
        filtered = sorted(
            [r for r in regions if float(r.get("severity", 0.0) or 0.0) >= 70.0],
            key=lambda r: float(r.get("severity", 0.0) or 0.0),
            reverse=True,
        )

        for region in regions:
            severity = float(region.get("severity", 0.0) or 0.0)
            if severity >= 90:
                categories.possibleTampering += 1
            elif severity >= 70:
                categories.irregularSpacing += 1
            else:
                categories.moderateAnomaly += 1

        for region in filtered:
            severity = float(region.get("severity", 0.0) or 0.0)
            category = str(region.get("ai_notes") or "").strip()
            if not category:
                if severity >= 90:
                    category = "Possible tampering near certificate header"
                elif severity >= 70:
                    category = "Irregular text spacing near certificate body"
                else:
                    category = "Moderate anomaly"
            output_regions.append(
                SuspiciousRegionView(
                    coordinates=RegionCoordinates(
                        left=int(region.get("left", 0) or 0),
                        top=int(region.get("top", 0) or 0),
                        width=int(region.get("width", 0) or 0),
                        height=int(region.get("height", 0) or 0),
                    ),
                    severity=round(severity, 2),
                    category=category,
                )
            )

        return FindingSummary(totalRegions=len(regions), categories=categories), output_regions

    def _compute_trust_score(
        self,
        ocr_confidence: float,
        anomaly_score: float,
        suspicious_regions: int,
        extracted_fields: int,
        findings: list[str],
    ) -> int:
        score = 100.0
        score -= min(30.0, anomaly_score * 0.4)

        if ocr_confidence < 0.45:
            score -= (0.45 - ocr_confidence) * 55
        elif ocr_confidence < 0.7:
            score -= (0.7 - ocr_confidence) * 20

        score -= min(18.0, suspicious_regions * 2.5)

        if extracted_fields == 0:
            score -= 10.0
        elif extracted_fields < 3:
            score -= 4.0

        findings_text = " ".join(findings).lower()
        if any(term in findings_text for term in ["duplicate", "duplicated", "copied", "tamper", "manipulation"]):
            score -= 8.0

        if ocr_confidence > 0.9 and extracted_fields > 0:
            score += 4.0

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
