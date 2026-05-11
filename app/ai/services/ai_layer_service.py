from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ai.classifiers.certificate_type_classifier import CertificateTypeClassifier
from app.ai.classifiers.institution_recognizer import InstitutionRecognizer
from app.ai.explainers.verification_explainer import VerificationExplainer
from app.ai.knowledge_base.store import KnowledgeBaseStore
from app.ai.models import AIInterpretationResponse, DocumentTypePrediction
from app.ai.scoring.refine_score import TrustScoreRefiner
from app.ai.utils.text_cleaner import OCRTextCleaner


class AILayerService:
    def __init__(self, storage_dir: Path = Path("storage/ai")) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_path = self.storage_dir / "knowledge_base.json"
        self.model_path = self.storage_dir / "certificate_type_model.joblib"
        bundled_seed = Path(__file__).resolve().parents[1] / "knowledge_base" / "knowledge_base.json"
        self.knowledge_store = KnowledgeBaseStore(self.knowledge_path, bundled_seed)
        self.knowledge_base = self.knowledge_store.load()
        self.cleaner = OCRTextCleaner()
        self.type_classifier = CertificateTypeClassifier(self.knowledge_base, self.model_path)
        self.institution_recognizer = InstitutionRecognizer(self.knowledge_base)
        self.scorer = TrustScoreRefiner()
        self.explainer = VerificationExplainer()

    def interpret(self, ml_payload: Dict[str, Any]) -> AIInterpretationResponse:
        ocr_result = ml_payload.get("ocrResult", {}) or {}
        anomaly_result = ml_payload.get("anomalyResult", {}) or {}
        raw_text = self.cleaner.merge_lines(self._ocr_lines(ocr_result))
        cleaned_text = self.cleaner.clean(raw_text)

        layout_hints = self._derive_layout_hints(ocr_result)
        certificate_type = self.type_classifier.predict(cleaned_text, layout_hints)
        document_type = self._detect_document_type(cleaned_text, certificate_type, layout_hints, ocr_result)
        institution = self.institution_recognizer.predict(cleaned_text, extracted_fields=ocr_result.get("extracted_fields", {}))

        refinement = self.scorer.refine(
            ml_payload,
            certificate_type_confidence=certificate_type.confidence,
            institution_confidence=institution.confidence,
        )

        anomaly_notes = self.explainer.build_anomaly_notes(anomaly_result.get("suspicious_regions", []) or [], ocr_result)
        summary, detailed_explanation, recommended_action = self.explainer.summarize(
            refinement.refined_trust_score,
            document_type.display_name,
            certificate_type.display_name,
            institution.canonical_name,
            refinement.reasoning,
            anomaly_notes,
        )

        return AIInterpretationResponse(
            scanId=str(ml_payload.get("scanId", "")),
            documentType=document_type,
            certificateType=certificate_type,
            institution=institution,
            refinedTrustScore=refinement.refined_trust_score,
            riskLevel=refinement.risk_level,
            confidenceLevel=refinement.confidence_level,
            verificationSummary=summary,
            detailedExplanation=detailed_explanation,
            recommendedAction=recommended_action,
            anomalyNotes=anomaly_notes,
            cleanedOCRText=cleaned_text,
            reasoning=refinement.reasoning,
            metadata={
                "originalTrustScore": int(ml_payload.get("trustScore", 0) or 0),
                "verificationStatus": ml_payload.get("verificationStatus", "inconclusive"),
                "layoutHints": layout_hints,
                "documentTypeLabel": document_type.label,
                "certificateTypeLabel": certificate_type.label,
            },
        )

    def _ocr_lines(self, ocr_result: Dict[str, Any]) -> List[str]:
        raw_text = str(ocr_result.get("raw_text", "") or "")
        lines = [line for line in raw_text.splitlines() if line.strip()]
        extracted_fields = ocr_result.get("extracted_fields", {}) or {}
        for value in extracted_fields.values():
            if isinstance(value, str) and value.strip():
                lines.append(value.strip())
        return lines

    def _derive_layout_hints(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        blocks = ocr_result.get("blocks", []) or []
        has_signature_area = False
        has_table_structure = False
        if blocks:
            left_positions = [float(block.get("left", 0) or 0) for block in blocks if isinstance(block, dict)]
            top_positions = [float(block.get("top", 0) or 0) for block in blocks if isinstance(block, dict)]
            has_signature_area = any(top > 700 or left > 650 for top, left in zip(top_positions, left_positions))
            has_table_structure = len(blocks) >= 8
        return {
            "has_signature_area": has_signature_area,
            "has_table_structure": has_table_structure,
            "blocks_count": len(blocks),
        }

    def _detect_document_type(
        self,
        cleaned_text: str,
        certificate_type,
        layout_hints: Dict[str, Any],
        ocr_result: Dict[str, Any],
    ) -> DocumentTypePrediction:
        text = cleaned_text.lower()
        extracted_fields = ocr_result.get("extracted_fields", {}) or {}
        evidence = []

        id_terms = ["student id", "id card", "matric", "registration number", "student identity card", "call up number"]
        certificate_terms = ["certificate", "transcript", "degree", "convocation", "award", "diploma", "waec", "neco", "nysc", "discharge certificate"]

        id_hits = [term for term in id_terms if term in text]
        certificate_hits = [term for term in certificate_terms if term in text]

        if certificate_type.label in {"student_id_card"} or id_hits:
            confidence = 0.95 if (layout_hints.get("has_table_structure") or id_hits) else 0.82
            evidence.extend(id_hits[:3] or ["student ID indicators detected"])
            if extracted_fields.get("matric_number") or extracted_fields.get("student_photo"):
                evidence.append("ID-specific fields detected")
            return DocumentTypePrediction(label="id_card", display_name="ID Card", confidence=round(confidence, 2), evidence=evidence)

        if certificate_type.label != "unknown_document" and (certificate_hits or certificate_type.confidence >= 0.55):
            confidence = 0.93 if certificate_hits else 0.8
            evidence.extend(certificate_hits[:4] or ["certificate indicators detected"])
            if layout_hints.get("has_signature_area"):
                evidence.append("formal signature layout detected")
            return DocumentTypePrediction(label="certificate", display_name="Certificate", confidence=round(confidence, 2), evidence=evidence)

        if len(text.split()) < 8 or certificate_type.confidence < 0.35:
            evidence.append("insufficient document signals detected")
            return DocumentTypePrediction(label="not_document", display_name="Not a Document", confidence=0.9, evidence=evidence)

        if layout_hints.get("has_signature_area") or layout_hints.get("has_table_structure"):
            evidence.append("document-like structure detected")
            return DocumentTypePrediction(label="certificate", display_name="Certificate", confidence=0.58, evidence=evidence)

        evidence.append("mixed signals, defaulting to not a document")
        return DocumentTypePrediction(label="not_document", display_name="Not a Document", confidence=0.55, evidence=evidence)
