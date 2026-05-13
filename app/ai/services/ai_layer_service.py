from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import re

from rapidfuzz import fuzz

from app.ai.classifiers.certificate_type_classifier import CertificateTypeClassifier
from app.ai.classifiers.institution_recognizer import InstitutionRecognizer
from app.ai.explainers.verification_explainer import VerificationExplainer
from app.ai.knowledge_base.store import KnowledgeBaseStore
from app.ai.models import AIInterpretationResponse, CertificateTypePrediction, DocumentTypePrediction, InstitutionPrediction
from app.ai.scoring.refine_score import TrustScoreRefiner
from app.ai.utils.text_cleaner import OCRTextCleaner


@lru_cache(maxsize=4)
def _shared_components(storage_dir: str, bundled_seed: str) -> Dict[str, Any]:
    storage_path = Path(storage_dir)
    storage_path.mkdir(parents=True, exist_ok=True)

    knowledge_path = storage_path / "knowledge_base.json"
    model_path = storage_path / "certificate_type_model.joblib"
    knowledge_store = KnowledgeBaseStore(knowledge_path, Path(bundled_seed))
    knowledge_base = knowledge_store.load()

    return {
        "knowledge_base": knowledge_base,
        "cleaner": OCRTextCleaner(),
        "type_classifier": CertificateTypeClassifier(knowledge_base, model_path),
        "institution_recognizer": InstitutionRecognizer(knowledge_base),
        "scorer": TrustScoreRefiner(),
        "explainer": VerificationExplainer(),
    }


class AILayerService:
    def __init__(self, storage_dir: Path = Path("storage/ai")) -> None:
        self.storage_dir = storage_dir
        bundled_seed = Path(__file__).resolve().parents[1] / "knowledge_base" / "knowledge_base.json"
        components = _shared_components(str(storage_dir), str(bundled_seed))

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_path = self.storage_dir / "knowledge_base.json"
        self.model_path = self.storage_dir / "certificate_type_model.joblib"
        self.knowledge_base = components["knowledge_base"]
        self.cleaner = components["cleaner"]
        self.type_classifier = components["type_classifier"]
        self.institution_recognizer = components["institution_recognizer"]
        self.scorer = components["scorer"]
        self.explainer = components["explainer"]

    def interpret(self, ml_payload: Dict[str, Any]) -> AIInterpretationResponse:
        ocr_result = ml_payload.get("ocrResult", {}) or {}
        anomaly_result = ml_payload.get("anomalyResult", {}) or {}

        raw_text = self.cleaner.merge_lines(self._ocr_lines(ocr_result))
        cleaned_text = self._clean_ocr_text(self.cleaner.clean(raw_text))
        layout_hints = self._derive_layout_hints(ocr_result)

        certificate_type = self.type_classifier.predict(cleaned_text, layout_hints)
        document_type = self._detect_document_type(cleaned_text, certificate_type, layout_hints, ocr_result)
        institution = self._refine_institution_prediction(
            self.institution_recognizer.predict(cleaned_text, extracted_fields=ocr_result.get("extracted_fields", {})),
            cleaned_text,
            ocr_result,
        )

        refinement = self.scorer.refine(
            ml_payload,
            certificate_type_confidence=certificate_type.confidence,
            institution_confidence=institution.confidence,
            document_type_confidence=document_type.confidence,
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
            metadata=self._build_metadata(
                ml_payload=ml_payload,
                ocr_result=ocr_result,
                anomaly_result=anomaly_result,
                layout_hints=layout_hints,
                document_type=document_type,
                certificate_type=certificate_type,
                institution=institution,
                preprocessing_metadata=ml_payload.get("preprocessingMetadata", {}) or {},
                reasoning=refinement.reasoning,
            ),
        )

    def _clean_ocr_text(self, text: str) -> str:
        cleaned = re.sub(r"[^\w\s/&.,:;()\-]", " ", str(text or ""))
        replacements = {
            r"\b@irth\b": "Birth",
            r"\bBinth\b": "Birth",
            r"\b0f\b": "of",
            r"\bCettificate\b": "Certificate",
            r"\bCertiflcate\b": "Certificate",
            r"\bUnlversity\b": "University",
            r"\bDepa rtment\b": "Department",
            r"\blD\b": "ID",
        }
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(\w)\1{2,}", r"\1\1", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    def _refine_institution_prediction(
        self,
        prediction: InstitutionPrediction,
        cleaned_text: str,
        ocr_result: Dict[str, Any],
    ) -> InstitutionPrediction:
        name = (prediction.canonical_name or prediction.name or "").strip()
        if not name or "unknown" in name.lower():
            return InstitutionPrediction(evidence=["no stable institution match"], confidence=0.0)

        text = cleaned_text.lower()
        support_terms = ["university", "college", "polytechnic", "institute", "school", "academy", "board", "council", "ministry"]
        support_hits = sum(1 for term in support_terms if term in text)
        fuzzy_score = fuzz.partial_ratio(name.lower(), text) if text else 0

        confidence = round(min(1.0, (prediction.confidence * 0.7) + (fuzzy_score / 100.0 * 0.3)), 2)
        evidence = list(prediction.evidence or [])
        evidence.append(f"fuzzy match score: {fuzzy_score}/100")

        if support_hits == 0 and confidence < 0.75:
            return InstitutionPrediction(name="Unknown Institution", canonical_name="Unknown Institution", confidence=0.0, matched_text="", evidence=["low-confidence institution match"])

        if confidence < 0.65:
            return InstitutionPrediction(name="Unknown Institution", canonical_name="Unknown Institution", confidence=0.0, matched_text="", evidence=evidence or ["low-confidence institution match"])

        return prediction.model_copy(update={"confidence": confidence, "evidence": evidence})

    def _build_metadata(
        self,
        *,
        ml_payload: Dict[str, Any],
        ocr_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        layout_hints: Dict[str, Any],
        document_type: DocumentTypePrediction,
        certificate_type: CertificateTypePrediction,
        institution: InstitutionPrediction,
        preprocessing_metadata: Dict[str, Any],
        reasoning: List[str],
    ) -> Dict[str, Any]:
        return {
            "originalTrustScore": int(ml_payload.get("trustScore", 0) or 0),
            "verificationStatus": ml_payload.get("verificationStatus", "inconclusive"),
            "layoutHints": layout_hints,
            "blocksCount": int(layout_hints.get("blocks_count", len(ocr_result.get("blocks", []) or []))),
            "ocrConfidence": round(float(ocr_result.get("average_confidence", 0.0) or 0.0), 2),
            "documentType": document_type.model_dump(),
            "certificateType": certificate_type.model_dump(),
            "institution": institution.model_dump(),
            "reasoningSummary": reasoning[:3],
            "anomalySummary": {
                "score": round(float(anomaly_result.get("overall_score", 0.0) or 0.0), 2),
                "regions": len(anomaly_result.get("suspicious_regions", []) or []),
            },
            "preprocessingMetadata": preprocessing_metadata,
            "heatmapNotes": ml_payload.get("heatmapNotes", []) or [],
        }

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
        if not blocks:
            return {"has_signature_area": False, "has_table_structure": False, "has_header_area": False, "blocks_count": 0}

        left_positions = [float(block.get("left", 0) or 0) for block in blocks if isinstance(block, dict)]
        top_positions = [float(block.get("top", 0) or 0) for block in blocks if isinstance(block, dict)]

        return {
            "has_signature_area": any(top > 700 or left > 650 for top, left in zip(top_positions, left_positions)),
            "has_table_structure": len(blocks) >= 8,
            "has_header_area": any(top < 180 for top in top_positions),
            "has_footer_area": any(top > 700 for top in top_positions),
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
        field_text = " ".join(str(value) for value in extracted_fields.values() if str(value).strip()).lower()
        corpus = f"{text} {field_text}".strip()

        candidates = {
            "university_degree": {
                "display_name": "University Degree",
                "keywords": [("degree certificate", 0.18), ("award of degree", 0.16), ("bachelor of", 0.14), ("master of", 0.14), ("convocation", 0.08), ("university", 0.1)],
                "signals": ["graduation", "faculty", "department"],
            },
            "transcript": {
                "display_name": "Transcript",
                "keywords": [("transcript", 0.26), ("academic record", 0.16), ("course title", 0.1), ("semester", 0.08), ("grade", 0.06)],
                "signals": ["cgpa", "result", "session"],
            },
            "student_id_card": {
                "display_name": "Student ID Card",
                "keywords": [("student id", 0.24), ("identity card", 0.2), ("matric", 0.14), ("registration number", 0.12), ("student photograph", 0.08)],
                "signals": ["photo", "expiry", "signature"],
            },
            "birth_certificate": {
                "display_name": "Birth Certificate",
                "keywords": [("birth certificate", 0.26), ("date of birth", 0.16), ("place of birth", 0.12), ("live birth", 0.12), ("registrar", 0.08)],
                "signals": ["mother", "father", "local government"],
            },
            "waec_certificate": {
                "display_name": "WAEC Certificate",
                "keywords": [("waec", 0.22), ("west african examinations council", 0.2), ("ssce", 0.12), ("candidate number", 0.1), ("result slip", 0.08)],
                "signals": ["examination", "grade", "subject"],
            },
            "neco_certificate": {
                "display_name": "NECO Certificate",
                "keywords": [("neco", 0.22), ("national examination council", 0.2), ("senior secondary certificate", 0.14), ("candidate number", 0.1)],
                "signals": ["examination", "result", "subject"],
            },
            "nysc_certificate": {
                "display_name": "NYSC Certificate",
                "keywords": [("nysc", 0.24), ("national youth service corps", 0.24), ("discharge certificate", 0.18), ("passing out parade", 0.08)],
                "signals": ["service year", "certificate of national service"],
            },
            "professional_certificate": {
                "display_name": "Professional Certificate",
                "keywords": [("professional certificate", 0.22), ("membership certificate", 0.16), ("chartered", 0.12), ("institute", 0.08), ("license", 0.08)],
                "signals": ["diploma", "accreditation", "board"],
            },
        }

        best_label = "unknown_document"
        best_display = "Unknown Document"
        best_score = 0.0
        best_keywords: List[str] = []
        best_evidence: List[str] = []

        for label, spec in candidates.items():
            score = 0.0
            matched_keywords: List[str] = []
            evidence: List[str] = []

            for phrase, weight in spec["keywords"]:
                match_score = self._match_phrase(corpus, phrase)
                if match_score:
                    score += weight * match_score
                    matched_keywords.append(phrase)

            if certificate_type.label == label:
                score += 0.12
                evidence.append(f"certificate classifier suggested {label}")

            if label == "transcript" and layout_hints.get("has_table_structure"):
                score += 0.08
                evidence.append("table-like layout detected")
            if label == "student_id_card" and layout_hints.get("has_header_area"):
                score += 0.05
                evidence.append("ID-style header layout detected")
            if label in {"university_degree", "professional_certificate", "nysc_certificate"} and layout_hints.get("has_signature_area"):
                score += 0.04
                evidence.append("formal signature area detected")

            for signal in spec["signals"]:
                if signal in corpus:
                    score += 0.04
                    evidence.append(signal)

            if score > best_score:
                best_label = label
                best_display = spec["display_name"]
                best_score = score
                best_keywords = matched_keywords
                best_evidence = evidence

        if best_score < 0.35:
            return DocumentTypePrediction(label="unknown_document", display_name="Unknown Document", confidence=0.0, matched_keywords=[], evidence=["insufficient document signals detected"])

        confidence = round(min(0.99, best_score), 2)
        if confidence < 0.45:
            return DocumentTypePrediction(label="unknown_document", display_name="Unknown Document", confidence=confidence, matched_keywords=best_keywords[:6], evidence=best_evidence or ["weak document signals detected"])

        return DocumentTypePrediction(
            label=best_label,
            display_name=best_display,
            confidence=confidence,
            matched_keywords=best_keywords[:8],
            evidence=best_evidence[:8],
        )

    def _match_phrase(self, text: str, phrase: str) -> float:
        if phrase in text:
            return 1.0
        score = fuzz.partial_ratio(phrase, text)
        return score / 100.0 if score >= 82 else 0.0
