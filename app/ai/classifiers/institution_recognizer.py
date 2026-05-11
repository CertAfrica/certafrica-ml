from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from app.ai.models import InstitutionPrediction
from app.ai.utils.nlp_tools import LightweightNLPTools
from app.ai.utils.text_cleaner import OCRTextCleaner


class InstitutionRecognizer:
    def __init__(self, knowledge_base: Dict[str, Any]) -> None:
        self.knowledge_base = knowledge_base
        self.cleaner = OCRTextCleaner()
        self.institutions = knowledge_base.get("institutions", [])
        self.nlp_tools = LightweightNLPTools(knowledge_base)

    def predict(self, text: str, extracted_fields: Optional[Dict[str, str]] = None) -> InstitutionPrediction:
        cleaned = self.cleaner.normalize_for_matching(text)
        extracted_fields = extracted_fields or {}
        evidence: List[str] = []
        best_name = "Unknown Institution"
        canonical_name = "Unknown Institution"
        best_score = 0.0
        matched_text = ""

        candidates: List[Tuple[str, List[str], List[str]]] = []
        for item in self.institutions:
            canonical = item.get("canonical_name", "")
            aliases = item.get("aliases", []) or []
            keywords = item.get("keywords", []) or []
            candidates.append((canonical, aliases, keywords))

        search_space = [cleaned]
        for field_value in extracted_fields.values():
            search_space.append(self.cleaner.normalize_for_matching(str(field_value)))

        for entity in self.nlp_tools.extract_institution_entities(text):
            canonical = entity.get("canonical_name", "Unknown Institution")
            if canonical != "Unknown Institution":
                return InstitutionPrediction(
                    name=canonical,
                    canonical_name=canonical,
                    confidence=0.95,
                    matched_text=entity.get("text", ""),
                    evidence=["spaCy entity ruler match"],
                )

        for candidate in search_space:
            if not candidate:
                continue
            choice = self._match_candidate(candidate)
            if choice and choice[1] > best_score:
                canonical_name, best_score, matched_text, evidence = choice
                best_name = canonical_name

        if best_score < 0.55 and extracted_fields.get("institution"):
            fallback = self._match_candidate(self.cleaner.normalize_for_matching(extracted_fields["institution"]))
            if fallback:
                canonical_name, best_score, matched_text, evidence = fallback
                best_name = canonical_name

        if best_score == 0.0 and any(word in cleaned for word in ["lagos", "ibadan", "covenant", "waec", "neco", "nysc"]):
            inferred = self._infer_from_fragments(cleaned)
            if inferred:
                canonical_name, matched_text, evidence = inferred
                best_name = canonical_name
                best_score = 0.6

        return InstitutionPrediction(
            name=best_name,
            canonical_name=canonical_name if canonical_name != "Unknown Institution" else best_name,
            confidence=round(best_score, 2),
            matched_text=matched_text,
            evidence=evidence,
        )

    def _match_candidate(self, text: str) -> Optional[Tuple[str, float, str, List[str]]]:
        best_name = None
        best_score = 0.0
        matched_text = ""
        evidence: List[str] = []

        for item in self.institutions:
            canonical = item.get("canonical_name", "")
            aliases = item.get("aliases", []) or []
            keywords = item.get("keywords", []) or []
            all_forms = [canonical] + aliases + keywords
            result = process.extractOne(text, all_forms, scorer=fuzz.partial_ratio)
            if not result:
                continue
            match_text, score, _ = result
            normalized_score = score / 100.0
            if normalized_score > best_score:
                best_name = canonical
                best_score = normalized_score
                matched_text = match_text
                evidence = [f"fuzzy match against {match_text}"]
                if any(alias in match_text.lower() for alias in aliases):
                    evidence.append("matched institution alias")

        if best_name:
            return best_name, best_score, matched_text, evidence
        return None

    def _infer_from_fragments(self, text: str) -> Optional[Tuple[str, str, List[str]]]:
        if "of lagos" in text or text.endswith("lagos") or "unilag" in text or "lagos state university" in text:
            return "University of Lagos", "of lagos", ["inferred from fragmented OCR text"]
        if "of ibadan" in text or text.endswith("ibadan") or "unibadan" in text or "ui" == text.strip():
            return "University of Ibadan", "of ibadan", ["inferred from fragmented OCR text"]
        if "covenant" in text or "c u" in text:
            return "Covenant University", "covenant", ["inferred from fragmented OCR text"]
        if "waec" in text or "wassce" in text or "w a e c" in text:
            return "WAEC", "waec", ["exam body keyword detected"]
        if "neco" in text or "n e c o" in text:
            return "NECO", "neco", ["exam body keyword detected"]
        if "nysc" in text or "n y s c" in text:
            return "NYSC", "nysc", ["corps keyword detected"]
        return None
