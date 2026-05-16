"""
Unified classifier: document type, certificate type, and institution in one pass.

Replaces the previous split between:
- app/ai/services/ai_layer_service.py::_detect_document_type
- app/ai/classifiers/certificate_type_classifier.py
- app/ai/classifiers/institution_recognizer.py

All three operated on the same OCR text. This module reads the text once,
extracts shared features, and runs all three classifications against those
features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.schemas import Classification, LayoutHints
from app.utils.text import OCRTextCleaner


# Document type rules. Each maps to keywords with weights and supporting signals.
_DOC_TYPE_RULES: Dict[str, Dict[str, Any]] = {
    "university_degree": {
        "display": "University Degree",
        "keywords": [("degree certificate", 0.25), ("award of degree", 0.20), ("bachelor of", 0.18),
                     ("master of", 0.18), ("convocation", 0.10), ("vice chancellor", 0.15), ("registrar", 0.08)],
        "signals": ["graduation", "faculty", "department", "honours"],
        "layout_signal": "has_signature_area",
    },
    "transcript": {
        "display": "Transcript",
        "keywords": [("academic transcript", 0.30), ("transcript", 0.20), ("cgpa", 0.18),
                     ("course code", 0.12), ("semester", 0.10), ("credit unit", 0.12)],
        "signals": ["result", "session", "grade point"],
        "layout_signal": "has_table_structure",
    },
    "student_id_card": {
        "display": "Student ID Card",
        "keywords": [("student id", 0.28), ("identity card", 0.22), ("matric no", 0.18),
                     ("matric number", 0.18), ("registration number", 0.12)],
        "signals": ["faculty", "department", "valid until"],
        "layout_signal": None,
    },
    "birth_certificate": {
        "display": "Birth Certificate",
        "keywords": [("birth certificate", 0.30), ("certificate of birth", 0.28),
                     ("date of birth", 0.15), ("place of birth", 0.12), ("national population", 0.20)],
        "signals": ["mother", "father", "registration number", "town"],
        "layout_signal": None,
    },
    "waec_certificate": {
        "display": "WAEC Certificate",
        "keywords": [("west african examinations council", 0.28), ("waec", 0.22),
                     ("wassce", 0.18), ("ssce", 0.10), ("candidate number", 0.10)],
        "signals": ["examination", "subject", "grade"],
        "layout_signal": None,
    },
    "neco_certificate": {
        "display": "NECO Certificate",
        "keywords": [("national examinations council", 0.28), ("neco", 0.22),
                     ("senior secondary certificate", 0.15), ("candidate number", 0.10)],
        "signals": ["examination", "subject"],
        "layout_signal": None,
    },
    "nysc_certificate": {
        "display": "NYSC Certificate",
        "keywords": [("national youth service corps", 0.28), ("nysc", 0.22),
                     ("discharge certificate", 0.20), ("call up", 0.12), ("service year", 0.10)],
        "signals": ["corps member", "state coordinator"],
        "layout_signal": "has_signature_area",
    },
    "professional_certificate": {
        "display": "Professional Certificate",
        "keywords": [("certificate of completion", 0.25), ("professional certification", 0.20),
                     ("chartered", 0.15), ("certified", 0.10)],
        "signals": ["membership", "license"],
        "layout_signal": "has_signature_area",
    },
}


@dataclass
class _DocTypeScore:
    label: str
    display: str
    score: float
    matched: List[str]
    evidence: List[str]


class UnifiedClassifier:
    """One-pass classifier. Call once per analysis."""

    def __init__(self, knowledge_base: Dict[str, Any], model_path: Path) -> None:
        self.kb = knowledge_base
        self.model_path = model_path
        self.cleaner = OCRTextCleaner()
        self.institutions = knowledge_base.get("institutions", [])
        self._ml_model: Optional[Pipeline] = self._load_ml_model()

    def classify(self, ocr_text: str, fields: Dict[str, str], layout: LayoutHints) -> Classification:
        normalized = self.cleaner.normalize(ocr_text)
        field_text = " ".join(str(v) for v in fields.values() if v).lower()
        corpus = f"{normalized} {field_text}".strip()

        doc_type = self._score_document_type(corpus, layout)
        cert_type = self._score_certificate_type(corpus, layout, doc_type)
        institution = self._match_institution(corpus, fields)

        return Classification(
            document_type=doc_type.label,
            document_type_display=doc_type.display,
            document_type_confidence=round(min(0.99, doc_type.score), 2),
            certificate_type=cert_type.label,
            certificate_type_display=cert_type.display,
            certificate_type_confidence=round(min(0.99, cert_type.score), 2),
            institution=institution[0],
            institution_confidence=institution[1],
            institution_evidence=institution[2],
            classification_evidence=doc_type.evidence[:5],
        )

    def _score_document_type(self, corpus: str, layout: LayoutHints) -> _DocTypeScore:
        best = _DocTypeScore("unknown_document", "Unknown Document", 0.0, [], [])

        for label, spec in _DOC_TYPE_RULES.items():
            score = 0.0
            matched: List[str] = []
            evidence: List[str] = []

            for phrase, weight in spec["keywords"]:
                similarity = self._fuzzy_phrase(corpus, phrase)
                if similarity:
                    score += weight * similarity
                    matched.append(phrase)

            for signal in spec["signals"]:
                if signal in corpus:
                    score += 0.04
                    evidence.append(signal)

            layout_signal = spec.get("layout_signal")
            if layout_signal and getattr(layout, layout_signal, False):
                score += 0.06
                evidence.append(f"layout: {layout_signal}")

            if score > best.score:
                best = _DocTypeScore(label, spec["display"], score, matched, evidence)

        # ML model gives a soft vote (low weight)
        if self._ml_model is not None and corpus:
            try:
                proba = self._ml_model.predict_proba([corpus])[0]
                classes = self._ml_model.named_steps["classifier"].classes_
                idx = int(proba.argmax())
                ml_label = classes[idx]
                ml_conf = float(proba[idx])
                if ml_label == best.label:
                    best.score += 0.08 * ml_conf
                    best.evidence.append(f"ml model agrees: {ml_conf:.2f}")
                elif ml_conf > 0.7 and best.score < 0.4:
                    spec = _DOC_TYPE_RULES.get(ml_label)
                    if spec:
                        best = _DocTypeScore(ml_label, spec["display"], 0.45,
                                             [], [f"ml model override: {ml_conf:.2f}"])
            except Exception:
                pass

        if best.score < 0.35:
            return _DocTypeScore("unknown_document", "Unknown Document", 0.0,
                                 [], ["insufficient document signals"])
        return best

    def _score_certificate_type(self, corpus: str, layout: LayoutHints,
                                doc_type: _DocTypeScore) -> _DocTypeScore:
        # Certificate type is a narrower lookup against the knowledge base rules.
        # For now it mirrors document type. Keep the separation so we can refine later.
        cert_rules = self.kb.get("certificate_types", [])
        best = _DocTypeScore(doc_type.label, doc_type.display,
                             doc_type.score * 0.9, doc_type.matched, [])

        for rule in cert_rules:
            label = str(rule.get("label", ""))
            display = str(rule.get("display_name", label.replace("_", " ").title()))
            keywords = rule.get("keywords", [])
            score = 0.0
            matched: List[str] = []

            for kw in keywords:
                term = kw.get("value", "") if isinstance(kw, dict) else str(kw)
                weight = kw.get("weight", 0.15) if isinstance(kw, dict) else 0.15
                if term and term.lower() in corpus:
                    score += min(0.20, float(weight) * 0.20)
                    matched.append(term)

            if score > best.score:
                best = _DocTypeScore(label, display, score, matched, [f"keyword matches: {len(matched)}"])

        if best.score < 0.20:
            return _DocTypeScore("unknown_document", "Unknown Document", 0.0, [], [])
        return best

    def _match_institution(self, corpus: str,
                           fields: Dict[str, str]) -> Tuple[str, float, List[str]]:
        if not corpus and not fields:
            return "Unknown Institution", 0.0, []

        best_name = ""
        best_score = 0.0
        best_evidence: List[str] = []

        search_targets = [corpus]
        for v in fields.values():
            if v:
                search_targets.append(self.cleaner.normalize(str(v)))

        for item in self.institutions:
            canonical = item.get("canonical_name", "")
            if not canonical:
                continue
            forms = [canonical] + item.get("aliases", []) + item.get("keywords", [])

            for target in search_targets:
                if not target:
                    continue
                for form in forms:
                    if not form:
                        continue
                    form_lower = form.lower()
                    if form_lower in target:
                        score = 0.95 if form_lower == canonical.lower() else 0.85
                        if score > best_score:
                            best_name = canonical
                            best_score = score
                            best_evidence = [f"exact match: {form}"]
                    else:
                        partial = fuzz.partial_ratio(form_lower, target) / 100.0
                        if partial > 0.85 and partial > best_score:
                            best_name = canonical
                            best_score = partial * 0.9  # discount fuzzy
                            best_evidence = [f"fuzzy match: {form} ({int(partial*100)}%)"]

        if best_score < 0.65:
            return "Unknown Institution", 0.0, ["no confident institution match"]

        return best_name, round(best_score, 2), best_evidence

    def _fuzzy_phrase(self, corpus: str, phrase: str) -> float:
        if not corpus or not phrase:
            return 0.0
        if phrase in corpus:
            return 1.0
        score = fuzz.partial_ratio(phrase, corpus)
        return (score / 100.0) if score >= 85 else 0.0

    def _load_ml_model(self) -> Optional[Pipeline]:
        if self.model_path.exists():
            try:
                return joblib.load(self.model_path)
            except Exception:
                return None
        return None

    def reload_model(self) -> None:
        self._ml_model = self._load_ml_model()
