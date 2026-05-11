from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ai.models import CertificateTypePrediction
from app.ai.utils.text_cleaner import OCRTextCleaner


@dataclass
class CertificateTypeRule:
    label: str
    display_name: str
    keywords: List[str]
    patterns: List[str]
    weight: float = 1.0


class CertificateTypeClassifier:
    def __init__(self, knowledge_base: Dict[str, Any], model_path: Path) -> None:
        self.knowledge_base = knowledge_base
        self.model_path = model_path
        self.cleaner = OCRTextCleaner()
        self.model: Optional[Pipeline] = self._load_model()
        self.rules = [self._normalize_rule(item) for item in knowledge_base.get("certificate_types", [])]

    def predict(self, text: str, layout_hints: Optional[Dict[str, Any]] = None) -> CertificateTypePrediction:
        normalized = self.cleaner.normalize_for_matching(text)
        candidate_scores: List[Tuple[str, str, float, List[str], List[str]]] = []

        if self.model is not None:
            proba_text = self.model.predict_proba([normalized])[0]
            classes = self.model.named_steps["classifier"].classes_
            best_index = int(proba_text.argmax())
            candidate_scores.append(
                (
                    classes[best_index],
                    classes[best_index].replace("_", " ").title(),
                    float(proba_text[best_index]),
                    [],
                    ["trained text model"],
                )
            )

        for rule in self.rules:
            score, matched_keywords, evidence = self._rule_score(normalized, rule, layout_hints)
            candidate_scores.append((rule.label, rule.display_name, score, matched_keywords, evidence))

        if not candidate_scores:
            return CertificateTypePrediction()

        best = max(candidate_scores, key=lambda item: item[2])
        return CertificateTypePrediction(
            label=best[0],
            display_name=best[1],
            confidence=round(min(0.99, best[2]), 2),
            matched_keywords=best[3],
            evidence=best[4],
        )

    def train(self, samples: List[Dict[str, str]]) -> Path:
        texts = [self.cleaner.normalize_for_matching(sample.get("text", "")) for sample in samples]
        labels = [sample.get("label", "unknown_document") for sample in samples]
        pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )
        pipeline.fit(texts, labels)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_path)
        self.model = pipeline
        return self.model_path

    def _rule_score(self, normalized_text: str, rule: CertificateTypeRule, layout_hints: Optional[Dict[str, Any]]) -> Tuple[float, List[str], List[str]]:
        score = 0.1
        matched_keywords: List[str] = []
        evidence: List[str] = []

        for keyword in rule.keywords:
            if keyword and keyword in normalized_text:
                matched_keywords.append(keyword)
                score += 0.18 * rule.weight

        for pattern in rule.patterns:
            if pattern and pattern in normalized_text:
                matched_keywords.append(pattern)
                score += 0.22 * rule.weight

        if layout_hints:
            if layout_hints.get("has_signature_area") and rule.label in {"professional_certification", "university_degree", "nysc_certificate"}:
                score += 0.08
                evidence.append("layout suggests formal signed document")
            if layout_hints.get("has_table_structure") and rule.label in {"transcript", "student_id_card"}:
                score += 0.08
                evidence.append("layout suggests tabular document")

        if matched_keywords:
            evidence.append(f"matched keywords: {', '.join(sorted(set(matched_keywords))[:4])}")

        return min(score, 0.98), matched_keywords, evidence

    def _normalize_rule(self, item: Dict[str, Any]) -> CertificateTypeRule:
        keywords = self._normalize_terms(item.get("keywords", []))
        patterns = self._normalize_terms(item.get("patterns", []))
        return CertificateTypeRule(
            label=str(item.get("label", "unknown_document")),
            display_name=str(item.get("display_name", "Unknown Document")),
            keywords=keywords,
            patterns=patterns,
            weight=float(item.get("weight", item.get("confidence_weight", 1.0)) or 1.0),
        )

    def _normalize_terms(self, values: Any) -> List[str]:
        normalized: List[str] = []
        for value in values or []:
            if isinstance(value, str):
                term = value.strip().lower()
            elif isinstance(value, dict):
                term = str(value.get("value") or value.get("pattern") or value.get("text") or "").strip().lower()
            else:
                term = str(value).strip().lower()
            if term:
                normalized.append(term)
        return normalized

    def _load_model(self) -> Optional[Pipeline]:
        if self.model_path.exists():
            try:
                return joblib.load(self.model_path)
            except Exception:
                return None
        return None
