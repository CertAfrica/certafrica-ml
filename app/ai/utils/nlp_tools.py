from __future__ import annotations

from typing import Any, Dict, List

try:
    import spacy
    from spacy.language import Language
    from spacy.pipeline import EntityRuler
except Exception:  # pragma: no cover - optional dependency
    spacy = None
    Language = Any
    EntityRuler = Any


class LightweightNLPTools:
    def __init__(self, knowledge_base: Dict[str, Any]) -> None:
        self.knowledge_base = knowledge_base
        self.nlp = self._build_pipeline()

    def _build_pipeline(self):
        if spacy is None:
            return None
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler")
        patterns: List[Dict[str, Any]] = []
        for item in self.knowledge_base.get("institutions", []):
            canonical = item.get("canonical_name", "")
            aliases = item.get("aliases", []) or []
            keywords = item.get("keywords", []) or []
            for form in [canonical] + aliases + keywords:
                if form:
                    patterns.append({"label": "INSTITUTION", "pattern": form, "id": canonical})
        if patterns:
            ruler.add_patterns(patterns)
        return nlp

    def extract_institution_entities(self, text: str) -> List[Dict[str, str]]:
        if self.nlp is None or not text:
            return []
        doc = self.nlp(text)
        entities: List[Dict[str, str]] = []
        for ent in doc.ents:
            if ent.label_ == "INSTITUTION":
                entities.append({"text": ent.text, "canonical_name": ent.ent_id_ or ent.text})
        return entities
