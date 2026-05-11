from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class KnowledgeBaseStore:
    def __init__(self, base_path: Path, seed_path: Optional[Path] = None) -> None:
        self.base_path = base_path
        self.seed_path = seed_path
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.base_path.exists():
            self._write_seed_or_default()
        else:
            try:
                current = json.loads(self.base_path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
            if self.seed_path and self.seed_path.exists() and self._should_refresh_runtime_data(current):
                self.base_path.write_text(self.seed_path.read_text(encoding="utf-8"), encoding="utf-8")

    def _write_seed_or_default(self) -> None:
        if self.seed_path and self.seed_path.exists():
            self.base_path.write_text(self.seed_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            self.base_path.write_text(json.dumps(self._default_data(), indent=2), encoding="utf-8")

    def load(self) -> Dict[str, Any]:
        base_data = json.loads(self.base_path.read_text(encoding="utf-8"))
        seed_data = self._load_seed_data()
        return self._merge(seed_data, base_data)

    def save(self, data: Dict[str, Any]) -> None:
        self.base_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_seed_data(self) -> Dict[str, Any]:
        if self.seed_path and self.seed_path.exists():
            try:
                return json.loads(self.seed_path.read_text(encoding="utf-8"))
            except Exception:
                return self._default_data()
        return self._default_data()

    def _merge(self, seed: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(seed)
        for key, value in runtime.items():
            if key not in merged:
                merged[key] = value
                continue

            seed_value = merged[key]
            if isinstance(seed_value, dict) and isinstance(value, dict):
                merged[key] = self._merge_dicts(seed_value, value)
            elif isinstance(seed_value, list) and isinstance(value, list):
                merged[key] = self._merge_lists(seed_value, value)
            else:
                merged[key] = value
        return merged

    def _merge_dicts(self, seed_value: Dict[str, Any], runtime_value: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(seed_value)
        for key, value in runtime_value.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_dicts(merged[key], value)
            elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = self._merge_lists(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _should_refresh_runtime_data(self, current: Dict[str, Any]) -> bool:
        if not current:
            return True
        if not current.get("metadata"):
            return True

        current_certificate_types = current.get("certificate_types", []) or []
        if current_certificate_types:
            first_rule = next((item for item in current_certificate_types if isinstance(item, dict)), None)
            if first_rule:
                keywords = first_rule.get("keywords", []) or []
                if keywords and isinstance(keywords[0], str):
                    return True

        if isinstance(current.get("layout_patterns"), list):
            return True
        if isinstance(current.get("fraud_detection_rules"), list):
            return True
        if isinstance(current.get("security_features"), list):
            return True
        return False

    def _merge_lists(self, seed_value: List[Any], runtime_value: Any) -> List[Any]:
        merged: List[Any] = []
        seen = set()
        for item in list(seed_value) + list(runtime_value or []):
            marker = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(item)
        return merged

    def add_certificate_keyword(self, keyword: str) -> None:
        data = self.load()
        keywords = data.setdefault("certificate_keywords", [])
        normalized = keyword.strip().lower()
        if normalized and normalized not in keywords:
            keywords.append(normalized)
            self.save(data)

    def add_institution_pattern(self, canonical_name: str, aliases: List[str], keywords: List[str]) -> None:
        data = self.load()
        institutions = data.setdefault("institutions", [])
        institutions.append(
            {
                "canonical_name": canonical_name,
                "aliases": aliases,
                "keywords": keywords,
            }
        )
        self.save(data)

    def _default_data(self) -> Dict[str, Any]:
        return {
            "certificate_types": [],
            "institutions": [],
            "layout_patterns": {},
            "certificate_keywords": [],
            "common_phrases": [],
            "metadata": {},
            "fraud_rules": [],
            "ocr_corrections": {},
            "confidence_scoring": {},
            "risk_levels": [],
            "temporal_validation": {},
            "fraud_detection_rules": {},
            "region_intelligence": [],
            "security_features": {},
            "known_forger_patterns": {},
            "verification_workflow": {},
            "explanation_templates": {},
            "audit_trail_template": {},
        }
