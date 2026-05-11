from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from app.ai.classifiers.certificate_type_classifier import CertificateTypeClassifier
from app.ai.knowledge_base.store import KnowledgeBaseStore


class LightweightTrainer:
    def __init__(self, storage_dir: Path = Path("storage/ai")) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_path = self.storage_dir / "knowledge_base.json"
        self.model_path = self.storage_dir / "certificate_type_model.joblib"
        bundled_seed = Path(__file__).resolve().parents[1] / "knowledge_base" / "knowledge_base.json"
        self.knowledge_store = KnowledgeBaseStore(self.knowledge_path, bundled_seed)
        self.classifier = CertificateTypeClassifier(self.knowledge_store.load(), self.model_path)

    def train_certificate_type_model(self, samples: List[Dict[str, str]]) -> Path:
        return self.classifier.train(samples)

    def add_institution(self, canonical_name: str, aliases: List[str], keywords: List[str]) -> None:
        self.knowledge_store.add_institution_pattern(canonical_name, aliases, keywords)

    def add_keyword(self, keyword: str) -> None:
        self.knowledge_store.add_certificate_keyword(keyword)
