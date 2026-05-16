from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


EMBEDDING_DIM = 576


class VisualEmbedder:
    """Single-instance image embedder. Load once at service startup."""

    def __init__(self) -> None:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        # Replace classifier head with identity so model.forward returns
        # the 576-dim pooled features instead of 1000-dim class logits.
        model.classifier = nn.Identity()
        model.eval()
        self.model = model
        self.transform = weights.transforms()

    @torch.no_grad()
    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        """Embed an RGB image array. Returns L2-normalized 576-dim vector."""
        pil = Image.fromarray(image_rgb.astype(np.uint8))
        tensor = self.transform(pil).unsqueeze(0)
        features = self.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(features)) + 1e-8
        return features / norm


@dataclass
class VisualMatch:
    sample_id: str
    label: str
    institution: str
    similarity: float


@dataclass
class VisualSimilarity:
    """Result of visual matching for an analyzed scan."""
    score: Optional[float]  # 0-100, or None if no references available
    best_similarity: float  # 0-1 cosine similarity to top match
    reference_count: int    # how many references existed for this label
    matches: List[VisualMatch] = field(default_factory=list)
    note: str = ""


class VisualIndex:
    """Append-only vector index of training sample embeddings."""

    def __init__(self, storage_dir: Path) -> None:
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = storage_dir / "embeddings.npy"
        self.metadata_path = storage_dir / "embedding_metadata.jsonl"
        self._embeddings: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.embeddings_path.exists():
            try:
                self._embeddings = np.load(self.embeddings_path)
            except Exception:
                self._embeddings = None
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as f:
                self._metadata = [
                    json.loads(line) for line in f if line.strip()
                ]

    def add(self, embedding: np.ndarray, sample_id: str, label: str,
            institution: str = "") -> None:
        vec = embedding.reshape(1, -1).astype(np.float32)
        if self._embeddings is None:
            self._embeddings = vec
        else:
            self._embeddings = np.vstack([self._embeddings, vec])

        # Atomic-ish write: save to temp then rename
        tmp_path = self.embeddings_path.with_suffix(".npy.tmp")
        np.save(tmp_path, self._embeddings)
        tmp_path.replace(self.embeddings_path)

        meta = {
            "sample_id": sample_id,
            "label": label,
            "institution": institution,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._metadata.append(meta)
        with self.metadata_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")

    def search(self, query: np.ndarray, label: Optional[str] = None,
               k: int = 5) -> List[VisualMatch]:
        if self._embeddings is None or not self._metadata:
            return []

        # Cosine similarity. Vectors are L2-normalized already.
        sims = self._embeddings @ query.astype(np.float32)

        if label:
            mask = np.array([m["label"] == label for m in self._metadata])
            if not mask.any():
                return []
            valid_indices = np.where(mask)[0]
            valid_sims = sims[valid_indices]
            order = np.argsort(-valid_sims)[:k]
            indices = valid_indices[order]
        else:
            indices = np.argsort(-sims)[:k]

        return [
            VisualMatch(
                sample_id=self._metadata[i]["sample_id"],
                label=self._metadata[i]["label"],
                institution=self._metadata[i].get("institution", ""),
                similarity=float(sims[i]),
            )
            for i in indices
        ]

    def count_by_label(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for m in self._metadata:
            counts[m["label"]] = counts.get(m["label"], 0) + 1
        return counts

    @property
    def total(self) -> int:
        return len(self._metadata)


class VisualSimilarityScorer:
    """Converts cosine similarities into a 0-100 trust component."""

    def score(self, embedding: np.ndarray, predicted_label: str,
              index: VisualIndex) -> VisualSimilarity:
        if index.total == 0:
            return VisualSimilarity(
                score=None,
                best_similarity=0.0,
                reference_count=0,
                note="no training references uploaded yet",
            )

        # Try filtered search first
        matches = index.search(embedding, label=predicted_label, k=5)
        reference_count_for_label = index.count_by_label().get(predicted_label, 0)

        if not matches:
            # No references for the predicted type. Fall back to cross-label
            # search just for diagnostic info, but don't score it.
            cross_matches = index.search(embedding, k=3)
            return VisualSimilarity(
                score=None,
                best_similarity=cross_matches[0].similarity if cross_matches else 0.0,
                reference_count=0,
                matches=cross_matches,
                note=f"no references for predicted type '{predicted_label}'",
            )

        top_sims = [m.similarity for m in matches[:3]]
        avg = sum(top_sims) / len(top_sims)
        best = top_sims[0]

        component_score = self._map_similarity_to_score(avg)

        # Penalize if we only have 1-2 references (less reliable signal)
        if reference_count_for_label < 3:
            component_score = component_score * 0.7 + 30  # dampen toward neutral
            note = f"only {reference_count_for_label} reference(s) for this type"
        else:
            note = f"{reference_count_for_label} references for this type"

        return VisualSimilarity(
            score=round(component_score, 1),
            best_similarity=round(best, 3),
            reference_count=reference_count_for_label,
            matches=matches,
            note=note,
        )

    def _map_similarity_to_score(self, avg_similarity: float) -> float:
        """Map cosine similarity (0-1) to a 0-100 trust score.

        Why these breakpoints:
        - Two scans of the same template typically score 0.85 to 0.98
        - Same document family (e.g. two different birth certificates) is 0.70 to 0.85
        - Unrelated documents land below 0.5
        """
        if avg_similarity >= 0.90:
            return 90 + (avg_similarity - 0.90) * 100  # 90-100
        if avg_similarity >= 0.75:
            return 70 + (avg_similarity - 0.75) / 0.15 * 20  # 70-90
        if avg_similarity >= 0.60:
            return 50 + (avg_similarity - 0.60) / 0.15 * 20  # 50-70
        if avg_similarity >= 0.40:
            return 30 + (avg_similarity - 0.40) / 0.20 * 20  # 30-50
        return max(0, avg_similarity / 0.40 * 30)  # 0-30