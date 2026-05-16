from __future__ import annotations

import json
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from fastapi import UploadFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline

from app.config import settings
from app.pipeline.ocr import OCRExtractor
from app.pipeline.preprocess import ImagePreprocessor
from app.pipeline.visual import VisualEmbedder, VisualIndex
from app.schemas import TrainingStatsResponse, TrainingUploadResponse
from app.utils.image import read_image
from app.utils.text import OCRTextCleaner
from app.utils.validation import validate_upload


AUTO_RETRAIN_THRESHOLD = 10


class Trainer:
    def __init__(self, embedder: VisualEmbedder, index: VisualIndex) -> None:
        training_dir = settings.storage_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path = training_dir / "samples.jsonl"
        self.model_path = settings.storage_dir / "classifier.joblib"
        self.last_trained_path = training_dir / "last_trained.txt"
        self.preprocessor = ImagePreprocessor()
        self.ocr = OCRExtractor()
        self.cleaner = OCRTextCleaner()
        self.embedder = embedder
        self.index = index

    async def ingest(self, certificate: UploadFile, cert_type: str,
                     institution: Optional[str] = None) -> TrainingUploadResponse:
        validate_upload(certificate, settings.max_upload_mb * 1024 * 1024)

        with tempfile.TemporaryDirectory(prefix="train-") as work_dir:
            work = Path(work_dir)
            suffix = Path(certificate.filename or "sample").suffix.lower() or ".png"
            tmp_path = work / f"sample{suffix}"
            with tmp_path.open("wb") as buf:
                certificate.file.seek(0)
                while True:
                    chunk = await certificate.read(1024 * 1024)
                    if not chunk:
                        break
                    buf.write(chunk)
            await certificate.close()

            image = read_image(tmp_path)
            preprocessed = self.preprocessor.preprocess(image)

            # OCR for text classifier
            ocr = self.ocr.extract(preprocessed["rgb"])

            # Visual embedding for similarity index
            visual_indexed = False
            sample_id = uuid.uuid4().hex
            try:
                embedding = self.embedder.embed(preprocessed["rgb"])
                self.index.add(
                    embedding=embedding,
                    sample_id=sample_id,
                    label=cert_type,
                    institution=institution or "",
                )
                visual_indexed = True
            except Exception:
                # Don't fail the whole upload if embedding fails. The text
                # classifier sample still gets stored.
                visual_indexed = False

        cleaned_text = self.cleaner.normalize(ocr.raw_text)

        sample = {
            "id": sample_id,
            "label": cert_type,
            "institution": institution or "",
            "text": cleaned_text,
            "ocr_confidence": ocr.average_confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.samples_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample) + "\n")

        label_count = sum(1 for s in self._iter_samples() if s.get("label") == cert_type)

        retrained = False
        if self._samples_since_last_train() >= AUTO_RETRAIN_THRESHOLD:
            self.retrain()
            retrained = True

        return TrainingUploadResponse(
            sample_id=sample_id,
            label=cert_type,
            extracted_text_preview=cleaned_text[:200],
            total_samples_for_label=label_count,
            retrained=retrained,
            visual_indexed=visual_indexed,
        )

    def retrain(self) -> Dict[str, Any]:
        samples = list(self._iter_samples())
        if len(samples) < 3:
            return {"trained": False, "reason": "need at least 3 samples"}

        labels = [s["label"] for s in samples]
        if len(set(labels)) < 2:
            return {"trained": False, "reason": "need at least 2 distinct labels"}

        texts = [s["text"] for s in samples]
        pipeline = SKPipeline([
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipeline.fit(texts, labels)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_path)
        self.last_trained_path.write_text(datetime.now(timezone.utc).isoformat())

        return {
            "trained": True,
            "total_samples": len(samples),
            "label_distribution": dict(Counter(labels)),
        }

    def stats(self) -> TrainingStatsResponse:
        samples = list(self._iter_samples())
        counts = Counter(s["label"] for s in samples)
        last_trained = None
        if self.last_trained_path.exists():
            try:
                last_trained = datetime.fromisoformat(
                    self.last_trained_path.read_text().strip()
                )
            except Exception:
                pass

        return TrainingStatsResponse(
            samples_per_label=dict(counts),
            total_samples=len(samples),
            visual_references_per_label=self.index.count_by_label(),
            total_visual_references=self.index.total,
            last_trained_at=last_trained,
            model_exists=self.model_path.exists(),
        )

    def _iter_samples(self):
        if not self.samples_path.exists():
            return
        with self.samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _samples_since_last_train(self) -> int:
        if not self.last_trained_path.exists():
            return sum(1 for _ in self._iter_samples())
        try:
            last_trained = datetime.fromisoformat(
                self.last_trained_path.read_text().strip()
            )
        except Exception:
            return sum(1 for _ in self._iter_samples())

        count = 0
        for sample in self._iter_samples():
            try:
                ts = datetime.fromisoformat(sample.get("timestamp", ""))
                if ts > last_trained:
                    count += 1
            except Exception:
                continue
        return count