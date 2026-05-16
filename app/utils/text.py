from __future__ import annotations

import re
from typing import Iterable, List


class OCRTextCleaner:
    """Single source of text cleaning. Used by OCR extractor and classifier."""

    def clean(self, text: str) -> str:
        text = text or ""
        text = text.replace("|", "I")
        text = re.sub(r"\b0f\b", "of", text, flags=re.IGNORECASE)
        text = re.sub(r"\bCettificate\b", "Certificate", text, flags=re.IGNORECASE)
        text = re.sub(r"\bCertiflcate\b", "Certificate", text, flags=re.IGNORECASE)
        text = re.sub(r"\bUnlversity\b", "University", text, flags=re.IGNORECASE)
        text = re.sub(r"\b@irth\b", "Birth", text, flags=re.IGNORECASE)
        text = re.sub(r"\bBinth\b", "Birth", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdegre\b", "degree", text, flags=re.IGNORECASE)
        text = re.sub(r"\bbachalor\b", "bachelor", text, flags=re.IGNORECASE)
        text = re.sub(r"(\w)\1{2,}", r"\1\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize(self, text: str) -> str:
        """Lowercase + strip punctuation, for matching."""
        text = self.clean(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        # Collapse spaced-out letters like "u n i l a g" -> "unilag"
        text = re.sub(
            r"\b(?:[a-z]\s+){2,}[a-z]\b",
            lambda m: m.group(0).replace(" ", ""),
            text,
        )
        return text.strip()

    def merge_lines(self, lines: Iterable[str]) -> str:
        out: List[str] = []
        for line in lines:
            cleaned = self.clean(line)
            if cleaned:
                out.append(cleaned)
        return "\n".join(out)
