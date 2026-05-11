from __future__ import annotations

import re
from typing import Iterable, List


class OCRTextCleaner:
    def clean(self, text: str) -> str:
        text = text or ""
        text = text.replace("|", "I")
        text = text.replace("0f ", "of ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([A-Za-z])\1{2,}", r"\1\1", text)
        return text.strip()

    def merge_lines(self, lines: Iterable[str]) -> str:
        cleaned_lines: List[str] = []
        previous = ""
        for line in lines:
            stripped = self.clean(line)
            if not stripped:
                continue
            if previous and previous.endswith("-"):
                cleaned_lines[-1] = previous[:-1] + stripped
            else:
                cleaned_lines.append(stripped)
            previous = cleaned_lines[-1]
        return "\n".join(cleaned_lines)

    def normalize_for_matching(self, text: str) -> str:
        text = self.clean(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\b(?:[a-z]\s+){2,}[a-z]\b", lambda match: match.group(0).replace(" ", ""), text)
        return text.strip()
