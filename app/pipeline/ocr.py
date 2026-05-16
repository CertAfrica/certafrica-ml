"""
OCR extraction. Lifted out of the old detection/anomaly_detector.py where
it lived as a separate class but shared a module with anomaly detection
for no good reason.

The OCRExtractor was already fine, just relocated and lightly cleaned.
The cleaned_text field is now populated here so the classifier does not
re-clean it downstream.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import easyocr
import numpy as np

from app.schemas import OCRBlock, OCRResult
from app.utils.text import OCRTextCleaner


class OCRExtractor:
    def __init__(self) -> None:
        self._reader: Optional[easyocr.Reader] = None
        self._cleaner = OCRTextCleaner()

    def _get_reader(self) -> easyocr.Reader:
        if self._reader is None:
            self._reader = easyocr.Reader(["en"], gpu=False)
        return self._reader

    def extract(self, image: np.ndarray) -> OCRResult:
        raw_results = self._get_reader().readtext(image)

        blocks: List[OCRBlock] = []
        confidences: List[float] = []
        text_lines: List[str] = []

        for bbox, text, confidence in raw_results:
            cleaned = str(text).strip()
            if not cleaned:
                continue
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            blocks.append(OCRBlock(
                text=cleaned,
                confidence=float(confidence),
                left=int(min(x_coords)),
                top=int(min(y_coords)),
                width=int(max(x_coords) - min(x_coords)),
                height=int(max(y_coords) - min(y_coords)),
            ))
            text_lines.append(cleaned)
            confidences.append(float(confidence))

        raw_text = "\n".join(text_lines)
        cleaned_text = self._cleaner.clean(raw_text)
        fields = self._extract_fields(text_lines, raw_text)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return OCRResult(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            extracted_fields=fields,
            blocks=blocks,
            average_confidence=round(avg_conf, 2),
            language="en",
        )

    def _extract_fields(self, lines: List[str], raw_text: str) -> Dict[str, str]:
        combined = "\n".join(lines + [raw_text])
        fields: Dict[str, str] = {}

        patterns = {
            "certificate_number": [
                r"(?:certificate|cert|reg(?:istration)?)\s*(?:no|number|#)[:\s-]*([A-Z0-9\-\/ ]{4,})",
                r"#\s*([A-Z0-9\-\/]+)",
                r"\b([A-Z]{2,4}\d{6,10})\b",
            ],
            "issue_date": [
                r"(?:issued?|date of issue)[:\s-]*([A-Za-z0-9,\-\/ ]{5,})",
                r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
            ],
            "name": [
                r"(?:name|candidate|holder|awarded to)[:\s-]*([A-Z][A-Za-z'\-. ]{3,})",
            ],
            "institution": [
                r"(?:institution|school|university|college)[:\s-]*([A-Za-z0-9&',\-./ ]{4,})",
            ],
        }

        for field, regexes in patterns.items():
            for regex in regexes:
                match = re.search(regex, combined, flags=re.IGNORECASE | re.MULTILINE)
                if match:
                    fields[field] = match.group(1).strip() if match.lastindex else match.group(0).strip()
                    break

        if "institution" not in fields:
            for line in lines:
                lower = line.lower()
                if any(k in lower for k in ["university", "college", "polytechnic", "institute"]):
                    fields["institution"] = line.strip()
                    break

        return fields
