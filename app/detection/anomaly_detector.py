from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Tuple

import cv2
import numpy as np
import easyocr

from app.models.schemas import AnomalyResult, OCRBlock, OCRResult


@dataclass
class AnomalyEvidence:
    findings: List[str]
    suspicious_regions: List[Dict[str, float]]
    overall_score: float


class OCRExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def extract(self, image: np.ndarray, language: str = "eng") -> OCRResult:
        """Extract text using EasyOCR"""
        results = self.reader.readtext(image)
        
        # Build raw text
        raw_text = '\n'.join([text[1] for text in results])
        
        # Convert results to OCRBlock format
        blocks: List[OCRBlock] = []
        confidences: List[float] = []
        text_lines: List[str] = []
        
        for (bbox, text, confidence) in results:
            cleaned = str(text).strip()
            if not cleaned:
                continue
            
            # Extract bounding box coordinates
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            left = int(min(x_coords))
            top = int(min(y_coords))
            width = int(max(x_coords) - left)
            height = int(max(y_coords) - top)
            
            blocks.append(OCRBlock(
                text=cleaned,
                confidence=float(confidence),
                left=left,
                top=top,
                width=width,
                height=height
            ))
            text_lines.append(cleaned)
            confidences.append(float(confidence))
        
        extracted_fields = self._extract_fields(text_lines, raw_text)
        average_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        return OCRResult(
            raw_text=raw_text.strip(),
            extracted_fields=extracted_fields,
            blocks=blocks,
            average_confidence=round(average_confidence, 2),
            language=language,
        )
    
    def _extract_fields(self, text_lines: List[str], raw_text: str) -> Dict[str, str]:
        combined = "\n".join(text_lines + [raw_text])
        fields: Dict[str, str] = {}

        patterns = {
            "certificate_number": [
                r"(?:certificate|cert|reg(?:istration)?|cert\.?|no\.?)\s*(?:no|number|#|num)[:\s-]*([A-Z0-9\-\/]{4,})",
                r"(?:cert|ref)\s*[#:=]?\s*([A-Z0-9]{4,}[-/]?[A-Z0-9\-\/]+)",
                r"([A-Z0-9]{6,}[-/][A-Z0-9\-\/]+)",
                r"#\s*([A-Z0-9\-\/]+)",
            ],
            "issue_date": [
                r"(?:issued?\s*(?:on|date)?|date\s*(?:of\s*)?issue)[:\s-]*([A-Za-z0-9,\-\/ ]{5,})",
                r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            ],
            "expiry_date": [
                r"(?:expir(?:y|es?)|valid\s*(?:until|till|to)|expire[ds]?)[:\s-]*([A-Za-z0-9,\-\/ ]{5,})",
                r"valid.*?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            ],
            "name": [
                r"(?:name|candidate|holder|recipient|awarded\s*to)[:\s-]*([A-Z][A-Za-z'\-\. ]{3,})",
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            ],
            "institution": [
                r"(?:institution|school|college|university|organization|body|academy|institute)[:\s-]*([A-Za-z0-9&',\-\.\/ ]{4,})",
                r"(?:by|from)[:\s-]*([A-Za-z0-9&',\-\.\/ ]{4,})",
            ],
        }

        def safe_extract(match: re.Match) -> str:
            if match.lastindex and match.lastindex >= 1:
                return match.group(1).strip()
            return match.group(0).strip()

        for field, regexes in patterns.items():
            for regex in regexes:
                match = re.search(regex, combined, flags=re.IGNORECASE | re.MULTILINE)
                if match:
                    extracted = safe_extract(match)
                    if field == "institution":
                        fields[field] = extracted
                    else:
                        fields[field] = extracted
                    break

        if "name" not in fields:
            for line in text_lines:
                cleaned = line.strip()
                if len(cleaned) >= 8 and cleaned.isupper():
                    fields["name"] = cleaned
                    break
                if len(cleaned.split()) >= 2 and all(word[:1].isupper() for word in cleaned.split() if word):
                    fields["name"] = cleaned
                    break

        if "institution" not in fields:
            for i, line in enumerate(text_lines):
                lower = line.lower()
                if any(k in lower for k in ["university", "college", "school", "academy", "institute", "polytechnic"]):
                    fields["institution"] = line.strip()
                    break
                if i > 0:
                    prev = text_lines[i - 1].strip()
                    if prev and len(prev) > 3 and not prev.lower().startswith(("certificate", "name", "date")):
                        fields.setdefault("institution", f"{prev} {line.strip()}".strip())
                        break

        if any("signature" in line.lower() for line in text_lines):
            fields.setdefault("signature_present", "possible")

        return fields


class AnomalyDetector:
    def detect(self, rgb_image: np.ndarray, gray_image: np.ndarray, ocr_result: OCRResult) -> AnomalyResult:
        findings: List[str] = []
        suspicious_regions: List[Dict[str, float]] = []
        scores: List[float] = []

        ela_score, ela_regions = self._error_level_analysis(rgb_image)
        if ela_regions:
            findings.append("Compression inconsistency hotspots detected via ELA.")
            suspicious_regions.extend(ela_regions)
        scores.append(ela_score)

        text_score, text_regions, text_findings = self._ocr_consistency_checks(ocr_result)
        findings.extend(text_findings)
        suspicious_regions.extend(text_regions)
        scores.append(text_score)

        contour_score, contour_regions = self._contour_checks(gray_image)
        if contour_regions:
            findings.append("Irregular contour density found in one or more regions.")
            suspicious_regions.extend(contour_regions)
        scores.append(contour_score)

        duplicate_score, duplicate_regions = self._duplicate_region_checks(gray_image)
        if duplicate_regions:
            findings.append("Possible duplicated or copied regions detected.")
            suspicious_regions.extend(duplicate_regions)
        scores.append(duplicate_score)

        suspicion_score = float(np.clip(np.mean(scores) if scores else 0.0, 0, 100))
        suspicious_regions = self._deduplicate_regions(suspicious_regions)
        heatmap_notes = ["Green = low concern", "Yellow = moderate anomaly", "Red = high suspicion"]
        return AnomalyResult(
            overall_score=round(suspicion_score, 2),
            findings=findings or ["No major anomaly signals detected."],
            suspicious_regions=suspicious_regions,
            heatmap_notes=heatmap_notes,
        )

    def _error_level_analysis(self, rgb_image: np.ndarray) -> Tuple[float, List[Dict[str, float]]]:
        temp = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", temp, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            return 0.0, []
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(temp, decoded)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        threshold = max(int(np.mean(diff_gray) + np.std(diff_gray)), 12)
        mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)[1]
        regions = self._mask_to_regions(mask, severity_base=70, max_regions=8)
        score = float(np.clip(np.mean(diff_gray) / 3.0, 0, 100))
        return score, regions

    def _ocr_consistency_checks(self, ocr_result: OCRResult) -> Tuple[float, List[Dict[str, float]], List[str]]:
        if not ocr_result.blocks:
            return 45.0, [], ["OCR confidence is low or text extraction returned no blocks."]

        confidences = np.array([block.confidence for block in ocr_result.blocks if block.confidence >= 0], dtype=float)
        low_confidence_blocks = [block for block in ocr_result.blocks if block.confidence < 0.45]
        regions: List[Dict[str, float]] = []
        for block in low_confidence_blocks[:10]:
            regions.append(
                {
                    "left": block.left,
                    "top": block.top,
                    "width": block.width,
                    "height": block.height,
                    "severity": 55 + max(0.0, 45.0 - block.confidence * 100),
                }
            )

        score = float(np.clip(70 - (np.mean(confidences) if confidences.size else 0) * 55 - len(low_confidence_blocks) * 1.5, 0, 100))
        findings = []
        if low_confidence_blocks:
            findings.append(f"{len(low_confidence_blocks)} OCR blocks have low recognition confidence.")
        if not ocr_result.extracted_fields:
            findings.append("Structured fields could not be reliably extracted.")
            score = min(100.0, score + 8.0)
        return score, regions, findings

    def _contour_checks(self, gray_image: np.ndarray) -> Tuple[float, List[Dict[str, float]]]:
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray_image.shape[:2]
        regions: List[Dict[str, float]] = []
        small_irregular = 0
        for contour in contours[:200]:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area < 200:
                continue
            aspect = max(width / max(height, 1), height / max(width, 1))
            if area < (h * w) * 0.012 and aspect > 4:
                small_irregular += 1
                regions.append({"left": x, "top": y, "width": width, "height": height, "severity": 50 + min(35, aspect * 6)})
        score = float(np.clip(small_irregular * 12 + np.mean(edges) / 4.0, 0, 100))
        return score, regions

    def _duplicate_region_checks(self, gray_image: np.ndarray) -> Tuple[float, List[Dict[str, float]]]:
        resized = cv2.resize(gray_image, (256, 256), interpolation=cv2.INTER_AREA)
        patch_size = 32
        patches = []
        positions = []
        for top in range(0, 256, patch_size):
            for left in range(0, 256, patch_size):
                patch = resized[top : top + patch_size, left : left + patch_size]
                if patch.size == 0:
                    continue
                vector = patch.astype(np.float32).flatten()
                vector = vector / (np.linalg.norm(vector) + 1e-6)
                patches.append(vector)
                positions.append((left, top))

        suspicious_pairs: List[Tuple[int, int]] = []
        for index, vector in enumerate(patches):
            for other_index in range(index + 1, len(patches)):
                left_a, top_a = positions[index]
                left_b, top_b = positions[other_index]
                if abs(left_a - left_b) < patch_size * 2 and abs(top_a - top_b) < patch_size * 2:
                    continue
                similarity = float(np.dot(vector, patches[other_index]))
                if similarity > 0.985:
                    suspicious_pairs.append((index, other_index))

        regions: List[Dict[str, float]] = []
        for index, other_index in suspicious_pairs[:6]:
            left_a, top_a = positions[index]
            left_b, top_b = positions[other_index]
            regions.append({"left": int(left_a * 2), "top": int(top_a * 2), "width": patch_size * 2, "height": patch_size * 2, "severity": 72})
            regions.append({"left": int(left_b * 2), "top": int(top_b * 2), "width": patch_size * 2, "height": patch_size * 2, "severity": 72})

        score = float(min(100, len(suspicious_pairs) * 14))
        return score, regions

    def _mask_to_regions(self, mask: np.ndarray, severity_base: float, max_regions: int) -> List[Dict[str, float]]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions: List[Dict[str, float]] = []
        for label in range(1, num_labels):
            x, y, width, height, area = stats[label]
            if area < 80:
                continue
            severity = severity_base + min(30, area / 20)
            regions.append({"left": float(x), "top": float(y), "width": float(width), "height": float(height), "severity": float(severity)})
        regions.sort(key=lambda item: item["severity"], reverse=True)
        return regions[:max_regions]

    def _deduplicate_regions(self, regions: List[Dict[str, float]]) -> List[Dict[str, float]]:
        unique = []
        seen = set()
        for region in regions:
            key = (
                int(region.get("left", 0) // 8),
                int(region.get("top", 0) // 8),
                int(region.get("width", 0) // 8),
                int(region.get("height", 0) // 8),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(region)
        return unique[:24]
