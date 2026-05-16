"""
Anomaly detection. Same core algorithms as before (ELA, contour checks,
duplicate region checks, OCR confidence) but emits typed SuspiciousRegion
with placeholder category/area. The pipeline reassigns those based on
image dimensions and OCR text.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.schemas import AnomalyResult, OCRResult, SuspiciousRegion


class AnomalyDetector:
    def detect(self, rgb: np.ndarray, gray: np.ndarray,
               ocr: OCRResult) -> AnomalyResult:
        findings: List[str] = []
        regions: List[SuspiciousRegion] = []
        scores: List[float] = []

        ela_score, ela_regions = self._error_level_analysis(rgb)
        regions.extend(ela_regions)
        if ela_regions:
            findings.append("Compression inconsistency hotspots detected.")
        scores.append(ela_score)

        text_score, text_regions, text_findings = self._ocr_consistency(ocr)
        regions.extend(text_regions)
        findings.extend(text_findings)
        scores.append(text_score)

        contour_score, contour_regions = self._contour_checks(gray)
        regions.extend(contour_regions)
        if contour_regions:
            findings.append("Irregular contour density detected.")
        scores.append(contour_score)

        duplicate_score, duplicate_regions = self._duplicate_regions(gray)
        regions.extend(duplicate_regions)
        if duplicate_regions:
            findings.append("Possible duplicated or copied regions detected.")
        scores.append(duplicate_score)

        regions = self._deduplicate(regions)
        overall = float(np.clip(np.mean(scores) if scores else 0.0, 0, 100))

        return AnomalyResult(
            overall_score=round(overall, 2),
            suspicious_regions=regions,
            findings=findings or ["No major anomaly signals detected."],
        )

    def _error_level_analysis(self, rgb: np.ndarray) -> Tuple[float, List[SuspiciousRegion]]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            return 0.0, []
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(bgr, decoded)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
        threshold = max(int(np.mean(diff_gray) + np.std(diff_gray)), 12)
        mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)[1]
        regions = self._mask_to_regions(mask, severity_base=70, limit=8)
        score = float(np.clip(np.mean(diff_gray) / 3.0, 0, 100))
        return score, regions

    def _ocr_consistency(self, ocr: OCRResult) -> Tuple[float, List[SuspiciousRegion], List[str]]:
        if not ocr.blocks:
            return 45.0, [], ["OCR returned no text blocks."]

        confidences = np.array([b.confidence for b in ocr.blocks], dtype=float)
        low_conf_blocks = [b for b in ocr.blocks if b.confidence < 0.45]
        regions: List[SuspiciousRegion] = []
        for block in low_conf_blocks[:10]:
            regions.append(SuspiciousRegion(
                left=block.left, top=block.top,
                width=block.width, height=block.height,
                severity=55 + max(0.0, 45.0 - block.confidence * 100),
                category="pending", area="pending",
            ))

        score = float(np.clip(
            70 - (np.mean(confidences) if confidences.size else 0) * 55
            - len(low_conf_blocks) * 1.5, 0, 100,
        ))
        findings: List[str] = []
        if low_conf_blocks:
            findings.append(f"{len(low_conf_blocks)} low-confidence OCR blocks.")
        if not ocr.extracted_fields:
            findings.append("No structured fields extracted.")
        return score, regions, findings

    def _contour_checks(self, gray: np.ndarray) -> Tuple[float, List[SuspiciousRegion]]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        regions: List[SuspiciousRegion] = []
        small_irregular = 0

        for contour in contours[:200]:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area < 200:
                continue
            aspect = max(width / max(height, 1), height / max(width, 1))
            if area < (h * w) * 0.012 and aspect > 4:
                small_irregular += 1
                regions.append(SuspiciousRegion(
                    left=x, top=y, width=width, height=height,
                    severity=min(85.0, 50 + aspect * 6),
                    category="pending", area="pending",
                ))

        score = float(np.clip(small_irregular * 12 + np.mean(edges) / 4.0, 0, 100))
        return score, regions

    def _duplicate_regions(self, gray: np.ndarray) -> Tuple[float, List[SuspiciousRegion]]:
        resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        patch = 32
        patches = []
        positions = []
        for top in range(0, 256, patch):
            for left in range(0, 256, patch):
                p = resized[top:top + patch, left:left + patch]
                if p.size == 0:
                    continue
                v = p.astype(np.float32).flatten()
                v = v / (np.linalg.norm(v) + 1e-6)
                patches.append(v)
                positions.append((left, top))

        suspicious_pairs = []
        for i, vec in enumerate(patches):
            for j in range(i + 1, len(patches)):
                la, ta = positions[i]
                lb, tb = positions[j]
                if abs(la - lb) < patch * 2 and abs(ta - tb) < patch * 2:
                    continue
                if float(np.dot(vec, patches[j])) > 0.985:
                    suspicious_pairs.append((i, j))

        regions: List[SuspiciousRegion] = []
        for i, j in suspicious_pairs[:6]:
            la, ta = positions[i]
            lb, tb = positions[j]
            regions.append(SuspiciousRegion(
                left=int(la * 2), top=int(ta * 2),
                width=patch * 2, height=patch * 2,
                severity=72.0, category="pending", area="pending",
            ))
            regions.append(SuspiciousRegion(
                left=int(lb * 2), top=int(tb * 2),
                width=patch * 2, height=patch * 2,
                severity=72.0, category="pending", area="pending",
            ))

        score = float(min(100, len(suspicious_pairs) * 14))
        return score, regions

    def _mask_to_regions(self, mask: np.ndarray, severity_base: float,
                         limit: int) -> List[SuspiciousRegion]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions: List[SuspiciousRegion] = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < 80:
                continue
            regions.append(SuspiciousRegion(
                left=int(x), top=int(y), width=int(w), height=int(h),
                severity=float(severity_base + min(30, area / 20)),
                category="pending", area="pending",
            ))
        regions.sort(key=lambda r: r.severity, reverse=True)
        return regions[:limit]

    def _deduplicate(self, regions: List[SuspiciousRegion]) -> List[SuspiciousRegion]:
        seen = set()
        unique: List[SuspiciousRegion] = []
        for r in regions:
            key = (r.left // 8, r.top // 8, r.width // 8, r.height // 8)
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        return unique[:24]
