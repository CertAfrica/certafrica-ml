from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


class HeatmapGenerator:
    def build_overlay(self, image: np.ndarray, suspicious_regions: List[Dict[str, float]]) -> np.ndarray:
        base = image.copy()
        overlay = np.zeros_like(base)

        for region in suspicious_regions:
            x = int(region.get("left", 0))
            y = int(region.get("top", 0))
            width = int(region.get("width", 0))
            height = int(region.get("height", 0))
            severity = float(region.get("severity", 0.0))
            color = self._severity_color(severity)
            alpha = 0.18 + min(severity / 120.0, 0.45)
            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, thickness=-1)
            base = cv2.addWeighted(base, 1.0, overlay, alpha, 0)

        if suspicious_regions:
            return cv2.addWeighted(image, 0.65, base, 0.35, 0)
        return image.copy()

    def _severity_color(self, severity: float) -> Tuple[int, int, int]:
        if severity >= 75:
            return (255, 0, 0)
        if severity >= 40:
            return (255, 255, 0)
        return (0, 255, 0)
