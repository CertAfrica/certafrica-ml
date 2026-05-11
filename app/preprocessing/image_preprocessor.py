from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    def preprocess(self, image: np.ndarray) -> dict:
        rgb = self._ensure_rgb(image)
        resized = self._resize(rgb, max_dimension=1600)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        sharpened = cv2.GaussianBlur(normalized, (0, 0), 1.0)
        sharpened = cv2.addWeighted(normalized, 1.4, sharpened, -0.4, 0)
        return {
            "rgb": resized,
            "gray": gray,
            "denoised": denoised,
            "normalized": normalized,
            "enhanced": sharpened,
        }

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image.copy()

    def _resize(self, image: np.ndarray, max_dimension: int) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(max_dimension / max(height, width), 1.0)
        if scale >= 1.0:
            return image
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
