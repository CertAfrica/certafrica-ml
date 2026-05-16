from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

import fitz


def read_image_from_upload(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return render_pdf_first_page(file_path)
    with Image.open(file_path) as image:
        rgb_image = image.convert("RGB")
        return np.array(rgb_image)


def read_image(file_path: Path) -> np.ndarray:
    return read_image_from_upload(file_path)


def render_pdf_first_page(file_path: Path) -> np.ndarray:
    document = fitz.open(file_path)
    try:
        page = document[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        return np.array(image)
    finally:
        document.close()


def save_numpy_image(array: np.ndarray, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array.astype(np.uint8))
    image.save(target_path)


def save_image(array: np.ndarray, target_path: Path) -> None:
    save_numpy_image(array, target_path)
