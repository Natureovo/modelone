from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def to_uint8(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    return (values * 255.0 + 0.5).astype(np.uint8)


def normalize(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    hi = float(np.percentile(values, percentile))
    if hi <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / hi, 0.0, 1.0).astype(np.float32)


def save_gray(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(normalize(values))).save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
