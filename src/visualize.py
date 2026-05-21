from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.image_io import normalize, to_uint8


def turbo_colormap(values: np.ndarray) -> np.ndarray:
    x = normalize(values)
    stops = np.array(
        [
            [0.18995, 0.07176, 0.23217],
            [0.25107, 0.25237, 0.63374],
            [0.27628, 0.48753, 0.96507],
            [0.15844, 0.73551, 0.92305],
            [0.18995, 0.83966, 0.54029],
            [0.53695, 0.84977, 0.18804],
            [0.97323, 0.74682, 0.11670],
            [0.95801, 0.41020, 0.08024],
            [0.47960, 0.01583, 0.01055],
        ],
        dtype=np.float32,
    )
    scaled = x * (len(stops) - 1)
    idx = np.floor(scaled).astype(np.int32)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = scaled[..., None] - idx[..., None]
    rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    return np.clip(rgb, 0.0, 1.0)


def save_heatmap(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(turbo_colormap(values))).save(path)


def overlay_heatmap(path: Path, rgb: np.ndarray, values: np.ndarray, alpha: float = 0.45) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heat = turbo_colormap(values)
    mixed = np.clip((1.0 - alpha) * rgb + alpha * heat, 0.0, 1.0)
    Image.fromarray(to_uint8(mixed)).save(path)
