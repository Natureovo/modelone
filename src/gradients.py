from __future__ import annotations

import numpy as np

from src.image_io import normalize


SOBEL_X = np.array(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
SOBEL_Y = np.array(
    [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ],
    dtype=np.float32,
)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            out += kernel[y, x] * padded[y : y + image.shape[0], x : x + image.shape[1]]
    return out


def sobel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = convolve2d(image, SOBEL_X)
    gy = convolve2d(image, SOBEL_Y)
    magnitude = np.sqrt(gx * gx + gy * gy)
    angle = np.arctan2(gy, gx)
    return gx, gy, magnitude, angle


def gradient_metrics(ref_y: np.ndarray, cmp_y: np.ndarray) -> dict[str, np.ndarray]:
    gx_ref, gy_ref, mag_ref, angle_ref = sobel(ref_y)
    gx_cmp, gy_cmp, mag_cmp, angle_cmp = sobel(cmp_y)

    gradient_loss = np.maximum(mag_ref - mag_cmp, 0.0)
    gradient_gain = np.maximum(mag_cmp - mag_ref, 0.0)
    eps = 1e-6
    direction_agreement = (gx_ref * gx_cmp + gy_ref * gy_cmp) / ((mag_ref * mag_cmp) + eps)
    direction_agreement = np.clip(direction_agreement, -1.0, 1.0)
    direction_change = 1.0 - ((direction_agreement + 1.0) * 0.5)

    return {
        "gx_ref": gx_ref,
        "gy_ref": gy_ref,
        "mag_ref": mag_ref,
        "angle_ref": angle_ref,
        "gx_cmp": gx_cmp,
        "gy_cmp": gy_cmp,
        "mag_cmp": mag_cmp,
        "angle_cmp": angle_cmp,
        "gradient_loss": normalize(gradient_loss),
        "gradient_gain": normalize(gradient_gain),
        "direction_change": normalize(direction_change),
    }
