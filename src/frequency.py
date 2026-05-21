from __future__ import annotations

import math

import numpy as np

from src.image_io import normalize


def dct_matrix(n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float32)
    factor = math.pi / (2.0 * n)
    scale0 = math.sqrt(1.0 / n)
    scale = math.sqrt(2.0 / n)
    for k in range(n):
        alpha = scale0 if k == 0 else scale
        for i in range(n):
            matrix[k, i] = alpha * math.cos((2 * i + 1) * k * factor)
    return matrix


def pad_to_blocks(image: np.ndarray, block_size: int) -> tuple[np.ndarray, int, int]:
    h, w = image.shape
    padded_h = int(math.ceil(h / block_size) * block_size)
    padded_w = int(math.ceil(w / block_size) * block_size)
    padded = np.pad(image, ((0, padded_h - h), (0, padded_w - w)), mode="edge")
    return padded, h, w


def band_masks(block_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:block_size, 0:block_size]
    radius = (xx + yy) / (2.0 * (block_size - 1))
    low = radius <= 0.18
    mid = (radius > 0.18) & (radius <= 0.50)
    high = radius > 0.50
    low[0, 0] = False
    return low, mid, high


def block_frequency_metrics(ref_y: np.ndarray, cmp_y: np.ndarray, block_size: int = 32) -> dict[str, np.ndarray]:
    if block_size < 8:
        raise ValueError("block_size should be at least 8.")

    ref_pad, h, w = pad_to_blocks(ref_y, block_size)
    cmp_pad, _, _ = pad_to_blocks(cmp_y, block_size)
    rows = ref_pad.shape[0] // block_size
    cols = ref_pad.shape[1] // block_size
    transform = dct_matrix(block_size)
    low_mask, mid_mask, high_mask = band_masks(block_size)

    high_loss = np.zeros((rows, cols), dtype=np.float32)
    high_gain = np.zeros((rows, cols), dtype=np.float32)
    mid_loss = np.zeros((rows, cols), dtype=np.float32)
    variance_loss = np.zeros((rows, cols), dtype=np.float32)
    high_ratio = np.zeros((rows, cols), dtype=np.float32)
    high_ref_energy = np.zeros((rows, cols), dtype=np.float32)
    high_cmp_energy = np.zeros((rows, cols), dtype=np.float32)
    mid_ref_energy = np.zeros((rows, cols), dtype=np.float32)
    mid_cmp_energy = np.zeros((rows, cols), dtype=np.float32)

    for by in range(rows):
        for bx in range(cols):
            y0 = by * block_size
            x0 = bx * block_size
            ref_block = ref_pad[y0 : y0 + block_size, x0 : x0 + block_size]
            cmp_block = cmp_pad[y0 : y0 + block_size, x0 : x0 + block_size]

            ref_centered = ref_block - float(ref_block.mean())
            cmp_centered = cmp_block - float(cmp_block.mean())
            d_ref = transform @ ref_centered @ transform.T
            d_cmp = transform @ cmp_centered @ transform.T
            e_ref = d_ref * d_ref
            e_cmp = d_cmp * d_cmp

            ref_high = float(e_ref[high_mask].sum())
            cmp_high = float(e_cmp[high_mask].sum())
            ref_mid = float(e_ref[mid_mask].sum())
            cmp_mid = float(e_cmp[mid_mask].sum())

            high_loss[by, bx] = max(ref_high - cmp_high, 0.0)
            high_gain[by, bx] = max(cmp_high - ref_high, 0.0)
            mid_loss[by, bx] = max(ref_mid - cmp_mid, 0.0)
            high_ratio[by, bx] = cmp_high / (ref_high + 1e-8)
            variance_loss[by, bx] = max(float(ref_block.var() - cmp_block.var()), 0.0)
            high_ref_energy[by, bx] = ref_high
            high_cmp_energy[by, bx] = cmp_high
            mid_ref_energy[by, bx] = ref_mid
            mid_cmp_energy[by, bx] = cmp_mid

    return {
        "high_loss": normalize(high_loss),
        "high_gain": normalize(high_gain),
        "mid_loss": normalize(mid_loss),
        "variance_loss": normalize(variance_loss),
        "high_ratio": np.clip(high_ratio, 0.0, 2.0),
        "high_ref_energy": high_ref_energy,
        "high_cmp_energy": high_cmp_energy,
        "mid_ref_energy": mid_ref_energy,
        "mid_cmp_energy": mid_cmp_energy,
        "shape": np.array([h, w], dtype=np.int32),
        "block_size": np.array([block_size], dtype=np.int32),
    }
