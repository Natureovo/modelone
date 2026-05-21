from __future__ import annotations

from pathlib import Path

import numpy as np

from src.gradients import convolve2d
from src.image_io import normalize


def resize_block_map(block_map: np.ndarray, shape: tuple[int, int], block_size: int) -> np.ndarray:
    expanded = np.repeat(np.repeat(block_map, block_size, axis=0), block_size, axis=1)
    return expanded[: shape[0], : shape[1]].astype(np.float32)


def block_artifact_score(cmp_y: np.ndarray, block_size: int) -> np.ndarray:
    h, w = cmp_y.shape
    diff_x = np.zeros_like(cmp_y, dtype=np.float32)
    diff_y = np.zeros_like(cmp_y, dtype=np.float32)

    for x in range(block_size, w, block_size):
        diff_x[:, x] = np.abs(cmp_y[:, x] - cmp_y[:, x - 1])
    for y in range(block_size, h, block_size):
        diff_y[y, :] = np.abs(cmp_y[y, :] - cmp_y[y - 1, :])

    kernel = np.ones((5, 5), dtype=np.float32) / 25.0
    return normalize(convolve2d(diff_x + diff_y, kernel))


def detail_loss_score(
    grad: dict[str, np.ndarray],
    freq: dict[str, np.ndarray],
    block_size: int,
) -> dict[str, np.ndarray]:
    h, w = grad["mag_ref"].shape
    highfreq_loss = resize_block_map(freq["high_loss"], (h, w), block_size)
    variance_loss = resize_block_map(freq["variance_loss"], (h, w), block_size)
    block_artifact = block_artifact_score(grad["mag_cmp"], block_size)

    detail_loss = (
        0.35 * grad["gradient_loss"]
        + 0.40 * highfreq_loss
        + 0.15 * grad["direction_change"]
        + 0.10 * variance_loss
    )
    detail_loss = normalize(detail_loss, percentile=98.0)

    return {
        "gradient_loss": grad["gradient_loss"],
        "gradient_gain": grad["gradient_gain"],
        "direction_change": grad["direction_change"],
        "highfreq_loss": highfreq_loss,
        "variance_loss": variance_loss,
        "block_artifact": block_artifact,
        "detail_loss": detail_loss,
    }


def summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def block_mean_map(values: np.ndarray, block_size: int) -> np.ndarray:
    h, w = values.shape
    rows = int(np.ceil(h / block_size))
    cols = int(np.ceil(w / block_size))
    out = np.zeros((rows, cols), dtype=np.float32)
    for by in range(rows):
        for bx in range(cols):
            y0 = by * block_size
            x0 = bx * block_size
            patch = values[y0 : min(y0 + block_size, h), x0 : min(x0 + block_size, w)]
            out[by, bx] = float(patch.mean())
    return out


def classify_region(detail: float, highfreq: float, gradient: float, block_artifact: float, threshold: float) -> str:
    if detail < threshold:
        return "保持或轻度修复"
    if block_artifact >= 0.45 and highfreq < 0.45:
        return "优先去块效应"
    if highfreq >= 0.55 and gradient >= 0.35:
        return "高频生成候选"
    if gradient >= 0.55:
        return "结构保真修复"
    if highfreq >= 0.55:
        return "纹理生成候选"
    return "局部增强候选"


def region_records(
    block_size: int,
    threshold: float,
    grad: dict[str, np.ndarray],
    freq: dict[str, np.ndarray],
    score: dict[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    h, w = score["detail_loss"].shape
    rows = int(np.ceil(h / block_size))
    cols = int(np.ceil(w / block_size))
    gradient_loss_blocks = block_mean_map(score["gradient_loss"], block_size)
    direction_change_blocks = block_mean_map(score["direction_change"], block_size)
    block_artifact_blocks = block_mean_map(score["block_artifact"], block_size)
    detail_loss_blocks = block_mean_map(score["detail_loss"], block_size)

    records: list[dict[str, float | int | str]] = []
    for by in range(rows):
        for bx in range(cols):
            y0 = by * block_size
            x0 = bx * block_size
            height = min(block_size, h - y0)
            width = min(block_size, w - x0)
            high_ref = float(freq["high_ref_energy"][by, bx])
            high_cmp = float(freq["high_cmp_energy"][by, bx])
            mid_ref = float(freq["mid_ref_energy"][by, bx])
            mid_cmp = float(freq["mid_cmp_energy"][by, bx])
            high_loss_ratio = max(high_ref - high_cmp, 0.0) / (high_ref + 1e-8)
            mid_loss_ratio = max(mid_ref - mid_cmp, 0.0) / (mid_ref + 1e-8)
            detail = float(detail_loss_blocks[by, bx])
            highfreq = float(freq["high_loss"][by, bx])
            gradient = float(gradient_loss_blocks[by, bx])
            block_artifact = float(block_artifact_blocks[by, bx])
            records.append(
                {
                    "block_id": by * cols + bx,
                    "x": int(x0),
                    "y": int(y0),
                    "width": int(width),
                    "height": int(height),
                    "detail_loss_score": detail,
                    "gradient_loss_score": gradient,
                    "direction_change_score": float(direction_change_blocks[by, bx]),
                    "dct_highfreq_loss_score": highfreq,
                    "dct_highfreq_ref_energy": high_ref,
                    "dct_highfreq_cmp_energy": high_cmp,
                    "dct_highfreq_loss_ratio": float(high_loss_ratio),
                    "dct_midfreq_ref_energy": mid_ref,
                    "dct_midfreq_cmp_energy": mid_cmp,
                    "dct_midfreq_loss_ratio": float(mid_loss_ratio),
                    "block_artifact_score": block_artifact,
                    "is_candidate_region": int(detail >= threshold),
                    "region_decision": classify_region(detail, highfreq, gradient, block_artifact, threshold),
                }
            )
    records.sort(key=lambda item: float(item["detail_loss_score"]), reverse=True)
    return records


def top_blocks(detail_loss: np.ndarray, block_size: int, limit: int = 20) -> list[dict[str, float | int]]:
    h, w = detail_loss.shape
    rows = int(np.ceil(h / block_size))
    cols = int(np.ceil(w / block_size))
    blocks = []
    for by in range(rows):
        for bx in range(cols):
            y0 = by * block_size
            x0 = bx * block_size
            patch = detail_loss[y0 : min(y0 + block_size, h), x0 : min(x0 + block_size, w)]
            blocks.append(
                {
                    "x": int(x0),
                    "y": int(y0),
                    "width": int(patch.shape[1]),
                    "height": int(patch.shape[0]),
                    "detail_loss_mean": float(patch.mean()),
                    "detail_loss_max": float(patch.max()),
                }
            )
    blocks.sort(key=lambda item: item["detail_loss_mean"], reverse=True)
    return blocks[:limit]


def build_report(
    ref_path: Path,
    cmp_path: Path,
    block_size: int,
    threshold: float,
    grad: dict[str, np.ndarray],
    freq: dict[str, np.ndarray],
    score: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict:
    mag_ref_mean = float(grad["mag_ref"].mean())
    mag_cmp_mean = float(grad["mag_cmp"].mean())
    high_ref_mean = float(freq["high_ref_energy"].mean())
    high_cmp_mean = float(freq["high_cmp_energy"].mean())
    mid_ref_mean = float(freq["mid_ref_energy"].mean())
    mid_cmp_mean = float(freq["mid_cmp_energy"].mean())

    records = region_records(block_size, threshold, grad, freq, score)
    severe_regions = [item for item in records if int(item["is_candidate_region"]) == 1]
    highfreq_regions = [item for item in records if float(item["dct_highfreq_loss_score"]) >= 0.55]

    return {
        "inputs": {
            "reference": str(ref_path),
            "compressed": str(cmp_path),
            "block_size": block_size,
            "threshold": threshold,
        },
        "global": {
            "detail_loss_mean": float(score["detail_loss"].mean()),
            "gradient_loss_mean": float(score["gradient_loss"].mean()),
            "highfreq_loss_mean": float(score["highfreq_loss"].mean()),
            "block_artifact_mean": float(score["block_artifact"].mean()),
            "candidate_area_ratio": float(mask.mean()),
        },
        "regional": {
            "total_region_count": len(records),
            "severe_region_count": len(severe_regions),
            "severe_region_ratio": len(severe_regions) / max(len(records), 1),
            "highfreq_loss_region_count": len(highfreq_regions),
            "highfreq_loss_region_ratio": len(highfreq_regions) / max(len(records), 1),
            "top_10_regions": records[:10],
        },
        "numeric_changes": {
            "gradient": {
                "reference_mean": mag_ref_mean,
                "compressed_mean": mag_cmp_mean,
                "absolute_change": mag_cmp_mean - mag_ref_mean,
                "relative_change_ratio": (mag_cmp_mean - mag_ref_mean) / (mag_ref_mean + 1e-8),
                "relative_loss_ratio": max(mag_ref_mean - mag_cmp_mean, 0.0) / (mag_ref_mean + 1e-8),
            },
            "frequency_dct": {
                "high_frequency_reference_mean": high_ref_mean,
                "high_frequency_compressed_mean": high_cmp_mean,
                "high_frequency_absolute_change": high_cmp_mean - high_ref_mean,
                "high_frequency_relative_change_ratio": (high_cmp_mean - high_ref_mean) / (high_ref_mean + 1e-8),
                "high_frequency_relative_loss_ratio": max(high_ref_mean - high_cmp_mean, 0.0) / (high_ref_mean + 1e-8),
                "mid_frequency_reference_mean": mid_ref_mean,
                "mid_frequency_compressed_mean": mid_cmp_mean,
                "mid_frequency_absolute_change": mid_cmp_mean - mid_ref_mean,
                "mid_frequency_relative_change_ratio": (mid_cmp_mean - mid_ref_mean) / (mid_ref_mean + 1e-8),
                "mid_frequency_relative_loss_ratio": max(mid_ref_mean - mid_cmp_mean, 0.0) / (mid_ref_mean + 1e-8),
            },
        },
        "distributions": {
            "detail_loss": summary(score["detail_loss"]),
            "gradient_loss": summary(score["gradient_loss"]),
            "highfreq_loss": summary(score["highfreq_loss"]),
            "block_artifact": summary(score["block_artifact"]),
        },
        "top_loss_blocks": top_blocks(score["detail_loss"], block_size),
    }
