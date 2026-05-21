from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.frequency import block_frequency_metrics
from src.gradients import gradient_metrics
from src.image_io import load_rgb, rgb_to_luma, save_gray, save_mask
from src.scoring import build_report, detail_loss_score, region_records
from src.visualize import overlay_heatmap, save_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gradient and frequency degradation between original and compressed images."
    )
    parser.add_argument("--ref", required=True, help="Path to the original/reference image.")
    parser.add_argument("--cmp", required=True, help="Path to the compressed image.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument("--block", type=int, default=32, help="Block size for frequency/statistical analysis.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Candidate mask threshold in [0, 1].")
    parser.add_argument("--case-name", default=None, help="Optional output subfolder name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref_path = Path(args.ref)
    cmp_path = Path(args.cmp)
    case_name = args.case_name or f"{ref_path.stem}_vs_{cmp_path.stem}"
    out_dir = Path(args.out) / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_rgb = load_rgb(ref_path)
    cmp_rgb = load_rgb(cmp_path, size=(ref_rgb.shape[1], ref_rgb.shape[0]))

    ref_y = rgb_to_luma(ref_rgb)
    cmp_y = rgb_to_luma(cmp_rgb)

    save_gray(out_dir / "原图_亮度通道.png", ref_y)
    save_gray(out_dir / "压缩图_亮度通道.png", cmp_y)

    grad = gradient_metrics(ref_y, cmp_y)
    freq = block_frequency_metrics(ref_y, cmp_y, block_size=args.block)
    score = detail_loss_score(grad, freq, block_size=args.block)

    save_gray(out_dir / "原图_梯度幅值.png", grad["mag_ref"])
    save_gray(out_dir / "压缩图_梯度幅值.png", grad["mag_cmp"])
    save_heatmap(out_dir / "梯度损失热力图.png", score["gradient_loss"])
    save_heatmap(out_dir / "高频损失热力图.png", score["highfreq_loss"])
    save_heatmap(out_dir / "块效应热力图.png", score["block_artifact"])
    save_heatmap(out_dir / "综合细节损失热力图.png", score["detail_loss"])
    overlay_heatmap(out_dir / "综合细节损失叠加图.png", ref_rgb, score["detail_loss"])

    mask = score["detail_loss"] >= args.threshold
    save_mask(out_dir / "候选增强区域_mask.png", mask)

    report = build_report(
        ref_path=ref_path,
        cmp_path=cmp_path,
        block_size=args.block,
        threshold=args.threshold,
        grad=grad,
        freq=freq,
        score=score,
        mask=mask,
    )
    (out_dir / "数值报告.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    records = region_records(
        block_size=args.block,
        threshold=args.threshold,
        grad=grad,
        freq=freq,
        score=score,
    )
    with (out_dir / "区域损失明细.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"分析完成: {out_dir}")
    print(f"平均细节损失: {report['global']['detail_loss_mean']:.4f}")
    print(f"候选增强区域占比: {report['global']['candidate_area_ratio']:.4f}")
    print(f"严重区域数量: {report['regional']['severe_region_count']}")


if __name__ == "__main__":
    main()
