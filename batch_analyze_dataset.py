from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.frequency import block_frequency_metrics
from src.gradients import gradient_metrics
from src.image_io import load_rgb, rgb_to_luma, save_gray, save_mask
from src.scoring import build_report, detail_loss_score, region_records
from src.visualize import overlay_heatmap, save_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch analyze frame pairs listed in dataset/frame_pairs.csv.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root directory.")
    parser.add_argument("--pairs", default="frame_pairs.csv", help="Frame-pair CSV under dataset root.")
    parser.add_argument("--out", default="outputs/批量视频帧测试", help="Output directory.")
    parser.add_argument("--block", type=int, default=32, help="Block size.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Candidate threshold.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of pairs to process.")
    return parser.parse_args()


def analyze_pair(ref_path: Path, cmp_path: Path, out_dir: Path, block: int, threshold: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_rgb = load_rgb(ref_path)
    cmp_rgb = load_rgb(cmp_path, size=(ref_rgb.shape[1], ref_rgb.shape[0]))
    ref_y = rgb_to_luma(ref_rgb)
    cmp_y = rgb_to_luma(cmp_rgb)

    grad = gradient_metrics(ref_y, cmp_y)
    freq = block_frequency_metrics(ref_y, cmp_y, block_size=block)
    score = detail_loss_score(grad, freq, block_size=block)
    mask = score["detail_loss"] >= threshold

    save_gray(out_dir / "原图_亮度通道.png", ref_y)
    save_gray(out_dir / "压缩图_亮度通道.png", cmp_y)
    save_heatmap(out_dir / "梯度损失热力图.png", score["gradient_loss"])
    save_heatmap(out_dir / "高频损失热力图.png", score["highfreq_loss"])
    save_heatmap(out_dir / "综合细节损失热力图.png", score["detail_loss"])
    overlay_heatmap(out_dir / "综合细节损失叠加图.png", ref_rgb, score["detail_loss"])
    save_mask(out_dir / "候选增强区域_mask.png", mask)

    report = build_report(ref_path, cmp_path, block, threshold, grad, freq, score, mask)
    (out_dir / "数值报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    records = region_records(block, threshold, grad, freq, score)
    with (out_dir / "区域损失明细.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    return report


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    pairs_path = dataset_root / args.pairs
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(pairs_path.open("r", encoding="utf-8-sig")))
    if args.limit is not None:
        rows = rows[: args.limit]

    summary_rows = []
    for row in rows:
        sample_id = row["sample_id"]
        ref_path = dataset_root / row["ref_frame"]
        cmp_path = dataset_root / row["compressed_frame"]
        if not ref_path.exists() or not cmp_path.exists():
            print(f"跳过 {sample_id}: 帧文件不存在")
            continue

        report = analyze_pair(ref_path, cmp_path, out_root / sample_id, args.block, args.threshold)
        summary_rows.append(
            {
                "sample_id": sample_id,
                "video_id": row.get("video_id", ""),
                "frame_idx": row.get("frame_idx", ""),
                "hm_qp": row.get("hm_qp", ""),
                "detail_loss_mean": report["global"]["detail_loss_mean"],
                "candidate_area_ratio": report["global"]["candidate_area_ratio"],
                "severe_region_count": report["regional"]["severe_region_count"],
                "highfreq_loss_region_count": report["regional"]["highfreq_loss_region_count"],
            }
        )
        print(f"完成 {sample_id}")

    if summary_rows:
        with (out_root / "批量汇总.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"批量分析完成: {out_root}")


if __name__ == "__main__":
    main()
