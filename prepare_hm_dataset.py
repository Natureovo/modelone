from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".y4m"}
YUV_EXTS = {".yuv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frame-pair dataset from raw videos and HM recon output.")
    parser.add_argument("--dataset", default="dataset", help="Dataset root.")
    parser.add_argument("--hm", default=None, help="Path to HM TAppEncoderStatic.exe. If omitted, only commands are generated.")
    parser.add_argument("--hm-config", default=None, help="Path to HM encoder config, e.g. encoder_randomaccess_main.cfg.")
    parser.add_argument("--ffmpeg", default=None, help="Path to ffmpeg.exe. Defaults to PATH lookup.")
    parser.add_argument("--ffprobe", default=None, help="Path to ffprobe.exe. Defaults to PATH lookup.")
    parser.add_argument("--qps", default="22,27,32,37,42", help="Comma-separated QP list.")
    parser.add_argument("--frames", type=int, default=120, help="Frames to encode/extract per video.")
    parser.add_argument("--frame-step", type=int, default=15, help="Extract one PNG every N frames.")
    parser.add_argument("--width", type=int, default=None, help="Force width for raw YUV input.")
    parser.add_argument("--height", type=int, default=None, help="Force height for raw YUV input.")
    parser.add_argument("--fps", type=float, default=None, help="Force fps for raw YUV input.")
    parser.add_argument("--run-hm", action="store_true", help="Actually run HM. Otherwise writes commands only.")
    return parser.parse_args()


def tool_path(value: str | None, name: str) -> str:
    if value:
        return value
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"找不到 {name}，请用 --{name} 指定路径。")


def run(cmd: list[str]) -> None:
    print("运行:", " ".join(f'"{part}"' if " " in part else part for part in cmd))
    subprocess.run(cmd, check=True)


def probe_video(path: Path, ffprobe: str) -> dict[str, float | int]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True, encoding="utf-8"))
    stream = data["streams"][0]
    num, den = stream.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / max(float(den), 1.0)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": int(stream.get("nb_frames") or 0),
    }


def convert_to_yuv(src: Path, dst: Path, ffmpeg: str, frames: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-frames:v",
        str(frames),
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    run(cmd)


def extract_from_video(src: Path, dst_dir: Path, ffmpeg: str, frame_step: int, frames: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    select_expr = f"select='lt(n,{frames})*not(mod(n,{frame_step}))'"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-vf",
        select_expr,
        "-vsync",
        "vfr",
        str(dst_dir / "%06d.png"),
    ]
    run(cmd)


def extract_from_yuv(
    src: Path,
    dst_dir: Path,
    ffmpeg: str,
    width: int,
    height: int,
    fps: float,
    frame_step: int,
    frames: int,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    select_expr = f"select='lt(n,{frames})*not(mod(n,{frame_step}))'"
    cmd = [
        ffmpeg,
        "-y",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-i",
        str(src),
        "-vf",
        select_expr,
        "-vsync",
        "vfr",
        str(dst_dir / "%06d.png"),
    ]
    run(cmd)


def hm_command(
    hm: str,
    config: str,
    input_yuv: Path,
    bitstream: Path,
    recon_yuv: Path,
    width: int,
    height: int,
    fps: float,
    frames: int,
    qp: int,
) -> list[str]:
    return [
        hm,
        "-c",
        config,
        "-i",
        str(input_yuv),
        "-b",
        str(bitstream),
        "-o",
        str(recon_yuv),
        "-wdt",
        str(width),
        "-hgt",
        str(height),
        "-fr",
        str(round(fps)),
        "-f",
        str(frames),
        "-q",
        str(qp),
    ]


def write_csv(path: Path, rows: list[dict[str, str | int | float]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    raw_dir = dataset / "raw_videos"
    yuv_dir = dataset / "hm_results" / "input_yuv"
    bitstream_dir = dataset / "hm_results" / "bitstreams"
    recon_dir = dataset / "hm_results" / "recon_yuv"
    log_dir = dataset / "hm_results" / "logs"
    commands_path = dataset / "hm_results" / "hm_commands.txt"

    ffmpeg = tool_path(args.ffmpeg, "ffmpeg")
    ffprobe = tool_path(args.ffprobe, "ffprobe")
    hm = args.hm
    qps = [int(item.strip()) for item in args.qps.split(",") if item.strip()]

    if args.run_hm and (not hm or not args.hm_config):
        raise SystemExit("使用 --run-hm 时必须提供 --hm 和 --hm-config。")

    videos = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS | YUV_EXTS]
    if not videos:
        raise SystemExit(f"没有在 {raw_dir} 找到视频或 YUV 文件。")

    metadata_rows: list[dict[str, str | int | float]] = []
    pair_rows: list[dict[str, str | int]] = []
    hm_commands: list[str] = []

    for src in videos:
        video_id = src.stem
        if src.suffix.lower() in YUV_EXTS:
            if not args.width or not args.height or not args.fps:
                raise SystemExit("输入是 .yuv 时，请提供 --width --height --fps。")
            width, height, fps = args.width, args.height, args.fps
            input_yuv = src
            raw_extract_source = src
        else:
            info = probe_video(src, ffprobe)
            width, height, fps = int(info["width"]), int(info["height"]), float(info["fps"])
            input_yuv = yuv_dir / f"{video_id}_{width}x{height}_{round(fps)}fps.yuv"
            convert_to_yuv(src, input_yuv, ffmpeg, args.frames)
            raw_extract_source = src

        raw_frame_dir = dataset / "frame_pairs" / video_id / "原始帧"
        if src.suffix.lower() in YUV_EXTS:
            extract_from_yuv(input_yuv, raw_frame_dir, ffmpeg, width, height, fps, args.frame_step, args.frames)
        else:
            extract_from_video(raw_extract_source, raw_frame_dir, ffmpeg, args.frame_step, args.frames)

        for qp in qps:
            bitstream = bitstream_dir / f"{video_id}_QP{qp}.bin"
            recon_yuv = recon_dir / f"{video_id}_QP{qp}_rec.yuv"
            log_path = log_dir / f"{video_id}_QP{qp}.log"
            bitstream.parent.mkdir(parents=True, exist_ok=True)
            recon_yuv.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            if hm and args.hm_config:
                cmd = hm_command(hm, args.hm_config, input_yuv, bitstream, recon_yuv, width, height, fps, args.frames, qp)
                hm_commands.append(" ".join(f'"{part}"' if " " in part else part for part in cmd) + f" > \"{log_path}\" 2>&1")
                if args.run_hm:
                    with log_path.open("w", encoding="utf-8") as log:
                        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)

            if recon_yuv.exists():
                cmp_frame_dir = dataset / "frame_pairs" / video_id / f"HM_QP{qp}_压缩帧"
                extract_from_yuv(recon_yuv, cmp_frame_dir, ffmpeg, width, height, fps, args.frame_step, args.frames)
                ref_frames = sorted(raw_frame_dir.glob("*.png"))
                cmp_frames = sorted(cmp_frame_dir.glob("*.png"))
                for index, (ref_frame, cmp_frame) in enumerate(zip(ref_frames, cmp_frames), start=1):
                    sample_id = f"{video_id}_QP{qp}_{index:06d}"
                    pair_rows.append(
                        {
                            "sample_id": sample_id,
                            "video_id": video_id,
                            "frame_idx": index,
                            "category": "unknown",
                            "hm_qp": qp,
                            "ref_frame": ref_frame.relative_to(dataset).as_posix(),
                            "compressed_frame": cmp_frame.relative_to(dataset).as_posix(),
                        }
                    )

            metadata_rows.append(
                {
                    "video_id": video_id,
                    "category": "unknown",
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": args.frames,
                    "format": "yuv420p",
                    "hm_qp": qp,
                    "raw_video_path": src.relative_to(dataset).as_posix() if src.is_relative_to(dataset) else str(src),
                    "hm_bitstream_path": bitstream.relative_to(dataset).as_posix(),
                    "hm_recon_yuv_path": recon_yuv.relative_to(dataset).as_posix(),
                    "hm_log_path": log_path.relative_to(dataset).as_posix(),
                    "notes": "",
                }
            )

    if hm_commands:
        commands_path.parent.mkdir(parents=True, exist_ok=True)
        commands_path.write_text("\n".join(hm_commands) + "\n", encoding="utf-8")

    write_csv(
        dataset / "metadata.csv",
        metadata_rows,
        [
            "video_id",
            "category",
            "width",
            "height",
            "fps",
            "frame_count",
            "format",
            "hm_qp",
            "raw_video_path",
            "hm_bitstream_path",
            "hm_recon_yuv_path",
            "hm_log_path",
            "notes",
        ],
    )
    write_csv(
        dataset / "frame_pairs.csv",
        pair_rows,
        ["sample_id", "video_id", "frame_idx", "category", "hm_qp", "ref_frame", "compressed_frame"],
    )

    print("HM 数据集准备完成。")
    print(f"视频数量: {len(videos)}")
    print(f"QP: {qps}")
    print(f"帧对数量: {len(pair_rows)}")
    if hm_commands and not args.run_hm:
        print(f"HM 命令已写入: {commands_path}")


if __name__ == "__main__":
    main()
