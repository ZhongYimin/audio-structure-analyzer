"""批量处理歌曲库：把整个库全部 GPU 跑成驱动文件。

用法：
    python examples/build_drive_library.py <音频目录> <输出目录> [--device cuda] [--ext .wav,.mp3]

示例：
    python examples/build_drive_library.py music/ drives/ --device cuda

输出：每个音频生成一个同名 .drive.json 驱动文件。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="批量生成驱动文件（空间换时间）")
    ap.add_argument("input_dir", help="音频目录")
    ap.add_argument("output_dir", help="驱动文件输出目录")
    ap.add_argument("--device", default="cuda", help="推理设备，默认 cuda")
    ap.add_argument("--ext", default=".wav,.mp3,.flac,.m4a,.ogg", help="音频扩展名，逗号分隔")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {e.strip().lower() for e in args.ext.split(",")}
    audio_files = sorted(
        f for f in in_dir.rglob("*") if f.suffix.lower() in exts
    )
    if not audio_files:
        print(f"[drive] 未在 {in_dir} 找到音频文件（扩展名 {args.ext}）")
        sys.exit(1)

    print(f"[drive] 待处理 {len(audio_files)} 首，设备={args.device}")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from audio_structure_analyzer import analyze_and_save_drive

    ok = fail = 0
    t_start = time.time()
    for i, af in enumerate(audio_files, 1):
        rel = af.relative_to(in_dir)
        out_path = out_dir / (af.stem + ".drive.json")
        if out_path.exists():
            print(f"[{i}/{len(audio_files)}] 跳过(已存在) {rel}")
            continue
        t0 = time.time()
        try:
            analyze_and_save_drive(af, out_path, device=args.device)
            dt = time.time() - t0
            ok += 1
            print(f"[{i}/{len(audio_files)}] OK {rel}  ({dt:.1f}s)")
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(audio_files)}] FAIL {rel}: {e}")

    total = time.time() - t_start
    print(f"\n[drive] 完成：成功 {ok}，失败 {fail}，总耗时 {total:.1f}s")


if __name__ == "__main__":
    main()
