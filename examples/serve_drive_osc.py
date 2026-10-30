"""启动运行时驱动引擎的 OSC 服务。

用法：
    python examples/serve_drive_osc.py --map song_a=drives/a.drive.json --map song_b=drives/b.drive.json \
        --listen-port 9000 --out-port 9001

也可以直接编辑下面的 DRIVE_MAP 后运行。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_structure_analyzer.osc_engine import serve_osc


def main():
    ap = argparse.ArgumentParser(description="运行时驱动引擎 OSC 服务")
    ap.add_argument("--map", action="append", default=[], help="song_id=驱动文件路径，可多次")
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=9000)
    ap.add_argument("--out-host", default="127.0.0.1")
    ap.add_argument("--out-port", type=int, default=9001)
    args = ap.parse_args()

    drive_map = {}
    for m in args.map:
        song_id, path = m.split("=", 1)
        drive_map[song_id.strip()] = path.strip()

    if not drive_map:
        print("请用 --map song_id=路径 指定驱动文件，或直接编辑本脚本 DRIVE_MAP")
        sys.exit(1)

    serve_osc(
        drive_map,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        out_host=args.out_host,
        out_port=args.out_port,
    )


if __name__ == "__main__":
    main()
