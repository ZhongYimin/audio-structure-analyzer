"""示例：分析一首歌，输出段落标签。

用法：
    python examples/analyze_song.py path/to/song.mp3 [--json out.json] [--plot out.png]
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="分析歌曲结构（Ap/Dp/S/Cp）")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--json", help="输出 JSON 文件路径（可选）")
    parser.add_argument("--plot", help="输出可视化 PNG 路径（可选）")
    args = parser.parse_args()

    # 把项目根目录加入 sys.path，确保无论从哪个目录运行都能 import 包
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from audio_structure_analyzer import analyze

    result = analyze(args.audio)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if args.json:
        from audio_structure_analyzer.io import write_json

        write_json(result, args.json)
        print(f"\n[已写入] {args.json}")

    if args.plot:
        from audio_structure_analyzer.viz import plot_sections

        plot_sections(args.audio, result, args.plot)
        print(f"[已绘图] {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
