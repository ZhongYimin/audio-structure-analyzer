"""端到端验证 OrchestrationDetector 配器判据。

在指定歌曲上跑 orchestration 后处理，打印检测到的 pre-chorus 候选，
并与位置/能量传导（prechorus 方案）的结果对比。

运行（Python 3.12 + demucs）：
    python examples/verify_orchestration_detector.py \
        "test_audio/马也_Crabbit - 海屿你.wav" "test_audio/马也_Crabbit - 海屿你.json"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("json")
    ap.add_argument("--cache-dir", default="test_audio/stems_npy")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    import librosa

    from audio_structure_analyzer import AnalysisResult, load_result, run_postprocess

    result: AnalysisResult = load_result(args.json)

    y, sr = librosa.load(args.audio, sr=22050, mono=True)

    print("=" * 70)
    print(f"歌曲: {Path(args.audio).name}  BPM={result.bpm}")
    print("段落结构:")
    for i, s in enumerate(result.segments):
        print(f"  [{i:2d}] {s.label:8s} {s.start:7.2f} -> {s.end:7.2f}s")
    print("-" * 70)

    # 1) 位置 + 能量传导（原有方案）
    pc = run_postprocess("prechorus", result, y=y, sr=sr)
    print("\n[prechorus 方案] 候选:")
    if pc.candidates:
        for c in pc.candidates:
            print(f"  seg[{c.segment_index:2d}] conf={c.confidence:.2f}  {c.reason}")
    else:
        print("  (无)")

    # 2) 时空判据（新方案）
    orch = run_postprocess(
        "spatiotemporal", result, y=y, sr=sr, cache_dir=args.cache_dir
    )
    print("\n[spatiotemporal 方案] 候选:")
    if orch.candidates:
        for c in orch.candidates:
            e = c.extra
            print(
                f"  seg[{c.segment_index:2d}] conf={c.confidence:.2f}  "
                f"bass {e['bass_low'][0]:.1f}->{e['bass_low'][1]:.1f}  "
                f"kick {e['drums_low'][0]:.1f}->{e['drums_low'][1]:.1f}  "
                f"hihat {e['drums_high'][0]:.1f}->{e['drums_high'][1]:.1f}  "
                f"vocals-bright {e['vocals_brightness'][0]:.3f}->{e['vocals_brightness'][1]:.3f}  "
                f"pad {e['other_mid'][0]:.1f}->{e['other_mid'][1]:.1f}"
            )
            print(f"           reason: {c.reason}")
    else:
        print("  (无)")
    print(f"  meta: {orch.meta}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
