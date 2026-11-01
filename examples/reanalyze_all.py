"""批量预处理所有 test_audio 的音频，产出「结构 + 后处理」合并 JSON。

用 pipeline.preprocess 逐首：分析（allin1_infer 推理 + Demucs 分轨）
+ 跑全部后处理方案，把结果合并写进 <音频同名>.json，供播放器直接读取。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob

from audio_structure_analyzer import preprocess


def auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main():
    wavs = sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "test_audio" / "*.wav")))
    device = auto_device()
    print(f"[batch] 共 {len(wavs)} 首歌，设备={device}", flush=True)

    for i, w in enumerate(wavs, 1):
        w = Path(w)
        print(f"\n[{i}/{len(wavs)}] 预处理 {w.name} ...", flush=True)
        t0 = time.time()
        try:
            result = preprocess(w, device=device, force=True)
            dt = time.time() - t0
            print(
                f"  OK  BPM={result.bpm}  {len(result.segments)} 段  "
                f"后处理={list(result.postprocess.keys())}  用时 {dt:.1f}s",
                flush=True,
            )
        except Exception as e:
            dt = time.time() - t0
            print(f"  FAILED ({dt:.1f}s): {e}", flush=True)

    print("\n[batch] 全部完成", flush=True)


if __name__ == "__main__":
    main()
