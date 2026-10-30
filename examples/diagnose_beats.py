"""诊断节拍检测精度：对比多种参数下的拍点数量与位置，可视化 onset 强度。

用法：
    python examples/diagnose_beats.py <song.mp3> [--start 0] [--dur 40]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(y_path, sr_target=22050):
    y, sr = librosa.load(y_path, sr=sr_target, mono=True)
    return y, sr


def onset_strength(y, sr, hop=512):
    """onset strength envelope（对打击乐敏感）。"""
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)


def tempo_and_beats(y, sr, hop=512, tightness=100, start_bpm=120.0):
    tempo, bf = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop, tightness=tightness, start_bpm=start_bpm
    )
    tempo = float(np.atleast_1d(tempo)[0])
    times = librosa.frames_to_time(bf, sr=sr, hop_length=hop)
    return tempo, times


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--dur", type=float, default=40.0)
    args = p.parse_args()

    y, sr = load(args.audio)
    s0 = int(args.start * sr)
    s1 = int((args.start + args.dur) * sr)
    yseg = y[s0:s1]

    hop = 512
    oenv = onset_strength(yseg, sr, hop)
    print(f"采样率={sr}, 片段={args.start:.1f}s~{args.start+args.dur:.1f}s")

    # 全局 BPM 估计
    tempo_global = librosa.beat._tempo(y=yseg, sr=sr, hop_length=hop)
    print(f"全局 tempo 估计: {np.atleast_1d(tempo_global)}")

    # 对比不同参数
    print("\n=== 拍点数量对比（同一 40s 片段）===")
    configs = [
        ("tightness=100 (当前)", dict(tightness=100, start_bpm=120)),
        ("tightness=100 start=178", dict(tightness=100, start_bpm=178)),
        ("tightness=50  start=178", dict(tightness=50, start_bpm=178)),
        ("tightness=50  start=89", dict(tightness=50, start_bpm=89)),
        ("tightness=100 start=89", dict(tightness=100, start_bpm=89)),
        ("tightness=25  start=178", dict(tightness=25, start_bpm=178)),
        ("tightness=25  start=89", dict(tightness=25, start_bpm=89)),
    ]
    results = {}
    for name, kw in configs:
        try:
            tempo, times = tempo_and_beats(yseg, sr, hop, **kw)
            results[name] = (tempo, times)
            print(f"  {name:28s} -> BPM={tempo:6.2f}, 拍点数={len(times):4d}")
        except Exception as e:
            print(f"  {name:28s} -> ERROR {e}")

    # 可视化
    n = len(configs)
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 3.2 * (n + 1)), sharex=True)

    # 顶层：onset 包络
    ax = axes[0]
    times_env = librosa.frames_to_time(
        np.arange(len(oenv)), sr=sr, hop_length=hop
    )
    ax.plot(times_env, oenv, color="tab:gray", lw=0.8, alpha=0.8)
    ax.set_title(f"onset strength 包络（{args.start:.0f}s 起 {args.dur:.0f}s）", fontsize=11)
    ax.set_ylabel("onset")

    for i, (name, kw) in enumerate(configs):
        ax = axes[i + 1]
        tempo, times = results.get(name, (None, np.array([])))
        ax.plot(times_env, oenv, color="tab:gray", lw=0.5, alpha=0.5)
        if len(times):
            ax.vlines(times, 0, max(oenv) * 1.0, color="tab:red", lw=0.8, alpha=0.8)
        ax.set_ylabel(f"{name.split(' ')[0][-3:]}\nBPM={tempo:.0f}" if tempo else "")
        ax.set_title(f"{name}  —  {len(times)} 拍", fontsize=10, loc="left")

    axes[-1].set_xlabel("时间 (s)")
    fig.tight_layout()

    out = os.path.join(_root, "test_audio", "beat_diagnosis.png")
    fig.savefig(out, dpi=110)
    print(f"\n[已保存诊断图] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
