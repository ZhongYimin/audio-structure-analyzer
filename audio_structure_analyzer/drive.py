"""驱动文件生成器：把 all-in-one 分析结果转成运行时驱动文件。

设计哲学：空间换时间。所有重计算（声部分离 + Transformer 结构分析）都在
离线 GPU 阶段完成，输出一份自包含、可查表的"驱动文件"。运行时只做 O(log n)
二分查表 + 定时触发，无需模型、无需外部信号。

驱动文件格式（JSON，与 all-in-one 结果兼容扩展）：
    {
      "meta": {
        "source": "dont_look_back.wav",   # 源音频（时间基准）
        "duration": 287.89,
        "bpm": 82,
        "beats": 380,
        "downbeats": 95,
        "format": "drive-v1",
        "generated_by": "all-in-one harmonix-all",
        "generated_at": "2026-10-31T07:56:44"
      },
      "beats": [
        {"t": 0.19, "downbeat": true, "pos": 1, "strength": 0.82},
        {"t": 0.95, "downbeat": false, "pos": 2, "strength": 0.71},
        ...
      ],
      "segments": [
        {"label": "intro", "start": 0.0, "end": 11.9},
        ...
      ]
    }

beats[i].strength 是"这一拍的触发强度"（0~1），供效果层映射亮度/力度等。
默认从低频能量包络在拍点处采样得到；若无法计算则回退为 1.0。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from .analyzer import AnalysisResult


# 驱动文件格式版本
DRIVE_FORMAT = "drive-v1"


@dataclass
class DriveBeat:
    """驱动文件里的一个拍点。"""

    t: float          # 拍点时间（秒，相对源音频）
    downbeat: bool    # 是否强拍（小节首拍）
    pos: int          # 小节内拍号（1~N，强拍为 1）
    strength: float   # 触发强度 0~1

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_strength(y: np.ndarray, sr: int, beat_times: List[float]) -> List[float]:
    """在拍点处采样低频能量包络，归一化为 0~1 的强度。

    用低频（<120Hz）能量近似"这一拍的鼓点/低频冲击强度"。
    返回与 beat_times 等长的强度列表。
    """
    if len(beat_times) == 0:
        return []

    # 低频能量包络：STFT 后对 <120Hz 求和
    import librosa

    n_fft = 2048
    hop = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bass = S[freqs < 120.0].sum(axis=0)
    if bass.max() <= 0:
        return [1.0] * len(beat_times)

    # 平滑 + 归一化
    bass = bass / bass.max()
    frame_times = librosa.frames_to_time(np.arange(len(bass)), sr=sr, hop_length=hop)

    strengths = []
    for t in beat_times:
        idx = int(np.clip(np.searchsorted(frame_times, t) - 1, 0, len(bass) - 1))
        strengths.append(round(float(bass[idx]), 4))
    return strengths


def build_drive_file(
    result: AnalysisResult,
    source_path: PathLike = "",
    audio_path: Optional[PathLike] = None,
) -> dict:
    """从 all-in-one 结果构建驱动文件 dict。

    Parameters
    ----------
    result : AnalysisResult
        all-in-one 分析结果。
    source_path : PathLike
        源音频路径（写入 meta.source，仅作标识，时间基准为 0）。
    audio_path : PathLike, optional
        若提供，则读取音频计算每拍强度；否则强度回退为 1.0。
    """
    beats = np.asarray(result.beats, dtype=float)
    downbeats = set(round(float(x), 4) for x in result.downbeats)

    # 强度
    if audio_path is not None and len(beats) > 0:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        strengths = _compute_strength(y, sr, [float(x) for x in beats])
    else:
        strengths = [1.0] * len(beats)

    # 拍号：直接用 all-in-one 原生的 beat_positions（已正确标注弱起/强拍）
    positions = list(result.beat_positions) if result.beat_positions else [1] * len(beats)
    if len(positions) != len(beats):
        positions = [1] * len(beats)

    beat_list: List[DriveBeat] = []
    for i, t in enumerate(beats):
        t = float(t)
        is_db = round(t, 4) in downbeats
        beat_list.append(
            DriveBeat(
                t=round(t, 4),
                downbeat=is_db,
                pos=int(positions[i]),
                strength=strengths[i],
            )
        )

    return {
        "meta": {
            "source": str(source_path),
            "duration": round(result.duration, 4),
            "bpm": int(result.bpm),
            "beats": len(beats),
            "downbeats": len(result.downbeats),
            "format": DRIVE_FORMAT,
            "generated_by": "all-in-one harmonix-all",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "beats": [b.to_dict() for b in beat_list],
        "segments": [s.to_dict() for s in result.segments],
    }


def analyze_and_save_drive(
    audio_path: PathLike,
    drive_path: PathLike,
    device: str = "cuda",
    **kwargs,
) -> Path:
    """一站式：分析音频 + 生成驱动文件。

    Parameters
    ----------
    audio_path : PathLike
        音频文件路径。
    drive_path : PathLike
        输出驱动文件路径（JSON）。
    device : str
        推理设备，默认 "cuda"。
    """
    from .analyzer import analyze

    result = analyze(audio_path, device=device, **kwargs)
    drive = build_drive_file(result, source_path=audio_path, audio_path=audio_path)
    drive_path = Path(drive_path)
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    with open(drive_path, "w", encoding="utf-8") as f:
        json.dump(drive, f, ensure_ascii=False, indent=2)
    return drive_path
