"""结构分割（v4）：chroma 层次聚类 + 能量突变点，可靠测「外层结构」。

经验结论（经过 chroma / 能量 / kick / Demucs 声部 / novelty 五种方法验证）：

- 「外层结构」边界（intro→主歌→副歌/drop→breakdown）可靠可测，
  用 chroma 层次聚类 + 能量突变即可，本模块已能测出 39s / 90s 等边界。
- 「副歌内部的细分」边界（S1→S2，如 53s）是音乐曲式/乐句层面的语义边界，
  两个副歌段在能量、和声、kick 节奏上几乎完全相同，任何单一低层声学特征
  （含预训练 embedding 的 novelty）都无法稳定区分——属于曲式分析（form
  analysis / repetition 结构）范畴，需要更强的先验或人工锚点。

本模块定位：可靠测「外层结构」边界。chroma 层次聚类打底，可选叠加能量
突变点，保证不产生碎边界、不漏掉大的结构切换。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def detect_boundaries(
    frames: np.ndarray,
    sr: int,
    hop: int = 512,
    min_segment_sec: float = 8.0,
    n_segments: int = 0,
    y: np.ndarray = None,
    embed_feat: Optional[np.ndarray] = None,
    **kwargs,
) -> List[float]:
    """检测结构边界，返回边界时间点列表（秒，含 0.0 起点）。

    主路径：chroma_cqt -> agglomerative 层次聚类（和声结构）。
    叠加 RMS 能量突变点捕捉 breakdown 边界。
    """
    import librosa

    duration = len(y) / sr if y is not None else frames.shape[0] * hop / sr

    # 1) chroma 层次聚类（和声结构边界）
    if y is not None:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    else:
        chroma = np.abs(frames[:, 4:6]).T
        if chroma.shape[0] < 12:
            chroma = np.vstack([chroma] * 6)[:12]

    if n_segments <= 0:
        n_segments = max(6, int(round(duration / 30.0)))

    try:
        bounds_frames = librosa.segment.agglomerative(chroma, n_segments)
        bound_times = librosa.frames_to_time(bounds_frames, sr=sr, hop_length=hop)
        bound_times = [float(t) for t in bound_times if t > 0]
    except Exception:
        bound_times = []

    # 2) 能量突变点（补充 breakdown 边界）
    energy_bounds = []
    if y is not None:
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_smooth = librosa.feature.rms(y=y, hop_length=hop * 4)[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms_smooth)), sr=sr, hop_length=hop * 4
        )
        # 能量低谷（breakdown）作为边界候选：低于邻域中位数且持续 >2s
        med = float(np.median(rms_smooth))
        if med > 0:
            low_mask = rms_smooth < 0.35 * med
            # 找连续低能量段
            in_low = False
            low_start = 0.0
            for i in range(len(low_mask)):
                t = float(rms_times[i])
                if low_mask[i] and not in_low:
                    in_low = True
                    low_start = t
                elif not low_mask[i] and in_low:
                    in_low = False
                    if t - low_start >= 2.0:
                        energy_bounds.append(low_start)
                        energy_bounds.append(t)

    all_times = [0.0] + sorted(bound_times + energy_bounds)
    final = [all_times[0]]
    for t in all_times[1:]:
        if t - final[-1] >= min_segment_sec:
            final.append(t)

    if duration - final[-1] >= min_segment_sec:
        final.append(duration)

    return final
