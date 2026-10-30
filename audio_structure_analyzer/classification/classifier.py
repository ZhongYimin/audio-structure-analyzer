"""段落归并与分类（推倒重来版）：把块归并为段落，打 Ap/Dp/S/Cp 标签。

相较旧版「能量/带宽排名 + 硬阈值」，本版：
- 用完整的段落级特征（能量、频谱质心、带宽、频谱通量）+ 波形（y）计算
  更可靠的「能量高低」与「能量爬升/回落」；
- 用「段落重复度」（结构自相似）识别副歌 S（全曲重复出现的高能量段）；
- 时序约束：Dp 紧邻 S 之前（能量爬升），Cp 紧跟 S 之后（能量回落）。

标签定义（映射到 EDM / 流行歌）：
- S（副歌/drop）：全曲重复出现的高能量「高潮」段；
- Dp（铺垫/build）：能量爬升、紧邻 S 之前的过渡段；
- Ap（主歌/叙事）：中低能量、编曲较稀疏的段；
- Cp（副歌后）：紧跟 S 之后的能量回落段。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core import Section, Segment


def _group_segments(
    segments: List[Segment], boundaries: List[float]
) -> List[List[Segment]]:
    """按边界把块归并为段落组。"""
    groups: List[List[Segment]] = []
    cur: List[Segment] = []
    b = sorted(boundaries)
    bi = 0
    for seg in segments:
        while bi < len(b) - 1 and seg.start >= b[bi + 1]:
            if cur:
                groups.append(cur)
                cur = []
            bi += 1
        cur.append(seg)
    if cur:
        groups.append(cur)
    return [g for g in groups if g]


def _section_from_group(g: List[Segment], idx_offset: int) -> Section:
    return Section(
        start=g[0].start,
        end=g[-1].end,
        label="",
        segment_indices=list(range(idx_offset, idx_offset + len(g))),
    )


def classify(
    segments: List[Segment],
    boundaries: List[float],
    frames: Optional[np.ndarray] = None,
    extra: Optional[dict] = None,
    y: Optional[np.ndarray] = None,
    sr: int = 22050,
    **kwargs,
) -> List[Section]:
    """归并段落并分类（Ap/Dp/S/Cp）。

    y/sr：原始波形（可选，用于计算更准确的段落能量与重复度）。
    """
    groups = _group_segments(segments, boundaries)
    sections = [_section_from_group(g, 0) for g in groups]
    for s, g in zip(sections, groups):
        s.segment_indices = [segments.index(x) for x in g]

    if not sections:
        return []

    # --- 段落级能量（用波形计算，比块级特征均值更可靠）---
    if y is not None:
        _compute_energy_from_waveform(sections, y, sr)
    else:
        _compute_energy_from_features(sections, groups, frames)

    n = len(sections)

    # --- 重复度：识别全曲重复出现的段落组（副歌最强信号）---
    chorus_group = _find_recurring_sections(sections, y, sr, extra)

    # --- 能量排名（全局归一化）---
    energies = np.array([s.feature_summary.get("energy", 0.0) for s in sections])
    e_rank = _rank(energies)

    # --- 绝对能量分层阈值（EDM/流行歌的能量是绝对的，用绝对阈值更稳）---
    e_min = float(energies.min())
    e_max = float(energies.max())
    e_span = max(e_max - e_min, 1e-9)
    high_thr = e_min + 0.55 * e_span   # 高能量（drop/副歌）
    low_thr = e_min + 0.30 * e_span    # 低能量（breakdown/主歌）

    labels: List[str] = []
    for i in range(n):
        e = energies[i]
        r = e_rank[i]
        slope_v = sections[i].feature_summary.get("energy_slope", 0.0)
        in_chorus = chorus_group is not None and i in chorus_group

        is_high = e >= high_thr
        is_low = e <= low_thr
        is_building = slope_v > 0.05  # 能量爬升（build up）

        # 分类优先级：
        # 1. 副歌重复组（且中高能量）→ S
        # 2. 高能量 → S（EDM drop）
        # 3. 能量爬升 → Dp（build，紧邻高能量前）
        # 4. 低能量 → Ap
        if in_chorus and e >= low_thr:
            labels.append("S")
        elif is_high:
            labels.append("S")
        elif is_building:
            labels.append("Dp")
        else:
            labels.append("Ap")

    # --- 时序约束后处理（合并连续同标签 + 修正 Dp/Cp 位置）---
    labels = _apply_timeline_constraints(sections, labels, energies, high_thr, low_thr)

    for s, lab in zip(sections, labels):
        s.label = lab
        s.confidence = _confidence(s, e_rank, sections)

    return sections


def _compute_energy_from_waveform(sections: List[Section], y: np.ndarray, sr: int):
    """用波形计算每个段落的 RMS 能量与能量斜率。"""
    import librosa

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]

    for s in sections:
        i0 = int(s.start * sr / hop)
        i1 = int(s.end * sr / hop)
        i0 = max(0, min(i0, len(rms) - 1))
        i1 = max(i0 + 1, min(i1, len(rms)))
        seg = rms[i0:i1]
        s.feature_summary["energy"] = float(seg.mean())
        # 能量斜率（爬升/回落），用首尾对比
        if len(seg) >= 4:
            first = seg[: len(seg) // 3].mean()
            last = seg[-len(seg) // 3:].mean()
            s.feature_summary["energy_slope"] = float(last - first)
        else:
            s.feature_summary["energy_slope"] = 0.0


def _compute_energy_from_features(
    sections: List[Section], groups: List[List[Segment]], frames: Optional[np.ndarray]
):
    """退化路径：用块级 rms 特征估算能量。"""
    for s, g in zip(sections, groups):
        vals = []
        for seg in g:
            if seg.features is not None and len(seg.features) > 0:
                vals.append(float(seg.features[0]))  # rms_mean
        s.feature_summary["energy"] = float(np.mean(vals)) if vals else 0.0
        s.feature_summary["energy_slope"] = 0.0


def _find_recurring_sections(
    sections: List[Section],
    y: Optional[np.ndarray],
    sr: int,
    extra: Optional[dict],
) -> Optional[set]:
    """识别全曲「重复出现且相似」的段落组（副歌候选）。

    用 chroma 直方图 + MFCC 均值作为段落特征，算余弦相似度，用动态阈值聚类。
    """
    n = len(sections)
    if n < 2:
        return None

    feats = None
    if extra is not None and "chroma" in extra and "mfcc" in extra:
        chroma = extra["chroma"]
        mfcc = extra["mfcc"]
        hop = 512
        feats = []
        for s in sections:
            f0 = int(s.start * sr / hop)
            f1 = int(s.end * sr / hop)
            f0 = max(0, min(f0, chroma.shape[1] - 1))
            f1 = max(f0 + 1, min(f1, chroma.shape[1]))
            c_hist = chroma[:, f0:f1].sum(axis=1)
            c_hist = c_hist / (c_hist.sum() + 1e-9)
            m_mean = mfcc[:, f0:f1].mean(axis=1)
            m_mean = m_mean / (np.linalg.norm(m_mean) + 1e-9)
            feats.append(np.concatenate([c_hist, m_mean]))
        feats = np.array(feats)

    if feats is None:
        return None

    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
    fn = feats / norms
    sim = fn @ fn.T

    off_diag = sim[~np.eye(n, dtype=bool)]
    baseline = float(np.median(off_diag))
    mad = float(np.median(np.abs(off_diag - baseline)))
    threshold = baseline + 3.0 * mad

    visited = [False] * n
    best_group: Optional[set] = None
    for i in range(n):
        if visited[i]:
            continue
        cluster = {i}
        visited[i] = True
        for j in range(n):
            if j != i and sim[i, j] >= threshold:
                cluster.add(j)
                visited[j] = True
        if best_group is None or len(cluster) > len(best_group):
            best_group = cluster

    if best_group is not None and len(best_group) >= 2:
        return best_group
    return None


def _apply_timeline_constraints(
    sections: List[Section],
    labels: List[str],
    energies: np.ndarray,
    high_thr: float,
    low_thr: float,
) -> List[str]:
    """用音乐结构时序合法性修正标签。

    - Dp（铺垫）：紧邻 S 之前、能量爬升的中低能量段。
    - Cp（副歌后）：紧跟 S 之后、能量回落的段。
    - 连续多个 S（drop 内部被和声边界切开）保留为 S，但中间夹的低能量
      小段（若极短）并入上下文，避免 Ap/S 频繁跳动。
    """
    n = len(sections)
    out = list(labels)

    for i in range(n):
        if labels[i] == "S":
            # S 之后：若下一个是「中低能量且能量低于本 S」，标为 Cp
            if i + 1 < n and labels[i + 1] not in ("S",):
                if energies[i + 1] < high_thr:
                    out[i + 1] = "Cp"
            # S 之前：若上一个是中低能量且能量爬升，标为 Dp
            if i - 1 >= 0 and labels[i - 1] not in ("S", "Dp"):
                slope_v = sections[i - 1].feature_summary.get("energy_slope", 0.0)
                if energies[i - 1] < high_thr:
                    if slope_v > 0.03:
                        out[i - 1] = "Dp"
    return out


def _rank(values: np.ndarray) -> np.ndarray:
    """全曲归一化排名 [0,1]。"""
    if values.size == 0 or values.std() < 1e-9:
        return np.zeros_like(values)
    order = np.argsort(np.argsort(values))
    return order / max(1, values.size - 1)


def _confidence(s: Section, e_rank: np.ndarray, sections: List[Section]) -> float:
    idx = sections.index(s)
    r = e_rank[idx]
    if s.label == "S":
        return float(0.5 + 0.4 * r)
    if s.label == "Dp":
        return float(0.5 + 0.3 * max(0.0, s.feature_summary.get("energy_slope", 0.0)))
    if s.label == "Cp":
        return 0.6
    return float(0.5 + 0.3 * (1.0 - r))
