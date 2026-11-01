"""结构判据：只依赖段落标签、位置、时长、能量包络等结构级信息。

包含两个原子判据：

1. :class:`PositionCriterion` —— 位置规则：
   cur 是 verse、前一段也是 verse（连续）、后一段是 chorus，
   依据「时长比（越短越像 pre-chorus）+ 能量爬升（越高越像）」给基础置信度。

2. :class:`EnergyMatchCriterion` —— 能量相似传导：
   凡是被高置信判为 pre-chorus 的段，其 RMS 能量可作为「指纹」；
   其它尚未识别的 verse 段若同时满足：
   (a) 后一段是 chorus（基础位置判据）
   (b) RMS 与种子相近
   (c) 时长与种子相近
   则说明是同一段音乐材料，沿用该候选的置信度。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..analyzer import AnalysisResult
from ..candidate import PostProcessCandidate
from .base import Criterion, CriterionCategory, CriterionResult


class PositionCriterion(Criterion):
    """位置规则判据：verse → verse → chorus 中的中间 verse 可能是 pre-chorus。"""

    name = "position"
    category = CriterionCategory.STRUCTURE.value
    requires_audio = True  # 需要 RMS 能量算「能量爬升」信号

    # 默认超参
    BASE = 0.5            # 位置规则的基础置信度
    W_DUR = 0.30          # 时长比信号权重
    W_ENERGY = 0.20       # 能量爬升信号权重

    def judge(self, result, y=None, sr=None) -> CriterionResult:
        segments = result.segments
        n = len(segments)

        base = float(self.params.get("base", self.BASE))
        w_dur = float(self.params.get("w_dur", self.W_DUR))
        w_energy = float(self.params.get("w_energy", self.W_ENERGY))

        rms: Dict[int, float] = {}
        if y is not None and sr is not None and len(y):
            rms = _seg_rms_map(segments, y, sr)

        candidates = []
        for i in range(1, n - 1):
            prv, cur, nxt = segments[i - 1], segments[i], segments[i + 1]
            if cur.label != "verse" or prv.label != "verse" or nxt.label != "chorus":
                continue

            d_prev, d_cur = prv.duration, cur.duration
            dur_ratio = d_cur / d_prev if d_prev > 0 else 1.0
            score_dur = max(0.0, 1.0 - dur_ratio)

            score_energy = 0.0
            if i in rms and (i - 1) in rms:
                e_prev, e_cur = rms[i - 1], rms[i]
                if e_prev > 0:
                    ratio = e_cur / e_prev
                    score_energy = min(1.0, max(0.0, (ratio - 1.0) * 2.0))

            confidence = min(1.0, base + w_dur * score_dur + w_energy * score_energy)

            reasons = []
            if score_dur > 0.15:
                reasons.append(f"shorter ({d_cur:.0f}/{d_prev:.0f}s)")
            if score_energy > 0.15:
                reasons.append("energy rise")

            candidates.append(
                PostProcessCandidate(
                    segment_index=i,
                    start=cur.start,
                    end=cur.end,
                    label="pre-chorus?",
                    confidence=round(confidence, 2),
                    reason=", ".join(reasons) if reasons else "position only",
                    source=self.name,
                )
            )

        return CriterionResult(
            criterion=self.name,
            category=self.category,
            candidates=candidates,
            meta={"base": base, "w_dur": w_dur, "w_energy": w_energy},
        )


class EnergyMatchCriterion(Criterion):
    """能量相似传导判据：与已识别的 pre-chorus 段能量/时长相近的 verse 段。"""

    name = "energy_match"
    category = CriterionCategory.STRUCTURE.value
    requires_audio = True  # 需要 RMS 能量

    # 默认超参
    SEED_CONFIDENCE = 0.6
    RMS_LO, RMS_HI = 0.85, 1.18
    DUR_LO, DUR_HI = 0.8, 1.25

    def judge(self, result, y=None, sr=None) -> CriterionResult:
        segments = result.segments
        n = len(segments)

        seed_conf = float(self.params.get("seed_confidence", self.SEED_CONFIDENCE))
        rms_lo = float(self.params.get("rms_lo", self.RMS_LO))
        rms_hi = float(self.params.get("rms_hi", self.RMS_HI))
        dur_lo = float(self.params.get("dur_lo", self.DUR_LO))
        dur_hi = float(self.params.get("dur_hi", self.DUR_HI))

        rms: Dict[int, float] = {}
        if y is not None and sr is not None and len(y):
            rms = _seg_rms_map(segments, y, sr)

        # 种子：由其它判据（这里约定通过 params 传入）给出的高置信候选
        seeds = self.params.get("seeds", [])

        candidates = []
        for i in range(n):
            if segments[i].label != "verse" or i in rms is False:
                continue
            if i + 1 >= n or segments[i + 1].label != "chorus":
                continue

            d_cur = segments[i].duration
            best_conf = 0.0
            best_reason = ""
            for s_idx in seeds:
                if s_idx == i or s_idx not in rms or i not in rms:
                    continue
                e_seed, e_cur = rms[s_idx], rms[i]
                d_seed = segments[s_idx].duration
                if e_seed <= 0:
                    continue
                rms_ratio = e_cur / e_seed
                if not (rms_lo <= rms_ratio <= rms_hi):
                    continue
                dur_ratio = d_cur / d_seed if d_seed > 0 else 1.0
                if not (dur_lo <= dur_ratio <= dur_hi):
                    continue
                conf = self.params.get("seed_conf_map", {}).get(s_idx, seed_conf)
                if conf > best_conf:
                    best_conf = conf
                    best_reason = (
                        f"similar to seg[{s_idx}] (RMS {rms_ratio:.2f}, dur {dur_ratio:.2f})"
                    )
            if best_conf > 0:
                candidates.append(
                    PostProcessCandidate(
                        segment_index=i,
                        start=segments[i].start,
                        end=segments[i].end,
                        label="pre-chorus?",
                        confidence=round(best_conf, 2),
                        reason=best_reason,
                        source=self.name,
                    )
                )

        return CriterionResult(
            criterion=self.name,
            category=self.category,
            candidates=candidates,
            meta={"seed_confidence": seed_conf},
        )


def _seg_rms_map(segments, y: np.ndarray, sr: int) -> Dict[int, float]:
    """计算每段的 RMS 能量均值，返回 {segment_index: rms}。"""
    try:
        import librosa
    except Exception:
        return {}
    try:
        hop = 512
        rms_env = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(rms_env)), sr=sr, hop_length=hop
        )
        out: Dict[int, float] = {}
        for i, s in enumerate(segments):
            m = (rms_times >= s.start) & (rms_times < s.end)
            out[i] = float(rms_env[m].mean()) if m.any() else 0.0
        return out
    except Exception:
        return {}


__all__ = ["PositionCriterion", "EnergyMatchCriterion"]
