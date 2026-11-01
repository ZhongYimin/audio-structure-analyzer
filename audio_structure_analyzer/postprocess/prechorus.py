"""pre-chorus（前副歌）候选检测后处理。

Harmonix 的 10 类标签里没有 pre-chorus，all-in-one 会把 pre-chorus
误标成 verse（导致「连续 verse → chorus」或「chorus → verse → chorus」
中出现"假 verse"）。本方案用启发式规则 + 能量相似传导，把这些疑似
pre-chorus 的 verse 段标出来，并给出置信度。

两阶段：
阶段 1（位置规则）——cur 是 verse，前一段也是 verse（连续），后一段是 chorus：
    判据：时长比（越短越像）+ 能量爬升（越高越像），合成基础置信度。
阶段 2（能量相似传导）——凡是被高置信判为 pre-chorus 的段，其 RMS 能量
    可作为"指纹"；其它尚未识别的 verse 段若同时满足：
    (a) 后一段是 chorus（基础位置判据，pre-chorus 本质是"通向副歌"）
    (b) RMS 与种子相近（0.85~1.18）
    (c) 时长与种子相近（0.8~1.25）
    则说明是同一段音乐材料，沿用该候选的置信度（哪怕它前一段不是 verse）。

可调参数（构造时传入）：
    seed_confidence  阶段 1 候选作为传导种子的置信度阈值（默认 0.6）
    rms_lo / rms_hi  传导的 RMS 相似窗口（默认 0.85 / 1.18）
    dur_lo / dur_hi  传导的时长相似窗口（默认 0.8 / 1.25）
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..analyzer import AnalysisResult
from .base import PostProcessCandidate, PostProcessor, PostProcessResult


class PreChorusDetector(PostProcessor):
    name = "prechorus"
    requires_audio = True  # 阶段 2 需要 RMS 能量

    # 默认超参
    SEED_CONFIDENCE = 0.6
    RMS_LO, RMS_HI = 0.85, 1.18
    DUR_LO, DUR_HI = 0.8, 1.25

    def process(
        self,
        result: AnalysisResult,
        y: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> PostProcessResult:
        segments = result.segments
        bpm = result.bpm
        n = len(segments)

        seed_conf = float(self.params.get("seed_confidence", self.SEED_CONFIDENCE))
        rms_lo = float(self.params.get("rms_lo", self.RMS_LO))
        rms_hi = float(self.params.get("rms_hi", self.RMS_HI))
        dur_lo = float(self.params.get("dur_lo", self.DUR_LO))
        dur_hi = float(self.params.get("dur_hi", self.DUR_HI))

        # 预计算每段的 RMS 能量（需要 y/sr）
        rms: Dict[int, float] = {}
        if y is not None and sr is not None and len(y):
            rms = _seg_rms_map(segments, y, sr)

        candidates: Dict[int, PostProcessCandidate] = {}

        # ---- 阶段 1：位置规则 ----
        for i in range(1, n - 1):
            prv = segments[i - 1]
            cur = segments[i]
            nxt = segments[i + 1]
            if cur.label != "verse" or prv.label != "verse" or nxt.label != "chorus":
                continue

            d_prev = prv.duration
            d_cur = cur.duration
            dur_ratio = d_cur / d_prev if d_prev > 0 else 1.0

            # 信号 1：时长比（越短越像 pre-chorus）
            score_dur = max(0.0, 1.0 - dur_ratio)

            # 信号 2：能量爬升
            score_energy = 0.0
            if i in rms and (i - 1) in rms:
                e_prev, e_cur = rms[i - 1], rms[i]
                if e_prev > 0:
                    ratio = e_cur / e_prev
                    score_energy = min(1.0, max(0.0, (ratio - 1.0) * 2.0))

            confidence = min(1.0, 0.5 + 0.30 * score_dur + 0.20 * score_energy)

            reasons = []
            if score_dur > 0.15:
                reasons.append(f"shorter ({d_cur:.0f}/{d_prev:.0f}s)")
            if score_energy > 0.15:
                reasons.append("energy rise")

            candidates[i] = PostProcessCandidate(
                segment_index=i,
                start=cur.start,
                end=cur.end,
                label="pre-chorus?",
                confidence=round(confidence, 2),
                reason=", ".join(reasons) if reasons else "position only",
                source="position",
            )

        # ---- 阶段 2：能量相似传导 ----
        seeds = [
            i for i, c in candidates.items()
            if c.confidence >= seed_conf and i in rms
        ]
        for i in range(n):
            if segments[i].label != "verse" or i in candidates or i not in rms:
                continue
            # 基础判据：目标段后一段必须是 chorus
            if i + 1 >= n or segments[i + 1].label != "chorus":
                continue

            d_cur = segments[i].duration
            best_conf = 0.0
            best_reason = ""
            for s_idx in seeds:
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
                conf = candidates[s_idx].confidence
                if conf > best_conf:
                    best_conf = conf
                    best_reason = (
                        f"similar to seg[{s_idx}] (RMS {rms_ratio:.2f}, dur {dur_ratio:.2f})"
                    )
            if best_conf > 0:
                candidates[i] = PostProcessCandidate(
                    segment_index=i,
                    start=segments[i].start,
                    end=segments[i].end,
                    label="pre-chorus?",
                    confidence=round(best_conf, 2),
                    reason=best_reason,
                    source="energy-match",
                )

        return PostProcessResult(
            name=self.name,
            candidates=list(candidates.values()),
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
