"""和声判据：依赖 chroma（CENS）分布度量「和弦走向差异」。

核心是 :class:`HarmonicShiftCriterion` —— 识别「PC 段落与主歌和弦走向不同」
的歌曲（如 dont_look_back）。这类歌曲的 pre-chorus 不是靠配器（贝斯/鼓
进入）推进，而是靠**和声进行的变化**：真主歌的和弦进行丰富、多音级，
PC 段则换成另一套和弦（通常是持续音 / 长音铺垫，为推向副歌蓄力）。

实测（dont_look_back，BPM 82）：
- 真主歌 verse[1] 主音级 top3 = [0, 9, 10]，verse[5] = [0, 9, 7]；
- PC 段 verse[2] / verse[6] 主音级 top3 = [0, 2, 1]（两段完全一致）；
- 相邻 verse→PC 边界的 chroma 相关系数显著低于整首歌平均水平。

判据（对一个 verse 段 cur，其前一段 prv 也是 verse、后一段 nxt 是 chorus）：
用 cur 与 prv 的 chroma（CENS）逐段平均分布的相关性度量差异，差异大
（相关性低于阈值）→ 和弦走向变了 → 疑似 pre-chorus。

注意：这是「分类型」判据——只在「靠和声变化推进」的歌上有效。对海屿你、
许嵩这类靠配器/鼓推进的歌，verse→PC 的 chroma 相关性依然很高，本判据
自然给不出高置信（正确，不强判）。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..analyzer import AnalysisResult
from ..candidate import PostProcessCandidate
from .base import Criterion, CriterionCategory, CriterionResult


class HarmonicShiftCriterion(Criterion):
    """和声走向差异判据：相邻 verse 段 chroma 分布相关性低 → 疑似 pre-chorus。"""

    name = "harmonic_shift"
    category = CriterionCategory.HARMONIC.value
    requires_audio = True  # 需要波形算 chroma

    # 默认超参
    CORR_THRESHOLD = 0.55   # 相关系数低于此值视为「和弦走向明显不同」
    MAX_CONFIDENCE = 0.70   # 该判据单条能贡献的最高置信度
    HOP = 512

    def judge(self, result, y=None, sr=None) -> CriterionResult:
        segments = result.segments
        n = len(segments)

        corr_thresh = float(self.params.get("corr_threshold", self.CORR_THRESHOLD))
        max_conf = float(self.params.get("max_confidence", self.MAX_CONFIDENCE))
        hop = int(self.params.get("hop", self.HOP))

        meta: Dict[str, object] = {"corr_threshold": corr_thresh, "max_confidence": max_conf}

        if y is None or sr is None or len(y) == 0:
            meta["skipped"] = "no_audio"
            return CriterionResult(
                criterion=self.name, category=self.category, candidates=[], meta=meta
            )

        # 每段 chroma（CENS）逐段平均分布
        chromas = _seg_chroma_dist(segments, y, sr, hop=hop)
        if len(chromas) != n:
            meta["skipped"] = "chroma_failed"
            return CriterionResult(
                criterion=self.name, category=self.category, candidates=[], meta=meta
            )

        candidates = []
        for i in range(1, n - 1):
            prv, cur, nxt = segments[i - 1], segments[i], segments[i + 1]
            # 基础位置：cur 是 verse，且后接 chorus
            if cur.label != "verse" or nxt.label != "chorus":
                continue
            # 前一段需也是 verse（PC 是「连续 verse」的后者）
            if prv.label != "verse":
                continue

            corr = _chroma_corr(chromas[i - 1], chromas[i])
            if corr is None:
                continue

            # 相关性越低 → 和弦走向差异越大 → 越像 pre-chorus
            # 映射到置信度：corr <= corr_thresh 时给接近 max_conf
            if corr >= corr_thresh:
                continue

            shift = corr_thresh - corr  # 差异程度（0 ~ corr_thresh）
            confidence = min(max_conf, max_conf * (shift / corr_thresh) * 2.0)

            candidates.append(
                PostProcessCandidate(
                    segment_index=i,
                    start=cur.start,
                    end=cur.end,
                    label="pre-chorus?",
                    confidence=round(confidence, 2),
                    reason=f"harmonic shift (corr {corr:.2f})",
                    source=self.name,
                    extra={"chroma_corr": corr},
                )
            )

        return CriterionResult(
            criterion=self.name,
            category=self.category,
            candidates=candidates,
            meta=meta,
        )


def _seg_chroma_dist(segments, y: np.ndarray, sr: int, hop: int = 512):
    """每段的 CENS chroma 逐段平均分布（12 维，归一化）。"""
    try:
        import librosa
    except Exception:
        return []
    try:
        out = []
        for s in segments:
            i0, i1 = int(s.start * sr), int(s.end * sr)
            seg_y = y[i0:i1]
            if seg_y.size < hop:
                out.append(np.zeros(12))
                continue
            C = librosa.feature.chroma_cens(y=seg_y, sr=sr, hop_length=hop)
            d = C.mean(axis=1) + 1e-9
            d = d / d.sum()
            out.append(d)
        return out
    except Exception:
        return []


def _chroma_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """两个 12 维 chroma 分布的皮尔逊相关系数。"""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return None
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return None
    return float(c)


__all__ = ["HarmonicShiftCriterion"]
