"""pre-chorus（前副歌）后处理：组合三类判据识别 pre-chorus。

Harmonix 的 10 类标签里没有 pre-chorus，all-in-one 会把 pre-chorus 误标成
verse（导致「连续 verse → chorus」中出现「假 verse」）。本后处理组合三类
判据，把疑似 pre-chorus 的 verse 段标出来，并给出融合置信度。

组合的判据（见 :mod:`criteria`）：

1. :class:`~criteria.structure.PositionCriterion`（结构）——
   verse→verse→chorus 位置规则 + 时长比 + 能量爬升；
2. :class:`~criteria.structure.EnergyMatchCriterion`（结构）——
   与已识别 PC 段能量/时长相近的传导；
3. :class:`~criteria.orchestration.OrchestrationCriterion`（配器）——
   贝斯/底鼓进入 + 开镲/人声变亮/铺底加厚；
4. :class:`~criteria.harmonic.HarmonicShiftCriterion`（和声）——
   相邻 verse 段和弦走向差异。

融合策略：按段落下标聚合各判据的候选，置信度取「判据间互补叠加」——
即不同判据命中同一段时，置信度叠加（封顶 1.0），体现「多判据交叉验证
更可信」；单判据命中则保留该判据置信度。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..analyzer import AnalysisResult
from ..criteria import (
    EnergyMatchCriterion,
    HarmonicShiftCriterion,
    OrchestrationCriterion,
    PositionCriterion,
)
from .base import PostProcessCandidate, PostProcessor, PostProcessResult


class PreChorusProcessor(PostProcessor):
    """组合结构 + 配器 + 和声三类判据识别 pre-chorus。"""

    name = "prechorus"
    requires_audio = True  # 结构（RMS）+ 配器（分轨）+ 和声（chroma）都需要波形

    def __init__(self, **params):
        super().__init__(**params)
        cache_dir = params.get("cache_dir")
        # 结构判据
        self.position = PositionCriterion(
            **{k: params[k] for k in ("base", "w_dur", "w_energy") if k in params}
        )
        self.energy_match = EnergyMatchCriterion(
            **{k: params[k] for k in ("seed_confidence", "rms_lo", "rms_hi", "dur_lo", "dur_hi") if k in params}
        )
        # 配器判据
        self.orchestration = OrchestrationCriterion(
            **{k: params[k] for k in ("enter_ratio", "enhance_ratio", "cache_dir") if k in params}
        )
        # 和声判据
        self.harmonic = HarmonicShiftCriterion(
            **{k: params[k] for k in ("corr_threshold", "max_confidence", "hop") if k in params}
        )
        self.criteria = [self.position, self.orchestration, self.harmonic]

    def process(self, result, y=None, sr=None) -> PostProcessResult:
        # 1) 位置规则判据（作为基础候选 + 传导种子）
        pos = self.position.judge(result, y=y, sr=sr)
        position_candidates = {c.segment_index: c for c in pos.candidates}

        # 2) 能量传导判据（种子 = 位置规则的高置信候选）
        seeds = [
            i for i, c in position_candidates.items()
            if c.confidence >= self.energy_match.SEED_CONFIDENCE
        ]
        self.energy_match.params["seeds"] = seeds
        self.energy_match.params["seed_conf_map"] = {
            i: c.confidence for i, c in position_candidates.items()
        }
        em = self.energy_match.judge(result, y=y, sr=sr)

        # 3) 配器判据
        orch = self.orchestration.judge(result, y=y, sr=sr)

        # 4) 和声判据
        harm = self.harmonic.judge(result, y=y, sr=sr)

        # ---- 融合 ----
        fused: Dict[int, PostProcessCandidate] = {}

        def _add(cand: PostProcessCandidate):
            i = cand.segment_index
            # 记录这条判据的独立贡献（供播放器按判据分别配色 / 标注）
            contrib = {
                "source": cand.source,
                "confidence": cand.confidence,
                "reason": cand.reason,
            }
            if i not in fused:
                # 首次命中：以该判据为「主判据」，并建立 sources 列表
                cand.sources = [contrib]
                fused[i] = cand
            else:
                # 多判据交叉验证：置信度互补叠加（封顶 1.0），
                # 但**保留每条判据的独立贡献**，不吞掉来源。
                prev = fused[i]
                prev.sources.append(contrib)
                # 主判据取「置信度最高」的判据（首次命中可能不是最强的）
                best = max(prev.sources, key=lambda s: s["confidence"])
                prev.source = best["source"]
                new_conf = min(1.0, prev.confidence + cand.confidence)
                prev.confidence = round(new_conf, 2)
                prev.reason = prev.reason + "; " + cand.reason
                # 合并 extra
                prev.extra = {**prev.extra, **cand.extra}

        for c in position_candidates.values():
            _add(c)
        for c in em.candidates:
            _add(c)
        for c in orch.candidates:
            _add(c)
        for c in harm.candidates:
            _add(c)

        meta = {
            "criteria": [c.name for c in self.criteria],
            "position": len(pos.candidates),
            "energy_match": len(em.candidates),
            "orchestration": len(orch.candidates),
            "harmonic": len(harm.candidates),
        }

        return PostProcessResult(
            name=self.name,
            candidates=list(fused.values()),
            meta=meta,
        )


__all__ = ["PreChorusProcessor"]
