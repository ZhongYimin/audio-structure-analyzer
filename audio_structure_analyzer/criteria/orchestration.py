"""配器判据：依赖 Demucs 分轨 + 分频段能量。

核心是 :class:`OrchestrationCriterion` —— 用「时域（从无到有 / 爬升）+
空域（四轨频段分布）」识别 pre-chorus 的配器推进特征。

判据信号（对一个 verse 段 cur，其前一段 prv、后一段 nxt）：

- **空域进入信号（强）**：
  - 贝斯进入：bass 低频数量级跳升；
  - 底鼓进入：drums 低频数量级跳升。
- **空域增强信号（弱）**：
  - 开镲：drums 高频抬升；
  - 人声变亮：vocals 高频/低频比增大；
  - 铺底加厚：other 中频抬升。

信号全拉满的歌（海屿你式）→ 高置信；只有弱信号的歌 → 低置信。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..analyzer import AnalysisResult
from ..candidate import PostProcessCandidate
from ..stems import (
    BAND_BASS_LOW,
    BAND_DRUMS_HIGH,
    BAND_DRUMS_LOW,
    segment_band_energy,
)
from .base import Criterion, CriterionCategory, CriterionResult


class OrchestrationCriterion(Criterion):
    """配器判据：四轨分频段能量的「进入 / 增强」识别 pre-chorus 推进。"""

    name = "orchestration"
    category = CriterionCategory.ORCHESTRATION.value
    requires_audio = True  # 需要波形做 Demucs 分轨

    # 默认超参
    ENTER_RATIO = 8.0     # 进入信号：数量级跳升阈值
    ENHANCE_RATIO = 1.5   # 增强信号：抬升阈值
    CACHE_DIR = None

    # 各信号的置信度权重（合计 ≤ 1.0）
    W_BASS_ENTER = 0.30    # 贝斯进入（强）
    W_KICK_ENTER = 0.25    # 底鼓进入（强）
    W_HIHAT = 0.15         # 开镲（中）
    W_VOCAL_BRIGHT = 0.15  # 人声变亮（中）
    W_OTHER_THICKEN = 0.15 # 铺底加厚（中）

    def judge(self, result, y=None, sr=None) -> CriterionResult:
        segments = result.segments
        n = len(segments)

        enter_ratio = float(self.params.get("enter_ratio", self.ENTER_RATIO))
        enhance_ratio = float(self.params.get("enhance_ratio", self.ENHANCE_RATIO))
        cache_dir = self.params.get("cache_dir", self.CACHE_DIR)
        cache_dir = Path(cache_dir) if cache_dir else None

        meta: Dict[str, object] = {
            "enter_ratio": enter_ratio,
            "enhance_ratio": enhance_ratio,
        }

        if y is None or sr is None or len(y) == 0:
            meta["skipped"] = "no_audio"
            return CriterionResult(
                criterion=self.name, category=self.category, candidates=[], meta=meta
            )

        try:
            from ..stems import separate_stems_with_sr

            stems, stems_sr = separate_stems_with_sr(result.path, cache_dir=cache_dir)
        except Exception as e:
            meta["skipped"] = f"separation_failed: {e}"
            return CriterionResult(
                criterion=self.name, category=self.category, candidates=[], meta=meta
            )

        starts = np.array([s.start for s in segments], dtype=np.float64)
        ends = np.array([s.end for s in segments], dtype=np.float64)

        def seg_energy(stem: str, lo: float, hi: float) -> np.ndarray:
            return segment_band_energy(stems[stem], stems_sr, lo, hi, starts, ends)

        bass_low = seg_energy("bass", *BAND_BASS_LOW)
        drums_low = seg_energy("drums", *BAND_DRUMS_LOW)
        drums_high = seg_energy("drums", *BAND_DRUMS_HIGH)
        vocals_low = seg_energy("vocals", 80.0, 300.0)
        vocals_high = seg_energy("vocals", 2000.0, 8000.0)
        other_mid = seg_energy("other", 200.0, 4000.0)

        candidates = []
        for i in range(1, n - 1):
            cur, nxt = segments[i], segments[i + 1]
            if cur.label != "verse" or nxt.label != "chorus":
                continue
            p = i - 1

            bass_enter = bass_low[i] >= enter_ratio * max(bass_low[p], 1e-6)
            kick_enter = drums_low[i] >= enter_ratio * max(drums_low[p], 1e-6)
            hihat_up = drums_high[i] >= enhance_ratio * max(drums_high[p], 1e-6)

            v_ratio_cur = vocals_high[i] / max(vocals_low[i], 1e-6)
            v_ratio_prv = vocals_high[p] / max(vocals_low[p], 1e-6)
            vocal_bright = v_ratio_cur >= enhance_ratio * max(v_ratio_prv, 1e-6)

            other_up = other_mid[i] >= enhance_ratio * max(other_mid[p], 1e-6)

            if not (bass_enter or kick_enter or hihat_up or vocal_bright or other_up):
                continue

            confidence = 0.0
            reasons = []

            def ratio(cur_v, prv_v):
                return cur_v / max(prv_v, 1e-6)

            if bass_enter:
                confidence += self.W_BASS_ENTER
                reasons.append(f"bass enters x{ratio(bass_low[i], bass_low[p]):.0f}")
            if kick_enter:
                confidence += self.W_KICK_ENTER
                reasons.append(f"kick enters x{ratio(drums_low[i], drums_low[p]):.0f}")
            if hihat_up:
                confidence += self.W_HIHAT
                reasons.append(f"hi-hat up x{ratio(drums_high[i], drums_high[p]):.1f}")
            if vocal_bright:
                confidence += self.W_VOCAL_BRIGHT
                reasons.append(f"vocal brightens x{ratio(v_ratio_cur, v_ratio_prv):.1f}")
            if other_up:
                confidence += self.W_OTHER_THICKEN
                reasons.append(f"pad thickens x{ratio(other_mid[i], other_mid[p]):.1f}")

            has_enter = bass_enter or kick_enter
            if not has_enter:
                confidence *= 0.6

            confidence = round(min(1.0, confidence), 2)
            if confidence <= 0.0:
                continue

            candidates.append(
                PostProcessCandidate(
                    segment_index=i,
                    start=cur.start,
                    end=cur.end,
                    label="pre-chorus?",
                    confidence=confidence,
                    reason=", ".join(reasons),
                    source=self.name,
                    extra={
                        "bass_low": (float(bass_low[p]), float(bass_low[i])),
                        "drums_low": (float(drums_low[p]), float(drums_low[i])),
                        "drums_high": (float(drums_high[p]), float(drums_high[i])),
                        "vocals_brightness": (float(v_ratio_prv), float(v_ratio_cur)),
                        "other_mid": (float(other_mid[p]), float(other_mid[i])),
                    },
                )
            )

        return CriterionResult(
            criterion=self.name,
            category=self.category,
            candidates=candidates,
            meta=meta,
        )


__all__ = ["OrchestrationCriterion"]
