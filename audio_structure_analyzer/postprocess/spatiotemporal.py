"""时空判据后处理：用 Demucs 分轨的时域/空域配器特征识别 pre-chorus。

背景
----
Harmonix 10 类标签没有 pre-chorus，all-in-one 会把 pre-chorus 误标成
verse。:class:`PreChorusDetector` 用位置 + RMS 能量爬升，但 RMS 是混合信号，
在编曲复杂的歌上不够稳。

本方案引入**时空判据**（spatiotemporal）：一段音乐的「推进感」由两个维度决定——

- **空域（spectral / spatial）**：四轨（bass/drums/other/vocals）在频段空间
  上的分布。真 verse 是「稀疏坐标」（只有人声 + 铺底，贝斯/底鼓接近 0），
  pre-chorus / chorus 是「密集坐标」（贝斯、底鼓、镲、铺底同时出现）。
- **时域（temporal）**：这些频段能量相对前段的「从无到有」（数量级跳升）
  与「增强」（倍数抬升），即配器在时间轴上的推进过程。

实测（《海屿你》，verse 用木吉他 + 人声铺底，PC 才进贝斯/鼓）四轨能量从
verse₁ → PC 的跃迁：

    bass  低频 40-200Hz :    0.0  -> 6908   （贝斯进入，43 万倍）
    drums 低频 40-200Hz :    0.0  -> 11241  （底鼓进入）
    drums 高频 4k-12kHz :    2.5  -> 127.8  （开镲增强，51 倍）
    vocals 高频/低频比  :  0.029  -> 0.178  （人声频谱重心上移，变激昂）
    other 中频 200-4kHz :   3068  -> 5148   （吉他/键盘铺底加厚）

判据（对一个 verse 段 cur，其前一段 prv、后一段 nxt）：
1. 基础位置（必要条件）：cur.label == verse 且 nxt.label == chorus。
2. 空域进入信号（强，各加权重）：
   - 贝斯进入：bass 低频数量级跳升；
   - 底鼓进入：drums 低频数量级跳升。
3. 空域增强信号（弱，各加小权重）：
   - 开镲：drums 高频抬升；
   - 人声变亮：vocals 高频/低频比增大；
   - 铺底加厚：other 中频抬升。

信号全拉满的歌（海屿你式）→ conf≈1.0；只有弱信号的歌（如许嵩式，
verse 已有贝斯鼓）→ 低 conf，不强判。

可调参数（构造时传入）：
    enter_ratio    进入信号的数量级跳升阈值（默认 8.0 倍）
    enhance_ratio  增强信号的抬升阈值（默认 1.5 倍）
    cache_dir      分轨缓存目录（复用 :mod:`stems` 的磁盘缓存）
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..analyzer import AnalysisResult
from ..stems import (
    BAND_BASS_LOW,
    BAND_DRUMS_HIGH,
    BAND_DRUMS_LOW,
    segment_band_energy,
)
from .base import PostProcessCandidate, PostProcessor, PostProcessResult


class SpatiotemporalDetector(PostProcessor):
    name = "spatiotemporal"
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

    def process(
        self,
        result: AnalysisResult,
        y: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> PostProcessResult:
        segments = result.segments
        n = len(segments)

        enter_ratio = float(self.params.get("enter_ratio", self.ENTER_RATIO))
        enhance_ratio = float(self.params.get("enhance_ratio", self.ENHANCE_RATIO))
        cache_dir = self.params.get("cache_dir", self.CACHE_DIR)
        cache_dir = Path(cache_dir) if cache_dir else None

        candidates: Dict[int, PostProcessCandidate] = {}
        meta: Dict[str, object] = {
            "enter_ratio": enter_ratio,
            "enhance_ratio": enhance_ratio,
        }

        if y is None or sr is None or len(y) == 0:
            meta["skipped"] = "no_audio"
            return PostProcessResult(name=self.name, candidates=[], meta=meta)

        # 分轨（带缓存）。注意：频段能量必须用分轨波形的真实 sr 计算，
        # 与传入 y/sr（可能被 librosa 重采样为 22050）无关，否则频点错位。
        try:
            from ..stems import separate_stems_with_sr

            stems, stems_sr = separate_stems_with_sr(result.path, cache_dir=cache_dir)
        except Exception as e:
            meta["skipped"] = f"separation_failed: {e}"
            return PostProcessResult(name=self.name, candidates=[], meta=meta)

        starts = np.array([s.start for s in segments], dtype=np.float64)
        ends = np.array([s.end for s in segments], dtype=np.float64)

        def seg_energy(stem: str, lo: float, hi: float) -> np.ndarray:
            return segment_band_energy(stems[stem], stems_sr, lo, hi, starts, ends)

        # 空域坐标：四轨 × 频段 → 每段能量
        bass_low = seg_energy("bass", *BAND_BASS_LOW)      # 贝斯
        drums_low = seg_energy("drums", *BAND_DRUMS_LOW)   # 底鼓
        drums_high = seg_energy("drums", *BAND_DRUMS_HIGH) # 镲片
        vocals_low = seg_energy("vocals", 80.0, 300.0)     # 人声胸腔/低频
        vocals_high = seg_energy("vocals", 2000.0, 8000.0) # 人声亮部
        other_mid = seg_energy("other", 200.0, 4000.0)     # 铺底

        for i in range(1, n - 1):
            cur = segments[i]
            nxt = segments[i + 1]
            # 基础判据：verse 且后面紧跟 chorus
            if cur.label != "verse" or nxt.label != "chorus":
                continue

            p = i - 1

            # ---- 空域进入信号（数量级跳升）----
            bass_enter = bass_low[i] >= enter_ratio * max(bass_low[p], 1e-6)
            kick_enter = drums_low[i] >= enter_ratio * max(drums_low[p], 1e-6)

            # ---- 空域增强信号（倍数抬升）----
            hihat_up = drums_high[i] >= enhance_ratio * max(drums_high[p], 1e-6)

            # 人声变亮：高频/低频比（频谱重心上移）
            v_ratio_cur = vocals_high[i] / max(vocals_low[i], 1e-6)
            v_ratio_prv = vocals_high[p] / max(vocals_low[p], 1e-6)
            vocal_bright = v_ratio_cur >= enhance_ratio * max(v_ratio_prv, 1e-6)

            # 铺底加厚
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

            # 若没有任何「进入」信号（只有增强型），整体打折：
            # 增强型信号在普通 verse→chorus 过渡里也常见，区分度弱。
            has_enter = bass_enter or kick_enter
            if not has_enter:
                confidence *= 0.6

            confidence = round(min(1.0, confidence), 2)
            if confidence <= 0.0:
                continue

            candidates[i] = PostProcessCandidate(
                segment_index=i,
                start=cur.start,
                end=cur.end,
                label="pre-chorus?",
                confidence=confidence,
                reason=", ".join(reasons),
                source="spatiotemporal",
                extra={
                    "bass_low": (float(bass_low[p]), float(bass_low[i])),
                    "drums_low": (float(drums_low[p]), float(drums_low[i])),
                    "drums_high": (float(drums_high[p]), float(drums_high[i])),
                    "vocals_brightness": (float(v_ratio_prv), float(v_ratio_cur)),
                    "other_mid": (float(other_mid[p]), float(other_mid[i])),
                },
            )

        return PostProcessResult(
            name=self.name,
            candidates=list(candidates.values()),
            meta=meta,
        )


# 兼容旧名（orchestration），避免下游引用失效
OrchestrationDetector = SpatiotemporalDetector


__all__ = ["SpatiotemporalDetector", "OrchestrationDetector"]
