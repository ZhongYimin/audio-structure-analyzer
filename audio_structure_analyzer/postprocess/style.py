"""流派识别后处理（基于 MAEST 模型，AI 推理）。

用 MAEST（MTG 的 400 风格细粒度分类模型）对一首歌做**多标签流派识别**，
输出 top-K 风格标签及置信度。

与早期规则式流派判据不同，这是**真正的 AI 推理**：模型在 330 万首 Discogs
曲目上学习「什么声音像 Pop Rock / J-pop / Synth-pop」，能给出规则无法表达
的文化 + 地域细粒度标签（如 dont_look_back → Pop Rock，Reason → J-pop）。

输出结构：一个 ``segment_index = -1`` 的全曲候选，``extra`` 里携带：
- ``genre_tags``：top-K 风格标签（形如 ``Rock---Pop Rock``）；
- ``genre_dist``：{标签: 置信度}；
- ``main_genre``：主流派（top-1 的大类，如 ``Rock``）。

依赖：transformers（MAEST 模型，首次运行会下载约 300MB 权重）。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..analyzer import AnalysisResult
from ..candidate import PostProcessCandidate
from ..genre import GenreClassifier
from .base import PostProcessor, PostProcessResult


class StyleProfileProcessor(PostProcessor):
    """用 MAEST 模型对歌曲做细粒度多标签流派识别。"""

    name = "style"
    requires_audio = True

    def __init__(self, **params):
        super().__init__(**params)
        self.top_k = int(params.get("top_k", 5))
        self._clf = GenreClassifier.get()

    def process(self, result, y=None, sr=None) -> PostProcessResult:
        meta: Dict[str, object] = {"criteria": ["maest_genre_model"]}

        if y is None or sr is None or len(y) == 0:
            meta["skipped"] = "no_audio"
            return PostProcessResult(name=self.name, candidates=[], meta=meta)

        try:
            tags = self._clf.classify(y, sr, top_k=self.top_k)
        except Exception as e:
            meta["skipped"] = f"genre_failed: {e}"
            return PostProcessResult(name=self.name, candidates=[], meta=meta)

        if not tags:
            return PostProcessResult(name=self.name, candidates=[], meta=meta)

        # 拆「大类---子类」
        main_genre = tags[0][0].split("---")[0] if "---" in tags[0][0] else tags[0][0]
        genre_dist = {label: conf for label, conf in tags}

        candidate = PostProcessCandidate(
            segment_index=-1,
            start=0.0,
            end=result.duration,
            label=f"genre:{main_genre}",
            confidence=round(tags[0][1], 2),
            reason="; ".join(f"{l} {c:.0%}" for l, c in tags),
            source=self.name,
            extra={
                "genre_tags": [l for l, _ in tags],
                "genre_dist": genre_dist,
                "main_genre": main_genre,
            },
        )

        return PostProcessResult(
            name=self.name,
            candidates=[candidate],
            meta={
                "criteria": ["maest_genre_model"],
                "main_genre": main_genre,
                "genre_dist": genre_dist,
                "genre_tags": [l for l, _ in tags],
            },
        )


__all__ = ["StyleProfileProcessor"]
