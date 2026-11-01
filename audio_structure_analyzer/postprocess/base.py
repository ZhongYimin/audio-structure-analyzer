"""后处理插件基类。

后处理（post-process）是对 all-in-one 原始结构分析结果的二次加工，
用于补充 / 修正原始标签体系没有覆盖或覆盖不足的语义。例如：

- pre-chorus（前副歌）：Harmonix 10 类标签里没有 pre-chorus，需要启发式推断；
- 未来的 EDM 段落（build-up / drop / break）、调式、能量曲线等。

所有后处理方案都实现 :class:`PostProcessor` 接口，通过
:func:`~audio_structure_analyzer.postprocess.run_postprocess` 统一调度，
方便按需增删、组合，而无需改动播放器或推理主流程。

设计原则：
- 每个后处理方案是一个独立类，职责单一；
- 输入统一为 :class:`~audio_structure_analyzer.analyzer.AnalysisResult`
  （可选附带音频波形 y/sr 以计算频谱特征）；
- 输出统一为 :class:`PostProcessResult`（带 name / 候选列表 / 元信息）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..analyzer import AnalysisResult


@dataclass
class PostProcessCandidate:
    """一条后处理候选结果，通常对应一个段落级或时间级的新标注。"""

    segment_index: int
    """在 ``AnalysisResult.segments`` 中的段落下标。"""

    start: float
    """候选起始时间（秒）。"""

    end: float
    """候选结束时间（秒）。"""

    label: str
    """候选的新标签（如 ``"pre-chorus?"``）。"""

    confidence: float
    """置信度，0~1。"""

    reason: str = ""
    """判据说明（供调试 / 展示）。"""

    source: str = ""
    """判据来源（如 ``"position"`` / ``"energy-match"``）。"""

    extra: Dict[str, Any] = field(default_factory=dict)
    """方案自定义的附加信息。"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "extra": self.extra,
        }


@dataclass
class PostProcessResult:
    """一个后处理方案的完整输出。"""

    name: str
    """方案名（如 ``"prechorus"``）。"""

    candidates: List[PostProcessCandidate] = field(default_factory=list)
    """候选列表。"""

    meta: Dict[str, Any] = field(default_factory=dict)
    """方案自定义的元信息。"""

    def by_segment(self) -> Dict[int, PostProcessCandidate]:
        """按段落下标索引候选，便于播放器按段查找。"""
        return {c.segment_index: c for c in self.candidates}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "candidates": [c.to_dict() for c in self.candidates],
            "meta": self.meta,
        }


class PostProcessor(ABC):
    """后处理方案接口。

    子类需实现 :meth:`process`。可选覆盖 ``name`` 与 ``requires_audio``。
    """

    #: 方案唯一名（用于注册与结果标识）。默认取类名小写。
    name: str = ""

    #: 是否需要音频波形（y/sr）。默认 False（仅依赖结构标签）。
    requires_audio: bool = False

    def __init__(self, **params: Any) -> None:
        """用可选参数初始化方案（阈值等可调超参）。"""
        self.params = params
        if not self.name:
            self.name = type(self).__name__.lower()

    @abstractmethod
    def process(
        self,
        result: AnalysisResult,
        y: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> PostProcessResult:
        """对分析结果做后处理，返回候选结果。"""
        raise NotImplementedError
