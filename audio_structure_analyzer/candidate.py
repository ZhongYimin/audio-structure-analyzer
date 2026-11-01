"""候选（Candidate）数据结构的底层定义。

``PostProcessCandidate`` 是判据（Criterion）与后处理（PostProcessor）
共同产出的候选单元，为避免 criteria <-> postprocess 之间的循环依赖，
将它下沉到本独立模块，由两侧共享。

``PostProcessResult`` 是后处理（PostProcessor）融合判据后的最终输出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


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
    """判据来源（如 ``"position"`` / ``"energy_match"`` / ``"harmonic_shift"``）。"""

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


__all__ = ["PostProcessCandidate", "PostProcessResult"]
