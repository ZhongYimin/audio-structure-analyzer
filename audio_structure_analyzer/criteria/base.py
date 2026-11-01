"""判据（Criterion）抽象基类。

判据是「原子级」的判定单元：每个判据针对一个**单一维度的特征**，
独立地对分析结果给出若干「候选 + 置信度」。判据本身**不做融合**，
只负责「我根据某条规律，认为这些段落像 pre-chorus（置信多少、理由是什么）」。

判据按**特征域**分为三类（见 :data:`CriterionCategory`）：

- ``structure``     结构判据：只依赖段落标签、位置、时长、能量包络等结构级信息；
- ``orchestration`` 配器判据：依赖 Demucs 分轨 + 分频段能量等配器信息；
- ``harmonic``      和声判据：依赖 chroma / 和弦走向等调性信息。

判据与后处理（PostProcessor）的关系是**组合**而非继承：一个后处理
（如「识别 pre-chorus」）会**组合多个判据**，把它们的候选**融合**成
一个统一结果。判据是「零件」，后处理是「装配」。

这样设计的好处：
- 判据可独立开发、独立测试、独立调阈值；
- 后处理可以自由增删、重排判据组合，而不改动判据本身；
- 未来新判据（如调式、EDM build-up）只需新增一个 Criterion 子类，
  并把它登记进某个后处理的判据列表即可。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..analyzer import AnalysisResult
from ..candidate import PostProcessCandidate


class CriterionCategory(str, Enum):
    """判据的特征域类别。"""

    STRUCTURE = "structure"
    ORCHESTRATION = "orchestration"
    HARMONIC = "harmonic"


@dataclass
class CriterionResult:
    """单个判据的输出：一组候选 + 元信息。

    与 PostProcessResult 的区别：CriterionResult 是「一个判据」的输出，
    尚未跨判据融合；PostProcessResult 是「一个后处理」融合后的最终输出。
    """

    criterion: str
    """判据唯一名（如 ``"position"`` / ``"harmonic_shift"``）。"""

    category: str
    """判据类别（structure / orchestration / harmonic）。"""

    candidates: List[PostProcessCandidate] = field(default_factory=list)
    """候选列表。"""

    meta: Dict[str, Any] = field(default_factory=dict)
    """判据自定义元信息（阈值、诊断值等）。"""


class Criterion(ABC):
    """判据抽象基类。

    子类需实现 :meth:`judge`。可选覆盖 ``name`` / ``category`` /
    ``requires_audio``。判据的 ``judge`` 返回 :class:`CriterionResult`，
    由上层后处理负责融合。
    """

    #: 判据唯一名。默认取类名转 snake_case。
    name: str = ""

    #: 判据类别（structure / orchestration / harmonic）。
    category: str = CriterionCategory.STRUCTURE.value

    #: 是否需要音频波形（y/sr）。配器 / 和声判据通常需要。
    requires_audio: bool = False

    def __init__(self, **params: Any) -> None:
        """用可选参数初始化判据（阈值等可调超参）。"""
        self.params = params
        if not self.name:
            self.name = _camel_to_snake(type(self).__name__)

    @abstractmethod
    def judge(
        self,
        result: AnalysisResult,
        y: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> CriterionResult:
        """对分析结果做判定，返回候选 + 置信度。

        Parameters
        ----------
        result : AnalysisResult
            all-in-one 结构分析结果。
        y, sr : optional
            音频波形与采样率（需要频谱特征的判据才需要）。
        """
        raise NotImplementedError


def _camel_to_snake(name: str) -> str:
    """CamelCase -> snake_case（去掉 Criterion 后缀）。"""
    import re

    name = re.sub(r"Criterion$", "", name)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return s


__all__ = [
    "Criterion",
    "CriterionResult",
    "CriterionCategory",
]
