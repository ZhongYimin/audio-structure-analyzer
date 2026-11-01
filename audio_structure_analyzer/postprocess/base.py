"""后处理（PostProcessor）基类。

后处理是对 all-in-one 原始结构分析结果的二次加工，用于补充 / 修正
原始标签体系没有覆盖或覆盖不足的语义。例如：

- pre-chorus（前副歌）：Harmonix 10 类标签里没有 pre-chorus，需要启发式推断；
- 未来的 EDM 段落（build-up / drop / break）、调式、能量曲线等。

**核心架构：后处理 = 判据组合器**

一个后处理（PostProcessor）**组合**多个判据（Criterion），把它们的候选
**融合**成一个统一结果。判据是原子判定单元（见 :mod:`criteria`），按
特征域分为 structure / orchestration / harmonic 三类。

关系是**组合而非继承**：

    PostProcessor（后处理，组合器）
        ├─ Criterion: PositionCriterion        (structure)
        ├─ Criterion: EnergyMatchCriterion     (structure)
        ├─ Criterion: OrchestrationCriterion   (orchestration)
        └─ Criterion: HarmonicShiftCriterion   (harmonic)

设计原则：
- 每个判据独立、可测、可调阈值；
- 后处理自由增删 / 重排判据组合，不改判据本身；
- 输入统一为 AnalysisResult（可选附带 y/sr）；
- 输出统一为 PostProcessResult（带 name / 候选列表 / 元信息）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ..analyzer import AnalysisResult
from ..candidate import PostProcessCandidate, PostProcessResult  # noqa: F401 (re-export)


class PostProcessor(ABC):
    """后处理方案接口（判据组合器）。

    子类需实现 :meth:`process`。可选覆盖 ``name`` 与 ``requires_audio``。

    一个后处理通常**组合**一组判据（见 :attr:`criteria`），并负责融合
    它们的候选。子类可通过 :meth:`run_criteria` 便捷地运行所有判据，
    拿到 ``{判据名: CriterionResult}``，再做融合。
    """

    #: 方案唯一名（用于注册与结果标识）。默认取类名小写。
    name: str = ""

    #: 是否需要音频波形（y/sr）。默认 False（仅依赖结构标签）。
    #: 若组合了需要波形的判据，应设为 True（或由 run_criteria 自动判断）。
    requires_audio: bool = False

    #: 组合的判据列表（Criterion 实例）。子类在 __init__ 里构造。
    criteria: List["Criterion"] = []

    def __init__(self, **params: Any) -> None:
        """用可选参数初始化方案（阈值等可调超参）。"""
        self.params = params
        if not self.name:
            self.name = type(self).__name__.lower()

    def run_criteria(
        self,
        result: AnalysisResult,
        y: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> Dict[str, "CriterionResult"]:
        """依次运行本后处理组合的所有判据，返回 {判据名: CriterionResult}。"""
        from ..criteria import CriterionResult

        out: Dict[str, CriterionResult] = {}
        for crit in self.criteria:
            out[crit.name] = crit.judge(result, y=y, sr=sr)
        return out

    @abstractmethod
    def process(
        self,
        result: AnalysisResult,
        y: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
    ) -> PostProcessResult:
        """对分析结果做后处理，返回候选结果。"""
        raise NotImplementedError
