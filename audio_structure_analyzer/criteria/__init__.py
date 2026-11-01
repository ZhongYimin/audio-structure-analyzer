"""判据（Criterion）包。

判据是原子级判定单元，按特征域分三类：
- structure     结构判据（位置、时长、能量传导）
- orchestration 配器判据（分轨频段能量）
- harmonic      和声判据（和弦走向差异）

判据由后处理（PostProcessor）组合使用，本包只提供判据本身与分类，
不负责融合。
"""

from __future__ import annotations

from .base import Criterion, CriterionCategory, CriterionResult
from .structure import EnergyMatchCriterion, PositionCriterion
from .orchestration import OrchestrationCriterion
from .harmonic import HarmonicShiftCriterion

__all__ = [
    "Criterion",
    "CriterionResult",
    "CriterionCategory",
    "PositionCriterion",
    "EnergyMatchCriterion",
    "OrchestrationCriterion",
    "HarmonicShiftCriterion",
]
