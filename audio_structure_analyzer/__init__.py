"""音频结构分析（基于 all-in-one 模型）。

项目唯一分析后端为 HuggingFace 的 all-in-one（`allin1_infer`，harmonix-all 模型），
负责节拍 / 强拍 / 段落结构分析。此包提供轻量封装。
"""

from .analyzer import (  # noqa: F401
    HARMONIX_LABELS,
    AnalysisResult,
    Segment,
    analyze,
    load_result,
)
from .criteria import (  # noqa: F401
    Criterion,
    CriterionCategory,
    CriterionResult,
    EnergyMatchCriterion,
    HarmonicShiftCriterion,
    OrchestrationCriterion,
    PositionCriterion,
)
from .postprocess import (  # noqa: F401
    PostProcessCandidate,
    PostProcessor,
    PostProcessResult,
    PreChorusProcessor,
    available_processors,
    run_all,
    run_postprocess,
)
from .pipeline import (  # noqa: F401
    find_result,
    load_audio,
    load_bundle,
    preprocess,
)

__version__ = "0.8.0"

__all__ = [
    "HARMONIX_LABELS",
    "AnalysisResult",
    "Segment",
    "analyze",
    "load_result",
    # 判据
    "Criterion",
    "CriterionResult",
    "CriterionCategory",
    "PositionCriterion",
    "EnergyMatchCriterion",
    "OrchestrationCriterion",
    "HarmonicShiftCriterion",
    # 后处理
    "PostProcessor",
    "PostProcessResult",
    "PostProcessCandidate",
    "PreChorusProcessor",
    "available_processors",
    "run_postprocess",
    "run_all",
    # 流水线（后端编排层）
    "preprocess",
    "load_bundle",
    "find_result",
    "load_audio",
    "__version__",
]
