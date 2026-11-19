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
    StyleProfileProcessor,
    available_processors,
    run_all,
    run_postprocess,
)
from .genre import GenreClassifier  # noqa: F401
from .edm import EDMFormer, analyze_edm  # noqa: F401
from .pipeline import (  # noqa: F401
    find_result,
    load_audio,
    load_bundle,
    preprocess,
    preprocess_edm,
)

__version__ = "0.9.0"

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
    "StyleProfileProcessor",
    "available_processors",
    "run_postprocess",
    "run_all",
    # 流水线（后端编排层）
    "preprocess",
    "preprocess_edm",
    "load_bundle",
    "find_result",
    "load_audio",
    # 流派识别（MAEST 模型）
    "GenreClassifier",
    # EDM 结构识别（EDMFormer 模型，外部依赖）
    "EDMFormer",
    "analyze_edm",
    "__version__",
]
