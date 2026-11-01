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
from .postprocess import (  # noqa: F401
    PostProcessCandidate,
    PostProcessor,
    PostProcessResult,
    PreChorusDetector,
    SpatiotemporalDetector,
    available_processors,
    run_all,
    run_postprocess,
)

__version__ = "0.6.0"

__all__ = [
    "HARMONIX_LABELS",
    "AnalysisResult",
    "Segment",
    "analyze",
    "load_result",
    # 后处理
    "PostProcessor",
    "PostProcessResult",
    "PostProcessCandidate",
    "PreChorusDetector",
    "SpatiotemporalDetector",
    "available_processors",
    "run_postprocess",
    "run_all",
    "__version__",
]
