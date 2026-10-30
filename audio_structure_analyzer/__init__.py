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

__version__ = "0.2.0"

__all__ = [
    "HARMONIX_LABELS",
    "AnalysisResult",
    "Segment",
    "analyze",
    "load_result",
    "__version__",
]
