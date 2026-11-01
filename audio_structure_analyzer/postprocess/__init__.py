"""后处理插件包。

提供统一的后处理接口与调度入口。后处理（PostProcessor）是**判据组合器**，
组合多个判据（Criterion，见 :mod:`criteria`）并融合其候选。

新增后处理方案只需：

1. 实现 :class:`~audio_structure_analyzer.postprocess.base.PostProcessor` 子类；
2. 在 :data:`REGISTRY` 里登记（一行）。

然后通过 :func:`run_postprocess` 或 :func:`run_all` 统一调用，
播放器无需感知具体方案实现。

用法：
    from audio_structure_analyzer.postprocess import run_postprocess, run_all

    # 只跑 pre-chorus 检测
    res = run_postprocess("prechorus", result, y=y, sr=sr)

    # 跑所有已注册方案
    results = run_all(result, y=y, sr=sr)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import numpy as np

from ..analyzer import AnalysisResult
from .base import (
    PostProcessCandidate,
    PostProcessor,
    PostProcessResult,
)
from .prechorus import PreChorusProcessor

# ---- 方案注册表：name -> PostProcessor 类 ----
# 新增方案在此登记即可被 run_postprocess / run_all 发现。
REGISTRY: Dict[str, Type[PostProcessor]] = {
    PreChorusProcessor.name: PreChorusProcessor,
}

# 默认执行顺序（按名字）。
DEFAULT_ORDER: List[str] = [PreChorusProcessor.name]


def available_processors() -> List[str]:
    """返回所有已注册的后处理方案名。"""
    return list(REGISTRY.keys())


def get_processor(name: str) -> Type[PostProcessor]:
    """按名字取后处理方案类。"""
    if name not in REGISTRY:
        raise KeyError(f"未知的后处理方案: {name!r}（可用: {available_processors()}）")
    return REGISTRY[name]


def run_postprocess(
    name: str,
    result: AnalysisResult,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    **params,
) -> PostProcessResult:
    """运行单个后处理方案。

    Parameters
    ----------
    name : str
        方案名（见 :data:`available_processors`）。
    result : AnalysisResult
        all-in-one 结构分析结果。
    y, sr : optional
        音频波形与采样率（需要频谱特征的方案才需要）。
    **params
        传给方案构造器的超参。
    """
    cls = get_processor(name)
    proc = cls(**params)
    return proc.process(result, y=y, sr=sr)


def run_all(
    result: AnalysisResult,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    names: Optional[List[str]] = None,
    **params,
) -> Dict[str, PostProcessResult]:
    """按顺序运行一组后处理方案，返回 {name: PostProcessResult}。

    names 缺省时按 DEFAULT_ORDER 执行全部已注册方案。
    """
    order = names if names is not None else DEFAULT_ORDER
    out: Dict[str, PostProcessResult] = {}
    for name in order:
        out[name] = run_postprocess(name, result, y=y, sr=sr, **params)
    return out


__all__ = [
    "REGISTRY",
    "DEFAULT_ORDER",
    "PostProcessor",
    "PostProcessResult",
    "PostProcessCandidate",
    "PreChorusProcessor",
    "available_processors",
    "get_processor",
    "run_postprocess",
    "run_all",
]
