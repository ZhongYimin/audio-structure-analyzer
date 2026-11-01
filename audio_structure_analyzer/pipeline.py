"""分析流水线（后端编排层）。

播放器（或任何前端）只应依赖这一层，而非直接调用 ``allin1_infer`` 或
手写后处理编排。本模块收敛三件事：

1. 推理 —— 调 :func:`analyzer.analyze` 分析音频并落盘 JSON（缓存复用）；
2. 加载 —— 用 :func:`analyzer.load_result` 读回结构化结果；
3. 后处理 —— 用 :func:`postprocess.run_all` 编排一组可配置的方案。

典型前端流程：

    from audio_structure_analyzer.pipeline import analyze_audio, build_result, run_analysis

    # 首次：分析 + 落盘（若已有 JSON 则跳过推理）
    result = analyze_audio("song.wav", device="cpu")

    # 加载音频波形 + 跑一组后处理
    result, y, sr, post = build_result("song.wav", processors=["prechorus", "spatiotemporal"])

设计原则：
- 前端不感知 ``allin1_infer``、Demucs 分轨、后处理注册等实现细节；
- 后处理方案名（见 ``postprocess.available_processors``）可配置，缺省跑全部；
- 后处理可能较重（如时空判据需 Demucs 分轨），前端应在后台线程调用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .analyzer import AnalysisResult, analyze, load_result
from .postprocess import PostProcessResult, available_processors, run_all


def load_audio(audio: Path, sr: int = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    """加载音频波形，返回 (y, sr)。仅在后处理需要波形时调用。"""
    import librosa

    y, _sr = librosa.load(str(audio), sr=sr, mono=mono)
    return y, _sr


def analyze_audio(
    audio,
    model: str = "harmonix-all",
    device: str = "cpu",
    out_json=None,
    **kwargs,
) -> AnalysisResult:
    """分析音频并把结果落盘为 JSON（与音频同目录同名 .json），返回结果。

    若目标 JSON 已存在，可选择复用（见 ``reuse_json``）。默认每次调用都重跑
    推理并覆盖落盘——调用方应自己判断是否已有结果（见 :func:`find_result`）。
    """
    result = analyze(audio, model=model, device=device, **kwargs)

    audio = Path(audio)
    out = Path(out_json) if out_json else audio.with_suffix(".json")
    result.save(out)
    return result


def find_result(audio) -> Optional[Path]:
    """查找音频对应的分析结果 JSON（同目录同名，或 allin1 子目录）。"""
    audio = Path(audio)
    candidates = [
        audio.with_suffix(".json"),
        audio.parent / "allin1" / f"{audio.stem}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_result(
    audio,
    processors: Optional[List[str]] = None,
    device: str = "cpu",
    model: str = "harmonix-all",
    cache_dir: Optional[Path] = None,
    load_audio_for_postprocess: bool = True,
) -> Tuple[AnalysisResult, Optional[np.ndarray], Optional[int], Dict[str, PostProcessResult]]:
    """加载（或按需分析）一首歌，并跑一组可配置的后处理方案。

    Parameters
    ----------
    audio : str | Path
        音频文件路径。
    processors : List[str], optional
        要运行的后处理方案名列表。缺省跑全部（postprocess.DEFAULT_ORDER）。
    device : str
        推理设备（仅在需要重新分析时用）。
    model : str
        模型名。
    cache_dir : Path, optional
        时空判据等方案的分轨缓存目录。
    load_audio_for_postprocess : bool
        是否加载音频波形（任何 requires_audio 的后处理方案都需要）。默认 True。

    Returns
    -------
    (result, y, sr, post_results)
        result : AnalysisResult
        y, sr : 音频波形与采样率（若无需波形则为 (None, None)）
        post_results : {方案名: PostProcessResult}
    """
    audio = Path(audio)

    jp = find_result(audio)
    if jp is None:
        result = analyze_audio(audio, model=model, device=device)
    else:
        result = load_result(jp)

    need_audio = load_audio_for_postprocess and _any_requires_audio(processors)
    y = sr = None
    if need_audio:
        y, sr = load_audio(audio)

    post = run_all(result, y=y, sr=sr, names=processors)

    return result, y, sr, post


def _any_requires_audio(names: Optional[List[str]]) -> bool:
    """判断给定方案（或默认全部）中是否有方案需要音频波形。"""
    from .postprocess import get_processor

    if names is None:
        from .postprocess import DEFAULT_ORDER

        names = DEFAULT_ORDER
    return any(get_processor(n).requires_audio for n in names)


__all__ = [
    "analyze_audio",
    "find_result",
    "build_result",
    "load_audio",
    "available_processors",
]
