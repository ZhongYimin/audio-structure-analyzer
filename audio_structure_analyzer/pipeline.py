"""预处理流水线（后端编排层）。

职责：把「结构分析 + 后处理」合并进**同一份 JSON** 落盘，供播放器直接
读取渲染，无需在点击歌曲时现场做任何重计算。

关键设计：
- 预处理是**独立的离线阶段**（可在播放器之外批量执行），产出完整 JSON；
- 播放器（前端）只 :func:`load_bundle` 读这一份 JSON，纯加载、秒开；
- 「用不用后处理」只是播放器 UI 的显示开关——所有方案的结果都已并入
  JSON，勾选/取消仅过滤显示，不触发任何计算。

产物 JSON 结构：
    {
        path, bpm, beats, downbeats, beat_positions, segments,  # 结构
        postprocess: { "<方案名>": PostProcessResult.to_dict(), ... }  # 后处理
    }
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


def find_result(audio) -> Optional[Path]:
    """查找音频对应的结果 JSON（同目录同名，或 allin1 子目录）。"""
    audio = Path(audio)
    candidates = [
        audio.with_suffix(".json"),
        audio.parent / "allin1" / f"{audio.stem}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def preprocess(
    audio,
    processors: Optional[List[str]] = None,
    device: str = "cpu",
    model: str = "harmonix-all",
    cache_dir: Optional[Path] = None,
    out_json=None,
    force: bool = False,
) -> AnalysisResult:
    """分析一首歌，跑全部（或指定）后处理，并把结果合并写进同一份 JSON。

    这是唯一的「重计算」入口，应在播放器之外离线调用（批处理 / CLI）。

    Parameters
    ----------
    audio : str | Path
        音频文件路径。
    processors : List[str], optional
        要运行的后处理方案名。缺省跑全部（postprocess.DEFAULT_ORDER）。
    device, model : str
        推理设备 / 模型名。
    cache_dir : Path, optional
        时空判据等方案的分轨缓存目录。
    out_json : str | Path, optional
        输出 JSON 路径。缺省写 <音频同名>.json。
    force : bool
        True 时无条件重跑推理；False 时若 JSON 已含后处理则跳过（见下）。

    Returns
    -------
    AnalysisResult
        已并入后处理的结果（同时也已落盘）。
    """
    audio = Path(audio)
    out = Path(out_json) if out_json else audio.with_suffix(".json")

    # 已有结果且不强制：加载；否则推理
    jp = find_result(audio) if out == audio.with_suffix(".json") else None
    if not force and jp is not None:
        result = load_result(jp)
    else:
        result = analyze(audio, model=model, device=device)

    # 加载音频（任一需波形的方案都需要）
    if _any_requires_audio(processors):
        y, sr = load_audio(audio)
    else:
        y = sr = None

    # 跑后处理，结果并入 result.postprocess
    post = run_all(result, y=y, sr=sr, names=processors)
    result.postprocess = {name: res.to_dict() for name, res in post.items()}

    # 合并落盘到同一份 JSON
    result.save(out)
    return result


def load_bundle(audio) -> AnalysisResult:
    """播放器读取：加载一份已预处理的 JSON（结构 + 后处理合并）。

    纯读取，不做任何计算。若 JSON 不存在则抛 FileNotFoundError。
    """
    audio = Path(audio)
    jp = find_result(audio)
    if jp is None:
        raise FileNotFoundError(f"未找到 {audio.name} 的预处理结果，请先运行 preprocess。")
    return load_result(jp)


def _any_requires_audio(names: Optional[List[str]]) -> bool:
    """判断给定方案（或默认全部）中是否有方案需要音频波形。"""
    from .postprocess import get_processor

    if names is None:
        from .postprocess import DEFAULT_ORDER

        names = DEFAULT_ORDER
    return any(get_processor(n).requires_audio for n in names)


__all__ = [
    "preprocess",
    "load_bundle",
    "find_result",
    "load_audio",
    "available_processors",
]
