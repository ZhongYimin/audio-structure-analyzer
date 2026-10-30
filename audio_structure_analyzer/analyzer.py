"""all-in-one 音乐结构分析封装。

本模块是项目唯一的分析后端：薄封装 HuggingFace 的 all-in-one
（`allin1_infer`，harmonix-all 模型）完成节拍 / 强拍 / 段落结构分析。

all-in-one 基于 Demucs 声部分离 + Transformer 逐帧预测，对真实演奏的
节奏浮动与编曲相位变化（进鼓 / 撤鼓）有强鲁棒性，输出：

- bpm / beats（逐拍）/ downbeats（强拍）/ beat_positions（拍号）
- segments（Harmonix 10 类标签：intro/verse/chorus/bridge/break/inst/solo/outro...）

用法：
    from audio_structure_analyzer import analyze
    result = analyze("song.mp3")
    print(result.bpm, result.segments)

    # 或读入已有的 all-in-one JSON 结果（无需重复推理）
    result = load_result("test_audio/allin1/guy_contact.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

PathLike = Union[str, Path]


# Harmonix 数据集 10 类结构标签（all-in-one 输出的语义标签）
HARMONIX_LABELS = (
    "start", "end", "intro", "outro", "verse",
    "chorus", "bridge", "break", "inst", "solo",
)


@dataclass
class Segment:
    """一个结构段落。"""

    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "label": self.label}


@dataclass
class AnalysisResult:
    """一首歌的 all-in-one 分析结果。"""

    path: str
    bpm: int
    beats: List[float] = field(default_factory=list)
    downbeats: List[float] = field(default_factory=list)
    beat_positions: List[int] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    activation_fps: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "bpm": self.bpm,
            "beats": self.beats,
            "downbeats": self.downbeats,
            "beat_positions": self.beat_positions,
            "activation_fps": self.activation_fps,
            "segments": [s.to_dict() for s in self.segments],
        }

    def save(self, path: PathLike) -> Path:
        """把结果写为 JSON（与 all-in-one 原生 JSON 兼容）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path


def _from_allin1(raw) -> AnalysisResult:
    """把 `allin1_infer` 的原生 AnalysisResult 转成本项目结构。"""
    return AnalysisResult(
        path=str(raw.path),
        bpm=int(raw.bpm),
        beats=[float(x) for x in raw.beats],
        downbeats=[float(x) for x in raw.downbeats],
        beat_positions=[int(x) for x in raw.beat_positions],
        segments=[Segment(start=s.start, end=s.end, label=s.label) for s in raw.segments],
        activation_fps=getattr(raw, "activation_fps", None),
    )


def analyze(
    path: PathLike,
    model: str = "harmonix-all",
    device: str = "cpu",
    out_dir: Optional[PathLike] = None,
    **kwargs,
) -> AnalysisResult:
    """用 all-in-one 模型分析一首歌。

    Parameters
    ----------
    path : PathLike
        音频文件路径（wav/mp3/flac 等）。
    model : str
        模型名，默认 "harmonix-all"。
    device : str
        推理设备，默认 "cpu"（有 CUDA 可传 "cuda"）。
    out_dir : PathLike, optional
        输出目录（保存 JSON / 激活 / embedding 等产物）。
    **kwargs
        透传给 `allin1_infer.analyze`（如 demix_dir、multiprocess、skip_separation 等）。

    Returns
    -------
    AnalysisResult
    """
    import allin1_infer

    results = allin1_infer.analyze(
        paths=[str(path)], model=model, device=device, out_dir=out_dir, **kwargs
    )
    result = results[0] if isinstance(results, list) else results

    return _from_allin1(result)


def load_result(path: PathLike) -> AnalysisResult:
    """从已有的 all-in-one JSON 结果文件加载（避免重复推理）。

    兼容 `allin1_infer` 原生保存的 JSON，以及本模块 `save()` 输出的 JSON。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return AnalysisResult(
        path=data["path"],
        bpm=int(data["bpm"]),
        beats=[float(x) for x in data.get("beats", [])],
        downbeats=[float(x) for x in data.get("downbeats", [])],
        beat_positions=[int(x) for x in data.get("beat_positions", [])],
        segments=[Segment(start=s["start"], end=s["end"], label=s["label"]) for s in data["segments"]],
        activation_fps=data.get("activation_fps"),
    )
