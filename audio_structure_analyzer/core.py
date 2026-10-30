"""音频结构分析框架核心数据结构与标签定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# 四类目标标签 + 非目标段落（前奏/间奏/尾奏/桥段）
LABELS = ("Ap", "Dp", "S", "Cp")
EXTRA_LABELS = ("Intro", "Interlude", "Outro", "Bridge")
ALL_LABELS = LABELS + EXTRA_LABELS


@dataclass
class Segment:
    """一个时间块（segment），用于特征提取与后续归并的最小单位。"""

    start: float          # 秒
    end: float            # 秒
    features: Optional[np.ndarray] = None  # 块级特征向量

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Section:
    """归并后的段落，携带标签、置信度与节拍信息。"""

    start: float
    end: float
    label: str
    confidence: float = 0.0
    feature_summary: dict = field(default_factory=dict)
    segment_indices: List[int] = field(default_factory=list)
    # 节奏信息（由 segmentation.beat 填充）
    tempo: float = 0.0                 # BPM（参考平均值）
    beat_count: int = 0                # 拍数（整数）
    bar_count: int = 0                 # 小节数（整数，= beat_count // 4）
    beat_times: List[float] = field(default_factory=list)      # 逐拍时间
    downbeat_times: List[float] = field(default_factory=list)  # 每小节强拍时间
    beat_durations: List[float] = field(default_factory=list)  # 每拍的拍长（秒，可浮动）
    # kick 事件（每一次底鼓击打，含正拍 + off-beat 弱 kick）
    kick_times: List[float] = field(default_factory=list)      # kick 事件时间
    kick_strengths: List[float] = field(default_factory=list)  # kick 强度（低频能量）
    kick_intervals: List[float] = field(default_factory=list)  # 到下一个 kick 的间隔（秒）

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AnalysisResult:
    """一首歌的完整分析结果。"""

    path: str
    duration: float
    sr: int
    bpm: float = 0.0
    sections: List[Section] = field(default_factory=list)
    beat_times: Optional[np.ndarray] = None
    ssm: Optional[np.ndarray] = None
    kick_times: List[float] = field(default_factory=list)       # 全曲 kick 事件时间
    kick_strengths: List[float] = field(default_factory=list)   # 全曲 kick 强度

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "duration": round(self.duration, 3),
            "sr": self.sr,
            "bpm": round(self.bpm, 2),
            "kick_count": len(self.kick_times),
            "sections": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "label": s.label,
                    "confidence": round(s.confidence, 3),
                    "tempo": round(s.tempo, 2),
                    "beat_count": s.beat_count,
                    "bar_count": s.bar_count,
                    "beat_times": [round(t, 3) for t in s.beat_times],
                    "downbeat_times": [round(t, 3) for t in s.downbeat_times],
                    # 每拍的拍长（秒，可浮动）；以毫秒形式额外给出，便于直观查看
                    "beat_durations_sec": [round(d, 4) for d in s.beat_durations],
                    "beat_durations_ms": [round(d * 1000.0, 1) for d in s.beat_durations],
                    # kick 事件（每一次底鼓击打）
                    "kick_times": [round(t, 3) for t in s.kick_times],
                    "kick_strengths": [round(v, 3) for v in s.kick_strengths],
                    "kick_intervals_ms": [round(v * 1000.0, 1) for v in s.kick_intervals],
                    "kick_count": len(s.kick_times),
                }
                for s in self.sections
            ],
        }
