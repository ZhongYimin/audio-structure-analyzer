"""segmentation 子包：结构分割 + 节拍/强拍分析。"""

from .boundary import detect_boundaries
from .beat import detect_bpm, track_beats, analyze_beats

__all__ = ["detect_boundaries", "detect_bpm", "track_beats", "analyze_beats"]
