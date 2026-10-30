"""音频结构分析框架主入口：编排完整分析管道。"""

from __future__ import annotations

import numpy as np

from .core import (  # noqa: F401
    ALL_LABELS,
    LABELS,
    AnalysisResult,
    Section,
    Segment,
)

# 显式暴露子包，方便 `asa.features` / `asa.segmentation` 等访问
from . import io, dsp, features, segmentation, classification, viz  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "LABELS",
    "ALL_LABELS",
    "AnalysisResult",
    "Section",
    "Segment",
    "analyze",
    "__version__",
]


def analyze(path: str, **kwargs) -> AnalysisResult:
    """完整分析一首歌的音频，返回段落标签结果。

    Parameters
    ----------
    path : str
        音频文件路径（mp3/wav/flac 等）。
    **kwargs
        传给各阶段的可选参数（如特征、分割、分类器的超参）。

    Returns
    -------
    AnalysisResult
    """
    from .io.loader import load_audio
    from .features.extractor import extract_features
    from .segmentation.boundary import detect_boundaries
    from .classification.classifier import classify

    # 1) 加载与预处理
    y, sr = load_audio(path)
    # 2) 特征提取（帧级 + 块级 + 完整chroma/mfcc）
    frames, segments, extra = extract_features(y, sr, **kwargs)
    # 2b) 声部分离 embedding（torch/Demucs 增强，可选；失败则回退 chroma）
    embed_feat = None
    if kwargs.get("use_embedding", True):
        try:
            from .features.embedding import extract_embedding_features
            embed_feat, _ = extract_embedding_features(
                y, sr, audio_path=path,
                use_cache=kwargs.get("use_stem_cache", True),
            )
        except Exception:
            embed_feat = None
    # 3) 结构分割（优先用 embedding 特征，回退 chroma）
    boundaries = detect_boundaries(frames, sr, y=y, embed_feat=embed_feat, **kwargs)
    # 4) 段落归并 + 分类（传入 y/sr 以便计算真实段落能量与重复度）
    sections = classify(segments, boundaries, frames, extra=extra, y=y, sr=sr, **kwargs)
    # 5) 节奏分析（BPM + kick 事件 + 拍点，含浮动拍长）
    from .segmentation.beat import (
        analyze_beats,
        detect_bpm,
        detect_kicks,
        _kick_envelope_full,
        _kick_strength_at_env,
    )

    tempo = kwargs.get("tempo", None)
    if tempo is None:
        tempo = detect_bpm(y, sr)

    # 全曲 kick 事件（每一次底鼓击打）
    kick_times = detect_kicks(y, sr)
    kick_env = _kick_envelope_full(y, sr)
    kick_strengths = [
        _kick_strength_at_env(kick_env, sr, 512, float(t)) for t in kick_times
    ]

    beat_infos = analyze_beats(y, sr, sections, tempo=tempo, **kwargs)
    for s, bi in zip(sections, beat_infos):
        # 段落边界对齐到拍：开端必然是一个拍子
        s.start = bi.start
        s.end = bi.end
        s.tempo = bi.beats_per_minute
        s.beat_count = bi.beat_count
        s.bar_count = bi.bar_count
        s.beat_times = bi.beat_times
        s.beat_durations = [b.duration for b in bi.beats]
        # 该段落内的 kick 事件
        seg_kick = kick_times[(kick_times >= s.start) & (kick_times < s.end)]
        s.kick_times = [float(t) for t in seg_kick]
        s.kick_strengths = [
            _kick_strength_at_env(kick_env, sr, 512, float(t)) for t in seg_kick
        ]
        if len(seg_kick) > 1:
            s.kick_intervals = list(np.diff(seg_kick))
            s.kick_intervals.append(float(s.kick_intervals[-1]))
        else:
            s.kick_intervals = [60.0 / tempo] * len(seg_kick)  # 0/1 个 kick 时按拍长占位

    result = AnalysisResult(
        path=path,
        duration=float(len(y)) / sr,
        sr=sr,
        bpm=float(tempo),
        sections=sections,
        kick_times=[float(t) for t in kick_times],
        kick_strengths=[float(v) for v in kick_strengths],
    )
    return result
