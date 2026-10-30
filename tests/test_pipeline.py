"""基础冒烟测试：验证包可导入、管道可对合成音频跑通。"""

import numpy as np
import pytest


def test_package_imports():
    import audio_structure_analyzer as asa

    assert asa.LABELS == ("Ap", "Dp", "S", "Cp")


def test_analyze_synthetic_tone():
    """用合成信号跑通完整管道，验证不报错且产出段落。"""
    import audio_structure_analyzer as asa

    sr = 22050
    t = np.linspace(0, 30, sr * 30)
    # 合成一段带节奏变化的信号
    y = 0.5 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    y = y.astype(np.float32)

    frames, segments, extra = asa.features.extractor.extract_features(y, sr)
    boundaries = asa.segmentation.boundary.detect_boundaries(frames, sr)
    sections = asa.classification.classifier.classify(
        segments, boundaries, frames, extra=extra
    )

    assert isinstance(sections, list)
    # 每个段落都有标签
    for s in sections:
        assert s.label in asa.ALL_LABELS
