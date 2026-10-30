"""底层 DSP 工具函数。"""

from __future__ import annotations

import numpy as np


def frame_windows(n_frames: int, hop: int = 1, frame_len: int = 1) -> np.ndarray:
    """生成帧索引窗口，便于把帧级特征聚合为块级。"""
    starts = np.arange(0, n_frames - frame_len + 1, hop)
    return starts


def slope(x: np.ndarray) -> float:
    """计算一维序列的线性斜率（用于检测能量爬升/回落）。"""
    if x.size < 2:
        return 0.0
    t = np.arange(x.size, dtype=np.float64)
    # 最小二乘斜率
    x = x.astype(np.float64)
    return float(np.polyfit(t, x, 1)[0])


def zscore(x: np.ndarray) -> np.ndarray:
    """标准归一化（去均值除标准差）。"""
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-9:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd
