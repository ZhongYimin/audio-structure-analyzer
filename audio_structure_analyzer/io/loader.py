"""音频加载与预处理。"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def load_audio(
    path: str,
    target_sr: int = 22050,
    mono: bool = True,
    normalize_lufs: float = -14.0,
) -> Tuple[np.ndarray, int]:
    """加载音频并做预处理。

    Parameters
    ----------
    path : str
        音频文件路径。
    target_sr : int
        目标采样率（默认 22050，够用且省算力）。
    mono : bool
        是否转单声道。
    normalize_lufs : float
        目标响度（LUFS），< -70 时跳过响度归一。

    Returns
    -------
    y : np.ndarray
        波形（float32，[-1, 1]）。
    sr : int
        采样率。
    """
    import librosa
    import soundfile as sf

    # 用 soundfile 读取原始数据（对 VBR mp3 更稳健，避免 librosa 因 Xing 头不准而提前截断），
    # 再手动重采样到目标采样率。mp3 之外（wav/flac）也走同一路径。
    data, native_sr = sf.read(path, dtype="float32", always_2d=False)

    # 转单声道
    if data.ndim > 1:
        data = data.mean(axis=1)

    # 重采样到目标采样率
    if native_sr != target_sr:
        y = librosa.resample(data, orig_sr=native_sr, target_sr=target_sr)
        sr = target_sr
    else:
        y = data
        sr = native_sr

    # 去掉首尾静音
    y, _ = librosa.effects.trim(y, top_db=30)

    # 响度归一（简化版：RMS 归一到目标，LUFS 精确计算后续可替换为 pyloudnorm）
    if normalize_lufs > -70.0 and y.size:
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms > 1e-6:
            target_rms = 10 ** (normalize_lufs / 20.0)
            y = y * (target_rms / rms)
            y = np.clip(y, -1.0, 1.0).astype(np.float32)

    return y.astype(np.float32), sr
