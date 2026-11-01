"""Demucs 声部分离 + 分频段能量分析能力模块。

这是「第一手配器判据」的数据地基：all-in-one 模型内部用 Demucs 分离出
bass/drums/other/vocals 四轨，但模型的**结构 embedding 是隐表示，其能量
并不等价于配器强度**（实测整首歌能量在 0.5~0.8 之间晃、无区分度）。

真正干净的配器信号来自**分离后的四轨波形**上的分频段能量：
- 判据① 贝斯进入：bass 轨 40~200Hz 低频能量，verse 段≈0、PC/chorus 段陡升；
- 判据② 鼓开镲：drums 轨 4~12kHz 高频能量（镲片/踩镲主频）阶梯抬升。

本模块职责：
1. :func:`separate_stems` —— 调 Demucs 分离四轨，带磁盘缓存（避免每个
   后处理插件重复跑分离这个重操作）；
2. :func:`band_energy` —— 对某轨波形算指定频段的逐帧能量包络；
3. :func:`segment_band_energy` —— 汇总到段落级（均值/中位数）。

注意：Demucs（htdemucs）要求**立体声**输入，单声道需复制为双声道。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


# Demucs 四轨顺序（与 allin1_infer 的 STEM_NAMES 一致）
STEM_NAMES = ("bass", "drums", "other", "vocals")

# 分频段定义（Hz）
BAND_BASS_LOW = (40.0, 200.0)    # 贝斯主频
BAND_DRUMS_HIGH = (4000.0, 12000.0)  # 镲片 / 踩镲 / 开镲高频
BAND_DRUMS_LOW = (40.0, 200.0)   # 底鼓低频（作对比：鼓一直在打）


def _load_demucs_model():
    """惰性加载 htdemucs 模型（避免 import 时开销）。"""
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.eval()
    return model


def _resolve_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def separate_stems(
    audio_path,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """用 Demucs 把音频分离成四轨，返回 {stem: 单声道波形}。

    Parameters
    ----------
    audio_path : str | Path
        音频文件路径。
    cache_dir : Path, optional
        缓存目录。给定且四轨 npy 已存在时直接读取，跳过分离。
    device : str, optional
        推理设备，默认自动选择 cuda/cpu。

    Returns
    -------
    dict[str, np.ndarray]
        {bass/drums/other/vocals: 波形（float32，单声道）}。
        同时返回 sr 见 :func:`separate_stems_with_sr`。
    """
    stems, _ = separate_stems_with_sr(audio_path, cache_dir=cache_dir, device=device)
    return stems


def separate_stems_with_sr(
    audio_path,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
):
    """同 :func:`separate_stems`，额外返回采样率 sr。"""
    audio_path = Path(audio_path)
    device = device or _resolve_device()

    # 缓存路径
    cache_files = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_files = {name: cache_dir / f"{audio_path.stem}.{name}.npy" for name in STEM_NAMES}
        if all(p.exists() for p in cache_files.values()):
            stems = {name: np.load(p) for name, p in cache_files.items()}
            # 采样率：从缓存里附带存个 sr 文件
            sr_file = cache_dir / f"{audio_path.stem}.sr.txt"
            sr = int(sr_file.read_text().strip()) if sr_file.exists() else 44100
            return stems, sr

    import torch
    import torchaudio

    from demucs.apply import apply_model

    model = _load_demucs_model()
    model.to(device)

    wav, sr = torchaudio.load(str(audio_path))
    wav = wav.to(device)
    # HTDemucs 要求立体声；单声道复制为双声道。
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, progress=False)[0]

    stems = {
        name: src.mean(dim=0).cpu().numpy().astype(np.float32)
        for name, src in zip(model.sources, sources)
    }

    if cache_files is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in stems.items():
            np.save(cache_files[name], arr)
        (cache_dir / f"{audio_path.stem}.sr.txt").write_text(str(sr))

    return stems, int(sr)


def band_energy(
    y: np.ndarray,
    sr: int,
    lo: float,
    hi: float,
    hop: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """逐帧频段能量包络（STFT 幅度平方在 [lo, hi] Hz 内求和）。

    Parameters
    ----------
    y : np.ndarray
        单声道波形。
    sr : int
        采样率。
    lo, hi : float
        频段下/上界（Hz）。
    hop, n_fft : int
        STFT 参数。

    Returns
    -------
    np.ndarray
        逐帧能量（长度 = 帧数）。
    """
    import librosa

    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= lo) & (freqs <= hi)
    return D[mask].sum(axis=0)


def segment_band_energy(
    y: np.ndarray,
    sr: int,
    lo: float,
    hi: float,
    starts: np.ndarray,
    ends: np.ndarray,
    hop: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """把逐帧频段能量汇总到段落级，返回每段的均值。

    Parameters
    ----------
    y, sr, lo, hi, hop, n_fft
        同 :func:`band_energy`。
    starts, ends : np.ndarray
        各段起止时间（秒），长度一致。

    Returns
    -------
    np.ndarray
        每段的频段能量均值。
    """
    env = band_energy(y, sr, lo, hi, hop=hop, n_fft=n_fft)
    fps = sr / hop
    out = np.zeros(len(starts), dtype=np.float64)
    for i, (s, e) in enumerate(zip(starts, ends)):
        i0, i1 = int(s * fps), int(e * fps)
        seg = env[i0:i1]
        out[i] = float(seg.mean()) if seg.size else 0.0
    return out


__all__ = [
    "STEM_NAMES",
    "BAND_BASS_LOW",
    "BAND_DRUMS_HIGH",
    "BAND_DRUMS_LOW",
    "separate_stems",
    "separate_stems_with_sr",
    "band_energy",
    "segment_band_energy",
]
