"""声部分离 + 结构 embedding（torch/Demucs 增强通道）。

路线 B（用户选定）：用预训练模型从音频提取比 chroma/mfcc 更丰富的结构特征，
目标是区分能量、chroma、kick 节奏都几乎相同的两个副歌段（S1 vs S2）。

实现：
1. 用 torchaudio HDEMUCS_HIGH_MUSDB 把混音分离成 4 个声部（drums/bass/other/vocals）；
2. 每个声部提取独立的能量包络、bass 声部额外算 chroma（和声）；
3. 融合成统一的帧级结构特征矩阵，供 boundary 阶段构建自相似矩阵（SSM）。

关键约束：
- torch 是 CPU-only（2.15.0+cpu），Demucs 全曲约 2 分钟，可接受，但结果需落盘缓存。
- 声部分离结果缓存到 <audio>.stems.npz，避免重复推理。
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional, Tuple

import numpy as np


# Demucs 输出的 4 个声部顺序（HDEMUCS_HIGH_MUSDB）
STEM_NAMES = ("drums", "bass", "other", "vocals")


def _cache_path(audio_path: str) -> str:
    """根据音频路径生成 stem 缓存文件路径。"""
    digest = hashlib.md5(os.path.abspath(audio_path).encode("utf-8")).hexdigest()[:16]
    d = os.path.join(os.path.dirname(os.path.abspath(audio_path)), ".asa_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"stems_{digest}.npz")


def separate_stems(
    y: np.ndarray,
    sr: int,
    audio_path: Optional[str] = None,
    use_cache: bool = True,
    segment_sec: Optional[float] = None,
    model=None,
) -> dict:
    """分离声部，返回 {stem_name: np.ndarray(mono, sr)}。

    Parameters
    ----------
    y, sr : 输入波形（sr 通常 22050，内部会重采样到 44100 喂 Demucs）。
    audio_path : 用于缓存路径；传 None 则不落盘。
    use_cache : 命中缓存则跳过推理。
    segment_sec : 若给定，只分离前 segment_sec 秒（调试用）。
    model : 预加载的 Demucs 模型（避免每次重新加载）。

    Returns
    -------
    dict
        {"drums": arr, "bass": arr, "other": arr, "vocals": arr}，mono @ sr。
        若 torch 不可用或推理失败，返回空 dict。
    """
    try:
        import torch
        import torchaudio
    except Exception:
        return {}

    cache_file = _cache_path(audio_path) if (audio_path and use_cache) else None
    if cache_file and os.path.exists(cache_file):
        data = np.load(cache_file)
        return {k: data[k] for k in STEM_NAMES if k in data}

    if model is None:
        try:
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
            model = bundle.get_model()
        except Exception:
            return {}
    model.eval()

    # Demucs 需要 44100Hz 立体声
    if sr != 44100:
        import librosa
        y44 = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr44 = 44100
    else:
        y44 = y
        sr44 = 44100

    if y44.ndim == 1:
        y44 = np.stack([y44, y44])

    wav = torch.from_numpy(y44.astype(np.float32)).float()

    # 分块处理：Demucs 对长音频内存敏感，按 60s 块滑窗（带重叠）拼接
    chunk = int(60 * sr44)
    overlap = int(2 * sr44)
    hop = chunk - overlap

    T = wav.shape[1]
    stems = {k: np.zeros(T, dtype=np.float32) for k in STEM_NAMES}
    weights = np.zeros(T, dtype=np.float32)

    if segment_sec is not None:
        T = min(T, int(segment_sec * sr44))

    start = 0
    with torch.no_grad():
        while start < T:
            end = min(start + chunk, T)
            seg = wav[:, start:end]
            # 右端 pad 到 chunk 的整数倍，避免 Demucs 尾部裁剪
            if seg.shape[1] < chunk:
                pad = chunk - seg.shape[1]
                seg = torch.nn.functional.pad(seg, (0, pad))
            out = model(seg[None])[0]  # (4, 2, T_out)
            n_out = min(end - start, out.shape[2])
            for i, name in enumerate(STEM_NAMES):
                mono = out[i].mean(dim=0).numpy()[:n_out]
                stems[name][start:start + n_out] += mono
            # 三角窗权重，重叠区平滑过渡
            win = np.ones(n_out, dtype=np.float32)
            if start > 0 and overlap > 0:
                w = min(overlap, n_out)
                win[:w] = np.linspace(0, 1, w)
            if end < T and overlap > 0:
                w = min(overlap, n_out)
                win[-w:] = np.linspace(1, 0, w)
            weights[start:start + n_out] += win
            if end >= T:
                break
            start += hop

    for name in STEM_NAMES:
        safe = np.where(weights > 1e-6, weights, 1.0)
        stems[name] = stems[name] / safe

    # 重采样回原始 sr（Demucs 输出 44100Hz），保证下游特征帧率正确
    if sr != 44100:
        import librosa
        stems = {
            name: librosa.resample(sig.astype(np.float32), orig_sr=44100, target_sr=sr)
            for name, sig in stems.items()
        }

    if cache_file:
        np.savez_compressed(cache_file, **stems)

    return stems


def stem_features(
    stems: dict,
    sr: int,
    hop: int = 512,
) -> np.ndarray:
    """从分离出的声部提取帧级结构特征矩阵。

    返回 shape=(n_frames, n_feats)，各列：
      [0]  drums 能量
      [1]  bass  能量
      [2]  other 能量
      [3]  vocals 能量
      [4]  bass  chroma 均值（和声亮度）
      [5]  bass  chroma 标准差（和声变化）
      [6]  other chroma 均值
      [7]  other chroma 标准差
      [8]  vocal 存在性（平滑后的 0/1 倾向）
    """
    import librosa

    feats_list = []
    for name in STEM_NAMES:
        sig = stems.get(name)
        if sig is None or len(sig) == 0:
            # 无声部：全 0
            n = max(1, int(len(stems.get("drums", np.zeros(sr))) / hop))
            feats_list.append(np.zeros(n, dtype=np.float32))
            continue
        rms = librosa.feature.rms(y=sig.astype(np.float32), hop_length=hop)[0]
        feats_list.append(rms)

    # bass 与 other 声部的 chroma
    for name in ("bass", "other"):
        sig = stems.get(name)
        if sig is None or len(sig) == 0:
            feats_list.append(np.zeros(feats_list[0].shape[0], dtype=np.float32))
            feats_list.append(np.zeros(feats_list[0].shape[0], dtype=np.float32))
            continue
        chroma = librosa.feature.chroma_stft(y=sig.astype(np.float32), sr=sr, hop_length=hop)
        feats_list.append(chroma.mean(axis=0))
        feats_list.append(chroma.std(axis=0))

    # vocal 存在性：vocal 能量平滑 + 归一化
    v = feats_list[3]
    if v.max() > 0:
        v_norm = v / (v.max() + 1e-9)
        # 平滑
        k = max(1, int(0.5 * sr / hop))
        v_sm = np.convolve(v_norm, np.ones(k) / k, mode="same")
        feats_list.append(v_sm)
    else:
        feats_list.append(np.zeros_like(v))

    # 对齐到统一帧数
    n_frames = max(len(f) for f in feats_list)
    aligned = []
    for f in feats_list:
        if len(f) < n_frames:
            f = np.pad(f, (0, n_frames - len(f)), mode="edge")
        aligned.append(f[:n_frames])

    return np.stack(aligned, axis=1).astype(np.float32)


def extract_embedding_features(
    y: np.ndarray,
    sr: int,
    audio_path: Optional[str] = None,
    hop: int = 512,
    use_cache: bool = True,
    model=None,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """主入口：分离声部并提取结构 embedding 特征。

    Returns
    -------
    feat : np.ndarray or None
        帧级结构特征 (n_frames, n_feats)；torch 不可用或失败时返回 None。
    stems : dict or None
        分离出的声部（mono @ sr），失败时 None。
    """
    stems = separate_stems(y, sr, audio_path=audio_path, use_cache=use_cache, model=model)
    if not stems:
        return None, None
    feat = stem_features(stems, sr, hop)
    return feat, stems
