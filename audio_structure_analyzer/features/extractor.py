"""特征提取：帧级特征 + 块级聚合。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core import Segment
from ..dsp.utils import slope, zscore

# 帧级特征名（顺序固定，供下游索引）
FRAME_FEATURES = [
    "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flux",
    "chroma_mean",
    "chroma_std",
    "mfcc_1",
    "mfcc_2",
]


def extract_frame_features(y: np.ndarray, sr: int) -> np.ndarray:
    """提取帧级特征，返回 shape=(n_frames, n_features)。"""
    return _extract_frame_features_internal(y, sr)[0]


def _extract_frame_features_internal(y: np.ndarray, sr: int):
    """提取帧级特征，返回 (标量特征, 完整chroma, 完整mfcc)。

    - 标量特征 (n_frames, 8)：用于能量/分类
    - chroma (12, n_frames)：用于结构相似度
    - mfcc (13, n_frames)：用于结构相似度
    """
    import librosa

    hop = 512
    n_fft = 2048

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    flux = librosa.onset.onset_strength(S=stft, sr=sr)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, hop_length=hop)
    chroma_mean = chroma.mean(axis=0)
    chroma_std = chroma.std(axis=0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)

    n_frames = rms.shape[0]

    def _align(x: np.ndarray) -> np.ndarray:
        if x.shape[0] < n_frames:
            x = np.pad(x, (0, n_frames - x.shape[0]), mode="edge")
        return x[:n_frames]

    feats = np.stack(
        [
            _align(rms),
            _align(centroid),
            _align(bandwidth),
            _align(flux),
            _align(chroma_mean),
            _align(chroma_std),
            _align(mfcc[1]),
            _align(mfcc[2]),
        ],
        axis=1,
    )

    # 对齐 chroma 和 mfcc 的帧数
    chroma = chroma[:, :n_frames]
    mfcc = mfcc[:, :n_frames]

    return feats, chroma, mfcc


def frames_to_segments(
    y: np.ndarray,
    sr: int,
    frame_feats: np.ndarray,
    segment_sec: float = 2.0,
    hop: int = 512,
) -> List[Segment]:
    """把帧级特征聚合为块级（segment）特征，返回 Segment 列表。

    块级特征在帧级特征基础上做：均值、标准差、能量斜率等统计量。
    """
    frames_per_seg = max(1, int(round(segment_sec * sr / hop)))
    n_frames = frame_feats.shape[0]

    segments: List[Segment] = []
    for i in range(0, n_frames, frames_per_seg):
        j = min(i + frames_per_seg, n_frames)
        blk = frame_feats[i:j]

        seg_feat = []
        seg_feat.append(blk[:, 0].mean())  # rms mean
        seg_feat.append(blk[:, 0].std())   # rms std
        seg_feat.append(slope(blk[:, 0]))  # rms slope（能量爬升/回落）
        seg_feat.append(blk[:, 1].mean())  # centroid mean
        seg_feat.append(blk[:, 2].mean())  # bandwidth mean
        seg_feat.append(blk[:, 3].mean())  # flux mean
        seg_feat.append(blk[:, 4].mean())  # chroma mean
        seg_feat.append(blk[:, 5].mean())  # chroma std

        start = i * hop / sr
        end = j * hop / sr
        segments.append(
            Segment(start=float(start), end=float(end), features=np.array(seg_feat))
        )

    return segments


def extract_features(
    y: np.ndarray,
    sr: int,
    segment_sec: float = 2.0,
    hop: int = 512,
    **kwargs,
) -> Tuple[np.ndarray, List[Segment], dict]:
    """提取帧级 + 块级特征。

    Returns
    -------
    frames : np.ndarray
        帧级标量特征 (n_frames, 8)。
    segments : List[Segment]
        块级段落（含块级特征）。
    extra : dict
        额外的完整特征，含 'chroma' (12, n_frames) 与 'mfcc' (13, n_frames)，
        供结构相似度聚类使用。
    """
    frame_feats, chroma, mfcc = _extract_frame_features_internal(y, sr)
    segments = frames_to_segments(y, sr, frame_feats, segment_sec, hop)
    extra = {"chroma": chroma, "mfcc": mfcc}
    return frame_feats, segments, extra
