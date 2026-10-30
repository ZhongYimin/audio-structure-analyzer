"""节奏分析（频谱 onset 相位 + 均匀拍网格，v3）。

核心设计（修正「用 kick 单点锚定拍」导致的错拍）：

1. **BPM 检测**（`detect_bpm`）：低频（kick/bass）能量自相关为主，onset 交叉验证。
   稳定测出 123 BPM 等真实 tempo。

2. **拍网格相位估计**（`_estimate_phase`）：
   **不用 kick 单点**（kick 有 off-beat 弱击打，会把网格拉偏），
   而是用**全频谱 onset_strength**（整体频谱能量的起音，即「一拍」的概念）
   做相位搜索，找到使 onset 峰值最对齐网格的 phase offset。
   —— 这符合用户「用频谱特征把握『一拍』而非 kick 单独」的直觉。

3. **均匀拍网格**（`track_beats`）：锁定 BPM + phase 后**全局均匀推进**，
   不再逐拍吸附 kick（避免 off-beat 拉偏），拍长恒定 ≈ 60/BPM。

4. **kick 事件检测**（`detect_kicks`）：低频（40~120Hz）能量峰值检测，
   得到每一次真实 kick（含正拍 + off-beat 弱击打），作为**独立事件列表**输出，
   与拍网格解耦。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Beat:
    """一个拍点，携带该拍到下一拍的实际时长（拍长，可浮动）。"""

    time: float                    # 拍点时间（秒）
    duration: float                # 本拍到下一拍的实际时长（秒，可浮动）
    has_kick: bool = False         # 该拍是否命中真实 kick（底鼓击打）
    kick_strength: float = 0.0     # 该拍的 kick 强度（低频能量，无 kick 为 0）

    def to_dict(self) -> dict:
        return {
            "time": round(self.time, 4),
            "duration_ms": round(self.duration * 1000.0, 2),
            "has_kick": self.has_kick,
            "kick_strength": round(self.kick_strength, 3),
        }


@dataclass
class BeatInfo:
    """一个段落内的节奏信息。"""

    start: float                          # 段落起始时间（对齐到拍后）
    end: float                            # 段落结束时间（对齐到拍后）
    beats_per_minute: float               # BPM
    beat_count: int                       # 段落内拍数（整数）
    bar_count: int                        # 段落内小节数（4拍/小节，整数）
    beats: List[Beat] = field(default_factory=list)        # 逐拍（含拍长/kick）
    beat_times: List[float] = field(default_factory=list)  # 每拍时间点（秒）
    kick_times: List[float] = field(default_factory=list)  # 命中 kick 的拍（秒）


# ---------------------------------------------------------------------------
# BPM 检测
# ---------------------------------------------------------------------------

def _autocorrelate_period(env: np.ndarray, sr: int, hop: int,
                          lo_sec: float, hi_sec: float) -> tuple:
    """在 [lo_sec, hi_sec] 拍长区间内，用自相关找主周期。"""
    import librosa

    ac = librosa.autocorrelate(env, max_size=int(2.0 * sr / hop))
    frame_dur = hop / sr
    lo = int(lo_sec / frame_dur)
    hi = int(hi_sec / frame_dur)
    hi = min(hi, len(ac))
    if lo >= hi:
        return (lo_sec + hi_sec) / 2.0, 0.0
    seg = ac[lo:hi]
    i = lo + int(np.argmax(seg))
    return i * frame_dur, float(ac[i])


def detect_bpm(y: np.ndarray, sr: int, hop: int = 512) -> float:
    """独立实现的 BPM 检测器。

    主通道：低频能量（kick/bass，<120Hz）自相关——kick 最规整，octave 歧义最小。
    辅通道：onset 强度包络自相关（窄拍长区间交叉验证）。
    """
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bass_env = S[freqs < 120].sum(axis=0)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    p_bass, _ = _autocorrelate_period(bass_env, sr, hop, 0.40, 0.58)
    p_onset, _ = _autocorrelate_period(onset_env, sr, hop, 0.40, 0.58)

    bpm_bass = 60.0 / p_bass
    bpm_onset = 60.0 / p_onset

    if abs(bpm_bass - bpm_onset) / max(bpm_bass, 1e-9) < 0.08:
        return float(bpm_bass)
    if 60.0 < bpm_bass < 200.0:
        return float(bpm_bass)
    if 60.0 < bpm_onset < 200.0:
        return float(bpm_onset)
    return 120.0


# ---------------------------------------------------------------------------
# kick 事件检测
# ---------------------------------------------------------------------------

def _kick_envelope(y: np.ndarray, sr: int, hop: int = 512,
                   lo_hz: float = 40.0, hi_hz: float = 120.0) -> np.ndarray:
    """低频（kick drum）能量包络。"""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    return S[mask].sum(axis=0)


def detect_kicks(
    y: np.ndarray,
    sr: int,
    hop: int = 512,
    min_distance_sec: float = 0.20,
    rel_prominence: float = 0.05,
    lo_hz: float = 40.0,
    hi_hz: float = 120.0,
) -> np.ndarray:
    """检测每一次真实的 kick drum（底鼓）击打。

    Returns
    -------
    np.ndarray kick 事件时间点（秒），升序。
    """
    from scipy.signal import find_peaks

    kick_env = _kick_envelope(y, sr, hop, lo_hz, hi_hz)
    times = np.arange(len(kick_env)) * hop / sr

    if kick_env.max() <= 0:
        return np.asarray([], dtype=float)

    peaks, _ = find_peaks(
        kick_env,
        distance=int(min_distance_sec * sr / hop),
        prominence=rel_prominence * float(kick_env.max()),
    )
    return times[peaks].astype(float)


# ---------------------------------------------------------------------------
# 拍网格 + kick 对齐
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 拍网格相位估计（基于全频谱 onset，非 kick 单点）
# ---------------------------------------------------------------------------

def _estimate_phase(y: np.ndarray, sr: int, hop: int, tempo: float) -> float:
    """估计拍网格的相位偏移（秒），用全频谱 onset 起音的圆平均相位。

    为什么用 onset 而非 kick 峰值：
    - kick 峰值检测在副歌段会因 off-beat 弱击打 + 低频能量形态变化而系统性偏移
      （实测 intro 相位 0.093s vs 副歌 0.139s，差 46ms）；
    - onset_strength（全频谱起音）反映「一拍」的感知起音，不受低频 off-beat 干扰，
      且与 intro kick 相位独立验证一致（0.094s ≈ 0.093s）。

    Returns
    -------
    float 相位偏移（秒），即第 0 拍的时间点。
    """
    import librosa
    from scipy.signal import find_peaks

    period = 60.0 / tempo
    duration = len(y) / sr

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop)

    # onset 峰值（较宽松）
    thr = float(onset.max()) * 0.20
    peaks, _ = find_peaks(onset, distance=max(1, int(0.20 * sr / hop)), height=thr)
    pt = times[peaks]

    if len(pt) < 5:
        return 0.0

    # 圆平均相位（对 off-beat / 漏检 robust：正拍 onset 相位一致，噪声均匀散布）
    phases = pt % period
    ang = phases / period * 2.0 * np.pi
    mean_phase = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / (2.0 * np.pi) * period
    if mean_phase < 0:
        mean_phase += period
    return mean_phase


def track_beats(
    y: np.ndarray,
    sr: int,
    hop: int = 512,
    tempo: Optional[float] = None,
    **kwargs,
) -> tuple:
    """拍点跟踪（均匀网格 + onset 相位），返回 (tempo_bpm, beat_times, kick_times)。

    1. 锁定 BPM；
    2. 用全频谱 onset 估计相位偏移（拍网格第 0 拍位置）；
    3. 全局均匀推进拍网格（不再吸附 kick，避免 off-beat 拉偏）；
    4. 独立检测 kick 事件（含正拍 + off-beat）。
    """
    if tempo is None:
        tempo = detect_bpm(y, sr, hop)
    tempo = float(tempo)

    period = 60.0 / tempo
    duration = len(y) / sr

    kick_times = detect_kicks(y, sr, hop)

    # 相位估计（用频谱 onset，非 kick）
    offset = _estimate_phase(y, sr, hop, tempo)

    # 均匀网格：从 offset 起，每 period 一拍
    n_beats = int((duration - offset) / period) + 2
    beat_arr = np.asarray([offset + i * period for i in range(n_beats)], dtype=float)
    beat_arr = beat_arr[beat_arr < duration]

    return tempo, beat_arr, kick_times.astype(float)


# ---------------------------------------------------------------------------
# 段落节奏切分
# ---------------------------------------------------------------------------

def _kick_strength_at(y: np.ndarray, sr: int, hop: int, t: float,
                      lo_hz: float = 40.0, hi_hz: float = 120.0) -> float:
    """返回时间 t 处的低频能量（kick 强度）。"""
    env = _kick_envelope(y, sr, hop, lo_hz, hi_hz)
    i = int(t * sr / hop)
    i = max(0, min(i, len(env) - 1))
    return float(env[i])


def _build_beats(
    y: np.ndarray, sr: int, hop: int,
    beat_times: np.ndarray, kick_times: np.ndarray,
    kick_match_tol_sec: float = 0.06,
) -> List[Beat]:
    """把拍点时间序列转为 Beat 列表，标注每拍是否命中 kick + kick 强度。"""
    n = len(beat_times)
    if n == 0:
        return []

    durations = np.diff(beat_times)
    last_dur = float(np.median(durations)) if len(durations) else 0.6
    durations = np.append(durations, last_dur)

    beats: List[Beat] = []
    for t, d in zip(beat_times, durations):
        # 该拍是否命中 kick
        hit = False
        if len(kick_times):
            dmin = np.abs(kick_times - t).min()
            hit = dmin <= kick_match_tol_sec
        strength = _kick_strength_at(y, sr, hop, float(t)) if hit else 0.0
        beats.append(
            Beat(
                time=float(t),
                duration=float(d),
                has_kick=hit,
                kick_strength=strength,
            )
        )
    return beats


def _kick_envelope_full(y: np.ndarray, sr: int, hop: int = 512,
                        lo_hz: float = 40.0, hi_hz: float = 120.0) -> np.ndarray:
    """低频（kick drum）能量包络（全曲），`_kick_envelope` 的语义别名。"""
    return _kick_envelope(y, sr, hop, lo_hz, hi_hz)


def _kick_strength_at_env(env: np.ndarray, sr: int, hop: int, t: float) -> float:
    """从预计算的能量包络取 t 处的 kick 强度。"""
    i = int(t * sr / hop)
    i = max(0, min(i, len(env) - 1))
    return float(env[i])


def analyze_beats(
    y: np.ndarray,
    sr: int,
    sections: List,
    hop: int = 512,
    tempo: Optional[float] = None,
    beats_per_bar: int = 4,
    **kwargs,
) -> List[BeatInfo]:
    """为每个段落计算节奏信息（拍点 + kick 事件），边界对齐到拍。"""
    tempo_bpm, beat_times, kick_times = track_beats(y, sr, hop, tempo=tempo)

    all_beats = _build_beats(y, sr, hop, beat_times, kick_times)
    n_beats = len(all_beats)

    # 全曲 kick 强度包络（供填充段落 kick 强度）
    kick_env = _kick_envelope_full(y, sr, hop)

    infos: List[BeatInfo] = []
    n = len(sections)

    cursor = 0
    for i, s in enumerate(sections):
        if cursor >= n_beats:
            infos.append(
                BeatInfo(
                    start=float(s.start), end=float(s.end),
                    beats_per_minute=tempo_bpm, beat_count=0, bar_count=0,
                )
            )
            continue

        # 起始拍：离段落 start 最近的拍点
        si = cursor
        best_d = float("inf")
        for j in range(cursor, n_beats):
            d = abs(all_beats[j].time - float(s.start))
            if d < best_d:
                best_d = d
                si = j
            else:
                break

        # 结束拍
        if i == n - 1:
            ei = n_beats - 1
        else:
            ei = si
            best_d = float("inf")
            for j in range(si, n_beats):
                d = abs(all_beats[j].time - float(s.end))
                if d < best_d:
                    best_d = d
                    ei = j
                elif d > best_d + 0.5 * (60.0 / tempo_bpm):
                    break

        seg_beats = all_beats[si: ei + 1]
        beat_count = len(seg_beats)

        aligned_start = float(seg_beats[0].time) if seg_beats else float(s.start)
        aligned_end = float(seg_beats[-1].time + seg_beats[-1].duration) if seg_beats else float(s.end)

        # 该段落内的 kick 事件（真实底鼓击打，含正拍 + off-beat）
        seg_kick = kick_times[(kick_times >= aligned_start) & (kick_times < aligned_end)]
        seg_kick_strengths = [
            _kick_strength_at_env(kick_env, sr, hop, float(t)) for t in seg_kick
        ]
        # 每个 kick 到下一个 kick 的间隔
        if len(seg_kick) > 1:
            seg_kick_intervals = list(np.diff(seg_kick))
            seg_kick_intervals.append(float(seg_kick_intervals[-1]))  # 末个用前一个间隔补
        else:
            seg_kick_intervals = [60.0 / tempo_bpm] * len(seg_kick)

        infos.append(
            BeatInfo(
                start=aligned_start,
                end=aligned_end,
                beats_per_minute=tempo_bpm,
                beat_count=beat_count,
                bar_count=beat_count // beats_per_bar,
                beats=list(seg_beats),
                beat_times=[round(float(b.time), 4) for b in seg_beats],
                kick_times=[round(float(t), 4) for t in seg_kick],
            )
        )
        # 补充 kick 强度/间隔到 BeatInfo（临时挂在 beats 之外的属性不好，用独立返回）
        cursor = ei + 1

    # 把 kick 强度/间隔写回 infos（通过扩展 BeatInfo 字段不方便，改用附加属性）
    # 这里直接返回 infos，kick 强度由上层从全曲 kick 序列切分
    return infos
