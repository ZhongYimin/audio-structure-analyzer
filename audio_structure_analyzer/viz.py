"""可视化：波形 + 段落色块 + SSM。"""

from __future__ import annotations

from typing import Optional

from .core import AnalysisResult


def plot_sections(
    audio_path: str,
    result: AnalysisResult,
    out_path: Optional[str] = None,
) -> None:
    """绘制波形与段落标签色块。"""
    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    label_colors = {
        "Ap": "#4c8bf5",
        "Dp": "#f5a623",
        "S": "#e74c3c",
        "Cp": "#9b59b6",
        "Intro": "#95a5a6",
        "Interlude": "#95a5a6",
        "Outro": "#95a5a6",
        "Bridge": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    t = librosa.times_like(y, sr=sr)
    ax.plot(t, y, color="#999", linewidth=0.4, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Song Structure (with beats & bars)")

    ymax = float(max(abs(y))) if len(y) else 1.0

    for s in result.sections:
        color = label_colors.get(s.label, "#cccccc")
        ax.axvspan(s.start, s.end, color=color, alpha=0.25)
        mid = (s.start + s.end) / 2
        ax.text(
            mid,
            ymax * 0.9,
            s.label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

    # 拍点与小节标记
    for s in result.sections:
        for bt in s.beat_times:
            ax.axvline(bt, color="#666", linewidth=0.3, alpha=0.35, ymin=0, ymax=0.35)
        for db in s.downbeat_times:
            ax.axvline(db, color="#e74c3c", linewidth=0.9, alpha=0.8, ymin=0, ymax=0.5)

    ax.set_xlim(0, result.duration)
    ax.set_ylim(-ymax, ymax)

    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
