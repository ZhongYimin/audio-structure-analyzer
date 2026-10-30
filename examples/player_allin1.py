"""可视化播放器：展示 all-in-one 框架的分析结果（段落 + 强拍 + 拍）。

用法：
    python examples/player_allin1.py path/to/song.wav --json test_audio/allin1/guy_contact.json

交互：
    - 空格：暂停/继续
    - 左右方向键：快退/快进 5 秒
    - 上/下方向键：调整播放速度
    - q 或 Esc：退出

窗口内容：
    - 波形图（灰色）
    - 段落色块 + 语义标签（intro/inst/chorus/outro/...）
    - 强拍 downbeat（红粗线）+ 拍 beat（灰细线）
    - 红色游标：当前播放位置
    - 顶部信息栏：时间 / 段落标签 / 拍号
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


# all-in-one (Harmonix) 标签颜色
LABEL_COLORS = {
    "start": "#b0b0b0",
    "end": "#b0b0b0",
    "intro": "#4c8bf5",   # 蓝
    "outro": "#4c8bf5",   # 蓝
    "verse": "#f5a623",   # 橙
    "chorus": "#e74c3c",  # 红
    "bridge": "#9b59b6",  # 紫
    "break": "#16a085",   # 青绿
    "inst": "#27ae60",    # 绿
    "solo": "#e67e22",    # 橙红
}


def mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="all-in-one 结果可视化播放器")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--json", help="all-in-one 结果 JSON 文件路径")
    args = parser.parse_args()

    # --- 1. 加载分析结果 ---
    if args.json and os.path.exists(args.json):
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = data["segments"]
        bpm = data.get("bpm", 0)
    else:
        print("未提供 JSON，请先运行 all-in-one 分析生成结果。")
        return 1

    beats = data.get("beats", [])
    downbeats = data.get("downbeats", [])
    beat_positions = data.get("beat_positions", [])

    # --- 2. 加载音频 ---
    import librosa
    y, sr = librosa.load(args.audio, sr=22050, mono=True)
    duration = len(y) / sr
    y = y.astype(np.float32)

    # --- 3. 提取段落信息 ---
    seg_starts = [s["start"] for s in segments]
    seg_ends = [s["end"] for s in segments]
    seg_labels = [s["label"] for s in segments]

    # --- 4. matplotlib 设置 ---
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.canvas.manager.set_window_title("all-in-one 结构播放器")

    t = librosa.times_like(y, sr=sr)
    ymax = float(max(abs(y))) if len(y) else 1.0

    # 波形
    ax.plot(t, y, color="#888", linewidth=0.4, alpha=0.8, zorder=1)

    # 段落色块 + 标签
    for i in range(len(segments)):
        color = LABEL_COLORS.get(seg_labels[i], "#cccccc")
        ax.axvspan(seg_starts[i], seg_ends[i], color=color, alpha=0.22, zorder=2)
        mid = (seg_starts[i] + seg_ends[i]) / 2
        # 太短的段落标签竖着放
        ax.text(
            mid, ymax * 0.90, seg_labels[i],
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=color, zorder=5,
        )
        ax.text(
            mid, ymax * 0.74, f"{seg_starts[i]:.1f}s",
            ha="center", va="center", fontsize=8, color="#555", zorder=5,
        )

    # 拍（灰细线，画在波形下半部分）
    for b in beats:
        ax.axvline(b, color="#999", linewidth=0.3, alpha=0.25, ymin=0.04, ymax=0.12, zorder=3)
    # 强拍 downbeat（红粗线）
    for db in downbeats:
        ax.axvline(db, color="#e74c3c", linewidth=1.2, alpha=0.65, ymin=0.04, ymax=0.16, zorder=3)

    # 播放游标
    cursor = ax.axvline(0, color="#000", linewidth=2, alpha=0.9, zorder=10)

    # 当前段落高亮描边
    highlight = ax.axvspan(0, 0, alpha=0.15, facecolor="#000", edgecolor="#000", linewidth=2.5, zorder=4)
    highlight.set_visible(False)

    # 信息文本
    info = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top", color="#111",
        bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9),
    )

    ax.set_xlim(0, duration)
    ax.set_ylim(-ymax * 1.05, ymax * 1.05)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("振幅")
    ax.set_title(
        f"all-in-one 结构分析 — BPM={bpm}  拍={len(beats)}  强拍={len(downbeats)}  "
        f"[空格]播放/暂停 [←→]快退/快进 [↑↓]变速 [q]退出"
    )

    # --- 5. 播放状态 ---
    state = {
        "pos": 0.0,
        "sample_pos": 0,   # 音频回调维护的样本级位置（唯一真相源）
        "playing": False,
        "stream": None,
        "speed": 1.0,
        "prev_seg_idx": -1,
        "prev_beat_idx": -1,
        "_last_draw": 0.0,
    }

    def get_seg_idx(pos):
        for i in range(len(segments)):
            if seg_starts[i] <= pos < seg_ends[i]:
                return i
        if pos < seg_starts[0]:
            return 0
        return len(segments) - 1

    def find_beat(pos):
        bi = -1
        for i, b in enumerate(beats):
            if b <= pos:
                bi = i
            else:
                break
        return bi

    def log_seg_change(idx):
        s = segments[idx]
        print(
            f"\n[段落切换] {mmss(s['start'])} -> {mmss(s['end'])}  标签={s['label']}"
        )

    def log_beat(idx):
        bt = beats[idx]
        pos = beat_positions[idx] if idx < len(beat_positions) else 0
        is_downbeat = pos == 1
        marker = "▍强拍" if is_downbeat else "· 拍"
        print(f"      {marker} @ {bt:.2f}s  (第{pos}拍)")

    # --- 6. 音频回调（独立线程） ---
    # 关键：用「回调实际已写入声卡缓冲的样本数」作为唯一播放位置来源，
    # 并降低 latency，让屏幕文字与耳朵听到的音频尽量同步。
    def callback(outdata, frames, time_info, status):
        if status:
            print("播放状态:", status, file=sys.stderr)
        if not state["playing"]:
            outdata[:] = 0
            return
        # 从当前位置按速度取音频；speed=1 时 step==frames，逐样本推进。
        i0 = state["sample_pos"]
        step = int(frames * state["speed"])
        i1 = i0 + step
        if i0 >= len(y):
            outdata[:] = 0
            state["playing"] = False
            return
        seg = y[i0:i1]
        n = min(frames, len(seg))
        outdata[:n, 0] = seg[:n]
        if n < frames:
            outdata[n:, 0] = 0
        # 用实际写出的样本数推进（而非 speed*step），保证 pos 与输出严格一致
        state["sample_pos"] += step
        state["pos"] = state["sample_pos"] / sr

    def ensure_stream():
        import sounddevice as sd
        if state["stream"] is None or state["stream"].closed:
            # 低延迟 + 中等块：既保证文字/音频同步，又留足够缓冲防 underrun 卡顿
            state["stream"] = sd.OutputStream(
                samplerate=sr, channels=1, dtype="float32",
                callback=callback, blocksize=2048, latency="low",
            )
            state["stream"].start()

    def update():
        if state["playing"] and state["pos"] >= duration:
            state["pos"] = duration
            state["playing"] = False
            if state["stream"] is not None:
                state["stream"].stop()

        pos = state["pos"]
        cursor.set_xdata([pos, pos])

        idx = get_seg_idx(pos)
        if idx != state["prev_seg_idx"]:
            log_seg_change(idx)
            state["prev_seg_idx"] = idx

        highlight.set_visible(True)
        highlight.set_x(seg_starts[idx])
        highlight.set_width(seg_ends[idx] - seg_starts[idx])

        bi = find_beat(pos)
        if bi != state["prev_beat_idx"] and bi >= 0:
            log_beat(bi)
            state["prev_beat_idx"] = bi

        s = segments[idx]
        pos_label = beat_positions[bi] if 0 <= bi < len(beat_positions) else 0
        info.set_text(
            f"时间 {mmss(pos)} / {mmss(duration)}    "
            f"段落: {s['label']}    "
            f"拍 {bi+1}/{len(beats)} (第{pos_label}拍)    "
            f"BPM {bpm}"
        )

        now = time.perf_counter()
        if now - state.get("_last_draw", 0.0) >= 0.033:
            fig.canvas.draw_idle()
            state["_last_draw"] = now

    def on_key(event):
        if event.key == " ":
            if state["playing"]:
                state["playing"] = False
                state["stream"].stop()
            else:
                ensure_stream()
                state["playing"] = True
                if state["stream"].stopped:
                    state["stream"].start()
        elif event.key == "right":
            state["pos"] = min(duration, state["pos"] + 5)
            state["sample_pos"] = int(state["pos"] * sr)
            state["prev_beat_idx"] = find_beat(state["pos"])
        elif event.key == "left":
            state["pos"] = max(0, state["pos"] - 5)
            state["sample_pos"] = int(state["pos"] * sr)
            state["prev_beat_idx"] = find_beat(state["pos"])
        elif event.key == "q" or event.key == "escape":
            plt.close(fig)
        elif event.key == "up":
            state["speed"] = min(2.0, state["speed"] + 0.25)
            print(f"速度: {state['speed']:.2f}x")
        elif event.key == "down":
            state["speed"] = max(0.25, state["speed"] - 0.25)
            print(f"速度: {state['speed']:.2f}x")

    fig.canvas.mpl_connect("key_press_event", on_key)

    timer = fig.canvas.new_timer(interval=20)
    timer.add_callback(update)
    timer.start()

    print("=" * 60)
    print(f"歌曲: {os.path.basename(args.audio)}  时长: {mmss(duration)}")
    print(f"BPM: {bpm}  段落数: {len(segments)}  拍点数: {len(beats)}  强拍: {len(downbeats)}")
    print("-" * 60)
    print("段落结构 (all-in-one):")
    for s in segments:
        print(f"  {s['label']:8s} {s['start']:7.2f} -> {s['end']:7.2f}s")
    print("-" * 60)
    print("操作: [空格]播放/暂停  [←→]快退/快进5s  [↑↓]变速  [q]退出")
    print("=" * 60)

    plt.show()

    if state["stream"] is not None:
        state["stream"].stop()
        state["stream"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
