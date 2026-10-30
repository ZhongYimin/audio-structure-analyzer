"""可视化播放器：播放音频并实时显示段落/拍子信息，验证分析结果是否对齐。

用法：
    python examples/player.py path/to/song.mp3 --json result.json

交互：
    - 空格：暂停/继续
    - 左右方向键：快退/快进 5 秒
    - 上/下方向键：调整播放速度
    - q 或 Esc：退出

窗口内容：
    - 波形图（灰色）
    - 段落色块（Ap蓝/Dp橙/S红/Cp紫）
    - 当前段落高亮（描边）
    - 拍点（灰细线）+ 小节强拍（红粗线）
    - 红色游标：当前播放位置
    - 顶部信息栏：当前时间 / 段落标签 / 小节号 / 拍号
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


# 段落标签颜色
LABEL_COLORS = {
    "Ap": "#4c8bf5",
    "Dp": "#f5a623",
    "S": "#e74c3c",
    "Cp": "#9b59b6",
    "Intro": "#95a5a6",
    "Interlude": "#95a5a6",
    "Outro": "#95a5a6",
    "Bridge": "#95a5a6",
}


def mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="可视化播放器（验证段落/拍子对齐）")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--json", help="分析结果 JSON 文件路径（可选，缺省自动分析）")
    parser.add_argument("--bpm", type=float, default=None, help="手动指定 BPM（可选）")
    args = parser.parse_args()

    # 项目根目录加入 sys.path
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import librosa
    import sounddevice as sd

    from audio_structure_analyzer import analyze

    # --- 1. 加载分析结果 ---
    if args.json and os.path.exists(args.json):
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        sections = data["sections"]
        sr = data.get("sr", 22050)
    else:
        print("未提供 JSON，正在实时分析……")
        if args.bpm:
            result = analyze(args.audio, tempo=args.bpm)
        else:
            result = analyze(args.audio)
        sections = result.to_dict()["sections"]
        sr = result.sr

    # --- 2. 加载音频 ---
    y, sr = librosa.load(args.audio, sr=22050, mono=True)
    duration = len(y) / sr
    y = y.astype(np.float32)

    # --- 3. 提取段落与拍点信息 ---
    sec_starts = [s["start"] for s in sections]
    sec_ends = [s["end"] for s in sections]
    sec_labels = [s["label"] for s in sections]
    sec_bars = [s.get("bar_count", 0) for s in sections]
    sec_beats = [s.get("beat_count", 0) for s in sections]
    sec_kicks = [s.get("kick_times", []) for s in sections]
    sec_kick_strengths = [s.get("kick_strengths", []) for s in sections]

    # 全曲所有拍点（用于显示）
    all_beats = []
    for s in sections:
        all_beats.extend(s.get("beat_times", []))
    all_beats = sorted(set(round(b, 3) for b in all_beats))

    # 全曲所有拍长（每拍的实际时长，可浮动），与拍点一一对应
    all_durations_ms = []
    for s in sections:
        durs = s.get("beat_durations_ms", [])
        all_durations_ms.extend(durs)

    # 全曲所有 kick 事件（每一次底鼓击打）
    all_kicks = []
    for s in sections:
        all_kicks.extend(s.get("kick_times", []))
    all_kicks = sorted(set(round(b, 3) for b in all_kicks))

    # --- 4. matplotlib 设置 ---
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    # 设置中文字体，避免中文显示为方块
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.canvas.manager.set_window_title("歌曲结构播放器 - 验证段落/拍子对齐")

    t = librosa.times_like(y, sr=sr)
    ymax = float(max(abs(y))) if len(y) else 1.0

    # 波形
    ax.plot(t, y, color="#888", linewidth=0.4, alpha=0.8, zorder=1)

    # 段落色块 + 标签
    sec_patches = []
    for i in range(len(sections)):
        color = LABEL_COLORS.get(sec_labels[i], "#cccccc")
        p = ax.axvspan(sec_starts[i], sec_ends[i], color=color, alpha=0.22, zorder=2)
        sec_patches.append(p)
        mid = (sec_starts[i] + sec_ends[i]) / 2
        ax.text(
            mid, ymax * 0.92, sec_labels[i],
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=color, zorder=5,
        )
        ax.text(
            mid, ymax * 0.78, f"{sec_bars[i]}小节·{sec_beats[i]}拍·{len(sec_kicks[i])}kick",
            ha="center", va="center", fontsize=8, color="#555", zorder=5,
        )

    # 拍点（灰细线）
    for b in all_beats:
        ax.axvline(b, color="#999", linewidth=0.3, alpha=0.25, ymin=0.72, ymax=0.78, zorder=3)
    # kick 事件（红色竖线，每一次底鼓击打）
    for k in all_kicks:
        ax.axvline(k, color="#e74c3c", linewidth=0.9, alpha=0.6, ymin=0.60, ymax=0.66, zorder=3)

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
    ax.set_title("按 空格 暂停/继续 | ←/→ 快退/快进 | q 退出")

    # --- 5. 播放状态 ---
    state = {
        "pos": 0.0,           # 当前播放位置（秒）
        "playing": False,
        "stream": None,
        "start_time": None,   # 播放开始时的 time.perf_counter
        "start_pos": 0.0,     # 播放开始时的 pos
        "speed": 1.0,
        "prev_section_idx": -1,
        "prev_beat_idx": -1,
        "prev_kick_idx": -1,
        "_last_draw": 0.0,
    }

    def get_section_idx(pos):
        for i in range(len(sections)):
            if sec_starts[i] <= pos < sec_ends[i]:
                return i
        # 越界
        if pos < sec_starts[0]:
            return 0
        return len(sections) - 1

    def _reset_log_state():
        """快进/快退后重置日志指针，避免打印回跳的历史拍/kick。"""
        state["prev_beat_idx"] = find_beat(state["pos"])
        state["prev_kick_idx"] = find_kick(state["pos"])

    def find_beat(pos):
        """返回当前最近的拍点索引（用于显示拍号）。"""
        bi = -1
        for i, b in enumerate(all_beats):
            if b <= pos:
                bi = i
            else:
                break
        return bi

    def find_kick(pos):
        """返回当前最近的 kick 事件索引（用于显示）。"""
        ki = -1
        for i, k in enumerate(all_kicks):
            if k <= pos:
                ki = i
            else:
                break
        return ki

    def log_section_change(idx):
        s = sections[idx]
        print(
            f"\n[段落切换] {mmss(s['start'])} -> {mmss(s['end'])}  标签={s['label']}  "
            f"拍数={s.get('beat_count',0)}  小节数={s.get('bar_count',0)}  "
            f"kick数={s.get('kick_count',0)}"
        )

    def log_beat(idx):
        """打印拍子信息（每拍），含该拍实际拍长。"""
        beat_time = all_beats[idx]
        dur_ms = all_durations_ms[idx] if idx < len(all_durations_ms) else None
        dstr = f" 拍长 {dur_ms:.0f}ms" if dur_ms else ""
        print(f"      · 拍 @ {beat_time:.2f}s{dstr}")

    def log_kick(idx):
        """打印 kick 事件（每一次底鼓击打）。"""
        kick_time = all_kicks[idx]
        print(f"      🥁 KICK @ {kick_time:.2f}s")

    # --- 6. 音频回调（独立于 GUI 线程，用当前播放指针取音频） ---
    def callback(outdata, frames, time_info, status):
        if status:
            print("播放状态:", status, file=sys.stderr)
        if not state["playing"]:
            outdata[:] = 0
            return
        pos = state["pos"]
        # 变速：每块推进 frames * speed 帧的音频，实现 0.25x~2x 变速
        step = int(frames * state["speed"])
        i0 = int(pos * sr)
        i1 = i0 + step
        if i0 >= len(y):
            outdata[:] = 0
            state["playing"] = False
            return
        seg = y[i0:i1]
        # 若目标帧数不等于 step，需重采样到 frames（简化：按速度调整读指针即可，
        # 输出长度固定 frames，直接切片；速度 <1 时可能取不够，末尾补零）
        n = min(frames, len(seg))
        outdata[:n, 0] = seg[:n]
        if n < frames:
            outdata[n:, 0] = 0
        state["pos"] += step / sr

    def ensure_stream():
        if state["stream"] is None or state["stream"].closed:
            # 用大 buffer + 高延迟，避免回调被 matplotlib 抢占导致卡顿
            state["stream"] = sd.OutputStream(
                samplerate=sr, channels=1, dtype="float32",
                callback=callback, blocksize=8192, latency="high",
            )
            state["stream"].start()

    def update():
        # pos 由音频回调线程唯一维护（避免与 GUI 线程竞争导致跳针/卡顿）。
        # 仅在暂停态用墙钟时间外推，播放态直接读回调写入的 pos。
        if state["playing"]:
            if state["pos"] >= duration:
                state["pos"] = duration
                state["playing"] = False
                if state["stream"] is not None:
                    state["stream"].stop()

        pos = state["pos"]
        cursor.set_xdata([pos, pos])

        # 段落高亮 + 信息
        idx = get_section_idx(pos)
        if idx != state["prev_section_idx"]:
            log_section_change(idx)
            state["prev_section_idx"] = idx

        # 高亮描边当前段落
        highlight.set_visible(True)
        highlight.set_x(sec_starts[idx])
        highlight.set_width(sec_ends[idx] - sec_starts[idx])

        # 拍子日志
        bi = find_beat(pos)
        if bi != state["prev_beat_idx"] and bi >= 0:
            log_beat(bi)
            state["prev_beat_idx"] = bi

        # kick 日志
        ki = find_kick(pos)
        if ki != state.get("prev_kick_idx", -1) and ki >= 0:
            log_kick(ki)
            state["prev_kick_idx"] = ki

        # 信息栏
        s = sections[idx]
        bi_cur = find_beat(pos)
        cur_dur_ms = None
        if bi_cur >= 0 and bi_cur < len(all_durations_ms):
            cur_dur_ms = all_durations_ms[bi_cur]

        dur_text = f"  拍长 {cur_dur_ms:.0f}ms" if cur_dur_ms else ""
        kick_text = f"  kick {ki+1}/{len(all_kicks)}" if ki >= 0 else ""
        info.set_text(
            f"时间 {mmss(pos)} / {mmss(duration)}    "
            f"段落: {s['label']}    "
            f"拍 {bi_cur+1}{dur_text}{kick_text}    "
            f"参考BPM {s.get('tempo', 0):.0f}"
        )

        # 节流重绘：每 100ms 才真正重绘一次，避免与音频回调抢 CPU 导致卡顿
        now = time.perf_counter()
        if now - state.get("_last_draw", 0.0) >= 0.1:
            fig.canvas.draw_idle()
            state["_last_draw"] = now

    def on_key(event):
        if event.key == " ":
            # 暂停/继续
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
            _reset_log_state()
        elif event.key == "left":
            state["pos"] = max(0, state["pos"] - 5)
            _reset_log_state()
        elif event.key == "q" or event.key == "escape":
            plt.close(fig)
        elif event.key == "up":
            state["speed"] = min(2.0, state["speed"] + 0.25)
            print(f"速度: {state['speed']:.2f}x")
        elif event.key == "down":
            state["speed"] = max(0.25, state["speed"] - 0.25)
            print(f"速度: {state['speed']:.2f}x")

    fig.canvas.mpl_connect("key_press_event", on_key)

    # 定时器（50ms 刷新一次光标，重绘在 update 内部进一步节流到 100ms）
    timer = fig.canvas.new_timer(interval=50)
    timer.add_callback(update)
    timer.start()

    print("=" * 60)
    print(f"歌曲: {os.path.basename(args.audio)}  时长: {mmss(duration)}")
    print(f"段落数: {len(sections)}  拍点数: {len(all_beats)}  kick数: {len(all_kicks)}")
    print("-" * 60)
    print("操作: [空格]播放/暂停  [←→]快退/快进5s  [↑↓]变速  [q]退出")
    print("播放时会实时打印段落切换、拍子和 KICK（底鼓）日志：")
    print("=" * 60)

    plt.show()

    # 清理
    if state["stream"] is not None:
        state["stream"].stop()
        state["stream"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
