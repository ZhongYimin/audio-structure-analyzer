"""all-in-one 结构分析 GUI 播放器（带歌曲选择 + 智能推理）。

功能：
1. 歌曲选择：内置歌曲目录 + 文件对话框选任意音频
2. 智能推理：若 <音频同目录>/<同名>.json 已存在 → 不显示推理按钮，直接可播放；
   若不存在 → 显示「推理」按钮，后台线程调用 all-in-one 分析并落盘 json。
3. 可视化播放：波形 + 段落色块 + 强拍/拍线 + 游标（复用 matplotlib 逻辑）。

推理设备：优先 CUDA，无 GPU 自动回退 CPU（见 auto_device()）。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np


# ---- 常量 ----

LABEL_COLORS = {
    "start": "#b0b0b0",
    "end": "#b0b0b0",
    "intro": "#4c8bf5",
    "outro": "#4c8bf5",
    "verse": "#f5a623",
    "chorus": "#e74c3c",
    "bridge": "#9b59b6",
    "break": "#16a085",
    "inst": "#27ae60",
    "solo": "#e67e22",
}

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aac"}

# 默认歌曲目录：优先 exe 旁边的 songs/，否则项目 test_audio/
def default_song_dir() -> Path:
    # 打包后：<exe目录>/songs
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        songs = base / "songs"
        if songs.is_dir():
            return songs
        return base
    # 开发环境：项目 test_audio
    here = Path(__file__).resolve().parent
    ta = here.parent / "test_audio"
    return ta if ta.is_dir() else here


def json_path_for(audio: Path) -> Path:
    """推理结果 json 的约定路径：<音频目录>/<同名>.json（新结果统一落这里）。"""
    return audio.with_suffix(".json")


def _candidate_result_paths(audio: Path):
    """查找结果 json 的所有候选位置（兼容旧的 allin1/ 子目录）。"""
    yield audio.with_suffix(".json")                       # 同目录
    yield audio.parent / "allin1" / f"{audio.stem}.json"   # 旧约定：allin1/ 子目录


def find_result(audio: Path) -> Path | None:
    """返回已存在的结果 json 路径；不存在返回 None。"""
    for p in _candidate_result_paths(audio):
        if p.exists():
            return p
    return None


def has_result(audio: Path) -> bool:
    return find_result(audio) is not None


def auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


# ---- 推理（后台线程调用） ----

def run_inference(audio: Path, device: str, progress_cb=None, done_cb=None):
    """在后台线程跑 all-in-one 分析，落盘 json。"""
    from allin1_infer import analyze

    def _work():
        try:
            if progress_cb:
                progress_cb(f"正在分离声部 + 分析（{device}）…")
            results = analyze(
                paths=[str(audio)],
                model="harmonix-all",
                device=device,
                multiprocess=False,          # 打包/多线程环境必须 False
                keep_byproducts=False,
            )
            result = results[0] if isinstance(results, list) else results
            data = {
                "path": str(result.path),
                "bpm": int(result.bpm),
                "beats": [float(x) for x in result.beats],
                "downbeats": [float(x) for x in result.downbeats],
                "beat_positions": [int(x) for x in result.beat_positions],
                "activation_fps": getattr(result, "activation_fps", None),
                "segments": [
                    {"start": s.start, "end": s.end, "label": s.label}
                    for s in result.segments
                ],
            }
            out = json_path_for(audio)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if done_cb:
                done_cb(True, f"推理完成：BPM={data['bpm']}，{len(data['segments'])} 段")
        except Exception as e:
            if done_cb:
                done_cb(False, f"推理失败：{e}")

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    return t


# ---- 可视化播放窗口（matplotlib） ----

class PlayerWindow:
    """独立的 matplotlib 播放窗口，展示一首歌的分析结果并可播放。"""

    def __init__(self, audio: Path, data: dict, title: str = ""):
        self.audio = audio
        self.data = data
        self.title = title or audio.name

    def show(self):
        import librosa
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import sounddevice as sd

        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"
        ]
        plt.rcParams["axes.unicode_minus"] = False

        data = self.data
        segments = data["segments"]
        beats = data.get("beats", [])
        downbeats = data.get("downbeats", [])
        beat_positions = data.get("beat_positions", [])
        bpm = data.get("bpm", 0)

        y, sr = librosa.load(str(self.audio), sr=22050, mono=True)
        duration = len(y) / sr
        y = y.astype(np.float32)

        seg_starts = [s["start"] for s in segments]
        seg_ends = [s["end"] for s in segments]
        seg_labels = [s["label"] for s in segments]

        fig, ax = plt.subplots(figsize=(15, 6))
        fig.canvas.manager.set_window_title(f"all-in-one 播放器 — {self.title}")

        # librosa 1.0 的 times_like 有坑，直接用样本索引算时间
        ymax = float(np.max(np.abs(y))) if len(y) else 1.0

        # 波形降采样显示：整首歌 600 万+ 样本点一次性 plot 会让 TkAgg 卡死，
        # 降到 ~4000 点视觉无损、绘制快 35 倍
        WAVE_POINTS = 4000
        if len(y) > WAVE_POINTS:
            idx = np.linspace(0, len(y) - 1, WAVE_POINTS).astype(int)
            y_plot = y[idx]
            t_plot = idx / sr
        else:
            y_plot = y
            t_plot = np.arange(len(y)) / sr

        # 段落色块放最底层（zorder=1），不遮挡波形
        for i in range(len(segments)):
            color = LABEL_COLORS.get(seg_labels[i], "#cccccc")
            ax.axvspan(seg_starts[i], seg_ends[i], color=color, alpha=0.18, zorder=1)
            mid = (seg_starts[i] + seg_ends[i]) / 2
            ax.text(mid, ymax * 0.90, seg_labels[i], ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color, zorder=6)
            ax.text(mid, ymax * 0.74, f"{seg_starts[i]:.1f}s", ha="center",
                    va="center", fontsize=8, color="#555", zorder=6)

        # 波形：深色实线（已降采样，绘制流畅）
        ax.plot(t_plot, y_plot, color="#2b2b2b", linewidth=0.7, alpha=1.0, zorder=2)

        for b in beats:
            ax.axvline(b, color="#999", linewidth=0.3, alpha=0.25, ymin=0.04, ymax=0.12, zorder=4)
        for db in downbeats:
            ax.axvline(db, color="#e74c3c", linewidth=1.2, alpha=0.65, ymin=0.04, ymax=0.16, zorder=4)

        cursor = ax.axvline(0, color="#d62728", linewidth=2, alpha=0.9, zorder=10, animated=True)

        info = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=13,
                       fontweight="bold", va="top", color="#111",
                       bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9),
                       animated=True)

        ax.set_xlim(0, duration)
        ax.set_ylim(-ymax * 1.05, ymax * 1.05)
        ax.set_xlabel("时间 (秒)")
        ax.set_ylabel("振幅")
        ax.set_title(
            f"{self.title} — BPM={bpm}  拍={len(beats)}  强拍={len(downbeats)}  "
            f"[空格]播放/暂停 [←→]快退/快进 [↑↓]变速 [q]退出"
        )

        state = {
            "pos": 0.0, "sample_pos": 0, "playing": False, "stream": None,
            "speed": 1.0, "prev_seg_idx": -1, "prev_beat_idx": -1, "_last_draw": 0.0,
            "_bg": None, "_blit_ready": False,
        }

        def get_seg_idx(pos):
            for i in range(len(segments)):
                if seg_starts[i] <= pos < seg_ends[i]:
                    return i
            return 0 if pos < seg_starts[0] else len(segments) - 1

        def find_beat(pos):
            # beats 已排序，二分查找最后一个 <= pos 的索引
            import bisect
            return bisect.bisect_right(beats, pos) - 1

        def callback(outdata, frames, time_info, status):
            if not state["playing"]:
                outdata[:] = 0
                return
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
            state["sample_pos"] += step
            state["pos"] = state["sample_pos"] / sr

        def ensure_stream():
            if state["stream"] is None or state["stream"].closed:
                state["stream"] = sd.OutputStream(
                    samplerate=sr, channels=1, dtype="float32",
                    callback=callback, blocksize=2048, latency="low",
                )
                state["stream"].start()

        def _refresh_artists(pos):
            cursor.set_xdata([pos, pos])
            idx = get_seg_idx(pos)
            bi = find_beat(pos)
            state["prev_seg_idx"] = idx
            state["prev_beat_idx"] = bi
            s = segments[idx]
            pos_label = beat_positions[bi] if 0 <= bi < len(beat_positions) else 0
            info.set_text(
                f"时间 {mmss(pos)} / {mmss(duration)}    段落: {s['label']}    "
                f"拍 {bi+1}/{len(beats)} (第{pos_label}拍)    BPM {bpm}"
            )

        def update():
            if state["playing"] and state["pos"] >= duration:
                state["pos"] = duration
                state["playing"] = False
                if state["stream"] is not None:
                    state["stream"].stop()

            pos = state["pos"]
            _refresh_artists(pos)

            # blit 局部刷新：只重绘游标 + 文本，不重绘背景
            canvas = fig.canvas
            if not state["_blit_ready"]:
                canvas.draw()
                state["_bg"] = canvas.copy_from_bbox(ax.bbox)
                state["_blit_ready"] = True
            canvas.restore_region(state["_bg"])
            ax.draw_artist(cursor)
            ax.draw_artist(info)
            canvas.blit(ax.bbox)

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
            elif event.key in ("q", "escape"):
                plt.close(fig)
            elif event.key == "up":
                state["speed"] = min(2.0, state["speed"] + 0.25)
            elif event.key == "down":
                state["speed"] = max(0.25, state["speed"] - 0.25)

        fig.canvas.mpl_connect("key_press_event", on_key)

        def _on_resize(event):
            # 窗口大小变化 → 缓存失效，下次 update 重新画背景
            state["_blit_ready"] = False

        fig.canvas.mpl_connect("resize_event", _on_resize)

        timer = fig.canvas.new_timer(interval=33)
        timer.add_callback(update)
        timer.start()

        plt.show()
        if state["stream"] is not None:
            state["stream"].stop()
            state["stream"].close()


# ---- GUI 主窗口 ----

class App:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox

        self.root = tk.Tk()
        self.root.title("all-in-one 结构分析播放器")
        self.root.geometry("720x520")
        self.root.minsize(560, 420)

        self.song_dir = default_song_dir()
        self.current_audio: Path | None = None
        self.inferring = False

        self._build_ui()
        self.refresh_songs()

    def _build_ui(self):
        tk = self.tk
        ttk = self.ttk

        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="歌曲目录:").pack(side="left")
        self.dir_var = tk.StringVar(value=str(self.song_dir))
        ttk.Entry(top, textvariable=self.dir_var, width=48).pack(side="left", padx=6)
        ttk.Button(top, text="浏览", command=self.choose_dir).pack(side="left", padx=4)
        ttk.Button(top, text="刷新", command=self.refresh_songs).pack(side="left")

        mid = ttk.Frame(self.root, padding=(12, 0, 12, 0))
        mid.pack(fill="both", expand=True)

        cols = ("name", "status")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=12)
        self.tree.heading("name", text="歌曲")
        self.tree.heading("status", text="分析状态")
        self.tree.column("name", width=420)
        self.tree.column("status", width=140)
        self.tree.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<Double-1>", lambda e: self.play_selected())

        bottom = ttk.Frame(self.root, padding=12)
        bottom.pack(fill="x")

        self.btn_add = ttk.Button(bottom, text="添加音频…", command=self.add_audio)
        self.btn_add.pack(side="left")

        self.btn_infer = ttk.Button(bottom, text="推理", command=self.infer_selected, state="disabled")
        self.btn_infer.pack(side="left", padx=6)

        self.btn_play = ttk.Button(bottom, text="播放", command=self.play_selected, state="disabled")
        self.btn_play.pack(side="left")

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(bottom, textvariable=self.status_var, foreground="#555").pack(side="right")

    def choose_dir(self):
        d = self.filedialog.askdirectory(initialdir=self.dir_var.get())
        if d:
            self.dir_var.set(d)
            self.song_dir = Path(d)
            self.refresh_songs()

    def refresh_songs(self):
        self.tree.delete(*self.tree.get_children())
        self.song_dir = Path(self.dir_var.get())
        self.song_items = []
        if self.song_dir.is_dir():
            for f in sorted(self.song_dir.iterdir()):
                if f.suffix.lower() in AUDIO_EXTS:
                    ok = has_result(f)
                    self.tree.insert("", "end", iid=str(f), values=(
                        f.name, "已分析 ✓" if ok else "未分析"
                    ))
                    self.song_items.append(f)
        self.status_var.set(f"共 {len(self.song_items)} 首歌")

    def add_audio(self):
        files = self.filedialog.askopenfilenames(
            title="选择音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.m4a *.ogg *.aiff *.aac"), ("所有文件", "*.*")],
        )
        for fp in files:
            p = Path(fp)
            if str(p) not in self.tree.get_children():
                ok = has_result(p)
                self.tree.insert("", "end", iid=str(p), values=(
                    p.name, "已分析 ✓" if ok else "未分析"
                ))
                self.song_items.append(p)
        self.status_var.set(f"共 {len(self.song_items)} 首歌")

    def selected_audio(self) -> Path | None:
        sel = self.tree.selection()
        if not sel:
            return None
        p = Path(sel[0])
        return p if p.exists() else None

    def _on_select(self, event=None):
        p = self.selected_audio()
        if p is None:
            self.btn_play.config(state="disabled")
            self.btn_infer.config(state="disabled")
            return
        self.current_audio = p
        # 智能：有结果 → 可直接播放，推理按钮禁用；无结果 → 启用推理
        if has_result(p):
            self.btn_play.config(state="normal")
            self.btn_infer.config(state="disabled")
            self.status_var.set(f"{p.name}：已分析，可直接播放")
        else:
            self.btn_play.config(state="disabled")
            self.btn_infer.config(state="normal")
            self.status_var.set(f"{p.name}：未分析，请先推理")

    def play_selected(self):
        p = self.selected_audio()
        if p is None:
            return
        jp = find_result(p)
        if jp is None:
            self.messagebox.showinfo("提示", "这首歌还没有分析结果，请先推理。")
            return
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.status_var.set(f"正在打开播放器：{p.name}")
        self.root.after(100, lambda: PlayerWindow(p, data, title=p.stem).show())

    def infer_selected(self):
        if self.inferring:
            return
        p = self.selected_audio()
        if p is None:
            return
        self.inferring = True
        self.btn_infer.config(state="disabled")
        self.btn_play.config(state="disabled")
        dev = auto_device()

        def progress(msg):
            self.status_var.set(msg)

        def done(ok, msg):
            self.inferring = False
            self.status_var.set(msg)
            if ok:
                # 刷新该行状态
                if str(p) in self.tree.get_children():
                    self.tree.set(str(p), "status", "已分析 ✓")
                self.btn_play.config(state="normal")
            else:
                self.btn_infer.config(state="normal")
                self.messagebox.showerror("推理失败", msg)

        run_inference(p, dev, progress_cb=progress, done_cb=done)

    def run(self):
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.root.mainloop()


def main():
    App().run()


if __name__ == "__main__":
    main()
