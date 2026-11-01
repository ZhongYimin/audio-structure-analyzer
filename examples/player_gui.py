"""all-in-one 结构分析播放器（纯前端：选歌 + 预处理 + 播放 + 后处理展示）。

定位：本文件是「前端」，只负责 UI 展示与播放控制。所有分析、后处理、落盘
都在后端包 ``audio_structure_analyzer``（尤其 ``pipeline`` 模块）里离线完成，
产出**一份合并 JSON**（结构 + 后处理）。播放器点击歌曲时只读这份 JSON 纯加载。

功能：
1. 歌曲选择：内置歌曲目录 + 文件对话框选任意音频
2. 预处理：若歌曲还没有合并结果 JSON → 显示「Preprocess」按钮，后台线程跑
   pipeline.preprocess（分析 + 后处理）并落盘；已有则直接可播放。
3. 可视化播放：波形 + 段落色块 + 强拍/拍线 + 游标（内嵌在同一个 tkinter 窗口）。
4. 后处理展示开关：所有方案结果已并入 JSON，勾选/取消只是显示/隐藏对应高亮，
   零计算、即时生效。

推理设备：优先 CUDA，无 GPU 自动回退 CPU（见 auto_device()）。
"""

from __future__ import annotations

import bisect
import sys
import threading
from pathlib import Path

import numpy as np

# 让 `examples/player_gui.py` 从任意目录运行时都能找到项目包
# （把 examples 的父目录，即包根，加入 sys.path）
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from audio_structure_analyzer import available_processors as list_processors
from audio_structure_analyzer import find_result, load_bundle, preprocess


# ---- 常量 ----

LABEL_COLORS = {
    "start": "#5f6b7a",
    "end": "#5f6b7a",
    "intro": "#39d0e8",
    "outro": "#39d0e8",
    "verse": "#5ce1a6",
    "chorus": "#ff5fa2",
    "bridge": "#a78bfa",
    "break": "#f5c84c",
    "inst": "#2de2b8",
    "solo": "#ff8a5c",
}

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aac"}

# ---- 赛博主题配色 ----
THEME = {
    "bg": "#0d1117",        # 深底
    "panel": "#161b22",      # 面板
    "panel2": "#1c2230",     # 次级面板
    "fg": "#e6f1ff",         # 前景文字
    "sub": "#8892a0",        # 次级文字
    "accent": "#39d0e8",     # 青色强调
    "accent2": "#ff5fa2",    # 粉红强调
    "grid": "#1f2937",       # 网格线
    "border": "#30363d",     # 边框
    "select": "#2d3a4a",     # 选中
}

# 背景图路径（全窗口铺底用）
MIKU_BG = Path(__file__).resolve().parent.parent / "assets" / "miku_bg.jpg"

# 字体：英文用 JetBrains Mono，中文回退到微软雅黑（保证中文歌名正常显示）
MONO_FONT = "JetBrains Mono"
CJK_FONT = "Microsoft YaHei"


def default_song_dir() -> Path:
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        songs = base / "songs"
        return songs if songs.is_dir() else base
    here = Path(__file__).resolve().parent
    ta = here.parent / "test_audio"
    return ta if ta.is_dir() else here


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


# ---- 预处理（后台线程调用 pipeline.preprocess） ----

def run_inference(audio: Path, device: str, processors: list[str], progress_cb=None, done_cb=None):
    def _work():
        try:
            if progress_cb:
                progress_cb(f"预处理中：分析 + 后处理（{device}）…")
            result = preprocess(audio, processors=processors, device=device)
            n_pp = len(result.postprocess)
            if done_cb:
                done_cb(
                    True,
                    f"预处理完成：BPM={result.bpm}，{len(result.segments)} 段，"
                    f"{n_pp} 个后处理方案",
                )
        except Exception as e:
            if done_cb:
                done_cb(False, f"预处理失败：{e}")

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    return t


# ---- 内嵌播放面板（matplotlib canvas 嵌入 tkinter） ----

class PlayerPanel:
    """内嵌到 tkinter 的波形播放面板，展示并播放一首歌的分析结果。

    纯加载：所有重计算（分析 + 后处理）已在预处理阶段并入 JSON，本面板
    只读那份 JSON + 加载波形，点击秒开、不做任何计算。
    """

    def __init__(self, parent, audio: Path, enabled_processors: list[str], on_status=None):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import sounddevice as sd

        self.parent = parent
        self.audio = audio
        self.enabled_processors = list(enabled_processors)
        self.on_status = on_status
        self.sd = sd

        plt.rcParams["font.sans-serif"] = [
            CJK_FONT, MONO_FONT, "Segoe UI", "Consolas", "DejaVu Sans Mono"
        ]
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.monospace"] = [MONO_FONT, "Consolas", "DejaVu Sans Mono"]
        plt.rcParams["axes.unicode_minus"] = False

        # 纯加载：读合并 JSON（结构 + 后处理），不跑任何计算
        result = load_bundle(audio)
        self.result = result
        self.bpm = result.bpm
        self.beats = result.beats
        self.downbeats = result.downbeats
        self.beat_positions = result.beat_positions

        self.seg_starts = [s.start for s in result.segments]
        self.seg_ends = [s.end for s in result.segments]
        self.seg_labels = [s.label for s in result.segments]

        # 后处理候选（已并入 JSON）：{方案名: {seg_index: 候选dict}}
        # 只加载「当前启用」的方案，未启用的不显示（数据仍在 result.postprocess 里）
        self.candidates_by_segment = {}
        for name in self.enabled_processors:
            pp = result.postprocess.get(name)
            if not pp:
                continue
            cand_map = {}
            for c in pp.get("candidates", []):
                cand_map[c["segment_index"]] = c
            self.candidates_by_segment[name] = cand_map

        # 加载波形（唯一可能稍慢的 IO，但远快于任何计算）
        from audio_structure_analyzer import load_audio
        self.y, self.sr = load_audio(audio)
        self.y = self.y.astype(np.float32)
        self.duration = len(self.y) / self.sr
        self.ymax = float(np.max(np.abs(self.y))) if len(self.y) else 1.0

        self._build_figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        # 播放状态
        self.state = {
            "pos": 0.0, "sample_pos": 0, "playing": False, "stream": None,
            "speed": 1.0, "_bg": None, "_blit_ready": False, "_dragging": False,
        }

        self._bind_events()
        self._start_timer()

    def _build_figure(self):
        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots(figsize=(14, 5), dpi=100)
        self.fig.patch.set_facecolor(THEME["bg"])

        BG = THEME["bg"]
        FG = THEME["fg"]
        SUB = THEME["sub"]
        GRID = THEME["grid"]
        ACCENT = THEME["accent"]

        ax = self.ax
        ax.set_facecolor(BG)

        # 初音背景（压暗版），与全窗口背景保持一致，铺满绘图区
        if MIKU_BG.exists():
            try:
                from PIL import Image, ImageEnhance
                _im = Image.open(MIKU_BG).convert("RGB")
                _im = ImageEnhance.Brightness(_im).enhance(0.45)
                _im = ImageEnhance.Color(_im).enhance(0.85)
                ax.imshow(
                    _im, aspect="auto", zorder=0, alpha=1.0,
                    extent=[0, self.duration, -self.ymax * 1.05, self.ymax * 1.05],
                )
            except Exception:
                pass

        for spine in ax.spines.values():
            spine.set_color(THEME["border"])
        ax.tick_params(colors=SUB, labelsize=9)
        ax.xaxis.label.set_color(SUB)
        ax.yaxis.label.set_color(SUB)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

        # 波形降采样
        WAVE_POINTS = 4000
        y = self.y
        if len(y) > WAVE_POINTS:
            idx = np.linspace(0, len(y) - 1, WAVE_POINTS).astype(int)
            y_plot = y[idx]
            t_plot = idx / self.sr
        else:
            y_plot = y
            t_plot = np.arange(len(y)) / self.sr

        ax.plot(t_plot, y_plot, color="#7fd6f0", linewidth=0.7, alpha=0.95, zorder=3)
        ax.fill_between(t_plot, y_plot, 0, color=THEME["accent"], alpha=0.12, linewidth=0, zorder=2)

        # 段落色块 + 标签
        for i in range(len(self.seg_starts)):
            color = LABEL_COLORS.get(self.seg_labels[i], "#5f6b7a")
            ax.axvspan(self.seg_starts[i], self.seg_ends[i], color=color, alpha=0.14, zorder=1)

            # 各后处理方案的候选高亮（不同方案用不同颜色，堆叠展示）
            for proc_name, cand_map in self.candidates_by_segment.items():
                if i in cand_map:
                    cand = cand_map[i]
                    self._draw_candidate(i, cand, proc_name)

            mid = (self.seg_starts[i] + self.seg_ends[i]) / 2
            ax.text(mid, self.ymax * 0.88, self.seg_labels[i], ha="center", va="top",
                    fontsize=11, color=color, zorder=6)

        # 拍线
        for b in self.beats:
            ax.axvline(b, color=THEME["sub"], linewidth=0.5, alpha=0.30, ymin=0.04, ymax=0.12, zorder=2)
        for db in self.downbeats:
            ax.axvline(db, color=ACCENT, linewidth=1.6, alpha=0.95, ymin=0.04, ymax=0.16, zorder=2)

        self.cursor = ax.axvline(0, color=THEME["accent2"], linewidth=2.5, alpha=0.95, zorder=10, animated=True)

        self.info = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=11,
                            va="bottom", color=FG, fontfamily="monospace",
                            bbox=dict(boxstyle="round,pad=0.4", fc=THEME["panel"], ec=THEME["border"], alpha=0.92),
                            animated=True)

        ax.set_xlim(0, self.duration)
        ax.set_ylim(-self.ymax * 1.05, self.ymax * 1.05)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        title_text = f"{self.audio.stem}   ·   BPM {self.bpm}"
        ax.set_title(title_text, color=FG, fontsize=13, pad=12, loc="left")

    def _draw_candidate(self, seg_idx: int, cand, proc_name: str):
        """在段落上绘制某后处理方案的候选高亮（虚线框 + 置信度标签）。"""
        # 每个方案一个高亮色，便于区分多个方案叠加
        colors = {
            "prechorus": "#ffd166",       # 黄（位置+能量判据）
            "spatiotemporal": "#5ce1a6",  # 绿（时空/配器判据）
        }
        hl = colors.get(proc_name, "#ffd166")
        ax = self.ax
        ax.axvspan(
            self.seg_starts[seg_idx], self.seg_ends[seg_idx],
            facecolor="none", edgecolor=hl, linewidth=1.4,
            linestyle="--", alpha=0.9, zorder=5,
        )
        mid = (self.seg_starts[seg_idx] + self.seg_ends[seg_idx]) / 2
        conf = cand.get("confidence", 0.0) if isinstance(cand, dict) else getattr(cand, "confidence", 0.0)
        # 不同方案标签纵向错开，避免重叠
        base_y = 0.72
        offset = {"prechorus": 0.0, "spatiotemporal": -0.14}.get(proc_name, 0.0)
        ax.text(
            mid, self.ymax * (base_y + offset),
            f"{proc_name} {conf*100:.0f}%",
            ha="center", va="top", fontsize=9,
            color=hl, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="#1c2230", ec=hl, alpha=0.85),
        )

    def set_processors(self, enabled: list[str]):
        """切换启用的后处理方案（纯显示切换，数据已在内存，零计算）。"""
        self.enabled_processors = list(enabled)
        # 从已加载的 result.postprocess 重新聚合候选
        self.candidates_by_segment = {}
        for name in self.enabled_processors:
            pp = self.result.postprocess.get(name)
            if not pp:
                continue
            cand_map = {}
            for c in pp.get("candidates", []):
                cand_map[c["segment_index"]] = c
            self.candidates_by_segment[name] = cand_map
        # 重绘（去掉旧高亮，重画当前高亮）
        self._redraw()

    def _redraw(self):
        """重绘图形以反映最新的候选高亮。"""
        try:
            self.ax.cla()
            self._build_figure()
            self.canvas.draw()
            self.state["_blit_ready"] = False
        except Exception:
            pass

    def _bind_events(self):
        self.canvas.mpl_connect("key_press_event", self._on_key)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        # 游标拖拽：点击/拖动波形 seek 播放进度
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_drag)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        # 让 canvas 能接收键盘焦点
        self.canvas.get_tk_widget().focus_set()

    def _start_timer(self):
        self.timer = self.fig.canvas.new_timer(interval=33)
        self.timer.add_callback(self._update)
        self.timer.start()

    def get_seg_idx(self, pos):
        for i in range(len(self.seg_starts)):
            if self.seg_starts[i] <= pos < self.seg_ends[i]:
                return i
        return 0 if pos < self.seg_starts[0] else len(self.seg_starts) - 1

    def find_beat(self, pos):
        return bisect.bisect_right(self.beats, pos) - 1

    def _audio_callback(self, outdata, frames, time_info, status):
        if not self.state["playing"]:
            outdata[:] = 0
            return
        i0 = self.state["sample_pos"]
        step = int(frames * self.state["speed"])
        i1 = i0 + step
        if i0 >= len(self.y):
            outdata[:] = 0
            self.state["playing"] = False
            return
        seg = self.y[i0:i1]
        n = min(frames, len(seg))
        outdata[:n, 0] = seg[:n]
        if n < frames:
            outdata[n:, 0] = 0
        self.state["sample_pos"] += step
        self.state["pos"] = self.state["sample_pos"] / self.sr

    def _ensure_stream(self):
        if self.state["stream"] is None or self.state["stream"].closed:
            self.state["stream"] = self.sd.OutputStream(
                samplerate=self.sr, channels=1, dtype="float32",
                callback=self._audio_callback, blocksize=2048, latency="low",
            )
            self.state["stream"].start()

    def _refresh_info(self, pos):
        try:
            self.cursor.set_xdata([pos, pos])
            idx = self.get_seg_idx(pos)
            bi = self.find_beat(pos)
            if 0 <= idx < len(self.seg_labels):
                s_label = self.seg_labels[idx]
            else:
                s_label = "?"
            pos_label = self.beat_positions[bi] if 0 <= bi < len(self.beat_positions) else 0
            self.info.set_text(
                f"TIME {mmss(pos)} / {mmss(self.duration)}   SEC {s_label.upper():<6}   "
                f"BEAT {bi+1}/{len(self.beats)} (pos {pos_label})   BPM {self.bpm}"
            )
        except Exception:
            pass

    def _update(self):
        # 窗口已关闭则跳过
        try:
            if not self.canvas.get_tk_widget().winfo_exists():
                return
        except Exception:
            return

        try:
            if self.state["playing"] and self.state["pos"] >= self.duration:
                self.state["pos"] = self.duration
                self.stop()

            pos = self.state["pos"]
            self._refresh_info(pos)

            canvas = self.canvas
            if not self.state["_blit_ready"]:
                canvas.draw()
                self.state["_bg"] = canvas.copy_from_bbox(self.ax.bbox)
                self.state["_blit_ready"] = True
            canvas.restore_region(self.state["_bg"])
            self.ax.draw_artist(self.cursor)
            self.ax.draw_artist(self.info)
            canvas.blit(self.ax.bbox)
        except Exception:
            # 任何异常都不应中断 timer，避免画面停止更新
            self.state["_blit_ready"] = False

    def _on_key(self, event):
        if event.key == " ":
            self.toggle_play()
        elif event.key == "right":
            self.seek(5)
        elif event.key == "left":
            self.seek(-5)
        elif event.key == "up":
            self.state["speed"] = min(2.0, self.state["speed"] + 0.25)
        elif event.key == "down":
            self.state["speed"] = max(0.25, self.state["speed"] - 0.25)

    def _on_resize(self, event):
        self.state["_blit_ready"] = False

    # ---- 游标拖拽（seek） ----
    def _on_press(self, event):
        # 仅左键，且点击落在 axes 内
        if event.button != 1 or event.inaxes is None or event.xdata is None:
            return
        self.state["_dragging"] = True
        self._seek_to(event.xdata)

    def _on_drag(self, event):
        if not self.state.get("_dragging") or event.inaxes is None or event.xdata is None:
            return
        self._seek_to(event.xdata)

    def _on_release(self, event):
        self.state["_dragging"] = False
        # 释放后强制整帧重绘，确保画面与进度同步
        self.state["_blit_ready"] = False

    def _seek_to(self, t: float):
        t = min(self.duration, max(0.0, float(t)))
        self.state["pos"] = t
        self.state["sample_pos"] = int(t * self.sr)
        # 不在这里重绘；由 timer 的 _update 每 33ms 统一刷新 cursor/info，
        # 避免与 blit 局部重绘竞争导致画面停滞。

    # ---- 供外部按钮调用的接口 ----
    def toggle_play(self):
        if self.state["playing"]:
            self.stop()
        else:
            self._ensure_stream()
            self.state["playing"] = True
            if self.state["stream"].stopped:
                self.state["stream"].start()

    def seek(self, delta):
        self._seek_to(self.state["pos"] + delta)

    def stop(self):
        """停止播放（仅停音频流；timer 保持运行以持续刷新画面）。"""
        self.state["playing"] = False
        stream = self.state["stream"]
        if stream is not None:
            try:
                if not stream.closed:
                    stream.stop()
                    stream.close()
            except Exception:
                pass
            self.state["stream"] = None

    def destroy(self):
        self.stop()
        # 销毁时停掉 timer，避免窗口已销毁后 timer 仍回调
        if hasattr(self, "timer"):
            try:
                self.timer.stop()
            except Exception:
                pass
        try:
            self.canvas.get_tk_widget().destroy()
        except Exception:
            pass


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
        self.root.title("all-in-one structure analyzer")
        self.root.geometry("1100x760")
        self.root.minsize(900, 620)
        self.root.configure(bg=THEME["bg"])

        self._load_bg_image()
        self._apply_style()

        self.song_dir = default_song_dir()
        self.current_audio: Path | None = None
        self.inferring = False
        self.panel: PlayerPanel | None = None
        self._closing = False

        # 后处理方案勾选状态（默认全部启用）
        self.processor_vars = {}
        for name in list_processors():
            self.processor_vars[name] = tk.BooleanVar(value=True)

        self._build_ui()
        self.refresh_songs()

        # 关闭窗口时务必停止音频流，避免后台还在播
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._closing = True
        if self.panel is not None:
            self.panel.destroy()
        self.root.destroy()

    def _load_bg_image(self):
        """加载初音背景图并压暗，作为全窗口氛围背景。"""
        self._bg_photo = None
        self._bg_img_orig = None
        try:
            from PIL import Image, ImageEnhance, ImageTk
            if MIKU_BG.exists():
                img = Image.open(MIKU_BG).convert("RGB")
                # 压暗 + 降饱和，做氛围底，避免与前景文字抢视觉
                img = ImageEnhance.Brightness(img).enhance(0.45)
                img = ImageEnhance.Color(img).enhance(0.85)
                self._bg_img_orig = img
                self._bg_photo = ImageTk.PhotoImage(img)
        except Exception:
            self._bg_photo = None
            self._bg_img_orig = None

    def _resize_bg(self, event=None):
        """窗口尺寸变化时重采样背景图铺满内容层。"""
        if not getattr(self, "_bg_img_orig", None):
            return
        try:
            from PIL import Image, ImageTk
            w = max(1, self.content.winfo_width())
            h = max(1, self.content.winfo_height())
            img = self._bg_img_orig.resize((w, h), Image.Resampling.LANCZOS)
            self._bg_photo = ImageTk.PhotoImage(img)
            self.bg_label.configure(image=self._bg_photo)
        except Exception:
            pass

    def _apply_style(self):
        """统一 ttk 组件深色赛博主题。"""
        tk = self.tk
        ttk = self.ttk
        self.root.option_add("*Font", (MONO_FONT, 10))

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        c = THEME
        style.configure(".", background=c["bg"], foreground=c["fg"],
                        fieldbackground=c["panel"], bordercolor=c["border"],
                        lightcolor=c["bg"], darkcolor=c["bg"], font=(MONO_FONT, 10))

        style.configure("TFrame", background=c["bg"])
        style.configure("TLabel", background=c["bg"], foreground=c["fg"])
        style.configure("Sub.TLabel", background=c["bg"], foreground=c["sub"])
        style.configure("Accent.TLabel", background=c["bg"], foreground=c["accent"])
        # 中文字体样式：用于显示中文歌名的控件（状态栏等）
        style.configure("CJK.TLabel", background=c["bg"], foreground=c["fg"], font=(CJK_FONT, 10))
        style.configure("CJKSub.TLabel", background=c["bg"], foreground=c["sub"], font=(CJK_FONT, 10))

        style.configure("TEntry", fieldbackground=c["panel"], foreground=c["fg"],
                        insertcolor=c["fg"], bordercolor=c["border"], padding=4)

        style.configure("TButton", background=c["panel2"], foreground=c["fg"],
                        bordercolor=c["border"], padding=6, relief="flat")
        style.map("TButton",
                  background=[("active", c["select"]), ("disabled", c["bg"])],
                  foreground=[("disabled", c["sub"])])

        style.configure("Accent.TButton", background=c["accent"], foreground=c["bg"],
                        padding=6)
        style.map("Accent.TButton", background=[("active", "#5be6f5")])

        # Treeview 数据列用雅黑（含中文歌名），表头保持 JetBrains Mono
        style.configure("Treeview", background=c["panel"], fieldbackground=c["panel"],
                        foreground=c["fg"], bordercolor=c["border"], rowheight=24,
                        font=(CJK_FONT, 10))
        style.map("Treeview", background=[("selected", c["select"])],
                  foreground=[("selected", c["accent"])])
        style.configure("Treeview.Heading", background=c["panel2"], foreground=c["accent"],
                        relief="flat", font=(MONO_FONT, 10))

        style.configure("Vertical.TScrollbar", background=c["panel2"],
                        troughcolor=c["bg"], bordercolor=c["bg"], arrowcolor=c["sub"])

    def _build_ui(self):
        tk = self.tk
        ttk = self.ttk

        # 内容层（窗口主体，铺满 miku 背景图）
        self.content = tk.Frame(self.root, bg=THEME["bg"])
        self.content.pack(fill="both", expand=True)

        # 背景层：miku 铺满内容层
        self.bg_label = tk.Label(self.content, bd=0)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        if self._bg_photo:
            self.bg_label.configure(image=self._bg_photo)
        self.content.bind("<Configure>", self._resize_bg)
        self.bg_label.lower()  # 置于最底层

        # ---- 顶部：歌曲目录 + 选歌 ----
        top = ttk.Frame(self.content, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="SONGS", style="Accent.TLabel").pack(side="left")
        self.dir_var = tk.StringVar(value=str(self.song_dir))
        ttk.Entry(top, textvariable=self.dir_var, width=42).pack(side="left", padx=6)
        ttk.Button(top, text="Browse", command=self.choose_dir).pack(side="left", padx=4)
        ttk.Button(top, text="Refresh", command=self.refresh_songs).pack(side="left", padx=4)
        ttk.Button(top, text="Add Audio", command=self.add_audio).pack(side="left", padx=4)

        self.btn_infer = ttk.Button(top, text="Analyze", style="Accent.TButton",
                                    command=self.infer_selected, state="disabled")
        self.btn_infer.pack(side="left", padx=4)

        self.btn_play = ttk.Button(top, text="Play/Pause", command=self.toggle_play, state="disabled")
        self.btn_play.pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var, style="CJKSub.TLabel").pack(side="right")

        # ---- 中部：歌曲列表 ----
        mid = ttk.Frame(self.content, padding=(10, 0, 10, 0))
        mid.pack(fill="x")

        cols = ("name", "status")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=6)
        self.tree.heading("name", text="SONG")
        self.tree.heading("status", text="STATUS")
        self.tree.column("name", width=640)
        self.tree.column("status", width=140)
        self.tree.pack(side="left", fill="x", expand=True)

        sb = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<Double-1>", lambda e: self.load_selected())
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # ---- 后处理方案勾选区（分析完成后可配置） ----
        pp = ttk.Frame(self.content, padding=(10, 4, 10, 0))
        pp.pack(fill="x")
        ttk.Label(pp, text="POST-PROCESS", style="Accent.TLabel").pack(side="left")
        for name, var in self.processor_vars.items():
            cb = ttk.Checkbutton(
                pp, text=name, variable=var,
                command=self._on_postprocess_change,
            )
            cb.pack(side="left", padx=8)

        # ---- 底部：波形播放区 ----
        self.player_frame = ttk.Frame(self.content, padding=(10, 6, 10, 10))
        self.player_frame.pack(fill="both", expand=True)

        self.placeholder = ttk.Label(
            self.player_frame,
            text="SELECT A SONG AND PRESS PLAY TO LOAD THE WAVEFORM",
            style="Sub.TLabel", anchor="center"
        )
        self.placeholder.pack(fill="both", expand=True)

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
                        f.name, "analyzed" if ok else "not analyzed"
                    ))
                    self.song_items.append(f)
        self.status_var.set(f"{len(self.song_items)} songs")

    def add_audio(self):
        files = self.filedialog.askopenfilenames(
            title="Select audio files",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.aiff *.aac"), ("All", "*.*")],
        )
        for fp in files:
            p = Path(fp)
            if str(p) not in self.tree.get_children():
                ok = has_result(p)
                self.tree.insert("", "end", iid=str(p), values=(
                    p.name, "analyzed" if ok else "not analyzed"
                ))
                self.song_items.append(p)
        self.status_var.set(f"{len(self.song_items)} songs")

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
        if has_result(p):
            self.btn_play.config(state="normal")
            self.btn_infer.config(state="disabled")
            self.status_var.set(f"{p.name}: analyzed, ready to play")
        else:
            self.btn_play.config(state="disabled")
            self.btn_infer.config(state="normal")
            self.status_var.set(f"{p.name}: not analyzed, analyze first")

    def load_selected(self):
        p = self.selected_audio()
        if p is None:
            return
        if find_result(p) is None:
            self.messagebox.showinfo("Notice", "This song has no analysis result yet. Analyze first.")
            return

        self.status_var.set(f"Loading: {p.name}")

        # 清掉旧面板
        if self.panel is not None:
            self.panel.destroy()
            self.panel = None
        self.placeholder.pack_forget()

        processors = self._selected_processors()
        self.panel = PlayerPanel(
            self.player_frame, p, processors, on_status=self.status_var.set
        )
        self.btn_play.config(state="normal")
        self.status_var.set(f"Loaded: {p.name}")

    def _selected_processors(self) -> list[str]:
        """返回当前勾选启用的后处理方案名列表。"""
        return [name for name, var in self.processor_vars.items() if var.get()]

    def _on_postprocess_change(self):
        """勾选变化时即时切换高亮显示（数据已在面板内存，零计算）。"""
        if self.panel is not None:
            self.panel.set_processors(self._selected_processors())

    def toggle_play(self):
        if self.panel is None:
            self.load_selected()
            if self.panel is None:
                return
        self.panel.toggle_play()

    def infer_selected(self):
        if self.inferring:
            return
        p = self.selected_audio()
        if p is None:
            return
        self.inferring = True
        self.btn_infer.config(state="disabled")
        self.btn_play.config(state="disabled")

        def done(ok, msg):
            self.inferring = False
            self.status_var.set(msg)
            if ok:
                if str(p) in self.tree.get_children():
                    self.tree.set(str(p), "status", "analyzed")
                self.btn_play.config(state="normal")
                self.load_selected()
            else:
                self.btn_infer.config(state="normal")
                self.messagebox.showerror("Analyze failed", msg)

        run_inference(
            p, auto_device(), self._selected_processors(),
            progress_cb=self.status_var.set, done_cb=done,
        )

    def run(self):
        self.root.mainloop()


def main():
    App().run()


if __name__ == "__main__":
    main()
