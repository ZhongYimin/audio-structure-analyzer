"""运行时驱动引擎：接收播放事件，查表触发，经 OSC 分发到效果层。

架构（空间换时间）：
    离线阶段：歌曲库全部跑 all-in-one → 生成驱动文件（见 drive.py）
    运行时：宿主（UE/播放器）只发事件 → 本引擎查表 + 定时触发 → OSC 分发

事件协议（宿主 → 本引擎，OSC 地址 /drive/control）：
    /drive/start   <song_id>           某歌开始播放（从 0 或恢复点）
    /drive/stop    <song_id>           某歌停止
    /drive/pause   <song_id>           暂停
    /drive/resume  <song_id> <pos>     从 pos 秒恢复
    /drive/seek    <song_id> <pos>     用户重定位到 pos 秒

输出协议（本引擎 → 效果层，OSC 地址）：
    /beat          <song_id> <t> <pos> <is_downbeat:int> <strength>
    /segment       <song_id> <label> <start> <end>
    /state         <song_id> <state>

设计要点：
- 用播放位置 pos 作为唯一时间基准（宿主提供的单调时间），O(log n) 二分查表。
- seek/pause/resume 都通过"重设当前 pos + 重新定位到最近拍"恢复同步，宿主无需理解节拍结构。
- 每首歌一个 DriveEngine 实例，支持多首歌并发（各自独立状态）。

用法：
    from audio_structure_analyzer.osc_engine import DriveEngine, serve_osc
    serve_osc({"song_a": "drive/song_a.json", ...}, port=9000)
"""

from __future__ import annotations

import bisect
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .drive import DRIVE_FORMAT


@dataclass
class DriveFile:
    """加载后的驱动文件（内存态）。"""

    source: str
    duration: float
    bpm: int
    beats: List[dict]          # 已按 t 升序
    beat_times: List[float]
    segments: List[dict]

    @classmethod
    def load(cls, path: str) -> "DriveFile":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["meta"]["format"] == DRIVE_FORMAT, "驱动文件格式版本不匹配"
        beats = data["beats"]
        beat_times = [b["t"] for b in beats]
        return cls(
            source=data["meta"]["source"],
            duration=data["meta"]["duration"],
            bpm=data["meta"]["bpm"],
            beats=beats,
            beat_times=beat_times,
            segments=data.get("segments", []),
        )

    def beat_index_at(self, t: float) -> int:
        """返回时间 t 之前最近的拍点索引（不含 t 之后的拍）。"""
        return bisect.bisect_right(self.beat_times, t) - 1

    def segment_at(self, t: float) -> Optional[dict]:
        """返回时间 t 所在段落。"""
        for s in self.segments:
            if s["start"] <= t < s["end"]:
                return s
        return None


class DriveEngine:
    """单首歌的运行时驱动引擎。

    线程模型：一个后台线程以固定 tick 推进，检查是否有新的拍/段落需要触发。
    外部通过事件方法（on_start/on_stop/on_pause/on_resume/on_seek）驱动状态。
    触发结果通过 `callback` 输出（默认经 OSC 分发）。
    """

    def __init__(
        self,
        song_id: str,
        drive: DriveFile,
        callback,
        tick_interval: float = 0.005,  # 5ms tick，足够精度且开销低
    ):
        self.song_id = song_id
        self.drive = drive
        self.callback = callback
        self.tick_interval = tick_interval

        self._lock = threading.Lock()
        self._playing = False
        self._pos = 0.0            # 当前播放位置（秒）
        self._last_tick = 0.0      # 上次 tick 的墙钟时间
        self._next_beat_idx = 0    # 下一个待触发的拍索引
        self._cur_segment = None   # 当前段落
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ---- 事件接口（宿主调用） ----

    def on_start(self, pos: float = 0.0):
        with self._lock:
            self._playing = True
            self._set_pos_locked(pos)
            self._emit_state("start")

    def on_stop(self):
        with self._lock:
            self._playing = False
            self._emit_state("stop")

    def on_pause(self):
        with self._lock:
            self._playing = False
            self._emit_state("pause")

    def on_resume(self, pos: Optional[float] = None):
        with self._lock:
            if pos is not None:
                self._set_pos_locked(pos)
            self._playing = True
            self._emit_state("resume")

    def on_seek(self, pos: float):
        with self._lock:
            self._set_pos_locked(pos)
            if self._playing:
                self._emit_state("seek")

    # ---- 内部 ----

    def _set_pos_locked(self, pos: float):
        """设置播放位置，并把下一个待触发拍定位到最近的拍（向前看）。"""
        self._pos = max(0.0, pos)
        # 定位到 >= pos 的第一拍
        self._next_beat_idx = bisect.bisect_left(self.drive.beat_times, self._pos)
        # 立即同步当前段落（seek 后恢复段落状态）
        self._sync_segment_locked()

    def _sync_segment_locked(self):
        seg = self.drive.segment_at(self._pos)
        if seg is not None and (self._cur_segment is None or seg["label"] != self._cur_segment["label"]):
            self._cur_segment = seg
            self._emit("segment", [self.song_id, seg["label"], seg["start"], seg["end"]])

    def _emit(self, addr: str, args: list):
        try:
            self.callback(addr, args)
        except Exception:
            pass

    def _emit_state(self, state: str):
        self._emit("state", [self.song_id, state])

    def _tick(self):
        """推进播放位置，触发到达的拍点。"""
        with self._lock:
            if not self._playing:
                return
            now = time.monotonic()
            dt = now - self._last_tick
            self._last_tick = now
            self._pos += dt

            # 触发所有已到达但未触发的拍
            bt = self.drive.beat_times
            while self._next_beat_idx < len(bt) and bt[self._next_beat_idx] <= self._pos:
                idx = self._next_beat_idx
                b = self.drive.beats[idx]
                self._emit(
                    "beat",
                    [self.song_id, b["t"], b["pos"], int(b["downbeat"]), b["strength"]],
                )
                self._next_beat_idx += 1

            # 段落切换检测
            self._sync_segment_locked()

            # 播放结束
            if self._pos >= self.drive.duration:
                self._playing = False
                self._emit_state("end")

    def _run(self):
        self._last_tick = time.monotonic()
        while not self._stop_event.is_set():
            self._tick()
            time.sleep(self.tick_interval)

    def start_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop_thread(self):
        self._stop_event.set()


class EngineRegistry:
    """多首歌的引擎注册表，按 song_id 路由事件。"""

    def __init__(self, callback):
        self.callback = callback
        self.engines: Dict[str, DriveEngine] = {}

    def load(self, song_id: str, drive_path: str):
        drive = DriveFile.load(drive_path)
        eng = DriveEngine(song_id, drive, self.callback)
        eng.start_thread()
        self.engines[song_id] = eng
        return eng

    def dispatch(self, event: str, song_id: str, *args):
        eng = self.engines.get(song_id)
        if eng is None:
            return
        if event == "start":
            eng.on_start(args[0] if args else 0.0)
        elif event == "stop":
            eng.on_stop()
        elif event == "pause":
            eng.on_pause()
        elif event == "resume":
            eng.on_resume(args[0] if args else None)
        elif event == "seek":
            eng.on_seek(args[0])


def make_osc_callback(client, out_prefix: str = "/drive/out"):
    """构造一个把触发事件转成 OSC 消息的回调。

    out_prefix 下会发：
        /drive/out/beat    <song_id> <t> <pos> <is_downbeat> <strength>
        /drive/out/segment <song_id> <label> <start> <end>
        /drive/out/state   <song_id> <state>
    """
    def cb(addr: str, args: list):
        # addr 是 "beat"/"segment"/"state"，拼到 out_prefix 下
        full = f"{out_prefix}/{addr}"
        # OSC 参数：全部转成浮点/字符串
        msg_args = []
        for a in args:
            if isinstance(a, bool):
                msg_args.append(int(a))
            elif isinstance(a, int):
                msg_args.append(a)
            else:
                msg_args.append(a)
        client.send_message(full, msg_args)

    return cb


def serve_osc(
    drive_map: Dict[str, str],
    listen_host: str = "127.0.0.1",
    listen_port: int = 9000,
    out_host: str = "127.0.0.1",
    out_port: int = 9001,
    control_prefix: str = "/drive/control",
):
    """启动 OSC 服务：监听宿主控制事件，分发到效果层。

    drive_map: {song_id: 驱动文件路径}
    """
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer
    from pythonosc.udp_client import SimpleUDPClient

    client = SimpleUDPClient(out_host, out_port)
    callback = make_osc_callback(client)
    registry = EngineRegistry(callback)

    for song_id, path in drive_map.items():
        registry.load(song_id, path)
        print(f"[drive] 加载 {song_id} <- {path}")

    dispatcher = Dispatcher()
    dispatcher.map(
        f"{control_prefix}/start", lambda a, *v: registry.dispatch("start", str(v[0]), *v[1:])
    )
    dispatcher.map(
        f"{control_prefix}/stop", lambda a, *v: registry.dispatch("stop", str(v[0]))
    )
    dispatcher.map(
        f"{control_prefix}/pause", lambda a, *v: registry.dispatch("pause", str(v[0]))
    )
    dispatcher.map(
        f"{control_prefix}/resume", lambda a, *v: registry.dispatch("resume", str(v[0]), *v[1:])
    )
    dispatcher.map(
        f"{control_prefix}/seek", lambda a, *v: registry.dispatch("seek", str(v[0]), *v[1:])
    )

    print(f"[drive] OSC 监听 {listen_host}:{listen_port}，输出 {out_host}:{out_port}")
    server = BlockingOSCUDPServer((listen_host, listen_port), dispatcher)
    server.serve_forever()
