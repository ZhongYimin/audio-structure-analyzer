"""EDM 结构分析封装（EDMFormer，外部依赖）。

与 ``allin1_infer``（all-in-one，10 类流行结构标签）并列，这是第二套
分析方式：特化电子音乐的 **EDMFormer**（6 类 EDM 语义标签）。

标签（``edm98.constants.EDM98_LABELS``）：
    intro / buildup / drop / breakdown / outro / silence

设计约定：
- EDMFormer 是**外部依赖**（``edm98`` PyPI 包 + MuQ + MusicFM 两个预训练
  音频编码器 + 约 1.7GB 权重），不 vendored 进本仓库。本模块只是轻量封装。
- 推理走 **fp16（半精度）+ persistent_models**，不用官方 slow 的 low_memory
  路径（low_memory 会在每个分块之间反复加载/卸载模型，极慢）。
- 权重与 MusicFM 源码路径由 edm98 包内部解析（默认指向其安装目录的
  data/checkpoints 与 third_party/musicfm）。

用法::

    from audio_structure_analyzer.edm import analyze_edm
    segs = analyze_edm("song.wav")   # -> [{"label": "drop", "start": .., "end": ..}, ...]
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class EDMFormer:
    """EDMFormer 推理管线的惰性单例封装（fp16 + persistent_models）。

    首次调用时加载三个模型（MuQ + MusicFM + EDMFormer），之后常驻内存复用。
    """

    MODEL_PKG = "edm98"
    _instance: Optional["EDMFormer"] = None
    _lock = threading.Lock()

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipeline = None

    @classmethod
    def get(cls, device: str = "cuda") -> "EDMFormer":
        """获取全局单例（线程安全）。"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(device)
            return cls._instance

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device is not None:
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_loaded(self):
        if self._pipeline is not None:
            return
        try:
            from edm98.inference import create_pipeline
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "EDMFormer 推理依赖未安装（edm98）。请先安装 edm98、MuQ、MusicFM 并下载权重。"
            ) from e

        # fp16 半精度 + persistent_models（常驻内存），不用 low_memory 慢路径。
        # offline=True：权重已缓存时跳过 HF 的 HEAD 请求（否则无代理会 ReadTimeout）。
        self._pipeline = create_pipeline(
            device=self.device,
            persistent_models=True,
            fp16=True,
            low_memory=False,
            offline=True,
        )

    def predict(self, audio_path, progress_cb=None) -> List[dict]:
        """对音频做 EDM 结构识别，返回段落列表。

        返回：``[{"label": "drop", "start": 30.0, "end": 97.0}, ...]``
        标签 ∈ {intro, buildup, drop, breakdown, outro, silence}。

        :param progress_cb: 可选回调 ``(progress: float, desc: str)``，
            进度 0~1 + 阶段描述，由 EDMFormer 的 predict_file 触发。
        """
        self._ensure_loaded()

        def _cb(progress, desc=None):
            if progress_cb:
                progress_cb(progress, desc)

        return self._pipeline.predict_file(str(audio_path), progress_callback=_cb)


def analyze_edm(audio_path, device: Optional[str] = None, progress_cb=None) -> List[dict]:
    """便捷入口：对一首歌做 EDM 结构识别。

    :param audio_path: 音频文件路径（wav/mp3 等，内部重采样到 24kHz）
    :param device: 推理设备（"cuda"/"cpu"/None=自动）
    :param progress_cb: 可选回调 ``(progress: float, desc: str)``，
        进度 0~1 + 阶段描述。
    :return: EDM 段落列表 ``[{"label", "start", "end"}, ...]``
    """
    clf = EDMFormer.get(device=_resolve_auto_device(device))
    return clf.predict(audio_path, progress_cb=progress_cb)


def _resolve_auto_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"
