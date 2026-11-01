"""音乐流派识别（基于 MAEST 模型）。

用 MTG（Music Technology Group，即 MTG-Jamendo 数据集团队）的 MAEST 模型
做**细粒度、多标签**音乐风格分类：

- 模型：``mtg-upf/discogs-maest-10s-dw-75e``（75M 参数，Audio Transformer）
- 标签：400 个 Discogs 音乐风格，形如 ``大类---子类``
  （如 ``Rock---Pop Rock``、``Pop---J-pop``、``Electronic---Synth-pop``）
- 多标签：``top_k`` 返回多个风格及置信度，一首歌可同时命中 pop + rock
  （如 "pop rock" 这类混合风格）

相比规则式流派判据，这是**真正的 AI 推理**：模型在 330 万首 Discogs 曲目
上学习「什么声音像 Pop Rock / J-pop / Synth-pop」，能给出规则无法表达的
文化 + 地域细粒度标签。

设计要点：
- 惰性加载 + 进程内单例缓存（模型较大，避免重复加载）；
- 走 transformers 的 ``audio-classification`` pipeline；
- 有 CUDA 时用 GPU 推理（每首约 0.2~0.4 秒），无 GPU 回退 CPU；
- 音频输入统一重采样到模型要求的 16kHz 单声道。

局限（需知晓）：
- MAEST 训练数据偏西方电子乐，对华语抒情 / 英伦摇滚等可能有偏差；
- 模型按 10 秒片段训练，长歌取整段作为输入。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

MODEL_NAME = "mtg-upf/discogs-maest-10s-dw-75e"
TARGET_SR = 16000


class GenreClassifier:
    """MAEST 细粒度流派分类器（惰性加载，进程内单例缓存）。"""

    _instance: Optional["GenreClassifier"] = None

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[int] = None):
        self.model_name = model_name
        # device: 0 = GPU, -1 = CPU；缺省自动选 CUDA
        if device is None:
            import torch

            device = 0 if torch.cuda.is_available() else -1
        self.device = device
        self._pipe = None

    @classmethod
    def get(cls, model_name: str = MODEL_NAME) -> "GenreClassifier":
        """获取进程内单例。"""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        from transformers import pipeline

        self._pipe = pipeline(
            "audio-classification",
            model=self.model_name,
            trust_remote_code=True,
            device=self.device,
        )

    def classify(
        self,
        y: np.ndarray,
        sr: int,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """对音频做细粒度流派分类。

        Parameters
        ----------
        y, sr : 音频波形与采样率（任意采样率，内部重采样到 16kHz）。
        top_k : 返回前 k 个最匹配的风格标签。

        Returns
        -------
        list[(str, float)]
            [(风格标签, 置信度)]，按置信度降序。标签形如 ``Rock---Pop Rock``。
        """
        import librosa

        self._ensure_loaded()
        if sr != TARGET_SR:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        out = self._pipe(y, top_k=top_k)
        return [(r["label"], round(float(r["score"]), 4)) for r in out]


__all__ = ["GenreClassifier", "MODEL_NAME", "TARGET_SR"]
