<div align="center">

# Audio Structure Analyzer 🎵

**音乐结构分析 · Music structure & beat analysis**

一个基于深度学习预训练模型、可扩展的**音乐结构分析**库：
从任意音频中提取**节拍、强拍、段落结构（主歌/副歌…）**，并叠加
可插拔的**流派识别、pre-chorus 检测、EDM 结构识别**等后处理能力，
内置波形可视化播放器。

_A deep-learning–based, extensible library for **music structure analysis**:
extract **beats, downbeats, and section labels (verse/chorus/…)** from any audio,
plus pluggable **genre classification, pre-chorus detection, and EDM structure**
post-processing, with a built-in waveform player._

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)

</div>

---

## 我们在做什么 / What we're building

这是一个**音乐结构理解工具**。给定一首歌（wav / mp3 / flac），它回答：

> 这首歌的 BPM 是多少？每一拍在哪？强拍在哪？
> 从哪到哪是主歌（verse），哪段是副歌（chorus），有没有 pre-chorus 前副歌？
> 如果是一首电子乐，drop / buildup / breakdown 分别在哪？

*This is a **music-structure understanding tool**. Given a track, it answers:
what's the BPM? where are the beats and downbeats? which sections are verse /
chorus, and is there a pre-chorus? For electronic music, where are the drop /
buildup / breakdown?*

技术上，它把一个学术 SOTA 的预训练模型（**all-in-one**, KAIST 的
harmonix-all 模型）薄封装成稳定、易用的 Python API，并在这之上构建了一层
**可插拔的判据（Criterion）+ 后处理（PostProcessor）架构**，用来补全预训练
模型无法直接给出的细粒度语义（如 pre-chorus、EDM drop），以及一个
**离线预处理 → 播放器秒开**的两阶段流水线。

*Technically, it wraps a state-of-the-art pretrained model (**all-in-one**,
KAIST's harmonix-all) into a stable, ergonomic Python API, and builds a
**pluggable Criterion + PostProcessor architecture** on top to recover the
fine-grained semantics the pretrained model can't emit directly (pre-chorus,
EDM drop), plus a **two-stage offline-preprocess → instant-load player** pipeline.*

## 核心能力 / Features

- 🎼 **节拍 / 强拍**：整曲 BPM、逐拍时间 `beats`、每小节强拍 `downbeats`、拍号 `beat_positions`
- 🧱 **段落结构**：Harmonix 10 类语义标签 `start / end / intro / outro / verse / chorus / bridge / break / inst / solo`
- 🔌 **可插拔后处理**：判据（Criterion）与后处理（PostProcessor）解耦，组合出细粒度语义
  - **Pre-chorus 前副歌检测**：融合结构位置、能量匹配、和声走向、配器（贝斯/开镲）多判据交叉验证
  - **流派识别**：MAEST 细粒度模型（`mtg-upf/discogs-maest-10s-dw-75e`），400 类 Discogs 风格多标签分类
  - **EDM 结构识别**：EDMFormer（`edm98`）输出 `intro / buildup / drop / breakdown / outro / silence`
- 🎚️ **可视化播放器**：波形 + 段落色块 + 强拍/拍标记 + 实时游标，同步显示段落切换与拍点日志（终端版 + GUI 版）
- ⚡ **离线预处理**：分析结果并入单份 JSON，播放器纯加载、秒开，无需现场重计算

## 安装 / Installation

```bash
pip install numpy librosa soundfile sounddevice matplotlib

# all-in-one 推理后端（本项目唯一结构分析后端）
pip install git+https://github.com/CPJKU/all-in-one
# 国内镜像（墙内）：
#   git clone https://ghfast.top/https://github.com/CPJKU/all-in-one && pip install -e all-in-one
# HuggingFace 模型若下载失败，用 hf-mirror.com 手动下载 checkpoint 到 torch hub 缓存目录
```

> **说明**：官方 `allin1` 包依赖 `madmom`（Windows/Python 3.12 需 C 扩展编译），
> 本项目改用其纯 PyTorch 推理分叉 `all-in-one-infer`（`allin1_infer`），
> 模型权重与算法仍为官方 harmonix-all（MIT）。详见 [NOTICE](NOTICE)。

可选依赖（按需安装）：

```bash
# 流派识别（MAEST，需 torch 2.x）
pip install "transformers>=4.46,<5.0" torch

# EDM 结构识别（EDMFormer，外部依赖，权重约 1.7GB）
pip install edm98
```

## 快速上手 / Quickstart

### 分析一首歌 / Analyze a track

```python
from audio_structure_analyzer import analyze, load_result

# 直接推理（CPU，几分钟；有 CUDA 传 device="cuda"）
result = analyze("song.mp3", device="cpu")
print(result.bpm)
for s in result.segments:
    print(s.label, s.start, "->", s.end)

# 保存 / 复用结果（避免重复推理）
result.save("out/song.json")
result2 = load_result("out/song.json")
```

### 完整流水线（结构 + 后处理）/ Full pipeline

```python
from audio_structure_analyzer import preprocess, load_bundle

# 一次跑完：结构分析 + 全部后处理，合并写入同一份 JSON
result = preprocess("song.wav", device="cpu")

# 播放器侧：纯读取、秒开
bundle = load_bundle("song.wav")
print(bundle.postprocess.keys())   # {'prechorus', 'style', ...}
```

### 播放器 / Player

```bash
python examples/player_allin1.py path/to/song.wav --json examples/sample_results/pop_structure.json
```

交互：`空格` 播放/暂停，`←→` 快退/快进 5s，`↑↓` 变速，`q` 退出。

## 参考结果 / Sample results

仓库内置两份**脱敏示例结果**（不含音频、只含结构数据），可用作快速体验：

| 文件 | 类型 | BPM | 说明 |
|---|---|---|---|
| `examples/sample_results/pop_structure.json` | 流行结构 | 82 | verse/chorus 结构 + pre-chorus/style 后处理 |
| `examples/sample_results/edm_structure.json` | EDM 结构 | 122 | `analysis_type="edm"`，含 drop/buildup 段落 |

> 结果 JSON 由 `preprocess()` 生成，路径为 `<音频同名>.json`。
> 上面两文件已脱敏（`path` 字段改为占位符），可直接 `load_result()` 加载。

## 项目结构 / Layout

```
audio_structure_analyzer/     # 核心库
  analyzer.py                 #   all-in-one 薄封装（唯一结构分析后端）
  pipeline.py                 #   离线预处理编排（结构+后处理 → 单份 JSON）
  criteria/                   #   可插拔判据（结构/能量/和声/配器）
  postprocess/                #   可插拔后处理（pre-chorus / 流派 / EDM）
  candidate.py                #   候选/结果数据结构
  stems.py                    #   Demucs 声部分离 + 分频段能量
  genre.py                    #   MAEST 流派识别
  edm.py                      #   EDMFormer EDM 结构识别
examples/
  player_allin1.py            # 终端可视化播放器
  player_gui.py               # GUI 播放器（波形+段落+游标）
  sample_results/             # 脱敏示例结果
docs/                         # 架构设计 / EDM 方案 / 微调方案
```

## 架构 / Architecture

```
音频文件 (wav/mp3/flac)
        │
        ▼
allin1_infer.analyze()          # all-in-one: Demucs 分轨 + Transformer 逐帧预测
        │                       # → bpm / beats / downbeats / segments
        ▼
AnalysisResult                  # 统一封装
        │
        ├── criteria/           # 判据层（位置/能量/和声/配器，交叉验证）
        ├── postprocess/        # 后处理层（pre-chorus / 流派 / EDM）
        └── pipeline.preprocess # 合并 → 单份 JSON 落盘
                  │
                  ▼
        player_allin1 / player_gui   # 播放器：纯加载、秒开
```

## 未来要做什么 / Roadmap

- [ ] **中文 / 华语音乐适配**：当前 MAEST 流派模型偏西方电子乐，计划对华语抒情、国风等做数据增强与迁移
- [ ] **微调 harmonix 权重**：用自有标注数据微调，提升 EDM（drop/buildup）结构边界精度（方案见 `docs/finetune.md`，RTX 4060 8GB 可跑）
- [ ] **CLI 批处理**：提供 `asa analyze songs/*.mp3` 一键离线批处理，配合多进程/CUDA 加速
- [ ] **导出标准格式**：输出 Audacity Label Track（`.txt`）、Reaper cue、MIDI tempo map 等 DAW 兼容格式
- [ ] **测试与 CI**：补齐单元测试、GitHub Actions 自动跑 lint + test
- [ ] **发布到 PyPI**：稳定 API 后发布 `pip install audio-structure-analyzer`
- [ ] **Web 播放器**：将 GUI 播放器移植为浏览器版，支持在线试听与结构可视化

## 许可证 / License

本项目 MIT，可商用。分析后端 all-in-one（Taejun Kim & Juhan Nam, KAIST）
与 Demucs（Meta AI）均为 MIT 许可，模型权重按需下载、不随项目分发。
详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

## 致谢 / Acknowledgements

- [All-In-One Music Structure Analyzer](https://github.com/mir-aidj/all-in-one) — Taejun Kim & Juhan Nam (KAIST)
- [Demucs](https://github.com/facebookresearch/demucs) — Meta AI
- [MAEST](https://huggingface.co/mtg-upf/discogs-maest-10s-dw-75e) — Music Technology Group, UPF
