# Audio Structure Analyzer

一个 Python 音频结构分析框架：输入一首歌的音频文件，输出**段落级标签、BPM、拍点（beat）、强拍（downbeat）与每一次 kick（底鼓）事件**。

> 方法论见 [`docs/methodology.md`](docs/methodology.md)，系统设计见 [`docs/design.md`](docs/design.md)，环境配置见 [`docs/setup.md`](docs/setup.md)。

## 特性

- **音频加载与预处理**：soundfile 稳健解码 → 单声道 → 重采样 → 静音裁剪 → 响度归一。
- **BPM 检测**：低频（kick/bass）能量自相关 + onset 交叉验证，稳定测出真实 tempo（如 123 BPM）。
- **拍网格（均匀网格 + onset 相位）**：用全频谱 onset 起音估计相位偏移，锁定 BPM 后全局均匀推进，**不再逐拍吸附 kick**，避免 off-beat 弱击打把网格拉偏。
- **kick 事件检测**：低频（40~120Hz）能量峰值检测，输出每一次底鼓击打（含正拍 + off-beat），与拍网格解耦。
- **结构边界检测**：chroma 层次聚类（和声结构）+ 能量低谷（breakdown）检测。
- **声部分离增强（可选）**：Demucs 分离 drums/bass/other/vocals，提取更丰富的结构 embedding（失败自动回退 chroma）。
- **段落分类**：能量分层 + 结构重复度（副歌识别）+ 时序约束 → Ap/Dp/S/Cp。
- **结构化输出 + 可视化播放器**：JSON 结果 + 波形/段落/拍点/kick 播放器。

## 目录结构

```
audio_structure_analyzer/
├── audio_structure_analyzer/     # 主包
│   ├── core.py                   # 数据结构（Segment/Section/AnalysisResult）
│   ├── io/                       # 音频加载、结果输出
│   ├── dsp/                      # 底层 DSP 工具（slope/zscore）
│   ├── features/                 # 特征提取 + 声部分离 embedding
│   ├── segmentation/             # 结构分割 + 节拍/kick 分析
│   └── classification/           # 段落归并 + 分类
├── docs/                         # 文档（methodology/design/setup）
├── examples/                     # 示例脚本（分析、诊断、播放器）
├── tests/                        # 测试
├── requirements.txt
└── pyproject.toml
```

## 安装

```bash
cd audio_structure_analyzer
pip install -r requirements.txt
```

> 可选增强（声部分离）需额外安装 torch/torchaudio，详见 `docs/setup.md`。

## 快速开始

```bash
python examples/analyze_song.py path/to/song.mp3
```

保存结果到 JSON：

```bash
python examples/analyze_song.py path/to/song.mp3 --json out.json
```

生成可视化图（需 matplotlib）：

```bash
python examples/analyze_song.py path/to/song.mp3 --plot out.png
```

输出示例（JSON，节选）：

```json
{
  "bpm": 123.08,
  "kick_count": 668,
  "sections": [
    {
      "start": 0.0, "end": 39.0, "label": "Ap",
      "beat_count": 80, "bar_count": 20,
      "beat_times": [1.02, 1.51, 2.0, "..."],
      "downbeat_times": [2.0, 3.95, "..."],
      "kick_times": [0.0, 0.24, 0.49, "..."]
    }
  ]
}
```

标签定义：

| 标签 | 含义 | 英文 |
|---|---|---|
| **Ap** | 主歌段落 | Verse |
| **Dp** | 铺垫段落 | Pre-Chorus / Build |
| **S** | 副歌段落 | Chorus |
| **Cp** | 副歌后段落 | Post-Chorus |

## 播放器（验证对齐）

```bash
python examples/player.py path/to/song.mp3 --json result.json
```

## 许可证

项目立项阶段，暂未指定许可证。
