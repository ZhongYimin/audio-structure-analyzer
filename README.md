# 音频结构分析（all-in-one）

基于 HuggingFace 的 **all-in-one**（`allin1_infer`，harmonix-all 模型）做音乐结构分析：
节拍（beat）、强拍（downbeat）、段落结构（verse/chorus/...）。

all-in-one 用 Demucs 声部分离 + Transformer 逐帧预测，对真实演奏的节奏浮动、
进鼓/撤鼓等编曲相位变化有强鲁棒性。

## 能力

- **BPM**：整曲速度（整数）
- **拍点 / 强拍**：逐拍时间 `beats`、每小节强拍 `downbeats`、拍号 `beat_positions`
- **段落结构**：Harmonix 10 类语义标签
  `start / end / intro / verse / chorus / bridge / break / inst / solo / outro`
- **可视化播放器**：波形 + 段落色块 + 强拍/拍标记 + 实时游标，同步显示段落切换与拍点日志

## 目录

```
audio_structure_analyzer/   # 轻量封装（analyzer.py 薄封装 allin1_infer）
examples/player_allin1.py   # 可视化播放器
test_audio/allin1/          # 参考分析结果（*.json）
```

## 安装

```bash
pip install numpy librosa soundfile sounddevice matplotlib
# all-in-one 推理后端
pip install git+https://github.com/CPJKU/all-in-one
# 国内镜像（墙内）：
#   git clone https://ghfast.top/https://github.com/CPJKU/all-in-one && pip install -e all-in-one
# HuggingFace 模型若下载失败，用 hf-mirror.com 手动下载 checkpoint 到 torch hub 缓存目录
```

## 用法

### 分析一首歌

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

### 播放器

```bash
python examples/player_allin1.py path/to/song.wav --json test_audio/allin1/song.json
```

交互：`空格` 播放/暂停，`←→` 快退/快进 5s，`↑↓` 变速，`q` 退出。

## 参考结果

- `test_audio/allin1/guy_contact.json`：电子乐，BPM=123，intro/inst/chorus 结构
- `test_audio/allin1/britpop.json`：Britpop，BPM=136，verse/chorus/inst/outro 结构
