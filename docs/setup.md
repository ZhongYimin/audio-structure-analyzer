# 环境配置与运行指南

> 本文档帮你从零配好环境并跑通音频结构分析框架。

## 1. 确认 Python（3.12 最佳）

```powershell
python --version
```

- 期望输出 `Python 3.12.x`。
- 若显示 `3.13.x` 且路径含 `.workbuddy`，那是 WorkBuddy 自带的解释器，建议另装官方 3.12。
- 本项目实际使用 `C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe`。

### 安装官方 Python 3.12（如需要）

1. 访问 https://www.python.org/downloads/release/python-3120/
2. 下载 **Windows installer (64-bit)**
3. 安装时**务必勾选 "Add python.exe to PATH"**
4. 装完**重开终端**，验证 `python --version`

## 2. 安装依赖

在项目根目录执行：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> 网络慢时加国内镜像：
> ```powershell
> python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

装完验证：

```powershell
python -c "import numpy, scipy, sklearn, librosa, soundfile; print('依赖 OK')"
```

### 可选：声部分离增强（Demucs）

需安装 CPU 版 torch/torchaudio（结构 embedding 通道，不装会自动回退 chroma）：

```powershell
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 3. 运行框架

```powershell
python examples\analyze_song.py 你的歌曲.mp3
```

保存结果到 JSON：

```powershell
python examples\analyze_song.py 你的歌曲.mp3 --json out.json
```

生成可视化图（需 matplotlib）：

```powershell
python examples\analyze_song.py 你的歌曲.mp3 --plot out.png
```

## 4. 播放器（验证段落/拍子/kick 对齐）

```powershell
python examples\player.py 你的歌曲.mp3 --json out.json
```

交互：`空格` 播放/暂停，`←/→` 快退/快进 5s，`↑/↓` 变速，`q` 退出。

## 5. 输出说明

结果 JSON 结构（节选）：

```json
{
  "bpm": 123.08,
  "kick_count": 668,
  "sections": [
    {
      "start": 0.0, "end": 39.0, "label": "Ap", "confidence": 0.72,
      "tempo": 123.08, "beat_count": 80, "bar_count": 20,
      "beat_times": [1.02, 1.51, 2.0, "..."],
      "downbeat_times": [2.0, 3.95, "..."],
      "kick_times": [0.0, 0.24, 0.49, "..."],
      "kick_strengths": [0.42, 0.31, 0.55, "..."],
      "kick_intervals_ms": [240.0, 250.0, "..."]
    }
  ]
}
```

- **Ap** = 主歌（Verse）　**Dp** = 铺垫（Build）　**S** = 副歌（Chorus）　**Cp** = 副歌后（Post-Chorus）
- `beat_times` = 每拍时间点　`downbeat_times` = 每小节强拍　`kick_times` = 每一次底鼓击打

## 6. 常见问题

| 问题 | 解决 |
|---|---|
| `ModuleNotFoundError: librosa` | 依赖没装全，重跑第 2 步 |
| librosa 装不上（numba 报错） | 改用 Python 3.12（3.13 太新，numba 可能未适配） |
| pip 下载慢/超时 | 加清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| mp3 读不了 / 被提前截断 | 确保装了 `soundfile`（框架用它而非 librosa 读 mp3） |
| torch 装不上（WinError 5） | 加 `--user`，或用 PyTorch CPU 官方源 |
| 播放器文字延迟 / beat 不匀 | 已修复：低延迟流 + 回调样本位置作为唯一真相源；确认 sounddevice 正常 |
