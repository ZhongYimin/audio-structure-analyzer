# 架构说明

本项目当前唯一分析后端是 **all-in-one**（`allin1_infer`，harmonix-all 模型），
不再包含任何自研的 BPM / 拍网格 / kick / 结构分割算法。

## 数据流

```
音频文件 (wav/mp3/flac)
        │
        ▼
allin1_infer.analyze()
  ├─ Demucs 声部分离（drums/bass/other/vocals）
  ├─ Transformer 逐帧预测节拍 + 结构标签
  └─ 后处理：BPM 取整、拍点/强拍、段落归并
        │
        ▼
AnalysisResult（封装）
  ├─ bpm / beats / downbeats / beat_positions
  └─ segments（Harmonix 10 类标签）
        │
        ├─ save() → JSON（供复用）
        └─ player_allin1.py → 可视化播放器
```

## 封装层

`audio_structure_analyzer/analyzer.py` 是对 `allin1_infer` 的薄封装，只做三件事：

1. `analyze()`：调 `allin1_infer.analyze(...)` 并转成统一 `AnalysisResult`。
2. `load_result()`：从 JSON 复用结果，避免重复推理。
3. `save()`：把结果写回 JSON（与 all-in-one 原生 JSON 兼容）。

## 为什么选 all-in-one（而非自研）

历史版本曾尝试自研「低频自相关 BPM + 频谱 onset 相位 + 均匀拍网格」方案，
实测发现两个不可克服的缺陷：

1. **相位漂移**：真实乐队（Britpop 等）「前奏无鼓 → 副歌进鼓 → breakdown 撤鼓」，
   重音落点相位随编曲变化（无鼓段 ~207ms vs 有鼓段 ~34ms），单一全局相位必然错位。
2. **死均匀网格忽略演奏呼吸**：真实拍点有 ±10~15ms 浮动，锁死 BPM 后机械均匀推进
   （std≈0.4ms）反而丢失了真实节奏信息。

all-in-one 用 Transformer 逐帧预测节拍概率，天然跟踪这些变化，拍间隔 std≈11ms
且逐拍贴合真实重音，是稳定可靠的选择。

## 标签

Harmonix 数据集 10 类结构标签：`start / end / intro / outro / verse / chorus /
bridge / break / inst / solo`。
