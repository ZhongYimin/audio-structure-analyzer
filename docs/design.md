# 音频结构分析框架 — 系统设计文档

> 版本：0.1.0　|　配套方法论见 [`methodology.md`](methodology.md)，环境配置见 [`setup.md`](setup.md)。

## 1. 设计目标

输入一首歌的音频文件，输出其时间轴上的：

1. **结构分段**，每段打上四类标签之一（Ap/Dp/S/Cp）；
2. **BPM**（每分钟拍数）；
3. **拍点（beat）与强拍（downbeat）**；
4. **每一次 kick（底鼓）击打事件**（时间 + 强度）。

| 标签 | 含义 | 英文 |
|---|---|---|
| **Ap** | 主歌段落 | Verse |
| **Dp** | 铺垫段落 | Pre-Chorus / Build |
| **S** | 副歌段落 | Chorus |
| **Cp** | 副歌后段落 | Post-Chorus |

**设计原则：**

1. **管道化（Pipeline）**：分析拆成可独立替换的多个阶段，每阶段只做一件事。
2. **可扩展（Extensible）**：规则分类器可替换/融合为 ML 模型；特征可插拔；声部分离作为可选增强。
3. **可解释（Interpretable）**：默认用可量化的特征 + 规则，输出带置信度。
4. **模块化（Modular）**：每个阶段是独立子包，通过明确的接口（函数签名 + 数据结构）衔接。

---

## 2. 总体架构

```
                     audio file
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │  1. io.load_audio()                        │
        │     soundfile 解码 → 单声道 → 重采样        │
        │     → 静音裁剪 → 响度归一 → (y, sr)         │
        └───────────────────────┬────────────────────┘
                                ▼
        ┌────────────────────────────────────────────┐
        │  2. features.extract_features()            │
        │     帧级特征 + 块级特征（Segment）           │
        │     features.embedding（可选 Demucs 声部）  │
        └───────────────────────┬────────────────────┘
                                ▼
        ┌────────────────────────────────────────────┐
        │  3. segmentation.detect_boundaries()       │
        │     chroma 层次聚类 + 能量低谷              │
        │     segmentation.beat（BPM/kick/拍网格）    │
        └───────────────────────┬────────────────────┘
                                ▼
        ┌────────────────────────────────────────────┐
        │  4. classification.classify()              │
        │     段落归并 + 能量分层 + 重复度 + 时序约束  │
        └───────────────────────┬────────────────────┘
                                ▼
        ┌────────────────────────────────────────────┐
        │  5. io.to_json() / viz.plot_sections()     │
        │     AnalysisResult → JSON / 可视化          │
        └────────────────────────────────────────────┘
```

主入口 `analyze()`（在 `__init__.py`）按上述顺序编排。

---

## 3. 模块职责与接口

### 3.1 `core.py` — 数据结构（契约层）

```python
LABELS = ("Ap", "Dp", "S", "Cp")
EXTRA_LABELS = ("Intro", "Interlude", "Outro", "Bridge")

@dataclass
class Segment:          # 时间块（特征提取的最小单位）
    start: float
    end: float
    features: np.ndarray

@dataclass
class Section:          # 归并后的段落（最终输出单元）
    start: float
    end: float
    label: str
    confidence: float
    tempo: float            # BPM
    beat_count: int         # 拍数
    bar_count: int          # 小节数
    beat_times: List[float]
    downbeat_times: List[float]
    kick_times: List[float]      # 每一次 kick 时间
    kick_strengths: List[float]  # kick 强度
    kick_intervals: List[float]  # kick 间隔

@dataclass
class AnalysisResult:   # 一首歌的完整结果
    path, duration, sr, bpm
    sections: List[Section]
    kick_times, kick_strengths  # 全曲 kick 事件
```

### 3.2 `io/` — 加载与输出

| 函数 | 职责 |
|---|---|
| `load_audio(path, target_sr=22050, mono, normalize_lufs)` | soundfile 解码 → 单声道 → 重采样 → 静音裁剪 → 响度归一 |
| `to_json(result)` / `write_json(result, path)` | 结果序列化为 JSON |

> 注：用 `soundfile` 而非 `librosa.load` 读 mp3，因为后者对 VBR mp3 的 Xing 头处理不准会提前截断。

### 3.3 `features/` — 特征提取

| 函数 | 职责 |
|---|---|
| `extract_frame_features(y, sr)` | 逐帧提取 8 维标量特征 |
| `frames_to_segments(...)` | 帧级 → 块级聚合（均值/标准差/斜率） |
| `extract_features(y, sr, ...)` | 组合以上两步，返回 `(frames, segments, extra)` |
| `extract_embedding_features(...)` | **可选**：Demucs 声部分离 → 结构 embedding（失败返回 None） |

**帧级 8 维特征**（顺序固定，见 `FRAME_FEATURES`）：

```
[0] rms                [1] spectral_centroid  [2] spectral_bandwidth
[3] spectral_flux      [4] chroma_mean        [5] chroma_std
[6] mfcc_1             [7] mfcc_2
```

**块级 8 维统计量**（`frames_to_segments` 生成）：

```
[0] rms_mean  [1] rms_std  [2] rms_slope  [3] centroid_mean
[4] bandwidth_mean  [5] flux_mean  [6] chroma_mean  [7] chroma_std
```

**声部分离 embedding**（`embedding.py`，可选，依赖 torch/torchaudio）：

- `separate_stems`：用 `HDEMUCS_HIGH_MUSDB` 分离 drums/bass/other/vocals，分块滑窗（60s + 2s 重叠 + 三角窗平滑）处理长音频，结果落盘缓存到 `<audio>/.asa_cache/stems_<hash>.npz`。
- `stem_features`：每个声部能量 + bass/other chroma + vocal 存在性，拼成帧级特征矩阵。

### 3.4 `segmentation/` — 结构分割

| 函数 | 职责 |
|---|---|
| `detect_boundaries(frames, sr, ...)` | chroma_cqt 层次聚类（和声结构）+ RMS 能量低谷（breakdown）检测 |

当前实现（v4）：

1. `chroma_cqt` → `librosa.segment.agglomerative` 层次聚类，得到和声结构边界；
2. RMS 能量包络找「低于邻域中位数 35% 且持续 >2s」的低谷段作为 breakdown 边界；
3. 合并、去重，`min_segment_sec`（默认 8s）约束最小段长。

### 3.5 `segmentation/beat.py` — 节拍 / kick / 拍网格

| 函数 | 职责 |
|---|---|
| `detect_bpm(y, sr)` | 低频能量自相关 + onset 交叉验证，估计全局 BPM |
| `_estimate_phase(y, sr, tempo)` | 用全频谱 onset 起音的圆平均相位估计拍网格偏移 |
| `track_beats(y, sr, tempo)` | 均匀拍网格推进，返回 `(tempo, beat_times, kick_times)` |
| `detect_kicks(y, sr)` | 低频（40~120Hz）能量峰值检测，输出 kick 事件 |
| `analyze_beats(y, sr, sections)` | 逐段落切分拍点/kick，边界对齐到拍 |

**核心设计（v3，修正「用 kick 单点锚定拍」导致的错拍）：**

1. **BPM 检测**：低频（kick/bass，<120Hz）能量自相关为主，onset 交叉验证。
2. **相位估计**：**不用 kick 单点**（kick 有 off-beat 弱击打，会把网格拉偏），
   而是用**全频谱 onset_strength**（整体频谱能量的起音，即「一拍」的概念）
   做圆平均相位搜索。这符合「用频谱特征把握『一拍』而非 kick 单独」的直觉。
3. **均匀拍网格**：锁定 BPM + phase 后全局均匀推进，拍长恒定 ≈ 60/BPM，
   不再逐拍吸附 kick（避免 off-beat 拉偏）。
4. **kick 事件**：独立检测每一次真实 kick（含正拍 + off-beat），与拍网格解耦。

> 实测：本方案在 `guy_contact.wav` 上测得 123.1 BPM，Ap（0–39s）80 拍、20 小节，
> 与金标准完全一致。

### 3.6 `classification/` — 段落归并与分类

| 函数 | 职责 |
|---|---|
| `_group_segments(segments, boundaries)` | 按边界把块归并为段落组 |
| `_find_recurring_sections(...)` | 段落相似度聚类，找「重复出现的段落组」（副歌候选） |
| `classify(...)` | 能量分层 + 重复度 + 时序约束，输出 `List[Section]` |
| `_apply_timeline_constraints(...)` | 用音乐结构时序合法性修正标签 |

### 3.7 `dsp/` — 底层工具

`slope()`（最小二乘斜率）、`zscore()`（标准归一化）、`frame_windows()`。

### 3.8 `viz.py` — 可视化

`plot_sections(audio_path, result, out_path)`：波形 + 段落色块 + 标签 + 拍点/强拍。

---

## 4. 核心算法：段落标签判定

### 4.1 副歌（S）识别 — 结构重复 + 能量

副歌最可靠的信号是「在全曲重复出现且彼此高度相似」。流程：

1. `_find_recurring_sections` 用 chroma 直方图 + MFCC 均值作为段落特征，算余弦相似度矩阵；
2. 用「中位数 + 3×MAD」动态阈值聚类；
3. 成员数 ≥ 2 的最大组 → 副歌候选组。

### 4.2 分类优先级（从高到低）

```
1. 属于重复段落组 且 能量 ≥ low_thr      → S
2. 能量 ≥ high_thr（EDM drop）           → S
3. 能量爬升（slope > 0.05）              → Dp
4. 能量 ≤ low_thr                        → Ap
5. 其余                                  → Ap（兜底）
```

### 4.3 时序约束后处理

```
- Dp 只能紧邻某个 S 之前（能量爬升）
- Cp 只能紧跟某个 S 之后（能量回落）
```

---

## 5. 数据流示例

```
y[22050Hz, 60s]  → 帧特征 (1292, 8)  → 块 Segment ×16
                 → chroma 层次聚类    → 边界 [0, 14.9, 30.1, 44.9]
                 → 段落组 ×4          → 相似度聚类 → 副歌组{1,3}
                 → 标签 [Ap/Dp, S, Ap/Dp, S]
                 → beat/kick 分析     → BPM + 拍点 + kick 事件
```

---

## 6. 扩展点（未来增强）

| 扩展方向 | 介入位置 | 说明 |
|---|---|---|
| 人声分离 | `features/` | 已接入 Demucs，可继续优化声部特征 |
| 节拍对齐边界 | `segmentation/` | 边界对齐到拍/小节网格，容差 ±1 小节 |
| ML 分类器 | `classification/` | 规则分类器替换/融合为 GBDT/序列模型，需标注数据 |
| 序列解码 | `classification/` | HMM/CRF 平滑标签序列 |
| 曲式细分（S1→S2） | `segmentation/` | 副歌内部乐句边界属曲式语义，需更强先验或人工锚点 |
| 多格式输入 | `io/` | 支持 MIDI、歌词时间轴等 |

---

## 7. 关键设计权衡

| 权衡 | 选择 | 理由 |
|---|---|---|
| 采样率 | 22050 Hz | 结构分析够用，省一半算力 |
| 块长 | 2 秒 | 平衡边界分辨率与特征稳定性 |
| 最小段长 | 8 秒 | 过滤噪声边界，避免过分割 |
| BPM 通道 | 低频自相关 | kick 最规整，octave 歧义最小 |
| 拍网格 | 均匀网格 + onset 相位 | 不吸附 kick，避免 off-beat 拉偏 |
| 特征归一化 | 全曲相对排名 | 降低曲风差异影响 |
