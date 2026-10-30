# 输出 EDM 结构标签的方案（build-up / drop / breakdown）

## 背景

all-in-one（harmonix-all）的 function 头是**固定 10 类**的 softmax 分类器
（intro/verse/chorus/bridge/break/inst/solo/outro/start/end）。它**无法直接输出**
电子乐/EDM 的 build-up / drop / breakdown 这套标签 —— 这套语义在 Harmonix 数据集中
不存在等价标签。

本文件梳理四条让模型输出目标分段信息的路线，并给出当前推荐方案。

## 四条路线（由轻到重）

| # | 路线 | 成本 | 能否输出 build-up/drop | 结论 |
|---|---|---|---|---|
| ① | 标签映射（Harmonix→自定义） | 分钟级、零训练 | ❌ 缺语义等价 | 只能换名，不能补语义 |
| ② | 逐帧激活值 + 特征阈值后处理 | 中等，无需训练 | ✅ 用能量/低频/kick 特征重定义 | **推荐（短期）** |
| ③ | 微调 function 头 | 高，需标注数据+GPU | ✅ 原生输出新标签 | 长期目标 |
| ④ | LLM 语义重标 | 中，引入外部依赖 | ✅ 灵活但不可控 | 备选 |

详见 `finetune.md`（对应路线 ③）。

## 推荐方案：路线 ② —— 模型当特征提取器，规则层定义 EDM 语义

核心洞察：`analyze(include_activations=True)` 可输出**逐帧 function 激活值**
（`activations` 字典：每个标签一帧一概率），比离散 `segments` 更细。在这之上叠加
声学特征，用规则重定义边界。

### 数据来源

- **模型输出**：节拍网格（bpm/beats/downbeats）+ 逐帧 function 激活值
- **声学特征**（本项目早期已验证有效）：
  - 低频能量包络（40~120Hz）→ 判断底鼓/低频强弱
  - kick 事件密度 → 鼓点稀疏/密集
  - 能量爬升/低谷 → build-up 的 riser、breakdown 的抽离
  - onset 强度 → 段落起音密度

### EDM 标签的判据

| 目标标签 | 判据（特征） |
|---|---|
| Intro | 能量低、无鼓或稀疏鼓、结构重复 |
| Build-up | 能量单调爬升 + 高频 riser 渐强 + kick 越来越密 |
| Drop | 能量峰值 + 低频强 kick 密集 |
| Breakdown | 低频能量骤降 + 鼓点撤走（本项早期实测：Cp 段低频均值仅 ~1.2）|
| Outro | 能量衰减 + 结构收束 |

### 优点 / 缺点

- 优点：无需标注数据、无需训练、可解释、复用早期已验证的特征。
- 缺点：本质仍是"规则"，边界精度取决于特征质量，阈值需调参。

## 关键事实（已实测确认）

- `allin1_infer.analyze` 支持 `include_activations=True`（逐帧概率）与
  `include_embeddings=True`（中间层 embedding）。
- `AnalysisResult` 含 `activations: Dict[str, NDArray]` + `activation_fps` 字段，
  激活值以 `.activ.npz` 落盘，可由 `from_json` 回读。

## 待办

1. [ ] 跑 `analyze(include_activations=True)`，dump 逐帧激活值确认其 shape 与语义
2. [ ] 复用低频能量 / kick 密度特征，实现 build-up / drop / breakdown 规则层
3. [ ] 用 Guy_Contact（电子乐）验证规则层输出是否符合预期
4. [ ] 长期：积累 EDM 标注数据后，转路线 ③ 微调 function 头替换规则层
