# 微调方案（Fine-tuning all-in-one harmonix 权重）

## 目标

用官方 harmonix 预训练权重 + 自有标注数据，微调以提升**电子乐/EDM** 曲风的结构分析效果。
服务方向：电子乐/EDM（Drop/Buildup 等结构语义在官方 Harmonix 标签体系中缺失或映射不当）。

## 硬件

- GPU：NVIDIA RTX 4060 Laptop（8GB 显存，CUDA 驱动 13.1，向下兼容 CUDA 12.x）
- 结论：**8GB 显存足够微调**。模型极小（见下），官方默认 batch_size=1、
  segment_size=300 即能在 8GB 内运行，还可用 FP16 + 梯度累积进一步放宽。

## 模型规模（为什么 4060 够用）

| 参数 | 值 |
|---|---|
| depth（层数） | 11 |
| dim_embed（嵌入维度） | 24 |
| num_heads | 2 |
| kernel_size | 5 |
| 单个 checkpoint 体积 | ~1.4 MB |

这是一个「小而深」的模型，参数量远小于常见 Transformer，是 4060 能微调的关键。

## 模型结构（微调切入点）

```
输入: (N, K=4声部, T时间帧, F=81频点)   # 4 声部来自 Demucs 分离
  ↓ AllInOneEmbeddings（卷积嵌入，C=16）
  ↓ AllInOneEncoder（11 层邻域注意力 Transformer）
  ↓ LayerNorm + Dropout
  ↓ 4 个分类头（各自输出 logits，可独立微调）:
    ├─ beat_classifier        (num_classes=1)  节拍
    ├─ downbeat_classifier    (num_classes=1)  强拍
    ├─ section_classifier     (num_classes=1)  段落边界
    └─ function_classifier    (num_classes=10) 功能标签 verse/chorus/...
```

**微调策略**：冻结 encoder，只微调 4 个分类头（或重点微调 function 头，
如果主要关心段落标签）。学习率降到 1e-4 ~ 1e-5（官方从头训练用 0.005）。

## 关键技术结论（重要）

### NATTEN 编译问题已绕开

- 官方 `allin1` 训练依赖 NATTEN（邻域注意力），Windows 需源码编译，本机 MSVC 缺 SDK 头文件。
- **`allin1_infer` 已内置纯 PyTorch NATTEN 替代**（`models/neighborhood_attention.py`：
  `na1d_qk/na1d_av/na2d_qk/na2d_av`），用 `gather + einsum` 精确复刻 NATTEN 0.17.x 语义，
  数值逐位一致（golden-fixture 测试背书）。
- 这些操作全部可微（`einsum`/`gather` 支持反向传播），**可直接用于训练/微调**，
  彻底绕开 NATTEN 编译。

### 训练数据是核心瓶颈

- Harmonix Set 只发布**标注**（JAMS），不发布音频（版权），音频需自行收集对齐。
- 微调效果 90% 取决于自有标注数据质量与规模，不取决于 GPU。
- 标注格式：beats/downbeats/segments 三层（Harmonix JAMS 格式，官方代码直接读）。

## 待办（按顺序）

1. [x] 换装 CUDA 版 torch（cu121）—— 让 4060 真正参与计算
2. [ ] 验证 torch.cuda.is_available() == True
3. [ ] 获取官方训练代码（allin1[train]）或基于 allin1_infer 的纯 PyTorch 模型搭建微调脚本
4. [ ] 准备 EDM 标注数据（音频 + beats/downbeats/segments）
5. [ ] 冻结 encoder，微调分类头，FP16 + 梯度累积
