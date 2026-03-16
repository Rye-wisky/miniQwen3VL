# MiniQwen3VL

本项目是对 [MiniMind](https://github.com/jingyaogong/minimind) 模型的深度魔改版。受到 QwenVL 系列（特别是 Qwen3VL）报告的启发，本项目在轻量级 decoder（~90M）的基础上，集成了多项多模态前沿技术，旨在探索超小规模参数下的视觉-语言理解极限。

## 🌟 项目亮点 (Highlights)

本项目不仅是简单的模型组合，而是对底层逻辑进行了大量重构：

- **交错式 MRoPE (Interleaved Multi-modal RoPE)**：参考 Qwen3VL 逻辑，引入交错分布的 MRoPE。通过交错频率分布使感知更加均匀，有效提升了多模态序列的建模能力。
- **YaRN 上下文扩展**：集成 YaRN (Yet another RoPE extension method) 缩放逻辑，使模型在推理时具备更长的上下文感知范围。
- **Qwen 版 SigLIP2 视觉编码器**：采用了从 `Qwen3VL-2B-Instruct` 中提取的魔改版 SigLIP2（约 400M）。该编码器支持**动态分辨率采样**，在视觉细粒度理解上显著优于原生版本。
- **DeepStack 注入机制**：借鉴 DeepStack 思想，将视觉编码器的中间层特征直接注入 Decoder 的浅层（本实现注入第一层），在 Attention 之后、FFN 之前通过残差连接融合，加强底层特征共享。
- **平方根重加权损失函数 (Sqrt Re-weighted Loss)**：采用平方根形式平衡长 token 样本与短 token 样本对 Loss 的贡献，防止在多模态对齐过程中退化文字生成能力。

## 📊 模型规格

| **组件**             | **参数量** | **备注**                         |
| -------------------- | ---------- | -------------------------------- |
| **Language Decoder** | ~90M       | 基于 MiniMind 修改               |
| **Vision Encoder**   | ~400M      | 取自 Qwen3VL-2B (魔改 SigLIP2)   |
| **Total VLM**        | ~500M      | 典型的“头重脚轻”结构，视觉端极强 |

> **总结自述**：目前模型在 Stage 0 表现最像“人”，由于语言侧参数仅 90M，Stage 1/2 后的对齐逻辑对 decoder 的冲击较大。结构流程上已尽可能对齐 Qwen 官方方案。

## 📂 数据集准备 (Dataset Preparation)

请在开始训练前下载并准备好以下数据集：

1. **Stage 0 (文本预训练数据)**
   - **来源**: [ModelScope - minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)
   - **文件**: `pretrain_hq.jsonl`
2. **Stage 1 & 2 (视觉-语言对齐与 SFT 数据)**
   - **来源**: [HuggingFace - minimind-v_dataset](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset/tree/main)
   - **文件**:
     - `pretrain_i2t.parquet` (用于 Stage 1 视觉投影层训练)
     - `sft_i2t.parquet` (用于 Stage 2 全量指令微调)

## 🚀 训练流程 (Workflow)

训练分为三个阶段，请严格按照以下顺序执行：

### 0. 准备工作

首先运行脚本提取 Qwen3VL 的视觉组件, 运行前确保将Qwen3VL-2B-Instruct下载到本地./Qwen3VL-2B：

```
python extract.py
```

### 1. Stage 0: 文本 Decoder 预训练

预训练装有交错式 MRoPE 逻辑的文本 Decoder。

- **注意**：在 `model/model_minimind_modforVL.py`的` MiniMindConfig` 中将默认`vocab_size` 设置为 **6400**（此时无视觉占位符）。

```
python trainer/train_stage0.py
```

### 2. 扩展词表与 Embedding

运行脚本以扩展视觉占位符。

- **操作**：建议将 `model/model_minimind_modforVL.py`的` MiniMindConfig` 中的默认 `vocab_size` 更新为 **6403**，不过扩展后会自动提取出config.json，后续训练脚本均读取的config.json，不修改默认config也可以。

```
python model/expand_tokenizer.py
```

### 3. Stage 1: 视觉-语言对齐 (Alignment)

挂载冻结的视觉编码器和 Decoder，训练两个 Projection 层：

- **Proj 1**: 映射视觉输出至 Embedding 空间。
- **Proj 2**: 映射 DeepStack 中间层特征至 Decoder 隐藏层。

```
python trainer/train_stage1.py
```

### 4. Stage 2: 全量指令微调 (SFT)

冻结视觉编码器，放开 Decoder 和 Projection 层进行多模态 SFT。

```
python trainer/train_stage2.py
```

## 📈 评估 (Evaluation)

项目提供了两个维度的评估脚本：

- **纯文本能力评估**：

  ```
  python eval_llm.py
  ```

- **多模态能力评估 (VLM)**：

  ```
  python eval_vlm.py
  ```

## 🛠️ 环境要求

- Python 3.9+
- PyTorch 2.0+
- Transformers
- (建议) CUDA 11.8+

## 🤝 致谢

感谢 [Qwen 团队](https://github.com/QwenLM/Qwen3-VL) 提供的优秀报告与开源权重，以及 [MiniMind](https://github.com/jingyaogong/minimind) 提供的轻量级实验框架。

**License**: MIT