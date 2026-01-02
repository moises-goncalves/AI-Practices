# 预训练模型知识点详解

> **知识密度**：⭐⭐⭐⭐⭐ | **实战价值**：⭐⭐⭐⭐⭐

---

## 目录

1. [预训练范式](#1-预训练范式)
2. [GPT系列](#2-gpt系列)
3. [BERT系列](#3-bert系列)
4. [LLaMA系列](#4-llama系列)
5. [缩放定律](#5-缩放定律)
6. [架构对比](#6-架构对比)
7. [常见问题FAQ](#7-常见问题faq)

---

## 1. 预训练范式

### 1.1 自监督学习

**核心思想**：从无标注数据中构造监督信号

| 任务类型 | 描述 | 代表模型 |
|---------|------|---------|
| **CLM** | 因果语言建模，预测下一个token | GPT |
| **MLM** | 掩码语言建模，预测被掩盖的token | BERT |
| **PLM** | 排列语言建模，随机顺序预测 | XLNet |
| **DAE** | 去噪自编码，重建被破坏的输入 | BART, T5 |

### 1.2 因果语言建模 (CLM)

$$\mathcal{L}_{CLM} = -\sum_{t=1}^{T} \log P(x_t | x_1, ..., x_{t-1}; \theta)$$

**特点**：
- 自回归生成
- 只能看到过去的token
- 适合文本生成任务

### 1.3 掩码语言建模 (MLM)

$$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}}; \theta)$$

**BERT掩码策略**：
- 15%的token被选中
- 其中80%替换为[MASK]
- 10%替换为随机token
- 10%保持不变

---

## 2. GPT系列

### 2.1 架构演进

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头 | 上下文长度 |
|------|--------|------|---------|---------|-----------|
| GPT-1 | 117M | 12 | 768 | 12 | 512 |
| GPT-2 | 1.5B | 48 | 1600 | 25 | 1024 |
| GPT-3 | 175B | 96 | 12288 | 96 | 2048 |
| GPT-4 | ~1.8T* | - | - | - | 8K/32K/128K |

### 2.2 GPT核心架构

```
输入 → Token Embedding + Position Embedding
     → [Decoder Block × N]
     → Layer Norm → LM Head → 输出概率
```

**Decoder Block**：
```
x → Layer Norm → Causal Self-Attention → + → Layer Norm → FFN → +
    └──────────────────────────────────────┘   └─────────────────┘
                   残差连接                          残差连接
```

### 2.3 GPT关键技术

| 技术 | GPT-1 | GPT-2 | GPT-3 |
|------|-------|-------|-------|
| **位置编码** | 可学习 | 可学习 | 可学习 |
| **归一化** | Post-LN | Pre-LN | Pre-LN |
| **激活函数** | GELU | GELU | GELU |
| **训练数据** | BookCorpus | WebText | 300B tokens |

### 2.4 In-Context Learning

GPT-3的核心能力：无需微调，通过提示完成任务

```
Zero-shot: 任务描述 → 输出
One-shot:  任务描述 + 1个示例 → 输出
Few-shot:  任务描述 + K个示例 → 输出
```

---

## 3. BERT系列

### 3.1 架构演进

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头 |
|------|--------|------|---------|---------|
| BERT-base | 110M | 12 | 768 | 12 |
| BERT-large | 340M | 24 | 1024 | 16 |
| RoBERTa | 355M | 24 | 1024 | 16 |
| DeBERTa | 1.5B | 48 | 1536 | 24 |

### 3.2 BERT预训练任务

**任务1：MLM (Masked Language Model)**
```
输入: The [MASK] sat on the mat
输出: cat (预测被掩盖的词)
```

**任务2：NSP (Next Sentence Prediction)**
```
输入: [CLS] 句子A [SEP] 句子B [SEP]
输出: IsNext / NotNext
```

### 3.3 BERT输入表示

$$E_{input} = E_{token} + E_{segment} + E_{position}$$

| 嵌入类型 | 作用 | 维度 |
|---------|------|------|
| Token Embedding | 词汇表示 | vocab_size × d_model |
| Segment Embedding | 区分句子A/B | 2 × d_model |
| Position Embedding | 位置信息 | max_len × d_model |

### 3.4 RoBERTa改进

| 改进点 | BERT | RoBERTa |
|--------|------|---------|
| NSP任务 | 使用 | 移除 |
| 掩码策略 | 静态 | 动态 |
| 训练数据 | 16GB | 160GB |
| Batch Size | 256 | 8K |
| 训练步数 | 1M | 500K |

### 3.5 DeBERTa创新

**解耦注意力 (Disentangled Attention)**：
$$A_{i,j} = \{H_i, P_{i|j}\} \times \{H_j, P_{j|i}\}^T$$

分离内容和位置的注意力计算：
- Content-to-Content
- Content-to-Position
- Position-to-Content

---

## 4. LLaMA系列

### 4.1 架构演进

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头 | 上下文 |
|------|--------|------|---------|---------|--------|
| LLaMA-7B | 7B | 32 | 4096 | 32 | 2K |
| LLaMA-13B | 13B | 40 | 5120 | 40 | 2K |
| LLaMA-2-7B | 7B | 32 | 4096 | 32 | 4K |
| LLaMA-2-70B | 70B | 80 | 8192 | 64 | 4K |
| LLaMA-3-8B | 8B | 32 | 4096 | 32 | 8K |
| LLaMA-3-70B | 70B | 80 | 8192 | 64 | 8K |

### 4.2 LLaMA关键技术

| 技术 | 说明 | 优势 |
|------|------|------|
| **RMSNorm** | 简化的层归一化 | 计算更快 |
| **SwiGLU** | 门控激活函数 | 性能更好 |
| **RoPE** | 旋转位置编码 | 外推能力强 |
| **GQA** | 分组查询注意力 | 推理更快 |

### 4.3 RMSNorm

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma$$

相比LayerNorm：
- 移除均值中心化
- 计算量减少
- 效果相当

### 4.4 SwiGLU激活

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes xW_3$$
$$\text{Swish}(x) = x \cdot \sigma(x)$$

**FFN维度调整**：
- 标准FFN: $d_{ff} = 4 \times d_{model}$
- SwiGLU: $d_{ff} = \frac{8}{3} \times d_{model}$ (保持参数量)

### 4.5 分组查询注意力 (GQA)

| 类型 | KV头数 | 内存占用 | 推理速度 |
|------|--------|---------|---------|
| MHA | = Q头数 | 高 | 慢 |
| MQA | 1 | 低 | 快 |
| GQA | 介于两者 | 中 | 中 |

LLaMA-2-70B使用GQA：
- Q头数: 64
- KV头数: 8
- 每8个Q头共享1组KV

---

## 5. 缩放定律

### 5.1 Kaplan缩放定律 (OpenAI)

$$L(N, D, C) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + \left(\frac{C_c}{C}\right)^{\alpha_C}$$

| 参数 | 指数 | 含义 |
|------|------|------|
| N | 0.076 | 模型参数量 |
| D | 0.095 | 数据量 |
| C | 0.050 | 计算量 |

### 5.2 Chinchilla缩放定律 (DeepMind)

**核心发现**：模型参数和训练数据应该同比例增长

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

| 计算预算 | 最优参数量 | 最优数据量 |
|---------|-----------|-----------|
| GPT-3级 | 175B | 3.7T tokens |
| Chinchilla | 70B | 1.4T tokens |

### 5.3 涌现能力

当模型规模超过某个阈值时，突然出现的能力：

| 能力 | 涌现规模 |
|------|---------|
| 思维链推理 | ~100B |
| 多步数学 | ~100B |
| 代码生成 | ~10B |
| 指令遵循 | ~1B |

---

## 6. 架构对比

### 6.1 Encoder vs Decoder

| 特性 | Encoder (BERT) | Decoder (GPT) |
|------|---------------|---------------|
| **注意力** | 双向 | 因果(单向) |
| **预训练** | MLM | CLM |
| **生成能力** | 弱 | 强 |
| **理解能力** | 强 | 中 |
| **典型任务** | 分类、NER | 生成、对话 |

### 6.2 现代LLM架构选择

| 组件 | GPT-2 | LLaMA | Qwen |
|------|-------|-------|------|
| 归一化 | Pre-LN | RMSNorm | RMSNorm |
| 激活 | GELU | SwiGLU | SwiGLU |
| 位置编码 | 可学习 | RoPE | RoPE |
| 注意力 | MHA | GQA | GQA |
| Bias | 有 | 无 | 无 |

### 6.3 参数效率对比

| 模型 | 参数量 | 训练数据 | 性能 |
|------|--------|---------|------|
| GPT-3 | 175B | 300B | 基准 |
| Chinchilla | 70B | 1.4T | ≈GPT-3 |
| LLaMA | 65B | 1.4T | >GPT-3 |
| LLaMA-2 | 70B | 2T | >>GPT-3 |

---

## 7. 常见问题FAQ

### Q1: 为什么GPT用Decoder，BERT用Encoder？

- **GPT目标**：生成文本，需要自回归
- **BERT目标**：理解文本，需要双向上下文
- **现代趋势**：Decoder-only统一生成和理解

### Q2: 为什么LLaMA不用Bias？

- 减少参数量
- 实验表明对性能影响很小
- 简化实现

### Q3: RoPE vs 可学习位置编码？

| 方面 | RoPE | 可学习 |
|------|------|--------|
| 参数量 | 0 | O(L×d) |
| 外推能力 | 好 | 差 |
| 相对位置 | 显式 | 隐式 |

### Q4: 为什么用Pre-LN而不是Post-LN？

- Pre-LN训练更稳定
- 可以训练更深的网络
- 不需要warmup

### Q5: 如何选择模型规模？

根据Chinchilla定律：
- 计算预算C确定后
- 参数量N ≈ √C
- 数据量D ≈ 20×N

---

## 参考文献

1. Radford et al. (2018). Improving Language Understanding by Generative Pre-Training
2. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
3. Brown et al. (2020). Language Models are Few-Shot Learners
4. Touvron et al. (2023). LLaMA: Open and Efficient Foundation Language Models
5. Hoffmann et al. (2022). Training Compute-Optimal Large Language Models

---

*最后更新：2026年1月*
