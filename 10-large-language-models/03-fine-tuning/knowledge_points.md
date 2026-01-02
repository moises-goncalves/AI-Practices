# 微调方法知识点详解

> **知识密度**：⭐⭐⭐⭐⭐ | **实战价值**：⭐⭐⭐⭐⭐

---

## 目录

1. [微调概述](#1-微调概述)
2. [全量微调](#2-全量微调)
3. [LoRA](#3-lora)
4. [QLoRA](#4-qlora)
5. [其他PEFT方法](#5-其他peft方法)
6. [指令微调](#6-指令微调)
7. [常见问题FAQ](#7-常见问题faq)

---

## 1. 微调概述

### 1.1 为什么需要微调？

| 场景 | 预训练模型 | 微调后 |
|------|-----------|--------|
| 领域适配 | 通用知识 | 专业领域 |
| 任务适配 | 生成能力 | 特定任务 |
| 风格适配 | 通用风格 | 定制风格 |
| 对齐 | 原始输出 | 符合人类偏好 |

### 1.2 微调方法分类

```
微调方法
├── 全量微调 (Full Fine-tuning)
│   └── 更新所有参数
├── 参数高效微调 (PEFT)
│   ├── LoRA / QLoRA
│   ├── Adapter
│   ├── Prefix Tuning
│   └── Prompt Tuning
└── 对齐微调
    ├── SFT (监督微调)
    ├── RLHF
    └── DPO
```

---

## 2. 全量微调

### 2.1 原理

更新模型所有参数：
$$\theta^* = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D})$$

### 2.2 显存分析

| 组件 | 显存占用 (7B模型, fp16) |
|------|------------------------|
| 模型参数 | 14 GB |
| 梯度 | 14 GB |
| 优化器状态 (Adam) | 28 GB |
| 激活值 | 可变 |
| **总计** | ~60+ GB |

### 2.3 优缺点

| 优点 | 缺点 |
|------|------|
| 性能最佳 | 显存需求大 |
| 完全适配 | 训练时间长 |
| 无额外推理开销 | 灾难性遗忘风险 |

---

## 3. LoRA

### 3.1 核心思想

**低秩分解**：用两个小矩阵近似权重更新

$$W' = W + \Delta W = W + BA$$

其中：
- $W \in \mathbb{R}^{d \times k}$：原始权重（冻结）
- $B \in \mathbb{R}^{d \times r}$：低秩矩阵
- $A \in \mathbb{R}^{r \times k}$：低秩矩阵
- $r \ll \min(d, k)$：秩

### 3.2 参数量对比

| 配置 | 原始参数 | LoRA参数 | 比例 |
|------|---------|---------|------|
| d=4096, k=4096, r=8 | 16.7M | 65.5K | 0.4% |
| d=4096, k=4096, r=16 | 16.7M | 131K | 0.8% |
| d=4096, k=4096, r=64 | 16.7M | 524K | 3.1% |

### 3.3 关键超参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `r` | 秩 | 8-64 |
| `alpha` | 缩放因子 | 16-32 |
| `target_modules` | 应用层 | q_proj, v_proj |
| `dropout` | Dropout | 0.05-0.1 |

**缩放公式**：
$$\Delta W = \frac{\alpha}{r} \cdot BA$$

### 3.4 应用位置

| 位置 | 效果 | 推荐 |
|------|------|------|
| Q, K, V | 最佳 | ✅ |
| O (输出投影) | 良好 | ✅ |
| FFN | 一般 | 可选 |
| Embedding | 较差 | ❌ |

---

## 4. QLoRA

### 4.1 核心创新

**4-bit量化 + LoRA**：
1. 基础模型4-bit量化存储
2. 计算时反量化到bf16
3. 只训练LoRA参数

### 4.2 关键技术

| 技术 | 说明 |
|------|------|
| **NF4** | 4-bit NormalFloat量化 |
| **双重量化** | 量化常数也量化 |
| **分页优化器** | 避免OOM |

### 4.3 显存对比

| 方法 | 7B模型显存 |
|------|-----------|
| 全量微调 (fp16) | ~60 GB |
| LoRA (fp16) | ~16 GB |
| QLoRA (4-bit) | ~6 GB |

### 4.4 NF4量化

**Normal Float 4-bit**：
- 假设权重服从正态分布
- 量化点均匀分布在正态分布的分位数上
- 比均匀量化更精确

---

## 5. 其他PEFT方法

### 5.1 Adapter

在Transformer层中插入小型网络：

```
x → LayerNorm → Attention → + → Adapter → + → LayerNorm → FFN → + → Adapter → +
                            ↑             ↑                     ↑             ↑
                          残差          残差                   残差          残差
```

**Adapter结构**：
$$\text{Adapter}(x) = x + f(xW_{down})W_{up}$$

### 5.2 Prefix Tuning

在每层添加可学习的前缀向量：

$$h = \text{Attention}([P_k; K], [P_v; V], Q)$$

### 5.3 Prompt Tuning

只在输入层添加可学习的软提示：

$$\text{input} = [P_1, P_2, ..., P_n, x_1, x_2, ...]$$

### 5.4 方法对比

| 方法 | 参数量 | 推理开销 | 效果 |
|------|--------|---------|------|
| LoRA | 0.1-1% | 无 | 最佳 |
| Adapter | 1-5% | 有 | 良好 |
| Prefix | 0.1% | 有 | 中等 |
| Prompt | <0.1% | 有 | 较差 |

---

## 6. 指令微调

### 6.1 数据格式

```json
{
  "instruction": "将以下文本翻译成英文",
  "input": "今天天气很好",
  "output": "The weather is nice today"
}
```

### 6.2 常用模板

**Alpaca格式**：
```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**ChatML格式**：
```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>
```

### 6.3 训练技巧

| 技巧 | 说明 |
|------|------|
| 只计算输出损失 | 忽略instruction部分 |
| 数据多样性 | 覆盖多种任务类型 |
| 数据质量 | 质量 > 数量 |
| 长度平衡 | 避免长度偏差 |

---

## 7. 常见问题FAQ

### Q1: LoRA的r应该设多大？

- **简单任务**：r=8
- **复杂任务**：r=16-32
- **领域适配**：r=64

### Q2: 应该对哪些层应用LoRA？

推荐顺序：
1. Q, V投影（必选）
2. K, O投影（推荐）
3. FFN（可选）

### Q3: QLoRA vs LoRA如何选择？

| 场景 | 推荐 |
|------|------|
| 显存充足 | LoRA |
| 显存受限 | QLoRA |
| 追求性能 | LoRA |
| 快速实验 | QLoRA |

### Q4: 如何避免灾难性遗忘？

- 使用较小学习率
- 混合原始数据
- 使用LoRA而非全量微调
- 正则化约束

### Q5: 微调数据量需要多少？

| 任务类型 | 推荐数据量 |
|---------|-----------|
| 简单分类 | 1K-10K |
| 指令遵循 | 10K-100K |
| 领域适配 | 100K+ |

---

## 参考文献

1. Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models
2. Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs
3. Houlsby et al. (2019). Parameter-Efficient Transfer Learning for NLP

---

*最后更新：2026年1月*
