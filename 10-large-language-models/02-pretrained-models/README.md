# 预训练模型 (Pretrained Models)

预训练语言模型的架构详解与实现，涵盖 GPT、BERT、LLaMA 系列。

## 核心概念

**预训练范式**: 在大规模无标注语料上进行自监督学习

$$\mathcal{L}_{LM} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

## 目录结构

```
02-pretrained-models/
├── src/                          # 源代码
│   ├── gpt_model.py              # GPT架构实现
│   ├── bert_model.py             # BERT架构实现
│   └── llama_model.py            # LLaMA架构实现
├── notebooks/                    # 交互式教程
│   ├── gpt_architecture.ipynb    # GPT系列详解
│   ├── bert_architecture.ipynb   # BERT系列详解
│   └── llama_architecture.ipynb  # LLaMA系列详解
├── knowledge_points.md           # 知识点详解
└── README.md
```

## 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **GPT** | `gpt_model.py` | Decoder-only, 自回归生成 |
| **BERT** | `bert_model.py` | Encoder-only, 双向理解 |
| **LLaMA** | `llama_model.py` | 高效开源大模型 |

## 模型对比

| 模型 | 架构 | 预训练任务 | 典型应用 |
|------|------|-----------|---------|
| GPT | Decoder | CLM | 文本生成 |
| BERT | Encoder | MLM + NSP | 文本理解 |
| LLaMA | Decoder | CLM | 通用任务 |

## 快速开始

```python
from src.gpt_model import GPTConfig, GPT
from src.llama_model import LLaMAConfig, LLaMA

# GPT-2 Small
gpt_config = GPTConfig(vocab_size=50257, n_layers=12, n_heads=12, d_model=768)
gpt = GPT(gpt_config)

# LLaMA-7B 风格
llama_config = LLaMAConfig(vocab_size=32000, n_layers=32, n_heads=32, d_model=4096)
llama = LLaMA(llama_config)
```

## 参考文献

1. Radford et al. (2018). Improving Language Understanding by Generative Pre-Training
2. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
3. Touvron et al. (2023). LLaMA: Open and Efficient Foundation Language Models
