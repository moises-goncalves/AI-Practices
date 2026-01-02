# LLM 基础 (Large Language Model Fundamentals)

生产级 LLM 基础组件实现，包含分词器和 Transformer 架构。

## 核心概念

**Transformer 架构**: 基于自注意力机制的序列建模

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## 目录结构

```
01-llm-fundamentals/
├── src/                          # 源代码
│   └── transformer_architecture_v2.py  # 生产级Transformer实现
├── notebooks/                    # 交互式教程
│   ├── tokenizer_architecture.ipynb    # 分词器原理
│   └── transformer_architecture.ipynb  # Transformer详解
└── README.md
```

## 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| **分词器** | `tokenizer_architecture.ipynb` | BPE/WordPiece/SentencePiece |
| **Transformer** | `transformer_architecture.ipynb` | Self-Attention/FFN/LayerNorm |
| **生产实现** | `transformer_architecture_v2.py` | 875行完整实现 |

## 快速开始

```python
from src.transformer_architecture_v2 import (
    TransformerConfig,
    TransformerEncoder,
    MultiHeadAttention
)

config = TransformerConfig(d_model=512, n_heads=8, n_layers=6)
encoder = TransformerEncoder(config)

x = torch.randn(2, 10, 512)
output = encoder(x)
```

## 参考文献

1. Vaswani et al. (2017). Attention Is All You Need
2. Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
