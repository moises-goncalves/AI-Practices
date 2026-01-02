# 模型剪枝 (Model Pruning)

## 概述

剪枝是通过移除神经网络中不重要的参数来减少模型大小和计算量的技术。

## 核心概念

| 剪枝类型 | 粒度 | 可逆性 | 硬件支持 | 压缩比 |
|:---------|:-----|:------:|:--------:|:------:|
| **非结构化** | 权重级别 | 是 | 差 | 10x+ |
| **结构化** | 通道/滤波器级别 | 否 | 好 | 2-5x |
| **渐进式** | 迭代增加稀疏度 | 是 | 中 | 5-10x |

## 剪枝方法分类

```
剪枝方法
├── 非结构化剪枝 (Unstructured)
│   ├── 幅度剪枝 (Magnitude)
│   ├── L1/L2 正则化
│   └── 渐进式剪枝 (Gradual Pruning)
├── 结构化剪枝 (Structured)
│   ├── 通道剪枝 (Channel Pruning)
│   ├── 滤波器剪枝 (Filter Pruning)
│   └── 层剪枝 (Layer Pruning)
└── 自动化剪枝
    ├── Learning based (Lottery Ticket)
    └── NAS based
```

## 核心公式

**幅度剪枝**:
$$\text{mask}_i = \mathbb{1}(|W_i| > \theta)$$

**稀疏度**:
$$\text{Sparsity} = \\frac{\\|W\\|_0}{\\|W\\|_{total}} = \\frac{\\text{非零参数数}}{\\text{总参数数}}$$

**Taylor 展开** (重要性估计):
$$\\mathcal{I}_i = |g_i \\cdot W_i| = \\left|\\frac{\\partial \\mathcal{L}}{\\partial W_i} \\cdot W_i\\right|$$

## 学习路径

| 序号 | Notebook | 内容 | 难度 |
|:----:|:---------|:-----|:----:|
| 1 | `pruning_fundamentals.ipynb` | 剪枝基础、稀疏度计算 | ⭐⭐ |
| 2 | `magnitude_pruning.ipynb` | 幅度剪枝、渐进式剪枝 | ⭐⭐⭐ |
| 3 | `structured_pruning.ipynb` | 结构化剪枝、通道剪枝 | ⭐⭐⭐⭐ |
| 4 | `lottery_ticket.ipynb` | 彩票假说、迭代剪枝 | ⭐⭐⭐⭐⭐ |

## 参考文献

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)
- [To Prune, or Not to Prune](https://arxiv.org/abs/1710.01878)
