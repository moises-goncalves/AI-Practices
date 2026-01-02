# 模型量化 (Model Quantization)

## 概述

量化是将神经网络从高精度（FP32）转换为低精度（INT8/FP16）表示的技术，可显著减少模型大小和推理延迟。

## 核心概念

| 量化类型 | 精度 | 压缩比 | 精度损失 | 适用场景 |
|:---------|:-----|:------:|:--------:|:---------|
| FP32 → FP16 | 半精度 | 2x | 极小 | GPU推理 |
| FP32 → INT8 | 8位整数 | 4x | 小 | 边缘部署 |
| FP32 → INT4 | 4位整数 | 8x | 中等 | LLM推理 |

## 量化方法分类

```
量化方法
├── 训练后量化 (PTQ)
│   ├── 动态量化 (Dynamic)
│   ├── 静态量化 (Static)
│   └── 仅权重量化 (Weight-only)
└── 量化感知训练 (QAT)
    ├── 伪量化节点
    └── 直通估计器 (STE)
```

## 学习路径

| 序号 | Notebook | 内容 | 难度 |
|:----:|:---------|:-----|:----:|
| 1 | `quantization_fundamentals.ipynb` | 量化基础理论、数学原理 | ⭐⭐ |
| 2 | `post_training_quantization.ipynb` | PTQ实现、校准策略 | ⭐⭐⭐ |
| 3 | `quantization_aware_training.ipynb` | QAT训练、STE原理 | ⭐⭐⭐⭐ |

## 核心公式

**均匀量化**:
$$Q(x) = \text{round}\left(\frac{x - z}{s}\right), \quad s = \frac{x_{max} - x_{min}}{2^b - 1}$$

**反量化**:
$$\hat{x} = s \cdot Q(x) + z$$

其中 $s$ 为缩放因子，$z$ 为零点，$b$ 为位宽。

## 参考文献

- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630)
- [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602)
