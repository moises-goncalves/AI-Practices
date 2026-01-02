# 知识蒸馏 (Knowledge Distillation)

## 概述

知识蒸馏通过将大型教师模型的知识转移到小型学生模型，实现模型压缩和加速。

## 核心概念

| 方法 | 教师 | 学生 | 损失函数 | 压缩比 |
|:-----|:-----|:-----|:---------|:------:|
| **响应式蒸馏** | 软标签 | 学生模型 | KL(学生\|教师) | 2-10x |
| **特征蒸馏** | 中间特征 | 学生特征 | MSE/Lp | 2-10x |
| **关系蒸馏** | 样本关系 | 学生关系 | 距离保持 | 2-5x |
| **自蒸馏** | 自身 | 自身 | 辅助头 | 1.5-3x |

## Hinton 蒸馏公式

$$L = \\alpha \\cdot L_{CE}(y, \\hat{y}) + (1-\\alpha) \\cdot T^2 \\cdot D_{KL}(\\sigma(z/T) \\| \\sigma(z_s/T))$$

其中：
- $T$: 温度参数，控制软标签平滑度
- $\\alpha$: 平衡系数
- $z, z_s$: 教师和学生的 logits

## 学习路径

| 序号 | Notebook | 内容 | 难度 |
|:----:|:---------|:-----|:----:|
| 1 | `distillation_basics.ipynb` | Hinton 蒸馏、温度参数 | ⭐⭐⭐ |
| 2 | `feature_distillation.ipynb` | 特征蒸馏、FitNets | ⭐⭐⭐⭐ |
| 3 | `self_distillation.ipynb` | 自蒸馏、辅助头 | ⭐⭐⭐⭐ |

## 参考文献

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)
- [Born-Again Neural Networks](https://arxiv.org/abs/1805.04770)
