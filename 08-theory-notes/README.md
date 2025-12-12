# 08-Theory Notes | 理论笔记

> 激活函数、损失函数与网络架构速查

---

## 目录结构

```
08-theory-notes/
├── activation-functions/  # 激活函数：30+ 函数详解与对比
├── loss-functions/        # 损失函数：分类/回归/排序 Loss
└── architectures/         # 架构笔记：网络选型、通用流程
```

---

## 核心内容

| 子模块 | 核心文档 | 说明 |
|--------|----------|------|
| 激活函数 | `activation-functions-complete.md` | ReLU、GELU、Swish 等 30+ 函数 |
| 损失函数 | `loss-functions-complete.md` | 交叉熵、Focal Loss、对比损失等 |
| 架构笔记 | 多个 Markdown | 网络选型建议、ML 通用流程 |

---

## 速查指南

### 激活函数选择
- **分类输出**：Softmax / Sigmoid
- **隐藏层**：ReLU / GELU / SiLU
- **梯度问题**：LeakyReLU / ELU

### 损失函数选择
- **二分类**：BCELoss / Focal Loss
- **多分类**：CrossEntropy / Label Smoothing
- **回归**：MSE / Huber / MAE

---

[返回主页](../README.md)
