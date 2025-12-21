# 项目结构

AI-Practices 的系统架构设计。

## 渐进式学习框架

```
┌─────────────────────────────────────────────────────────────┐
│                Progressive Learning Framework                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐        │
│   │ Theory │──▶│  Impl  │──▶│Framework│──▶│Practice│        │
│   │ First  │   │ Scratch│   │ Master  │   │Project │        │
│   └────────┘   └────────┘   └────────┘   └────────┘        │
│       │            │            │            │               │
│       ▼            ▼            ▼            ▼               │
│   数学推导      NumPy       TensorFlow    Kaggle            │
│   算法分析      从零实现    PyTorch       真实项目          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 模块依赖

```
Phase 1: Foundation
└── 01 Foundations

Phase 2: Core
├── 02 Neural Networks
├── 03 Computer Vision
└── 04 Sequence Models

Phase 3: Advanced
├── 05 Advanced Topics
├── 06 Generative Models
└── 07 Reinforcement Learning

Phase 4: Practice
└── 09 Practical Projects

Support
└── 08 Theory Notes
```

## 目录结构

```
AI-Practices/
├── 01-foundations/           # 机器学习基础
├── 02-neural-networks/       # 神经网络
├── 03-computer-vision/       # 计算机视觉
├── 04-sequence-models/       # 序列模型
├── 05-advanced-topics/       # 高级专题
├── 06-generative-models/     # 生成模型
├── 07-reinforcement-learning/# 强化学习
├── 08-theory-notes/          # 理论笔记
├── 09-practical-projects/    # 实战项目
└── utils/                    # 工具库
```

## 技术选型

| 场景 | 首选 | 备选 |
|:-----|:-----|:-----|
| 原型开发 | TensorFlow/Keras | PyTorch |
| 研究 | PyTorch | JAX |
| 生产部署 | TensorFlow | ONNX |
| NLP | Transformers | spaCy |
| 表格数据 | XGBoost/LightGBM | CatBoost |

## 代码规范

| 标准 | 工具 |
|:-----|:-----|
| 代码风格 | Black |
| 类型检查 | mypy |
| 文档字符串 | Google Style |
| 测试 | pytest |
