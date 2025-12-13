# 学习路径

<script setup>
import LearningRoadmap from './.vitepress/theme/components/LearningRoadmap.vue'
</script>

<LearningRoadmap title="AI-Practices 学习旅程" />

## 详细学习计划

### Phase 1: Foundation (3-4 周)

**目标**: 建立坚实的机器学习理论基础

| 周次 | 内容 | 重点 |
|:----:|:-----|:-----|
| Week 1 | 训练模型基础 | 梯度下降、正则化、交叉验证 |
| Week 2 | 分类与 SVM | 二分类、多分类、核方法 |
| Week 3 | 决策树与集成 | CART、XGBoost、LightGBM |
| Week 4 | 无监督学习 | K-Means、PCA、t-SNE |

**配套学习**: `08-theory-notes` 中的激活函数和损失函数速查

### Phase 2: Core (5-7 周)

**目标**: 掌握深度学习核心技术

| 周次 | 内容 | 重点 |
|:----:|:-----|:-----|
| Week 1-2 | 神经网络基础 | Keras、训练技巧、自定义模型 |
| Week 3-4 | 计算机视觉 | CNN 架构、迁移学习、Grad-CAM |
| Week 5-7 | 序列模型 | RNN/LSTM、词嵌入、TextCNN |

::: tip 选择建议
根据你的兴趣方向，可以在 CV 和 NLP 之间选择一个深入学习，另一个了解基础即可。
:::

### Phase 3: Advanced (4-6 周)

**目标**: 探索高级专题和前沿领域

| 周次 | 内容 | 重点 |
|:----:|:-----|:-----|
| Week 1-2 | 高级专题 | Functional API、超参调优、TensorBoard |
| Week 3-4 | 生成式模型 | GAN、文本生成、DeepDream |
| Week 5-6 | 强化学习 | MDP、Q-Learning、DQN、PPO |

### Phase 4: Practice (4-8 周)

**目标**: 通过实战项目检验学习成果

| 难度 | 项目 | 建议时长 |
|:----:|:-----|:--------:|
| 🟢 入门 | Titanic、MNIST、情感分析 | 1-2 周 |
| 🟡 中级 | Otto、NER、温度预测 | 2-4 周 |
| 🔴 高级 | 机器翻译、Kaggle 方案 | 4+ 周 |

## 学习建议

### 🎯 目标明确

1. 确定你的学习目标（求职、研究、竞赛）
2. 根据目标选择重点模块
3. 制定合理的学习计划

### 📝 动手实践

1. 不要只看代码，要亲手运行
2. 修改参数观察效果变化
3. 尝试在新数据集上应用

### 🔄 循序渐进

1. 从基础开始，不要跳过
2. 每个模块的 notebooks 按顺序学习
3. 遇到不懂的及时回顾理论笔记

### 💡 主动思考

1. 理解 "为什么" 而不只是 "怎么做"
2. 对比不同方法的优缺点
3. 思考如何改进现有方案

## 推荐资源

### 配套书籍

- 《Hands-On Machine Learning》 - Aurélien Géron
- 《Deep Learning》 - Ian Goodfellow
- 《统计学习方法》 - 李航

### 在线课程

- Coursera: Machine Learning by Andrew Ng
- Fast.ai: Practical Deep Learning for Coders
- Stanford CS231n/CS224n

### 论文阅读

- Papers With Code (https://paperswithcode.com)
- arXiv (https://arxiv.org)

## 下一步

准备好开始了吗？

- 📚 [01-Foundations](/modules/01-foundations) - 开始学习机器学习基础
- 🚀 [快速开始](/guide/getting-started) - 配置你的学习环境
