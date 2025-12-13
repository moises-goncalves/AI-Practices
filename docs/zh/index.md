---
layout: home

hero:
  name: AI-Practices
  text: 全栈 AI 学习实验室
  tagline: 系统化、工程化的人工智能学习与研究平台
  image:
    src: /logo.svg
    alt: AI-Practices
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/guide/getting-started
    - theme: alt
      text: 课程模块
      link: /zh/modules/
    - theme: alt
      text: GitHub
      link: https://github.com/zimingttkx/AI-Practices

features:
  - icon: 📊
    title: 113+ 可复现实验
    details: 每个算法都有完整的 Jupyter Notebook 实现，含详细注释、数学推导与可视化分析
  - icon: 🧠
    title: 9 大核心模块
    details: 渐进式课程设计，从机器学习基础到强化学习，覆盖 AI 全技术栈
  - icon: 🏆
    title: Kaggle 金牌方案
    details: 包含多个顶级竞赛的完整解决方案，学习工业级 AI 工程实践
  - icon: 🔬
    title: 理论与实践结合
    details: 数学推导 → NumPy 实现 → 框架应用 → 实战项目，培养独立解决问题的能力
  - icon: ⚡
    title: 生产级代码质量
    details: 149k+ 行高质量代码，遵循 PEP8 规范，完整类型注解与文档字符串
  - icon: 🌐
    title: 中英双语文档
    details: 完整的双语文档支持，方便国内外开发者学习交流
---

<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #007AFF 30%, #5856D6);
  --vp-home-hero-image-background-image: linear-gradient(-45deg, #007AFF 50%, #5856D6 50%);
  --vp-home-hero-image-filter: blur(44px);
}

.dark {
  --vp-home-hero-image-background-image: linear-gradient(-45deg, #007AFF 50%, #5856D6 50%);
}
</style>

## 渐进式学习框架

本项目采用 **Progressive Learning Framework** 方法论，构建从理论到实战的完整学习闭环：

| 阶段 | 原则 | 方法 | 产出 | 目标 |
|:----:|:-----|:-----|:-----|:-----|
| **Ⅰ** | 理论先行 | 数学推导 + 算法复杂度分析 | 理论笔记 | 🎯 理解原理 |
| **Ⅱ** | 从零实现 | NumPy 手写核心算法 | 核心代码 | 🔧 掌握细节 |
| **Ⅲ** | 框架精通 | PyTorch / TensorFlow 工程化 | 工程代码 | ⚡ 高效开发 |
| **Ⅳ** | 实战检验 | Kaggle 竞赛 + 工业项目 | 完整方案 | 🏆 实战能力 |

## 技术栈

<div class="tech-badges">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras&logoColor=white)

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FDD?style=flat-square&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.30+-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)

</div>

## 快速开始

::: code-group

```bash [conda]
# 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 创建 Conda 环境
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter Lab
jupyter lab
```

```bash [pip]
# 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter Lab
jupyter lab
```

:::

## 学习路线图

```
入门 ──► 01 机器学习基础 ──► 02 神经网络 ──┬──► 03 计算机视觉 ──┬──► 05 高级专题 ──┬──► 06 生成模型 ──┐
                                          │                    │                  │                  │
                                          └──► 04 序列模型 ────┘                  └──► 07 强化学习 ──┼──► 09 实战项目
                                                                                                     │
                                          08 理论笔记 ◄─────────────── 随时参考 ─────────────────────┘
```

## 竞赛成绩

| 竞赛 | 排名 | 奖牌 | 年份 |
|:-----|:----:|:----:|:----:|
| Feedback Prize - ELL | **Top 1%** | 🥇 金牌 | 2023 |
| RSNA Abdominal Trauma | **Top 1%** | 🥇 金牌 | 2023 |
| American Express Default | Top 5% | 🥈 银牌 | 2022 |
| RSNA Lumbar Spine | Top 10% | 🥉 铜牌 | 2024 |
