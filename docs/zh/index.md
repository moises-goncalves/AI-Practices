---
layout: home

hero:
  name: AI-Practices
  text: 全栈 AI 学习实验室
  tagline: 从零到一的 AI 全栈实战之旅
  image:
    src: /logo.svg
    alt: AI-Practices
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/guide/getting-started
    - theme: alt
      text: GitHub
      link: https://github.com/zimingttkx/AI-Practices

features:
  - icon: 📊
    title: 113+ Notebooks
    details: 可复现的 Jupyter 实验，含详细注释与可视化
  - icon: 🧠
    title: 9 核心模块
    details: 渐进式课程设计，从基础到进阶
  - icon: 🚀
    title: 19 实战项目
    details: 端到端项目，含 Kaggle 金牌方案
  - icon: 📝
    title: 149k+ 代码行
    details: 高质量代码，遵循 PEP8 规范
---

<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #007AFF 30%, #5856D6);
  --vp-home-hero-image-background-image: linear-gradient(-45deg, #007AFF 50%, #5856D6 50%);
  --vp-home-hero-image-filter: blur(44px);
}
</style>

## 设计理念

本项目遵循 **"理论驱动、实践为本、工程导向"** 的设计理念：

| 阶段 | 原则 | 方法 | 产出 |
|:----:|:-----|:-----|:-----|
| **Ⅰ** | 理论先行 | 数学推导 + 算法分析 | 理论笔记 |
| **Ⅱ** | 代码实现 | NumPy 从零实现 | 核心代码 |
| **Ⅲ** | 框架应用 | TensorFlow/PyTorch | 工程代码 |
| **Ⅳ** | 项目实战 | 真实项目 + 竞赛 | 完整方案 |

## 技术栈

<div class="tech-stack">

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000?style=flat-square&logo=keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)

</div>

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 创建环境
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# 安装依赖
pip install -r requirements.txt

# 启动 Jupyter
jupyter lab
```
