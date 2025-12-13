---
layout: home

hero:
  name: "AI-Practices"
  text: "Full-Stack AI Learning Laboratory"
  tagline: 从零到一的 AI 全栈实战之旅
  image:
    src: /logo.svg
    alt: AI-Practices
  actions:
    - theme: brand
      text: 快速开始
      link: /guide/getting-started
    - theme: alt
      text: 在 GitHub 上查看
      link: https://github.com/zimingttkx/AI-Practices

features:
  - icon: 📚
    title: 系统化课程体系
    details: 9 大核心模块，从机器学习基础到强化学习，渐进式学习路径设计
  - icon: 🧠
    title: 113+ Jupyter Notebooks
    details: 可复现的实验代码，详细的中文注释，理论与实践完美结合
  - icon: 🏆
    title: 19 个实战项目
    details: 涵盖 CV、NLP、时序预测，包含 Kaggle 金牌方案复现
  - icon: 📝
    title: 149k+ 行高质量代码
    details: 遵循 PEP8 规范，工程化项目模板，最佳实践指南
  - icon: 🎯
    title: 理论笔记速查
    details: 30+ 激活函数、损失函数全景速查，架构设计指南
  - icon: 🔧
    title: 完整工具链
    details: TensorFlow、PyTorch、scikit-learn 全栈覆盖
---

<script setup>
import StatsCard from './.vitepress/theme/components/StatsCard.vue'
import DesignGoals from './.vitepress/theme/components/DesignGoals.vue'
import ArchitectureDiagram from './.vitepress/theme/components/ArchitectureDiagram.vue'
import LearningRoadmap from './.vitepress/theme/components/LearningRoadmap.vue'
import TechStack from './.vitepress/theme/components/TechStack.vue'
</script>

## 📊 项目概览

<StatsCard />

---

## 🎯 设计目标

<DesignGoals title="渐进式学习框架 | Progressive Learning Framework" />

---

## 🏗️ 系统架构

<ArchitectureDiagram title="模块化架构设计 | Modular Architecture" />

---

## 🗺️ 学习路径

<LearningRoadmap title="推荐学习路线 | Recommended Learning Path" />

---

## 🛠️ 技术栈

<TechStack title="全栈技术生态 | Full-Stack Technology Ecosystem" />

---

## 🚀 快速开始

::: code-group

```bash [npm]
# 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# 创建虚拟环境
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# 安装依赖
pip install -r requirements.txt
```

```bash [Docker]
# 构建镜像
docker build -t ai-practices .

# 运行容器
docker run -it --gpus all -v $(pwd):/workspace ai-practices
```

:::

---

## 🏆 Kaggle 竞赛成绩

| 竞赛 | 排名 | 奖牌 |
|:-----|:----:|:----:|
| Feedback Prize - English Language Learning | Top 1% | 🥇 Gold |
| RSNA 2023 Abdominal Trauma Detection | Top 1% | 🥇 Gold |
| American Express Default Prediction | Top 5% | 🥈 Silver |

---

## 📄 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{ai-practices,
  author       = {zimingttkx},
  title        = {AI-Practices: A Comprehensive Full-Stack AI Learning Laboratory},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/zimingttkx/AI-Practices}}
}
```

---

<div style="text-align: center; padding: 40px 0;">

**Made with ❤️ by [zimingttkx](https://github.com/zimingttkx)**

MIT License © 2024

</div>
