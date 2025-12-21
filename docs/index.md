---
layout: home
title: AI-Practices
titleTemplate: A Systematic Approach to AI Research & Engineering

hero:
  name: AI-Practices
  text: Full-Stack AI Learning Laboratory
  tagline: 系统化、工程化的人工智能学习与研究平台
  image:
    src: /logo.svg
    alt: AI-Practices
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/guide/getting-started
    - theme: alt
      text: English
      link: /en/
    - theme: alt
      text: GitHub
      link: https://github.com/zimingttkx/AI-Practices

features:
  - icon: 📚
    title: 9 大核心模块
    details: 从机器学习基础到强化学习，覆盖 AI 全技术栈
  - icon: 🧪
    title: 113+ 可复现实验
    details: 每个算法都有完整的 Jupyter Notebook 实现
  - icon: 🏆
    title: Kaggle 金牌方案
    details: 包含多个顶级竞赛的完整解决方案
  - icon: 🔬
    title: 理论与实践结合
    details: 数学推导 → NumPy 实现 → 框架应用 → 实战项目
---

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  // 检测浏览器语言，自动重定向
  const lang = navigator.language || navigator.userLanguage
  if (lang && lang.startsWith('zh')) {
    // 中文用户保持在当前页面或跳转到中文版
  } else {
    // 非中文用户可以选择跳转到英文版
  }
})
</script>
