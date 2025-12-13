---
layout: home

hero:
  name: AI-Practices
  text: Full-Stack AI Learning Laboratory
  tagline: A Systematic Approach to AI Research & Engineering
  image:
    src: /logo.svg
    alt: AI-Practices
  actions:
    - theme: brand
      text: Get Started
      link: /en/guide/getting-started
    - theme: alt
      text: Modules
      link: /en/modules/
    - theme: alt
      text: GitHub
      link: https://github.com/zimingttkx/AI-Practices

features:
  - icon: 📊
    title: 113+ Reproducible Experiments
    details: Complete Jupyter Notebooks with detailed comments, mathematical derivations, and visualizations
  - icon: 🧠
    title: 9 Core Modules
    details: Progressive curriculum from ML fundamentals to reinforcement learning, covering the full AI stack
  - icon: 🏆
    title: Kaggle Gold Solutions
    details: Complete solutions from top-tier competitions, learn industry-grade AI engineering practices
  - icon: 🔬
    title: Theory Meets Practice
    details: Math derivation → NumPy implementation → Framework application → Real projects
  - icon: ⚡
    title: Production-Grade Code
    details: 149k+ lines of high-quality code following PEP8, with complete type annotations and docstrings
  - icon: 🌐
    title: Bilingual Documentation
    details: Full Chinese and English documentation support for global developers
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

## Progressive Learning Framework

This project adopts the **Progressive Learning Framework** methodology, building a complete learning loop from theory to practice:

| Phase | Principle | Method | Output | Goal |
|:-----:|:----------|:-------|:-------|:-----|
| **Ⅰ** | Theory First | Math derivation + Algorithm analysis | Theory notes | 🎯 Understand principles |
| **Ⅱ** | From Scratch | NumPy implementation from scratch | Core code | 🔧 Master details |
| **Ⅲ** | Framework | PyTorch / TensorFlow engineering | Production code | ⚡ Efficient development |
| **Ⅳ** | Practice | Kaggle competitions + Industry projects | Complete solutions | 🏆 Real-world skills |

## Tech Stack

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

## Quick Start

::: code-group

```bash [conda]
# Clone repository
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# Create Conda environment
conda create -n ai-practices python=3.10 -y
conda activate ai-practices

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

```bash [pip]
# Clone repository
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

:::

## Learning Roadmap

```
Start ──► 01 ML Foundations ──► 02 Neural Networks ──┬──► 03 Computer Vision ──┬──► 05 Advanced ──┬──► 06 Generative ──┐
                                                     │                         │                  │                   │
                                                     └──► 04 Sequence Models ──┘                  └──► 07 RL ─────────┼──► 09 Projects
                                                                                                                      │
                                                     08 Theory Notes ◄─────────────── Reference ──────────────────────┘
```

## Competition Results

| Competition | Rank | Medal | Year |
|:------------|:----:|:-----:|:----:|
| Feedback Prize - ELL | **Top 1%** | 🥇 Gold | 2023 |
| RSNA Abdominal Trauma | **Top 1%** | 🥇 Gold | 2023 |
| American Express Default | Top 5% | 🥈 Silver | 2022 |
| RSNA Lumbar Spine | Top 10% | 🥉 Bronze | 2024 |
