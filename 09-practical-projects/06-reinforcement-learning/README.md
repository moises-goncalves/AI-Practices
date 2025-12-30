<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/TensorFlow-2.0+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" alt="PRs Welcome">
</p>

# 🎮 强化学习实战项目集

> **从游戏AI到量化交易 —— 三个精心设计的强化学习实战项目**

本项目集包含三个由浅入深的强化学习实战项目，涵盖游戏AI和金融交易两大热门应用领域。每个项目都配有详细的教程文档、完整的代码实现和单元测试，非常适合强化学习初学者系统学习。

---

## 📋 目录

- [项目概览](#-项目概览)
- [学习路线图](#-学习路线图)
- [核心算法](#-核心算法)
- [快速开始](#-快速开始)
- [项目详情](#-项目详情)
- [技术栈](#-技术栈)
- [致谢](#-致谢)

---

## 🎯 项目概览

| 项目 | 难度 | 算法 | 框架 | 应用领域 |
|:-----|:----:|:----:|:----:|:--------:|
| [Flappy Bird DQN](./01-flappy-bird-dqn) | ⭐⭐⭐ | DQN | PyTorch | 游戏AI |
| [Chrome Dino DQN](./02-dino-run-dqn) | ⭐⭐⭐ | DQN | TensorFlow | 游戏AI |
| [Stock Trading RL](./03-stock-trading-rl) | ⭐⭐⭐⭐ | DQN/A2C | PyTorch | 量化交易 |

### 项目亮点

- ✅ **完整实现** - 每个项目都包含完整的训练、测试代码
- ✅ **详细教程** - 面向初学者的手把手教程，零基础也能上手
- ✅ **单元测试** - 33个单元测试确保代码质量
- ✅ **最佳实践** - 遵循工业界代码规范和项目结构

---

## 🗺️ 学习路线图

```
┌─────────────────────────────────────────────────────────────────┐
│                      强化学习实战学习路线                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   第一阶段: Flappy Bird DQN (1-2周)                              │
│   ├── 理解DQN核心原理                                            │
│   ├── 掌握经验回放机制                                           │
│   └── 学习图像状态处理                                           │
│                         ↓                                       │
│   第二阶段: Chrome Dino DQN (1-2周)                              │
│   ├── 实践不同深度学习框架                                        │
│   ├── 学习浏览器自动化                                           │
│   └── 对比模拟器与真实环境                                        │
│                         ↓                                       │
│   第三阶段: Stock Trading RL (2-3周)                             │
│   ├── 理解金融环境建模                                           │
│   ├── 掌握A2C算法                                               │
│   └── 学习技术指标应用                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 核心算法

### Deep Q-Network (DQN)

DQN是DeepMind在2013年提出的里程碑式算法，首次成功将深度学习与强化学习结合。

```
┌──────────────────────────────────────────────────────────┐
│                    DQN 算法架构                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   状态 s ──→ [卷积神经网络] ──→ Q(s,a) for all a         │
│                    ↑                                     │
│              经验回放池                                   │
│           (s, a, r, s')                                  │
│                                                          │
│   关键创新:                                               │
│   • 经验回放 - 打破样本相关性                              │
│   • 目标网络 - 稳定训练过程                               │
│   • ε-greedy - 平衡探索与利用                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Advantage Actor-Critic (A2C)

A2C是一种策略梯度方法，同时学习策略函数和价值函数。

```
┌──────────────────────────────────────────────────────────┐
│                    A2C 算法架构                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   状态 s ──→ [共享网络] ──┬──→ Actor: π(a|s)             │
│                          └──→ Critic: V(s)              │
│                                                          │
│   优势函数: A(s,a) = Q(s,a) - V(s)                       │
│                                                          │
│   特点:                                                  │
│   • 在线学习，无需经验池                                  │
│   • 收敛速度快                                           │
│   • 适合连续动作空间                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/zimingttkx/AI-Practices.git
cd AI-Practices/09-practical-projects/06-reinforcement-learning

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 选择项目并安装依赖
cd 01-flappy-bird-dqn
pip install -r requirements.txt

# 4. 运行测试
python -m unittest discover tests -v

# 5. 开始训练
python train.py
```

---

## 📁 项目详情

### 01-Flappy Bird DQN

使用PyTorch实现的DQN算法，训练AI玩Flappy Bird游戏。

```
01-flappy-bird-dqn/
├── src/
│   ├── dqn.py          # DQN网络定义
│   ├── game_env.py     # 游戏环境封装
│   ├── trainer.py      # 训练器
│   └── utils.py        # 工具函数
├── train.py            # 训练脚本
├── play.py             # 测试脚本
└── tests/              # 单元测试
```

**核心特性**：
- 卷积神经网络处理游戏画面
- 帧堆叠技术捕捉运动信息
- 经验回放提升样本效率

### 02-Chrome Dino DQN

使用TensorFlow实现的DQN算法，支持模拟器和真实浏览器两种模式。

```
02-dino-run-dqn/
├── src/
│   ├── dqn.py          # DQN智能体
│   ├── game_env.py     # 游戏环境(模拟器+浏览器)
│   └── utils.py        # 图像处理
├── train.py            # 训练脚本
└── tests/              # 单元测试
```

**核心特性**：
- 双模式支持（模拟器/浏览器）
- Selenium浏览器自动化
- 实时图像处理

### 03-Stock Trading RL

使用PyTorch实现的DQN和A2C算法，用于股票交易策略学习。

```
03-stock-trading-rl/
├── src/
│   ├── env.py          # 交易环境
│   ├── agents.py       # DQN/A2C智能体
│   └── data.py         # 数据处理
├── train.py            # 训练脚本
└── tests/              # 单元测试
```

**核心特性**：
- 自定义交易环境
- 多种技术指标支持
- DQN和A2C算法对比

---

## 🛠️ 技术栈

| 类别 | 技术 |
|:-----|:-----|
| 深度学习框架 | PyTorch, TensorFlow |
| 数值计算 | NumPy, Pandas |
| 图像处理 | OpenCV, Pillow |
| 游戏引擎 | Pygame |
| 浏览器自动化 | Selenium |
| 测试框架 | unittest |

---

## 📊 性能基准

| 项目 | 训练时间 | 收敛回合 | 最终性能 |
|:-----|:--------:|:--------:|:--------:|
| Flappy Bird | ~4h (CPU) | ~500K steps | 通过100+管道 |
| Chrome Dino | ~2h (CPU) | ~1000 episodes | 得分500+ |
| Stock Trading | ~30min | ~500 episodes | ROI 5-15% |

*注：性能数据基于默认参数，实际结果可能因硬件和随机种子而异*

---

## 📚 参考资料

### 论文
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - DQN原始论文
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Nature DQN
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) - A3C/A2C

### 教程
- [PyTorch强化学习教程](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)

---

## 🙏 致谢

本项目参考了以下优秀的开源项目，在此表示衷心感谢：

| 项目 | 作者 | 贡献 |
|:-----|:-----|:-----|
| [Flappy-bird-deep-Q-learning-pytorch](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch) | Viet Nguyen | Flappy Bird DQN实现 |
| [DinoRunTutorial](https://github.com/Paperspace/DinoRunTutorial) | Paperspace | Chrome Dino教程 |
| [FinRL-Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials) | AI4Finance Foundation | 金融强化学习 |

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](../../LICENSE) 文件

---

<p align="center">
  <b>如果这个项目对你有帮助，请给一个 ⭐ Star！</b>
</p>

---

[返回实战项目](../README.md) | [强化学习理论](../../07-reinforcement-learning/README.md)
