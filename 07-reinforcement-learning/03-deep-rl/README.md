# 深度强化学习 (Deep RL)

> 从 DQN 到 PPO：生产级深度强化学习实现

---

## 目录结构

```
03-deep-rl/
├── core/                           # 核心组件库
│   ├── __init__.py
│   ├── networks.py                 # 神经网络架构
│   ├── buffers.py                  # 经验回放缓冲区
│   └── utils.py                    # 工具函数
│
├── algorithms/                     # 算法实现
│   ├── __init__.py
│   ├── dqn.py                      # DQN 家族
│   └── policy_gradient.py          # A2C/PPO
│
├── notebooks/                      # 交互式教程
│   ├── 01-DQN-Tutorial.ipynb       # DQN 深度教程
│   └── 02-PolicyGradient-Tutorial.ipynb  # 策略梯度教程
│
├── docs/                           # 文档
│   └── 知识要点.md                  # 核心知识总结
│
└── README.md                       # 本文件
```

---

## 快速开始

### 环境要求

```bash
pip install torch numpy gymnasium matplotlib
```

### 训练 DQN

```python
from algorithms.dqn import train_dqn

# 标准 DQN
agent, rewards = train_dqn(num_episodes=300)

# Double Dueling DQN
agent, rewards = train_dqn(double_dqn=True, dueling=True)
```

### 训练 PPO

```python
from algorithms.policy_gradient import train_ppo

agent, rewards = train_ppo(total_steps=100000)
```

---

## 算法概览

### 价值方法 (DQN 家族)

| 变体 | 核心改进 | 效果 |
|------|---------|------|
| DQN | 经验回放 + 目标网络 | 基础稳定训练 |
| Double DQN | 分离选择与评估 | 减少过估计 |
| Dueling DQN | V-A 分解 | 加速学习 |
| PER | 优先级采样 | 提高样本效率 |

### 策略方法

| 算法 | 核心思想 | 特点 |
|------|---------|------|
| A2C | Actor-Critic | 简单高效 |
| PPO | 裁剪目标 | 稳定易调 |

---

## 核心公式

### DQN 损失

$$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$$

### PPO 目标

$$L^{CLIP} = \mathbb{E}[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)]$$

### GAE

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

---

## 学习路径

```
1. 阅读 docs/知识要点.md 了解理论基础
        ↓
2. 运行 notebooks/01-DQN-Tutorial.ipynb 学习 DQN
        ↓
3. 运行 notebooks/02-PolicyGradient-Tutorial.ipynb 学习 PPO
        ↓
4. 阅读 algorithms/ 源码深入理解实现
        ↓
5. 修改参数进行实验
```

---

## 运行测试

```bash
# 测试 DQN
python algorithms/dqn.py --test

# 测试策略梯度
python algorithms/policy_gradient.py --algo test

# 训练 DQN
python algorithms/dqn.py --train --episodes 300 --double --dueling

# 训练 PPO
python algorithms/policy_gradient.py --algo ppo --steps 100000
```

---

## 参考文献

1. Mnih et al., "Human-level control through deep RL", Nature 2015
2. van Hasselt et al., "Deep RL with Double Q-learning", AAAI 2016
3. Wang et al., "Dueling Network Architectures", ICML 2016
4. Schulman et al., "Proximal Policy Optimization", 2017
5. Hessel et al., "Rainbow: Combining Improvements", AAAI 2018

---

[返回上级](../README.md)
