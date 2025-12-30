# 强化学习

> MDP、Q-Learning 与深度强化学习

---

## 目录结构

```
07-reinforcement-learning/
├── 01-mdp-basics/              # MDP 基础
├── 02-temporal-difference/     # 时序差分学习
├── 03-q-learning/              # Q-Learning
├── 04-deep-q-learning/         # 深度 Q 学习
├── 05-policy-gradient/         # 策略梯度
├── 06-reward-optimization/     # 奖励优化
├── 07-popular-algorithms/      # 流行算法 (PPO, SAC, TD3)
└── tools/                      # 工具库
    ├── gymnasium/              # Gymnasium 环境
    └── tf-agents/              # TF-Agents 库
```

---

## 学习路线

```
┌─────────────────────────────────────────────────────────────────────┐
│                        强化学习学习路线                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  基础理论 (01-mdp-basics)                                           │
│  ├── 马尔可夫决策过程 (MDP)                                          │
│  ├── 贝尔曼方程                                                     │
│  └── 动态规划                                                       │
│         │                                                          │
│         ▼                                                          │
│  无模型方法 (02-temporal-difference, 03-q-learning)                 │
│  ├── 时序差分学习 (TD)                                              │
│  ├── Q-Learning / SARSA                                            │
│  └── 探索与利用                                                     │
│         │                                                          │
│         ▼                                                          │
│  深度强化学习 (04-deep-q-learning, 05-policy-gradient)              │
│  ├── DQN 及变体                                                     │
│  ├── 策略梯度 (REINFORCE)                                           │
│  └── Actor-Critic (A2C/A3C)                                        │
│         │                                                          │
│         ▼                                                          │
│  高级主题 (06-reward-optimization, 07-popular-algorithms)           │
│  ├── 奖励优化与塑形                                                 │
│  ├── PPO / TRPO / SAC / TD3                                        │
│  └── 多智能体 RL                                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心内容

| 子模块 | 核心概念 | 实践重点 |
|--------|----------|----------|
| 01-mdp-basics | 马尔可夫决策过程、贝尔曼方程 | 网格世界、动态规划 |
| 02-temporal-difference | TD 误差、TD(0)、TD(λ) | SARSA、n-step TD |
| 03-q-learning | 值迭代、ε-greedy | 表格型 Q-Learning |
| 04-deep-q-learning | 经验回放、目标网络 | DQN、Double DQN、Dueling DQN |
| 05-policy-gradient | REINFORCE、优势函数 | Actor-Critic、A2C |
| 06-reward-optimization | 奖励塑形、好奇心驱动 | HER、逆强化学习 |
| 07-popular-algorithms | PPO、SAC、TD3 | 生产级算法实现 |
| tools/gymnasium | 环境接口、标准化 | Gymnasium 使用 |
| tools/tf-agents | 高级 RL 库 | 生产级实现 |

---

## 环境配置

```bash
# 基础依赖
pip install numpy matplotlib

# PyTorch (根据你的系统选择)
pip install torch

# Gymnasium (OpenAI Gym 的维护版本)
pip install gymnasium
pip install gymnasium[classic-control]  # CartPole 等经典环境
pip install gymnasium[atari]            # Atari 游戏

# TensorFlow Agents (可选)
pip install tf-agents
```

---

## 推荐资源

### 书籍
- Sutton & Barto, "Reinforcement Learning: An Introduction" (圣经)
- 《深度强化学习》- 王树森

### 课程
- David Silver, UCL RL Course
- OpenAI Spinning Up
- 李宏毅深度强化学习

### 工具库
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [CleanRL](https://github.com/vwxyzjn/cleanrl)

---

[返回主页](../README.md)
