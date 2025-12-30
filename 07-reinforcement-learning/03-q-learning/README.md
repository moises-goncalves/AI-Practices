# Q-Learning & SARSA 实现模块

> 无模型时序差分控制算法的研究级实现

---

## 模块概览

本模块提供 Q-Learning、SARSA 及其变体的生产级实现，适用于学术研究和工程应用。

### 核心特性

- **算法完整性**: Q-Learning, SARSA, Expected SARSA, Double Q-Learning
- **工程质量**: 模块化设计，详尽文档，完整测试
- **教学导向**: 交互式 Notebook，知识点总结

---

## 目录结构

```
02-q-learning/
├── src/                          # 核心源代码
│   ├── __init__.py              # 模块入口
│   ├── agents.py                # 智能体实现
│   ├── environments.py          # 环境实现
│   ├── training.py              # 训练基础设施
│   ├── exploration.py           # 探索策略
│   └── utils.py                 # 可视化工具
│
├── notebooks/                    # 交互式教程
│   ├── 01-Q-Learning-Fundamentals.ipynb  # Q-Learning 基础
│   ├── 02-SARSA-Comparison.ipynb         # SARSA 对比
│   └── 03-Advanced-Techniques.ipynb      # 高级技巧
│
├── docs/                         # 文档
│   └── 知识点总结.md             # 核心知识点
│
├── tests/                        # 单元测试
│   └── test_q_learning.py       # 测试套件
│
└── README.md                     # 本文件
```

---

## 快速开始

### 安装依赖

```bash
pip install numpy matplotlib
pip install gymnasium  # 可选，用于标准环境
```

### 基础使用

```python
from src.agents import QLearningAgent
from src.environments import CliffWalkingEnv
from src.training import Trainer

# 创建环境和智能体
env = CliffWalkingEnv()
agent = QLearningAgent(n_actions=4, learning_rate=0.5, epsilon=0.1)

# 训练
trainer = Trainer(env, agent)
metrics = trainer.train(episodes=500)

# 评估
eval_reward = trainer.evaluate(episodes=100)
print(f"评估奖励: {eval_reward:.2f}")
```

### 算法对比

```python
from src.agents import QLearningAgent, SARSAAgent
from src.training import train_q_learning, train_sarsa
from src.utils import plot_learning_curves

# 训练两种算法
q_agent = QLearningAgent(n_actions=4)
sarsa_agent = SARSAAgent(n_actions=4)

q_metrics = train_q_learning(env, q_agent, episodes=500)
sarsa_metrics = train_sarsa(env, sarsa_agent, episodes=500)

# 可视化对比
plot_learning_curves({
    'Q-Learning': q_metrics,
    'SARSA': sarsa_metrics
})
```

---

## 核心算法

### Q-Learning (Off-Policy)

$$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \max_a Q(S',a) - Q(S,A)]$$

**特点**:
- 离策略：学习最优策略，与行为策略无关
- 使用 max 操作选择下一状态最优动作
- 可能存在过估计问题

### SARSA (On-Policy)

$$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma Q(S',A') - Q(S,A)]$$

**特点**:
- 在策略：学习当前策略的价值
- 使用实际采取的下一动作
- 更保守，考虑探索风险

### Expected SARSA

$$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \mathbb{E}[Q(S',A')] - Q(S,A)]$$

**特点**:
- 结合两者优点：在策略但低方差
- 使用期望而非采样

### Double Q-Learning

**特点**:
- 解耦动作选择和价值评估
- 消除过估计偏差
- 维护两个独立 Q 表

---

## 超参数指南

| 参数 | 典型值 | 说明 |
|------|--------|------|
| learning_rate | 0.1 ~ 0.5 | 表格型可用较大值 |
| discount_factor | 0.95 ~ 0.99 | 接近1重视长期奖励 |
| epsilon | 1.0 → 0.01 | 从探索到利用 |
| epsilon_decay | 0.99 ~ 0.999 | 衰减速度 |

---

## 运行测试

```bash
cd 02-q-learning
python -m tests.test_q_learning
```

---

## 学习路径

1. **基础** → `notebooks/01-Q-Learning-Fundamentals.ipynb`
2. **对比** → `notebooks/02-SARSA-Comparison.ipynb`
3. **进阶** → `notebooks/03-Advanced-Techniques.ipynb`
4. **总结** → `docs/知识点总结.md`

---

## 参考文献

1. Watkins (1989). Learning from Delayed Rewards.
2. Rummery & Niranjan (1994). On-Line Q-Learning.
3. Van Hasselt (2010). Double Q-learning.
4. Sutton & Barto (2018). Reinforcement Learning: An Introduction.

---

[返回上级](../README.md)
