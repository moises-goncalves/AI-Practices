# Actor-Critic 方法

结合价值函数和策略梯度的混合方法实现。

## 核心概念

**Actor-Critic架构**:
- **Actor**: 策略网络 $\pi_\theta(a|s)$，负责选择动作
- **Critic**: 价值网络 $V_\phi(s)$ 或 $Q_\phi(s,a)$，评估动作好坏

## 目录结构

```
06-actor-critic/
├── algorithms/         # A2C, PPO等算法
├── networks/           # Actor和Critic网络
├── buffers/            # 轨迹缓冲区
├── core/               # 核心配置
├── notebooks/          # 交互式教程
└── knowledge_points.md # 知识点总结
```

## 实现算法

| 算法 | 特点 |
|------|------|
| A2C | 同步优势Actor-Critic |
| PPO | 近端策略优化，稳定训练 |
| GAE | 广义优势估计 |

## 快速开始

```python
from algorithms import PPO
from core import ActorCriticConfig

config = ActorCriticConfig(state_dim=4, action_dim=2)
agent = PPO(config)
```

## 参考文献

1. Mnih et al. (2016). A3C
2. Schulman et al. (2017). PPO
