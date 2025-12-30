# 高级强化学习算法

连续动作空间的深度强化学习算法实现。

## 实现算法

| 算法 | 类型 | 特点 |
|------|------|------|
| DDPG | 确定性策略 | DQN + Actor-Critic |
| TD3 | 确定性策略 | 双Q网络 + 延迟更新 |
| SAC | 最大熵RL | 自动温度调节 |

## 目录结构

```
07-advanced-algorithms/
├── algorithms/         # DDPG, TD3, SAC实现
├── core/               # 基础组件
├── training/           # 训练工具
├── notebooks/          # 教程
└── tests/              # 单元测试
```

## 快速开始

```python
from algorithms import SAC
from core import ContinuousConfig

config = ContinuousConfig(state_dim=8, action_dim=2)
agent = SAC(config)
```

## 参考文献

1. Lillicrap et al. (2016). DDPG
2. Fujimoto et al. (2018). TD3
3. Haarnoja et al. (2018). SAC
