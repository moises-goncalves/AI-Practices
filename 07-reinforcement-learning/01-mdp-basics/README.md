# 马尔可夫决策过程 (MDP) 基础

生产级MDP实现，包含完整的理论基础和求解算法。

## 核心概念

**MDP五元组**: $(S, A, P, R, \gamma)$
- $S$: 状态空间
- $A$: 动作空间  
- $P(s'|s,a)$: 状态转移概率
- $R(s,a,s')$: 奖励函数
- $\gamma$: 折扣因子

## 目录结构

```
01-mdp-basics/
├── src/
│   ├── core/           # MDP核心组件
│   ├── solvers/        # 求解算法(VI/PI/LP)
│   ├── environments/   # 基准环境
│   └── utils/          # 可视化工具
├── notebooks/          # 交互式教程
└── knowledge_points.md # 知识点总结
```

## 求解算法

| 算法 | 思想 | 复杂度 |
|------|------|--------|
| 值迭代(VI) | 迭代应用贝尔曼最优算子 | O(k\|S\|²\|A\|) |
| 策略迭代(PI) | 交替策略评估和改进 | O(k(\|S\|³+\|S\|²\|A\|)) |
| 线性规划(LP) | 直接求解LP问题 | O(\|S\|³) |

## 快速开始

```python
from src.environments import GridWorld
from src.solvers import ValueIterationSolver

env = GridWorld(height=5, width=5)
env.set_goal(4, 4)
env.build_transitions()

solver = ValueIterationSolver(env)
value_fn, policy = solver.solve()
```

## 参考文献

1. Bellman (1957). Dynamic Programming
2. Sutton & Barto (2018). Reinforcement Learning: An Introduction
