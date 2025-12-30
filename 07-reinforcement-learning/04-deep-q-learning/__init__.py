"""
深度Q学习实现 - 主模块

============================================================
模块概述 (Module Overview)
============================================================
本模块提供完整的Deep Q-Network (DQN)实现，包括：

1. 核心算法
   - 标准DQN
   - Double DQN（解决过估计问题）
   - Dueling DQN（价值-优势分解）

2. 经验回放
   - 均匀采样回放缓冲区
   - 优先经验回放(PER)

3. 训练工具
   - 训练循环
   - 评估函数
   - 可视化工具

============================================================
快速开始 (Quick Start)
============================================================
>>> from agent import DQNAgent, create_dqn_agent
>>> from core import DQNConfig
>>> 
>>> # 创建agent
>>> agent = create_dqn_agent(state_dim=4, action_dim=2)
>>> 
>>> # 训练循环
>>> state = env.reset()
>>> action = agent.select_action(state)
>>> next_state, reward, done, _ = env.step(action)
>>> loss = agent.train_step(state, action, reward, next_state, done)

============================================================
目录结构 (Directory Structure)
============================================================
实现深度Q学习/
├── __init__.py          # 主模块入口
├── agent.py             # DQN Agent实现
├── train.py             # 训练脚本
├── core/                # 核心配置和类型
│   ├── config.py        # DQNConfig配置类
│   ├── types.py         # 数据类型定义
│   └── enums.py         # 枚举类型
├── buffers/             # 经验回放缓冲区
│   ├── base.py          # 均匀回放缓冲区
│   ├── sum_tree.py      # Sum Tree数据结构
│   └── prioritized.py   # 优先经验回放
├── networks/            # 神经网络架构
│   ├── base.py          # 标准DQN网络
│   └── dueling.py       # Dueling网络
├── utils/               # 工具函数
│   ├── training.py      # 训练配置和指标
│   └── visualization.py # 可视化工具
├── notebooks/           # Jupyter教程
└── tests/               # 单元测试
"""

from .agent import DQNAgent, create_dqn_agent
from .core import DQNConfig, Transition, NStepTransition
from .core import NetworkType, LossType, ExplorationStrategy
from .buffers import ReplayBuffer, PrioritizedReplayBuffer
from .networks import DQNNetwork, DuelingDQNNetwork

__version__ = "1.0.0"
__author__ = "AI-Practices Contributors"

__all__ = [
    # Agent
    "DQNAgent",
    "create_dqn_agent",
    # Config
    "DQNConfig",
    # Types
    "Transition",
    "NStepTransition",
    # Enums
    "NetworkType",
    "LossType",
    "ExplorationStrategy",
    # Buffers
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    # Networks
    "DQNNetwork",
    "DuelingDQNNetwork",
]
