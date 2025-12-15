"""
Deep Q-Network (DQN) 实现模块

本模块提供了完整的 DQN 算法实现，包括：

核心组件 (dqn_core.py):
- DQNConfig: 超参数配置
- DQNAgent: DQN 智能体
- DQNNetwork: 标准 Q 网络
- DuelingDQNNetwork: Dueling 架构网络
- ReplayBuffer: 均匀经验回放
- PrioritizedReplayBuffer: 优先经验回放

训练工具 (training_utils.py):
- train_dqn: 训练函数
- evaluate_agent: 评估函数
- TrainingConfig: 训练配置
- TrainingMetrics: 训练指标
- plot_training_curves: 可视化
- compare_algorithms: 算法对比

教程 (dqn_tutorial.ipynb):
- 交互式 Jupyter notebook
- 从理论到实践的完整指南

使用示例:
    >>> from dqn_core import DQNConfig, DQNAgent
    >>> from training_utils import train_dqn
    >>>
    >>> config = DQNConfig(state_dim=4, action_dim=2, double_dqn=True)
    >>> agent = DQNAgent(config)
    >>> metrics = train_dqn(agent, env_name="CartPole-v1")

Author: Ziming Ding
License: MIT
"""

from dqn_core import (
    DQNConfig,
    DQNAgent,
    DQNNetwork,
    DuelingDQNNetwork,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    NetworkType,
    LossType,
    Transition,
)

from training_utils import (
    TrainingConfig,
    TrainingMetrics,
    train_dqn,
    evaluate_agent,
    plot_training_curves,
    compare_algorithms,
)

__all__ = [
    # Core
    "DQNConfig",
    "DQNAgent",
    "DQNNetwork",
    "DuelingDQNNetwork",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "NetworkType",
    "LossType",
    "Transition",
    # Training
    "TrainingConfig",
    "TrainingMetrics",
    "train_dqn",
    "evaluate_agent",
    "plot_training_curves",
    "compare_algorithms",
]

__version__ = "1.0.0"
