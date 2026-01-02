"""
Advanced RL Algorithms - 高级强化学习算法

核心组件:
    - Algorithms: DDPG, TD3, SAC (连续控制算法)
    - Core: 基础配置、经验回放、网络架构、基类Agent
    - Training: 训练工具

数学基础:
    DDPG: 确定性策略梯度 μ_θ(s)
    TD3: 双Q网络 + 延迟策略更新 + 目标策略平滑
    SAC: 最大熵强化学习 J = E[Σ r + α H(π)]
"""

__version__ = "1.0.0"
__author__ = "AI-Practices"

from .algorithms import (
    DDPGConfig,
    DDPGAgent,
    TD3Config,
    TD3Agent,
    SACConfig,
    SACAgent,
)

from .core import (
    BaseConfig,
    DeviceMixin,
    ReplayBuffer,
    Transition,
    DeterministicActor,
    GaussianActor,
    QNetwork,
    TwinQNetwork,
    ValueNetwork,
    orthogonal_init,
    create_mlp,
    BaseAgent,
)

__all__ = [
    # Algorithms
    "DDPGConfig", "DDPGAgent",
    "TD3Config", "TD3Agent",
    "SACConfig", "SACAgent",
    # Core
    "BaseConfig", "DeviceMixin",
    "ReplayBuffer", "Transition",
    "DeterministicActor", "GaussianActor",
    "QNetwork", "TwinQNetwork", "ValueNetwork",
    "orthogonal_init", "create_mlp",
    "BaseAgent",
]
