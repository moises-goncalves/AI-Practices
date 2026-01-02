"""
神经网络架构模块

提供DQN算法的各种网络架构：
- DQNNetwork: 标准MLP网络
- DuelingDQNNetwork: Dueling架构网络
"""

from .base import DQNNetwork, create_mlp, init_weights_orthogonal, init_weights_xavier
from .dueling import DuelingDQNNetwork

__all__ = [
    "DQNNetwork",
    "DuelingDQNNetwork",
    "create_mlp",
    "init_weights_orthogonal",
    "init_weights_xavier",
]
