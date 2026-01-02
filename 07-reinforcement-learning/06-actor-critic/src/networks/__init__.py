"""
网络模块 (Networks Module)

提供策略梯度算法所需的神经网络架构。

核心思想 (Core Idea):
    神经网络是策略梯度方法的函数逼近器，用于参数化:
    1. 策略函数 π_θ(a|s): 给定状态输出动作分布
    2. 价值函数 V_φ(s): 估计状态的期望回报

模块组成:
    - base.py: 基础网络组件 (MLP, 初始化函数)
    - policy.py: 策略网络 (离散/连续)
    - value.py: 价值网络和Actor-Critic网络
"""

from .base import MLP, init_weights, get_activation
from .policy import DiscretePolicy, ContinuousPolicy, SquashedGaussianPolicy
from .value import ValueNetwork, QNetwork, ActorCriticNetwork

__all__ = [
    # Base
    "MLP",
    "init_weights",
    "get_activation",
    # Policy
    "DiscretePolicy",
    "ContinuousPolicy",
    "SquashedGaussianPolicy",
    # Value
    "ValueNetwork",
    "QNetwork",
    "ActorCriticNetwork",
]
