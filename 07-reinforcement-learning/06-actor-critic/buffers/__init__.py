"""
缓冲区模块 (Buffers Module)

提供策略梯度算法所需的经验存储和管理组件。

核心思想 (Core Idea):
    策略梯度方法需要收集轨迹数据来估计梯度。不同算法对数据的
    存储和访问模式有不同需求:
    - REINFORCE: 完整episode存储
    - A2C/PPO: 固定长度rollout存储
    - GAE计算: 需要价值估计和done标志

模块组成:
    - trajectory.py: 轨迹缓冲区实现
"""

from .trajectory import (
    EpisodeBuffer,
    RolloutBuffer,
    GAEBuffer,
)

__all__ = [
    "EpisodeBuffer",
    "RolloutBuffer",
    "GAEBuffer",
]
