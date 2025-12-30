"""
强化学习环境模块 (Environments Module)
=====================================

本模块提供用于测试和可视化TD学习算法的经典环境实现。

模块结构:
--------
- base.py: 基础空间定义和通用工具
- grid_world.py: 可配置的网格世界
- cliff_walking.py: 悬崖行走环境
- windy_grid.py: 有风的网格世界
- random_walk.py: 随机游走环境
- blackjack.py: 21点环境
"""

from .base import (
    DiscreteSpace,
    BoxSpace,
    Action,
    ACTION_DELTAS,
)

from .grid_world import (
    GridWorld,
    GridWorldConfig,
)

from .cliff_walking import CliffWalkingEnv

from .windy_grid import WindyGridWorld

from .random_walk import RandomWalk

from .blackjack import Blackjack

__all__ = [
    # 基础组件
    "DiscreteSpace",
    "BoxSpace",
    "Action",
    "ACTION_DELTAS",
    # 环境
    "GridWorld",
    "GridWorldConfig",
    "CliffWalkingEnv",
    "WindyGridWorld",
    "RandomWalk",
    "Blackjack",
]
