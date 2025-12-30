"""
环境基础组件模块 (Base Components)
=================================

核心思想 (Core Idea):
--------------------
提供强化学习环境的基础构建块，包括空间定义和通用工具。
设计兼容OpenAI Gym/Gymnasium API标准。

数学原理 (Mathematical Theory):
------------------------------
强化学习环境的核心要素:
- 状态空间 S: 所有可能状态的集合
- 动作空间 A: 所有可能动作的集合
- 转移函数 P: S × A → P(S) 状态转移概率
- 奖励函数 R: S × A × S → R 即时奖励
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple


class Action(IntEnum):
    """
    标准四方向动作枚举。

    用于网格世界类环境中的移动控制。

    值说明:
        UP = 0: 向上移动（行号减少）
        RIGHT = 1: 向右移动（列号增加）
        DOWN = 2: 向下移动（行号增加）
        LEFT = 3: 向左移动（列号减少）
    """
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# 动作对应的移动向量 (row_delta, col_delta)
ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}


@dataclass
class DiscreteSpace:
    """
    离散空间定义。

    核心思想 (Core Idea):
    --------------------
    表示有限的离散集合，常用于定义离散状态空间或动作空间。
    兼容Gymnasium的spaces.Discrete接口。

    数学原理 (Mathematical Theory):
    ------------------------------
    离散空间 S = {0, 1, 2, ..., n-1}
    采样分布: P(s) = 1/n (均匀分布)

    Attributes:
        n: 空间大小（元素数量）

    Example:
        >>> action_space = DiscreteSpace(4)  # 4个动作
        >>> action = action_space.sample()   # 随机采样
        >>> print(action_space.contains(2))  # True
    """
    n: int

    def sample(self) -> int:
        """
        均匀随机采样一个元素。

        Returns:
            [0, n)范围内的随机整数
        """
        return np.random.randint(0, self.n)

    def contains(self, x: int) -> bool:
        """
        检查元素是否在空间内。

        Args:
            x: 待检查的元素

        Returns:
            是否属于该空间
        """
        return 0 <= x < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


@dataclass
class BoxSpace:
    """
    连续/整数盒空间定义。

    核心思想 (Core Idea):
    --------------------
    表示有界的多维空间，常用于定义连续状态空间或坐标空间。
    兼容Gymnasium的spaces.Box接口的子集。

    数学原理 (Mathematical Theory):
    ------------------------------
    盒空间 S = {x ∈ R^n | low ≤ x ≤ high}
    采样分布: 在边界内均匀分布

    Attributes:
        low: 各维度的下界
        high: 各维度的上界
        shape: 空间形状
        dtype: 数据类型

    Example:
        >>> # 4x12网格的坐标空间
        >>> obs_space = BoxSpace(
        ...     low=np.array([0, 0]),
        ...     high=np.array([3, 11]),
        ...     shape=(2,)
        ... )
    """
    low: np.ndarray
    high: np.ndarray
    shape: Tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.int32))

    def sample(self) -> np.ndarray:
        """
        在空间内均匀随机采样。

        Returns:
            采样的点
        """
        return np.random.randint(
            self.low,
            self.high + 1,
            size=self.shape
        ).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        """
        检查点是否在空间内。

        Args:
            x: 待检查的点

        Returns:
            是否在边界内
        """
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self) -> str:
        return f"Box({self.low}, {self.high}, {self.shape})"
