"""
基础环境组件 (Base Environment Components)
=========================================
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple
import numpy as np


class Action(IntEnum):
    """标准四方向动作。"""
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}


@dataclass
class DiscreteSpace:
    """离散空间，兼容Gymnasium。"""
    n: int
    
    def sample(self) -> int:
        return np.random.randint(0, self.n)
    
    def contains(self, x: int) -> bool:
        return 0 <= x < self.n
