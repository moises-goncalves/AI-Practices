"""
有风网格世界 (Windy Grid World)
==============================

核心思想:
--------
在标准网格世界基础上增加"风"的影响。
某些列有向上的风，会将智能体额外向上推动。
用于测试TD算法在非确定性环境中的表现。

数学原理:
--------
转移函数: s' = s + action_delta + wind_delta
风强度: wind[col] ∈ {0, 1, 2}
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from .base import Action, ACTION_DELTAS, DiscreteSpace


class WindyGridWorld:
    """有风网格世界环境。"""
    
    HEIGHT = 7
    WIDTH = 10
    WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # 每列的风强度

    def __init__(self, stochastic_wind: bool = False) -> None:
        """
        Args:
            stochastic_wind: 是否使用随机风（风强度±1随机变化）
        """
        self.stochastic_wind = stochastic_wind
        self.observation_space = DiscreteSpace(self.HEIGHT * self.WIDTH)
        self.action_space = DiscreteSpace(4)
        self._start = (3, 0)
        self._goal = (3, 7)
        self._state = self._start

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.WIDTH + pos[1]

    def _get_wind(self, col: int) -> int:
        base_wind = self.WIND[col]
        if self.stochastic_wind and base_wind > 0:
            return base_wind + np.random.choice([-1, 0, 1])
        return base_wind

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[int, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self._state = self._start
        return self._pos_to_state(self._state), {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        delta = ACTION_DELTAS[Action(action)]
        wind = self._get_wind(self._state[1])
        
        new_row = self._state[0] + delta[0] - wind  # 风向上推
        new_col = self._state[1] + delta[1]
        
        new_row = np.clip(new_row, 0, self.HEIGHT - 1)
        new_col = np.clip(new_col, 0, self.WIDTH - 1)
        
        self._state = (new_row, new_col)
        terminated = self._state == self._goal
        return self._pos_to_state(self._state), -1.0, terminated, False, {"wind": wind}

    def render(self, mode: str = "human") -> Optional[str]:
        grid = []
        for r in range(self.HEIGHT):
            row = []
            for c in range(self.WIDTH):
                pos = (r, c)
                if pos == self._state:
                    row.append("A")
                elif pos == self._goal:
                    row.append("G")
                elif pos == self._start:
                    row.append("S")
                else:
                    row.append(".")
            grid.append(" ".join(row))
        wind_str = " ".join(str(w) for w in self.WIND)
        result = "\n".join(grid) + f"\n风力: {wind_str}\n"
        if mode == "human":
            print(result)
            return None
        return result
