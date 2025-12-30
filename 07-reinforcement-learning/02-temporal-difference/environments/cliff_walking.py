"""
悬崖行走环境 (Cliff Walking Environment)
=======================================

核心思想:
--------
展示SARSA与Q-Learning区别的经典环境。
智能体从左下角走到右下角，中间是悬崖。
掉入悬崖回到起点并受大惩罚。

数学原理:
--------
状态空间: 4×12网格 = 48状态
奖励: 每步-1, 掉崖-100
Q-Learning学到最短但危险路径
SARSA学到安全但较长路径
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional, Any
import numpy as np

from .base import Action, ACTION_DELTAS, DiscreteSpace


class CliffWalkingEnv:
    """悬崖行走环境。"""
    
    HEIGHT = 4
    WIDTH = 12

    def __init__(self) -> None:
        self.observation_space = DiscreteSpace(self.HEIGHT * self.WIDTH)
        self.action_space = DiscreteSpace(4)
        self._start = (3, 0)
        self._goal = (3, 11)
        self._cliff: Set[Tuple[int, int]] = {(3, c) for c in range(1, 11)}
        self._state = self._start
        self._step_count = 0

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.WIDTH + pos[1]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[int, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self._state = self._start
        self._step_count = 0
        return self._pos_to_state(self._state), {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        self._step_count += 1
        delta = ACTION_DELTAS[Action(action)]
        new_row = np.clip(self._state[0] + delta[0], 0, self.HEIGHT - 1)
        new_col = np.clip(self._state[1] + delta[1], 0, self.WIDTH - 1)
        new_pos = (new_row, new_col)

        if new_pos in self._cliff:
            self._state = self._start
            return self._pos_to_state(self._state), -100.0, False, False, {"fell": True}

        self._state = new_pos
        terminated = self._state == self._goal
        return self._pos_to_state(self._state), -1.0, terminated, False, {}

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
                elif pos in self._cliff:
                    row.append("C")
                else:
                    row.append(".")
            grid.append(" ".join(row))
        result = "\n".join(grid) + f"\n步数: {self._step_count}\n"
        if mode == "human":
            print(result)
            return None
        return result
