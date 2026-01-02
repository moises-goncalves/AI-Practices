"""
网格世界环境 (Grid World Environment)
====================================

核心思想:
--------
网格世界是强化学习最基本的测试环境。智能体在二维网格中移动，
目标是找到从起点到终点的最优路径。

数学原理:
--------
状态空间: S = {(row, col) | 0 ≤ row < H, 0 ≤ col < W}
动作空间: A = {UP, RIGHT, DOWN, LEFT}
转移: P(s'|s,a) = 1 (确定性) 或带滑动概率
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Optional, Any, List
import numpy as np

from .base import Action, ACTION_DELTAS, DiscreteSpace


@dataclass
class GridWorldConfig:
    """网格世界配置。"""
    height: int = 4
    width: int = 12
    start: Tuple[int, int] = (3, 0)
    goals: Dict[Tuple[int, int], float] = field(default_factory=lambda: {(3, 11): 0.0})
    obstacles: Set[Tuple[int, int]] = field(default_factory=set)
    step_cost: float = -1.0
    stochastic: bool = False
    slip_prob: float = 0.1


class GridWorld:
    """
    可配置的网格世界环境。
    
    API兼容Gymnasium: reset(), step(), render()
    """

    def __init__(self, config: Optional[GridWorldConfig] = None) -> None:
        self.config = config or GridWorldConfig()
        self._validate_config()
        self.observation_space = DiscreteSpace(self.config.height * self.config.width)
        self.action_space = DiscreteSpace(4)
        self._state: Tuple[int, int] = self.config.start
        self._step_count: int = 0

    def _validate_config(self) -> None:
        h, w = self.config.height, self.config.width
        sr, sc = self.config.start
        if not (0 <= sr < h and 0 <= sc < w):
            raise ValueError(f"起始位置 {self.config.start} 超出范围")
        for goal in self.config.goals:
            if not (0 <= goal[0] < h and 0 <= goal[1] < w):
                raise ValueError(f"目标位置 {goal} 超出范围")

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.config.width + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        return (state // self.config.width, state % self.config.width)

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        if not (0 <= r < self.config.height and 0 <= c < self.config.width):
            return False
        return pos not in self.config.obstacles

    def _get_next_pos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        action_enum = Action(action)
        if self.config.stochastic and np.random.random() < self.config.slip_prob:
            slip = np.random.choice([-1, 1])
            action_enum = Action((action + slip) % 4)
        delta = ACTION_DELTAS[action_enum]
        new_pos = (pos[0] + delta[0], pos[1] + delta[1])
        return new_pos if self._is_valid_pos(new_pos) else pos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[int, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self._state = self.config.start
        self._step_count = 0
        return self._pos_to_state(self._state), {"pos": self._state}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        self._step_count += 1
        next_pos = self._get_next_pos(self._state, action)
        reward = self.config.step_cost
        terminated = next_pos in self.config.goals
        if terminated:
            reward += self.config.goals[next_pos]
        self._state = next_pos
        return self._pos_to_state(self._state), reward, terminated, False, {"pos": self._state}

    def render(self, mode: str = "human") -> Optional[str]:
        grid = []
        for r in range(self.config.height):
            row = []
            for c in range(self.config.width):
                pos = (r, c)
                if pos == self._state:
                    row.append("A")
                elif pos in self.config.goals:
                    row.append("G")
                elif pos in self.config.obstacles:
                    row.append("X")
                elif pos == self.config.start:
                    row.append("S")
                else:
                    row.append(".")
            grid.append(" ".join(row))
        result = "\n".join(grid) + "\n"
        if mode == "human":
            print(result)
            return None
        return result

    def get_all_states(self) -> List[int]:
        return [self._pos_to_state((r, c)) 
                for r in range(self.config.height) 
                for c in range(self.config.width) 
                if (r, c) not in self.config.obstacles]
