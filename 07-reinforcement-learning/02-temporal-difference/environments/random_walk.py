"""
随机行走环境 (Random Walk Environment)
=====================================

核心思想:
--------
TD预测的标准测试环境。智能体在一维链上随机行走，
到达右端获得+1奖励，到达左端获得0奖励。
有解析解，便于验证算法正确性。

数学原理:
--------
状态空间: {0, 1, ..., n+1}，0和n+1是终止状态
策略: 随机策略，左右各50%
真实价值: V(s) = s/(n+1) (γ=1时)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from .base import DiscreteSpace


class RandomWalk:
    """随机行走环境。"""

    def __init__(self, n_states: int = 19) -> None:
        """
        Args:
            n_states: 非终止状态数量（不含两端终止状态）
        """
        self.n_states = n_states
        self.n_total_states = n_states + 2  # 含两端终止状态
        self.observation_space = DiscreteSpace(self.n_total_states)
        self.action_space = DiscreteSpace(2)  # 0=左, 1=右
        self._start = n_states // 2 + 1  # 中间状态
        self._state = self._start

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[int, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self._state = self._start
        return self._state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        # 随机行走：忽略动作，50%左右
        direction = np.random.choice([-1, 1])
        self._state += direction
        
        terminated = self._state <= 0 or self._state >= self.n_total_states - 1
        reward = 1.0 if self._state >= self.n_total_states - 1 else 0.0
        
        return self._state, reward, terminated, False, {}

    def get_true_values(self, gamma: float = 1.0) -> Dict[int, float]:
        """
        计算真实状态价值（解析解）。
        
        对于γ=1: V(s) = s / (n_states + 1)
        """
        if gamma == 1.0:
            return {s: s / (self.n_states + 1) for s in range(self.n_total_states)}
        
        # γ<1时需要迭代求解
        values = {s: 0.0 for s in range(self.n_total_states)}
        for _ in range(1000):
            new_values = {}
            for s in range(1, self.n_states + 1):
                left_val = values.get(s - 1, 0.0)
                right_val = values.get(s + 1, 0.0)
                right_reward = 1.0 if s + 1 >= self.n_total_states - 1 else 0.0
                new_values[s] = 0.5 * (gamma * left_val) + 0.5 * (right_reward + gamma * right_val)
            values.update(new_values)
        return values

    def render(self, mode: str = "human") -> Optional[str]:
        line = ["T"] + ["." if s != self._state else "A" for s in range(1, self.n_states + 1)] + ["T"]
        result = " ".join(line) + "\n"
        if mode == "human":
            print(result)
            return None
        return result
