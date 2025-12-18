"""
有风网格世界环境 (Windy Grid World)
==================================

核心思想 (Core Idea):
--------------------
在标准网格世界的基础上，某些列有向上吹的风。
风会将智能体向上推移，强度因列而异。
这增加了环境的随机性和规划难度。

数学原理 (Mathematical Theory):
------------------------------
转移动态包含风力影响:
    s' = s + δ_action + δ_wind(col)

其中 δ_wind(col) 是该列的风力向量（向上推）。

标准配置 (Sutton & Barto):
    列:  0  1  2  3  4  5  6  7  8  9
    风:  0  0  0  1  1  1  2  2  1  0

智能体需要学会"抵消"风力来有效地移动。

问题背景 (Problem Statement):
----------------------------
有风网格世界测试智能体在随机环境中的规划能力。
这是SARSA论文的原始测试环境。最优策略需要考虑风力，
在某些位置可能需要向下走来保持位置或缓慢向上移动。

复杂度 (Complexity):
-------------------
- 状态数: 7 × 10 = 70
- 动作数: 4 (或8，如果允许斜向移动)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from .base import DiscreteSpace, Action, ACTION_DELTAS


class WindyGridWorld:
    """
    有风网格世界环境实现。

    核心思想 (Core Idea):
    --------------------
    在网格世界中添加风力效果，测试智能体应对环境动态的能力。
    支持两种移动模式：标准4方向和国王8方向移动。

    数学原理 (Mathematical Theory):
    ------------------------------
    状态转移:
        next_row = current_row + action_delta_row - wind_strength
        next_col = current_col + action_delta_col

    风力向上吹（减少行号），需要裁剪到网格边界内。

    环境布局:
    --------
    ```
    风力: 0  0  0  1  1  1  2  2  1  0
         ───────────────────────────────
         .  .  .  .  .  .  .  .  .  .   <- 第0行
         .  .  .  .  .  .  .  .  .  .
         .  .  .  .  .  .  .  .  .  .
         S  .  .  .  .  .  .  G  .  .   <- 第3行
         .  .  .  .  .  .  .  .  .  .
         .  .  .  .  .  .  .  .  .  .
         .  .  .  .  .  .  .  .  .  .   <- 第6行
    ```
    - S: 起点 (3, 0)
    - G: 终点 (3, 7)
    - 中间列有向上的风

    算法对比 (Comparison):
    ---------------------
    学习效率对比（达到最优策略所需回合数）:
    - SARSA: ~170 回合
    - SARSA(λ): ~30 回合（λ=0.9时）

    Example:
        >>> env = WindyGridWorld()
        >>> state, _ = env.reset()
        >>> # 在有风区域向右移动会被向上吹
        >>> state, reward, done, _, info = env.step(Action.RIGHT)
        >>> print(f"风力: {info['wind']}")
    """

    HEIGHT = 7
    WIDTH = 10

    # 每列的风力（向上推的格子数）
    WIND_STRENGTH: List[int] = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def __init__(
        self,
        stochastic_wind: bool = False,
        king_moves: bool = False
    ) -> None:
        """
        初始化有风网格世界。

        Args:
            stochastic_wind: 是否使用随机风力（基础风力±1随机变化）
            king_moves: 是否允许8方向移动（国际象棋王的走法）
        """
        self.stochastic_wind = stochastic_wind
        self.king_moves = king_moves

        # 动作空间
        n_actions = 8 if king_moves else 4
        self.action_space = DiscreteSpace(n_actions)
        self.observation_space = DiscreteSpace(self.HEIGHT * self.WIDTH)

        # 8方向移动的向量
        self._action_deltas_8 = [
            (-1, 0),   # UP
            (0, 1),    # RIGHT
            (1, 0),    # DOWN
            (0, -1),   # LEFT
            (-1, 1),   # UP-RIGHT (对角)
            (1, 1),    # DOWN-RIGHT
            (1, -1),   # DOWN-LEFT
            (-1, -1),  # UP-LEFT
        ]

        # 起点和终点
        self._start = (3, 0)
        self._goal = (3, 7)

        self._state = self._start
        self._step_count = 0

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """位置转状态索引。"""
        return pos[0] * self.WIDTH + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """状态索引转位置。"""
        return (state // self.WIDTH, state % self.WIDTH)

    def _get_wind(self, col: int) -> int:
        """
        获取指定列的风力。

        Args:
            col: 列索引

        Returns:
            风力值（向上推的格子数）
        """
        base_wind = self.WIND_STRENGTH[col]

        if self.stochastic_wind and base_wind > 0:
            # 随机风力变化: 基础值 + {-1, 0, 1}
            variation = np.random.choice([-1, 0, 1])
            return max(0, base_wind + variation)

        return base_wind

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """重置环境。"""
        if seed is not None:
            np.random.seed(seed)

        self._state = self._start
        self._step_count = 0
        return self._pos_to_state(self._state), {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        执行动作。

        考虑风力影响的状态转移。

        Args:
            action: 动作索引

        Returns:
            (下一状态, 奖励, 是否终止, 是否截断, 信息字典)
        """
        self._step_count += 1

        # 获取动作移动向量
        if self.king_moves:
            delta = self._action_deltas_8[action]
        else:
            delta = ACTION_DELTAS[Action(action)]

        # 获取当前位置的风力
        wind = self._get_wind(self._state[1])

        # 计算新位置（风向上吹，减少行号）
        new_row = self._state[0] + delta[0] - wind
        new_col = self._state[1] + delta[1]

        # 边界裁剪
        new_row = int(np.clip(new_row, 0, self.HEIGHT - 1))
        new_col = int(np.clip(new_col, 0, self.WIDTH - 1))

        self._state = (new_row, new_col)

        terminated = self._state == self._goal
        reward = -1.0

        return (
            self._pos_to_state(self._state),
            reward,
            terminated,
            False,
            {"wind": wind, "step_count": self._step_count}
        )

    def render(self, mode: str = "human") -> Optional[str]:
        """渲染环境状态。"""
        grid = []

        # 顶部风力指示
        wind_str = "风力: " + " ".join(str(w) for w in self.WIND_STRENGTH)
        grid.append(wind_str)
        grid.append("-" * (self.WIDTH * 2 - 1))

        for r in range(self.HEIGHT):
            row = []
            for c in range(self.WIDTH):
                pos = (r, c)
                if pos == self._state:
                    row.append("A")
                elif pos == self._goal:
                    row.append("G")
                elif pos == self._start and pos != self._state:
                    row.append("S")
                else:
                    row.append(".")
            grid.append(" ".join(row))

        result = "\n".join(grid) + "\n"
        result += f"步数: {self._step_count}\n"

        if mode == "human":
            print(result)
            return None
        return result
