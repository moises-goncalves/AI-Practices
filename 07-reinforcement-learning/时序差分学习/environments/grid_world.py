"""
网格世界环境 (Grid World Environment)
====================================

核心思想 (Core Idea):
--------------------
网格世界是强化学习最基本的测试环境。智能体在二维网格中移动，
目标是找到从起点到终点的最优路径。通过配置障碍物、奖励和
随机性，可以创建各种复杂度的问题。

数学原理 (Mathematical Theory):
------------------------------
状态空间: S = {(row, col) | 0 ≤ row < H, 0 ≤ col < W}
动作空间: A = {UP, RIGHT, DOWN, LEFT}

确定性转移:
    P(s'|s, a) = 1 如果 s' = clip(s + δ_a)
               = 0 否则

随机转移 (带滑动):
    P(s'|s, a) = 1-p 如果 s' 是目标位置
               = p/2 如果 s' 是左/右滑动位置

问题背景 (Problem Statement):
----------------------------
网格世界是MDP的直观表示:
- 每个格子是一个状态
- 移动是动作
- 到达目标或碰撞障碍物获得奖励/惩罚
是测试和调试RL算法的理想选择。

复杂度 (Complexity):
-------------------
- 状态数: O(H × W)
- 动作数: O(1) = 4
- 单步转移: O(1)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

from .base import DiscreteSpace, Action, ACTION_DELTAS


@dataclass
class GridWorldConfig:
    """
    网格世界配置类。

    核心思想 (Core Idea):
    --------------------
    封装网格世界的所有可配置参数，支持创建各种不同特性的环境。

    Attributes:
        height: 网格高度（行数）
        width: 网格宽度（列数）
        start: 起始位置 (row, col)
        goals: 目标位置到奖励的映射
        obstacles: 障碍物位置集合
        default_reward: 默认步进奖励
        step_cost: 每步的代价（负奖励）
        stochastic: 是否随机转移
        slip_prob: 滑动概率（随机转移时）

    Example:
        >>> config = GridWorldConfig(
        ...     height=4,
        ...     width=12,
        ...     start=(3, 0),
        ...     goals={(3, 11): 0.0},
        ...     step_cost=-1.0
        ... )
    """
    height: int = 4
    width: int = 12
    start: Tuple[int, int] = (3, 0)
    goals: Dict[Tuple[int, int], float] = field(
        default_factory=lambda: {(3, 11): 0.0}
    )
    obstacles: Set[Tuple[int, int]] = field(default_factory=set)
    default_reward: float = -1.0
    step_cost: float = -1.0
    stochastic: bool = False
    slip_prob: float = 0.1


class GridWorld:
    """
    可配置的网格世界环境。

    核心思想 (Core Idea):
    --------------------
    提供灵活的网格世界实现，支持:
    - 自定义大小和形状
    - 多个目标点和不同奖励
    - 障碍物配置
    - 确定性或随机转移

    数学原理 (Mathematical Theory):
    ------------------------------
    MDP元组 (S, A, P, R, γ):
    - S: 所有非障碍格子的位置
    - A: {UP, RIGHT, DOWN, LEFT}
    - P: 确定性或带滑动的随机转移
    - R: 步进代价 + 目标奖励

    Bellman方程:
        V*(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V*(s')]

    问题背景 (Problem Statement):
    ----------------------------
    网格世界用于:
    1. 算法正确性验证（可计算精确解）
    2. 可视化学习过程
    3. 理解TD方法的行为

    API (Gymnasium风格):
    -------------------
    - reset() -> (state, info)
    - step(action) -> (next_state, reward, terminated, truncated, info)
    - render() -> None

    Example:
        >>> env = GridWorld(GridWorldConfig(height=4, width=4))
        >>> state, _ = env.reset()
        >>> next_state, reward, done, _, _ = env.step(Action.RIGHT)
    """

    def __init__(self, config: Optional[GridWorldConfig] = None) -> None:
        """
        初始化网格世界。

        Args:
            config: 环境配置，None使用默认配置
        """
        self.config = config or GridWorldConfig()
        self._validate_config()

        # 空间定义
        self.observation_space = DiscreteSpace(
            self.config.height * self.config.width
        )
        self.action_space = DiscreteSpace(4)

        # 状态初始化
        self._state: Tuple[int, int] = self.config.start
        self._step_count: int = 0

    def _validate_config(self) -> None:
        """验证配置的有效性。"""
        h, w = self.config.height, self.config.width
        start_r, start_c = self.config.start

        if not (0 <= start_r < h and 0 <= start_c < w):
            raise ValueError(f"起始位置 {self.config.start} 超出网格范围 ({h}x{w})")

        for goal in self.config.goals:
            if not (0 <= goal[0] < h and 0 <= goal[1] < w):
                raise ValueError(f"目标位置 {goal} 超出网格范围")

        if self.config.start in self.config.obstacles:
            raise ValueError("起始位置不能是障碍物")

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """将位置坐标转换为状态索引。"""
        return pos[0] * self.config.width + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """将状态索引转换为位置坐标。"""
        return (state // self.config.width, state % self.config.width)

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否有效（在网格内且非障碍物）。"""
        r, c = pos
        if not (0 <= r < self.config.height and 0 <= c < self.config.width):
            return False
        return pos not in self.config.obstacles

    def _get_next_pos(
        self,
        pos: Tuple[int, int],
        action: int
    ) -> Tuple[int, int]:
        """
        计算执行动作后的下一位置。

        Args:
            pos: 当前位置
            action: 动作索引

        Returns:
            下一位置（如果无效则返回当前位置）
        """
        action_enum = Action(action)

        # 随机滑动
        if self.config.stochastic and np.random.random() < self.config.slip_prob:
            slip_direction = np.random.choice([-1, 1])
            action_enum = Action((action + slip_direction) % 4)

        delta = ACTION_DELTAS[action_enum]
        new_pos = (pos[0] + delta[0], pos[1] + delta[1])

        if self._is_valid_pos(new_pos):
            return new_pos
        return pos

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        重置环境到初始状态。

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            (初始状态索引, 信息字典)
        """
        if seed is not None:
            np.random.seed(seed)

        self._state = self.config.start
        self._step_count = 0

        return self._pos_to_state(self._state), {"pos": self._state}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        执行动作，返回环境反馈。

        Args:
            action: 动作索引 (0-3)

        Returns:
            (下一状态, 奖励, 是否终止, 是否截断, 信息字典)
        """
        self._step_count += 1

        # 状态转移
        next_pos = self._get_next_pos(self._state, action)

        # 计算奖励
        reward = self.config.step_cost

        # 检查目标
        terminated = False
        if next_pos in self.config.goals:
            reward += self.config.goals[next_pos]
            terminated = True

        self._state = next_pos

        info = {
            "pos": self._state,
            "step_count": self._step_count,
        }

        return (
            self._pos_to_state(self._state),
            reward,
            terminated,
            False,  # truncated
            info
        )

    def render(self, mode: str = "human") -> Optional[str]:
        """
        渲染环境状态。

        Args:
            mode: "human"打印到控制台，"ansi"返回字符串

        Returns:
            如果mode="ansi"，返回渲染字符串
        """
        grid = []

        for r in range(self.config.height):
            row = []
            for c in range(self.config.width):
                pos = (r, c)
                if pos == self._state:
                    row.append("A")  # Agent
                elif pos in self.config.goals:
                    row.append("G")  # Goal
                elif pos in self.config.obstacles:
                    row.append("X")  # Obstacle
                elif pos == self.config.start:
                    row.append("S")  # Start
                else:
                    row.append(".")  # Empty
            grid.append(" ".join(row))

        result = "\n".join(grid) + "\n"

        if mode == "human":
            print(result)
            return None
        return result

    def get_all_states(self) -> List[int]:
        """获取所有有效状态索引。"""
        states = []
        for r in range(self.config.height):
            for c in range(self.config.width):
                if (r, c) not in self.config.obstacles:
                    states.append(self._pos_to_state((r, c)))
        return states

    def get_optimal_value(self, gamma: float = 0.99) -> Dict[int, float]:
        """
        使用值迭代计算最优状态价值（用于验证算法正确性）。

        Args:
            gamma: 折扣因子

        Returns:
            状态到最优价值的映射
        """
        V: Dict[int, float] = {s: 0.0 for s in self.get_all_states()}

        # 目标状态价值固定
        for goal_pos, goal_reward in self.config.goals.items():
            V[self._pos_to_state(goal_pos)] = goal_reward

        # 值迭代
        for _ in range(1000):
            delta = 0
            for state in self.get_all_states():
                pos = self._state_to_pos(state)
                if pos in self.config.goals:
                    continue

                old_v = V[state]
                max_v = float('-inf')

                for action in range(4):
                    # 保存当前状态
                    saved_state = self._state
                    self._state = pos

                    next_pos = self._get_next_pos(pos, action)
                    next_state = self._pos_to_state(next_pos)

                    reward = self.config.step_cost
                    if next_pos in self.config.goals:
                        reward += self.config.goals[next_pos]

                    value = reward + gamma * V[next_state]
                    max_v = max(max_v, value)

                    # 恢复状态
                    self._state = saved_state

                V[state] = max_v
                delta = max(delta, abs(old_v - max_v))

            if delta < 1e-6:
                break

        return V
