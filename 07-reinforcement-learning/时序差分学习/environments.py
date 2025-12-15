"""
强化学习实验环境 (Reinforcement Learning Environments)
=====================================================

核心思想 (Core Idea):
--------------------
本模块提供用于测试和可视化TD学习算法的经典环境实现。
这些环境是手工设计的，具有明确的结构，便于理解算法行为。

包含的环境:
1. GridWorld: 经典网格世界，可配置障碍物和终点
2. CliffWalking: 悬崖行走问题，展示on-policy与off-policy的区别
3. WindyGridWorld: 有风的网格世界，测试随机环境下的学习
4. RandomWalk: 随机游走问题，TD预测的标准测试床

设计原则:
--------
- 兼容Gymnasium API (reset, step, render)
- 状态空间离散化，便于表格方法
- 提供丰富的可视化支持
- 支持自定义配置
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any, Set
import sys


# =============================================================================
# 常量和枚举
# =============================================================================

class Action(IntEnum):
    """标准四方向动作。"""
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


# =============================================================================
# 空间定义
# =============================================================================

@dataclass
class DiscreteSpace:
    """
    离散空间定义。

    兼容Gymnasium的spaces.Discrete接口。
    """
    n: int

    def sample(self) -> int:
        """随机采样一个元素。"""
        return np.random.randint(0, self.n)

    def contains(self, x: int) -> bool:
        """检查元素是否在空间内。"""
        return 0 <= x < self.n


@dataclass
class BoxSpace:
    """
    连续空间定义（用于网格坐标）。

    兼容Gymnasium的spaces.Box接口的子集。
    """
    low: np.ndarray
    high: np.ndarray
    shape: Tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.int32))

    def sample(self) -> np.ndarray:
        """随机采样一个点。"""
        return np.random.randint(
            self.low,
            self.high + 1,
            size=self.shape
        ).astype(self.dtype)

    def contains(self, x: np.ndarray) -> bool:
        """检查点是否在空间内。"""
        return np.all(x >= self.low) and np.all(x <= self.high)


# =============================================================================
# 基础网格世界
# =============================================================================

@dataclass
class GridWorldConfig:
    """
    网格世界配置。

    Attributes:
        height: 网格高度（行数）
        width: 网格宽度（列数）
        start: 起始位置 (row, col)
        goals: 目标位置集合，映射到奖励值
        obstacles: 障碍物位置集合
        default_reward: 默认步进奖励
        obstacle_reward: 碰撞障碍物奖励（如果允许穿过）
        step_cost: 每步的代价（负奖励）
        stochastic: 是否随机转移
        slip_prob: 滑动概率（随机转移时生效）
    """
    height: int = 4
    width: int = 12
    start: Tuple[int, int] = (3, 0)
    goals: Dict[Tuple[int, int], float] = field(default_factory=lambda: {(3, 11): 0.0})
    obstacles: Set[Tuple[int, int]] = field(default_factory=set)
    default_reward: float = -1.0
    obstacle_reward: float = -100.0
    step_cost: float = -1.0
    stochastic: bool = False
    slip_prob: float = 0.1


class GridWorld:
    """
    可配置的网格世界环境。

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
                   = p/2 如果 s' 是左右滑动位置

    问题背景 (Problem Statement):
    ----------------------------
    网格世界是MDP的直观表示，每个格子是一个状态，
    移动是动作，到达目标或碰撞障碍物会获得奖励/惩罚。
    它是测试和调试RL算法的理想选择。

    API:
    ----
    - reset() -> (state, info)
    - step(action) -> (next_state, reward, terminated, truncated, info)
    - render() -> None (打印到控制台)

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
        """验证配置有效性。"""
        h, w = self.config.height, self.config.width
        start_r, start_c = self.config.start

        if not (0 <= start_r < h and 0 <= start_c < w):
            raise ValueError(f"起始位置 {self.config.start} 超出网格范围")

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
            action: 动作

        Returns:
            下一位置（如果无效则返回当前位置）
        """
        action_enum = Action(action)

        if self.config.stochastic and np.random.random() < self.config.slip_prob:
            # 随机滑动到相邻方向
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
        重置环境。

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
        执行动作。

        Args:
            action: 动作索引 (0-3)

        Returns:
            (下一状态索引, 奖励, 终止标志, 截断标志, 信息字典)
        """
        self._step_count += 1

        # 计算下一位置
        next_pos = self._get_next_pos(self._state, action)

        # 计算奖励
        reward = self.config.step_cost

        # 检查是否到达目标
        terminated = False
        if next_pos in self.config.goals:
            reward += self.config.goals[next_pos]
            terminated = True

        # 更新状态
        self._state = next_pos

        # 截断检查（可选的步数限制）
        truncated = False

        info = {
            "pos": self._state,
            "step_count": self._step_count,
        }

        return (
            self._pos_to_state(self._state),
            reward,
            terminated,
            truncated,
            info
        )

    def render(self, mode: str = "human") -> Optional[str]:
        """
        渲染环境状态。

        Args:
            mode: 渲染模式 ("human" 打印, "ansi" 返回字符串)

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
        使用值迭代计算最优状态价值（用于比较）。

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
                    next_pos = self._get_next_pos(pos, action)
                    next_state = self._pos_to_state(next_pos)

                    reward = self.config.step_cost
                    if next_pos in self.config.goals:
                        reward += self.config.goals[next_pos]

                    value = reward + gamma * V[next_state]
                    max_v = max(max_v, value)

                V[state] = max_v
                delta = max(delta, abs(old_v - max_v))

            if delta < 1e-6:
                break

        return V


# =============================================================================
# 悬崖行走环境
# =============================================================================

class CliffWalkingEnv:
    """
    悬崖行走环境。

    核心思想 (Core Idea):
    --------------------
    悬崖行走是展示SARSA与Q-Learning区别的经典环境。
    智能体需要从左下角走到右下角，中间是悬崖。
    掉入悬崖会回到起点并受到大惩罚。

    数学原理 (Mathematical Theory):
    ------------------------------
    状态空间: 4×12的网格 = 48个状态
    动作空间: {UP, RIGHT, DOWN, LEFT}

    奖励结构:
        - 每步: -1
        - 掉入悬崖: -100，回到起点
        - 到达目标: 0

    最优策略与安全策略:
        - 最优路径: 沿悬崖边缘走（最短）
        - 安全路径: 远离悬崖走（更长但更安全）

    Q-Learning学习最优但危险的路径
    SARSA学习安全但次优的路径

    问题背景 (Problem Statement):
    ----------------------------
    这个环境直观展示了on-policy和off-policy的区别:
    - Q-Learning (off-policy): 不考虑探索时的风险，学到贴着悬崖走
    - SARSA (on-policy): 考虑探索时可能掉落，学到远离悬崖的路径

    Example:
        >>> env = CliffWalkingEnv()
        >>> state, _ = env.reset()
        >>> # 训练SARSA和Q-Learning，比较它们学到的路径
    """

    HEIGHT = 4
    WIDTH = 12

    def __init__(self) -> None:
        """初始化悬崖行走环境。"""
        self.observation_space = DiscreteSpace(self.HEIGHT * self.WIDTH)
        self.action_space = DiscreteSpace(4)

        # 起点和终点
        self._start = (3, 0)
        self._goal = (3, 11)

        # 悬崖位置
        self._cliff: Set[Tuple[int, int]] = {
            (3, c) for c in range(1, 11)
        }

        self._state = self._start
        self._step_count = 0

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """位置转状态索引。"""
        return pos[0] * self.WIDTH + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """状态索引转位置。"""
        return (state // self.WIDTH, state % self.WIDTH)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        重置环境。

        Returns:
            (初始状态, 信息字典)
        """
        if seed is not None:
            np.random.seed(seed)

        self._state = self._start
        self._step_count = 0
        return self._pos_to_state(self._state), {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        执行动作。

        Args:
            action: 动作索引

        Returns:
            (下一状态, 奖励, 终止, 截断, 信息)
        """
        self._step_count += 1

        # 计算移动
        delta = ACTION_DELTAS[Action(action)]
        new_row = np.clip(self._state[0] + delta[0], 0, self.HEIGHT - 1)
        new_col = np.clip(self._state[1] + delta[1], 0, self.WIDTH - 1)
        new_pos = (new_row, new_col)

        # 检查悬崖
        if new_pos in self._cliff:
            # 掉入悬崖：大惩罚，回到起点
            self._state = self._start
            return (
                self._pos_to_state(self._state),
                -100.0,
                False,
                False,
                {"fell_off_cliff": True}
            )

        # 正常移动
        self._state = new_pos

        # 检查是否到达目标
        terminated = self._state == self._goal
        reward = -1.0

        return (
            self._pos_to_state(self._state),
            reward,
            terminated,
            False,
            {}
        )

    def render(self, mode: str = "human") -> Optional[str]:
        """渲染环境。"""
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

        result = "\n".join(grid) + "\n"
        result += f"步数: {self._step_count}\n"

        if mode == "human":
            print(result)
            return None
        return result


# =============================================================================
# 有风的网格世界
# =============================================================================

class WindyGridWorld:
    """
    有风的网格世界环境。

    核心思想 (Core Idea):
    --------------------
    在标准网格世界的基础上，某些列有向上吹的风。
    风会将智能体向上推移，强度因列而异。
    这增加了环境的随机性和规划难度。

    数学原理 (Mathematical Theory):
    ------------------------------
    转移动态:
        s' = s + δ_action + δ_wind(col)

    其中 δ_wind(col) 是该列的风力向量（向上推）。

    标准配置 (Sutton & Barto):
        列:  0  1  2  3  4  5  6  7  8  9
        风:  0  0  0  1  1  1  2  2  1  0

    问题背景 (Problem Statement):
    ----------------------------
    有风网格世界测试智能体在随机环境中的规划能力。
    最优策略需要"抵消"风力影响。这是SARSA论文的原始测试环境。

    Example:
        >>> env = WindyGridWorld()
        >>> state, _ = env.reset()
        >>> # 向右移动时会被风向上吹
    """

    HEIGHT = 7
    WIDTH = 10

    # 每列的风力（向上推的格子数）
    WIND_STRENGTH = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def __init__(
        self,
        stochastic_wind: bool = False,
        king_moves: bool = False
    ) -> None:
        """
        初始化有风网格世界。

        Args:
            stochastic_wind: 是否随机风力（±1变化）
            king_moves: 是否允许8方向移动（国际象棋王的走法）
        """
        self.stochastic_wind = stochastic_wind
        self.king_moves = king_moves

        # 动作空间
        n_actions = 8 if king_moves else 4
        self.action_space = DiscreteSpace(n_actions)
        self.observation_space = DiscreteSpace(self.HEIGHT * self.WIDTH)

        # 8方向动作的移动向量
        self._action_deltas_8 = [
            (-1, 0),   # UP
            (0, 1),    # RIGHT
            (1, 0),    # DOWN
            (0, -1),   # LEFT
            (-1, 1),   # UP-RIGHT
            (1, 1),    # DOWN-RIGHT
            (1, -1),   # DOWN-LEFT
            (-1, -1),  # UP-LEFT
        ]

        # 起点和终点
        self._start = (3, 0)
        self._goal = (3, 7)

        self._state = self._start

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
            风力（向上推的格子数）
        """
        base_wind = self.WIND_STRENGTH[col]

        if self.stochastic_wind and base_wind > 0:
            # 随机风力变化
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
        return self._pos_to_state(self._state), {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """执行动作。"""
        # 获取动作移动
        if self.king_moves:
            delta = self._action_deltas_8[action]
        else:
            delta = ACTION_DELTAS[Action(action)]

        # 获取风力影响
        wind = self._get_wind(self._state[1])

        # 计算新位置（风向上吹，所以是减少行号）
        new_row = self._state[0] + delta[0] - wind
        new_col = self._state[1] + delta[1]

        # 边界裁剪
        new_row = np.clip(new_row, 0, self.HEIGHT - 1)
        new_col = np.clip(new_col, 0, self.WIDTH - 1)

        self._state = (new_row, new_col)

        terminated = self._state == self._goal
        reward = -1.0

        return (
            self._pos_to_state(self._state),
            reward,
            terminated,
            False,
            {"wind": wind}
        )

    def render(self, mode: str = "human") -> Optional[str]:
        """渲染环境。"""
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
                elif pos == self._start:
                    row.append("S")
                else:
                    row.append(".")
            grid.append(" ".join(row))

        result = "\n".join(grid) + "\n"

        if mode == "human":
            print(result)
            return None
        return result


# =============================================================================
# 随机游走环境
# =============================================================================

class RandomWalk:
    """
    随机游走环境。

    核心思想 (Core Idea):
    --------------------
    随机游走是TD预测的标准测试床。智能体在一维链上随机移动，
    走到右端得到+1奖励，走到左端得到0奖励。
    真实状态价值有解析解，便于验证TD算法的收敛性。

    数学原理 (Mathematical Theory):
    ------------------------------
    状态: A B C D E F G (7个状态)
    起始: 中间状态 D
    终止: A (左端, 奖励0) 或 G (右端, 奖励+1)

    随机策略下转移:
        P(left) = P(right) = 0.5

    真实价值 (γ=1):
        V(A) = 0 (终止)
        V(B) = 1/6
        V(C) = 2/6 = 1/3
        V(D) = 3/6 = 1/2
        V(E) = 4/6 = 2/3
        V(F) = 5/6
        V(G) = 1 (终止)

    问题背景 (Problem Statement):
    ----------------------------
    这个简单环境常用于:
    1. 验证TD算法正确性（与解析解比较）
    2. 研究学习率和λ的影响
    3. 比较TD(0)和Monte Carlo的收敛特性

    Example:
        >>> env = RandomWalk(n_states=19)
        >>> state, _ = env.reset()
        >>> # 测试TD预测算法
    """

    def __init__(self, n_states: int = 5) -> None:
        """
        初始化随机游走环境。

        Args:
            n_states: 非终止状态数量（总状态数 = n_states + 2）
        """
        if n_states < 1:
            raise ValueError("至少需要1个非终止状态")

        self.n_states = n_states
        self.n_total_states = n_states + 2  # 包括两个终止状态

        self.observation_space = DiscreteSpace(self.n_total_states)
        self.action_space = DiscreteSpace(2)  # 左/右

        # 终止状态
        self._left_terminal = 0
        self._right_terminal = self.n_total_states - 1

        # 起始状态（中间）
        self._start = (n_states + 1) // 2

        self._state = self._start

    def get_true_values(self, gamma: float = 1.0) -> np.ndarray:
        """
        计算随机策略下的真实状态价值。

        对于γ=1，有解析解:
            V(i) = i / (n_states + 1)

        Args:
            gamma: 折扣因子

        Returns:
            各状态的真实价值（包括终止状态）
        """
        if np.isclose(gamma, 1.0):
            # 解析解
            values = np.arange(self.n_total_states) / (self.n_states + 1)
        else:
            # 迭代求解
            values = np.zeros(self.n_total_states)
            values[self._right_terminal] = 1.0

            for _ in range(1000):
                old_values = values.copy()
                for s in range(1, self.n_states + 1):
                    # 随机策略: 左右各50%概率
                    # 只有到达右端有+1奖励
                    v_left = values[s - 1]
                    v_right = values[s + 1]

                    # 如果s+1是右端点，有奖励
                    r_right = 1.0 if s + 1 == self._right_terminal else 0.0

                    values[s] = 0.5 * (gamma * v_left + r_right + gamma * v_right)

                if np.max(np.abs(values - old_values)) < 1e-8:
                    break

        return values

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """重置到起始状态。"""
        if seed is not None:
            np.random.seed(seed)

        self._state = self._start
        return self._state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        执行动作。

        注意: 在标准随机游走中，动作被忽略，转移完全随机。
        这里保留动作接口是为了兼容性。

        Args:
            action: 动作（0=左, 1=右），可被忽略

        Returns:
            (下一状态, 奖励, 终止, 截断, 信息)
        """
        # 随机转移
        if np.random.random() < 0.5:
            self._state -= 1  # 左
        else:
            self._state += 1  # 右

        # 检查终止
        terminated = False
        reward = 0.0

        if self._state == self._left_terminal:
            terminated = True
            reward = 0.0
        elif self._state == self._right_terminal:
            terminated = True
            reward = 1.0

        return self._state, reward, terminated, False, {}

    def render(self, mode: str = "human") -> Optional[str]:
        """渲染环境。"""
        # 状态标签
        labels = ['T'] + [chr(ord('A') + i) for i in range(self.n_states)] + ['T']

        # 当前位置标记
        display = []
        for i, label in enumerate(labels):
            if i == self._state:
                display.append(f"[{label}]")
            else:
                display.append(f" {label} ")

        result = "".join(display) + "\n"

        if mode == "human":
            print(result)
            return None
        return result


# =============================================================================
# 21点环境（简化版）
# =============================================================================

class Blackjack:
    """
    21点（Blackjack）环境。

    核心思想 (Core Idea):
    --------------------
    21点是一个经典的纸牌游戏，用于测试Monte Carlo和TD方法。
    状态空间由玩家手牌总和、庄家明牌、是否有可用Ace组成。
    这是一个回合制游戏，展示了MC方法的优势。

    数学原理 (Mathematical Theory):
    ------------------------------
    状态空间: S = (player_sum, dealer_showing, usable_ace)
        - player_sum ∈ [12, 21]
        - dealer_showing ∈ [1, 10] (Ace=1, J/Q/K=10)
        - usable_ace ∈ {True, False}

    动作空间: A = {STICK, HIT}
        - STICK: 停牌
        - HIT: 要牌

    奖励:
        - 赢: +1
        - 输: -1
        - 平: 0

    最优策略复杂，依赖于是否有可用Ace。

    问题背景 (Problem Statement):
    ----------------------------
    21点是强化学习的经典测试问题，因为:
    1. 状态空间适中，可以用表格方法
    2. 回合短，适合MC方法
    3. 有明确的最优策略可以比较

    Example:
        >>> env = Blackjack()
        >>> state, _ = env.reset()
        >>> # player_sum, dealer_showing, usable_ace = state
    """

    def __init__(self) -> None:
        """初始化21点环境。"""
        self.observation_space = DiscreteSpace(200)  # 近似
        self.action_space = DiscreteSpace(2)  # STICK, HIT

        self._deck: List[int] = []
        self._player_hand: List[int] = []
        self._dealer_hand: List[int] = []

    def _draw_card(self) -> int:
        """抽一张牌。"""
        # 1-10, J/Q/K都算10
        return min(np.random.randint(1, 14), 10)

    def _hand_value(self, hand: List[int]) -> Tuple[int, bool]:
        """
        计算手牌价值。

        Returns:
            (总和, 是否有可用Ace)
        """
        value = sum(hand)
        usable_ace = False

        # Ace可以算1或11
        if 1 in hand and value + 10 <= 21:
            value += 10
            usable_ace = True

        return value, usable_ace

    def _get_state(self) -> Tuple[int, int, bool]:
        """获取当前状态。"""
        player_value, usable_ace = self._hand_value(self._player_hand)
        dealer_showing = self._dealer_hand[0]
        return (player_value, dealer_showing, usable_ace)

    def _state_to_int(self, state: Tuple[int, int, bool]) -> int:
        """状态转整数索引。"""
        player_sum, dealer_showing, usable_ace = state
        return (
            (player_sum - 12) * 20 +
            (dealer_showing - 1) * 2 +
            int(usable_ace)
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tuple[int, int, bool], Dict[str, Any]]:
        """
        重置游戏。

        初始发牌：玩家2张，庄家2张（1张明牌）。
        """
        if seed is not None:
            np.random.seed(seed)

        # 发牌
        self._player_hand = [self._draw_card(), self._draw_card()]
        self._dealer_hand = [self._draw_card(), self._draw_card()]

        # 确保玩家初始手牌在12-21之间
        while self._hand_value(self._player_hand)[0] < 12:
            self._player_hand.append(self._draw_card())

        state = self._get_state()
        return state, {"player_hand": self._player_hand.copy()}

    def step(
        self,
        action: int
    ) -> Tuple[Tuple[int, int, bool], float, bool, bool, Dict[str, Any]]:
        """
        执行动作。

        Args:
            action: 0=STICK（停牌）, 1=HIT（要牌）

        Returns:
            (状态, 奖励, 终止, 截断, 信息)
        """
        if action == 1:  # HIT
            self._player_hand.append(self._draw_card())
            player_value, _ = self._hand_value(self._player_hand)

            if player_value > 21:
                # 爆牌，玩家输
                return (
                    self._get_state(),
                    -1.0,
                    True,
                    False,
                    {"result": "player_bust"}
                )
            else:
                return (
                    self._get_state(),
                    0.0,
                    False,
                    False,
                    {}
                )

        else:  # STICK
            # 庄家按规则要牌
            while True:
                dealer_value, _ = self._hand_value(self._dealer_hand)
                if dealer_value >= 17:
                    break
                self._dealer_hand.append(self._draw_card())

            dealer_value, _ = self._hand_value(self._dealer_hand)
            player_value, _ = self._hand_value(self._player_hand)

            # 判断胜负
            if dealer_value > 21:
                reward = 1.0
                result = "dealer_bust"
            elif player_value > dealer_value:
                reward = 1.0
                result = "player_win"
            elif player_value < dealer_value:
                reward = -1.0
                result = "dealer_win"
            else:
                reward = 0.0
                result = "draw"

            return (
                self._get_state(),
                reward,
                True,
                False,
                {"result": result, "dealer_hand": self._dealer_hand.copy()}
            )

    def render(self, mode: str = "human") -> Optional[str]:
        """渲染游戏状态。"""
        player_value, usable_ace = self._hand_value(self._player_hand)
        dealer_value, _ = self._hand_value(self._dealer_hand)

        result = (
            f"玩家手牌: {self._player_hand} = {player_value}"
            f"{' (可用Ace)' if usable_ace else ''}\n"
            f"庄家手牌: [{self._dealer_hand[0]}, ?]\n"
        )

        if mode == "human":
            print(result)
            return None
        return result


# =============================================================================
# 单元测试
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("环境测试")
    print("=" * 60)

    # 测试GridWorld
    print("\n1. GridWorld测试")
    print("-" * 40)
    config = GridWorldConfig(
        height=4,
        width=6,
        start=(3, 0),
        goals={(0, 5): 10.0},
        obstacles={(1, 2), (2, 2)},
    )
    env = GridWorld(config)
    state, _ = env.reset()
    env.render()

    # 执行几个动作
    for action in [Action.UP, Action.RIGHT, Action.RIGHT]:
        state, reward, done, _, info = env.step(action)
        print(f"动作: {Action(action).name}, 奖励: {reward}, 位置: {info['pos']}")

    # 测试CliffWalking
    print("\n2. CliffWalking测试")
    print("-" * 40)
    cliff_env = CliffWalkingEnv()
    state, _ = cliff_env.reset()
    cliff_env.render()

    # 测试掉入悬崖
    cliff_env.step(Action.DOWN)  # 不动（在边界）
    _, reward, _, _, info = cliff_env.step(Action.RIGHT)  # 掉入悬崖
    print(f"掉入悬崖! 奖励: {reward}, 回到起点: {info.get('fell_off_cliff', False)}")

    # 测试WindyGridWorld
    print("\n3. WindyGridWorld测试")
    print("-" * 40)
    windy_env = WindyGridWorld()
    state, _ = windy_env.reset()
    windy_env.render()

    # 测试风力影响
    for _ in range(3):
        state, reward, done, _, info = windy_env.step(Action.RIGHT)
        print(f"右移，风力: {info['wind']}")

    # 测试RandomWalk
    print("\n4. RandomWalk测试")
    print("-" * 40)
    rw_env = RandomWalk(n_states=5)
    state, _ = rw_env.reset()
    rw_env.render()

    # 显示真实价值
    true_values = rw_env.get_true_values()
    print(f"真实价值: {true_values}")

    # 测试Blackjack
    print("\n5. Blackjack测试")
    print("-" * 40)
    bj_env = Blackjack()
    state, info = bj_env.reset()
    print(f"初始状态: 玩家总和={state[0]}, 庄家明牌={state[1]}, 可用Ace={state[2]}")
    bj_env.render()

    # 停牌
    final_state, reward, done, _, info = bj_env.step(0)
    print(f"停牌结果: {info['result']}, 奖励: {reward}")

    print("\n" + "=" * 60)
    print("所有环境测试完成!")
    print("=" * 60)
