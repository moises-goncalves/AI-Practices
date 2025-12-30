"""
悬崖行走环境 (Cliff Walking Environment)
======================================

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
    - 到达目标: 0（回合结束）

最优策略 vs 安全策略:
    - 最优路径: 沿悬崖边缘走（13步，累积奖励-13）
    - 安全路径: 绕远路（更长但不会掉落）

Q-Learning学习最优但危险的路径（假设执行时是贪婪的）
SARSA学习安全但次优的路径（考虑探索时的风险）

问题背景 (Problem Statement):
----------------------------
这个环境直观展示了On-Policy和Off-Policy的区别:

Q-Learning (Off-Policy):
    - TD目标使用max_a Q(S',a)
    - 假设后续都是贪婪动作
    - 不考虑训练时的探索风险
    - 学到贴着悬崖走的最短路径

SARSA (On-Policy):
    - TD目标使用Q(S',A')，A'是实际采样动作
    - 知道自己会探索（可能失误）
    - 学到远离悬崖的安全路径
    - 训练时平均奖励更高

环境布局:
--------
```
. . . . . . . . . . . .   <- 第0行
. . . . . . . . . . . .   <- 第1行
. . . . . . . . . . . .   <- 第2行
S C C C C C C C C C C G   <- 第3行（底部）
```
- S: 起点 (3, 0)
- G: 终点 (3, 11)
- C: 悬崖 (3, 1) 到 (3, 10)

复杂度 (Complexity):
-------------------
- 状态数: 48
- 动作数: 4
- 最优解: 13步
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any, Set

from .base import DiscreteSpace, Action, ACTION_DELTAS


class CliffWalkingEnv:
    """
    悬崖行走环境实现。

    核心思想 (Core Idea):
    --------------------
    提供标准的悬崖行走环境，用于:
    1. 对比SARSA和Q-Learning的行为差异
    2. 理解On-Policy和Off-Policy的本质区别
    3. 探索安全强化学习的基本概念

    数学原理 (Mathematical Theory):
    ------------------------------
    最优Bellman方程:
        Q*(s,a) = E[R + γ max_{a'} Q*(s',a') | s, a]

    SARSA学习的方程:
        Q^π(s,a) = E[R + γ Q^π(s',a') | s, a, a'~π]

    区别在于Q-Learning假设后续是贪婪策略，
    而SARSA假设后续是当前策略（包含探索）。

    算法对比 (Comparison):
    ---------------------
    在Cliff Walking中:
    ┌───────────────┬─────────────────┬─────────────────┐
    │    指标       │     SARSA       │   Q-Learning    │
    ├───────────────┼─────────────────┼─────────────────┤
    │  学到的路径   │  远离悬崖       │   沿悬崖边缘    │
    │  路径长度     │     长          │      短         │
    │  训练时奖励   │     高          │      低         │
    │  评估时奖励   │     低          │      高         │
    │  风险考虑     │     是          │      否         │
    └───────────────┴─────────────────┴─────────────────┘

    Example:
        >>> env = CliffWalkingEnv()
        >>> state, _ = env.reset()
        >>> # 向右走会掉入悬崖
        >>> state, reward, done, _, info = env.step(Action.RIGHT)
        >>> print(f"掉入悬崖: {info.get('fell_off_cliff', False)}")
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

        # 悬崖位置: 底行中间部分
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
        重置环境到起点。

        Args:
            seed: 随机种子
            options: 额外选项

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
            action: 动作索引 (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)

        Returns:
            (下一状态, 奖励, 是否终止, 是否截断, 信息字典)

        奖励规则:
            - 正常移动: -1
            - 掉入悬崖: -100，回到起点，不终止
            - 到达目标: -1（最后一步），终止
        """
        self._step_count += 1

        # 计算移动后的位置
        delta = ACTION_DELTAS[Action(action)]
        new_row = np.clip(self._state[0] + delta[0], 0, self.HEIGHT - 1)
        new_col = np.clip(self._state[1] + delta[1], 0, self.WIDTH - 1)
        new_pos = (int(new_row), int(new_col))

        # 检查是否掉入悬崖
        if new_pos in self._cliff:
            # 大惩罚，回到起点，但不终止回合
            self._state = self._start
            return (
                self._pos_to_state(self._state),
                -100.0,
                False,  # 不终止
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
        """
        渲染环境状态。

        显示格式:
            A: 智能体位置
            G: 目标
            S: 起点
            C: 悬崖
            .: 普通格子
        """
        grid = []
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

    def get_optimal_path_length(self) -> int:
        """
        获取最优路径长度（理论最短路径）。

        最短路径: 先向上走到第0行，然后向右走到终点列，再向下走到终点
        实际最优: 沿悬崖边缘直走（如果不探索的话）
        """
        # 沿边缘走: 起点到终点的曼哈顿距离
        # 但要绕过悬崖，所以最短是上-右走到终点上方-下
        # 或者直接沿悬崖边缘（但这很危险）
        return 13  # 直接从(3,0)到(3,11)如果不掉入悬崖
