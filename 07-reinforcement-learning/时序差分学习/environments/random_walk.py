"""
随机游走环境 (Random Walk Environment)
=====================================

核心思想 (Core Idea):
--------------------
随机游走是TD预测的标准测试床。智能体在一维链上随机移动，
走到右端得到+1奖励，走到左端得到0奖励。
真实状态价值有解析解，便于验证TD算法的收敛性和准确性。

数学原理 (Mathematical Theory):
------------------------------
状态链结构:
    T(0) - A - B - C - D - E - T(1)
    (左终止)           (右终止)

随机策略下的转移:
    P(left) = P(right) = 0.5

奖励结构:
    R = 0 到达左终止状态
    R = 1 到达右终止状态
    R = 0 其他转移

真实价值函数 (γ=1时的解析解):
    V(s) = s / (n_states + 1)

例如5个非终止状态时:
    V(A) = 1/6, V(B) = 2/6, V(C) = 3/6, V(D) = 4/6, V(E) = 5/6

问题背景 (Problem Statement):
----------------------------
随机游走环境常用于:
1. 验证TD算法的正确性（与解析解比较）
2. 研究学习率α和λ的影响
3. 比较TD(0)和Monte Carlo的收敛特性
4. 分析偏差-方差权衡

Sutton & Barto教材中的经典实验表明:
- TD(0)在相同步数下误差更低（低方差优势）
- 中间的λ值（如0.8）通常表现最好

复杂度 (Complexity):
-------------------
- 状态数: n_states + 2
- 动作数: 2（但动作被忽略，转移是随机的）
- 平均回合长度: O(n_states²)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict, Any

from .base import DiscreteSpace


class RandomWalk:
    """
    随机游走环境实现。

    核心思想 (Core Idea):
    --------------------
    提供一维随机游走环境，特点:
    - 动作被忽略，转移完全随机
    - 有解析解，便于验证算法正确性
    - 简单但能展示TD的核心特性

    数学原理 (Mathematical Theory):
    ------------------------------
    状态编号: 0, 1, 2, ..., n_states, n_states+1
    其中0和n_states+1是终止状态。

    转移概率:
        P(s+1|s) = P(s-1|s) = 0.5

    Bellman方程 (γ=1):
        V(s) = 0.5 * V(s-1) + 0.5 * V(s+1)

    边界条件:
        V(0) = 0 (左终止)
        V(n+1) = 1 (右终止)

    解: V(s) = s / (n+1)

    算法对比 (Comparison):
    ---------------------
    在Random Walk上比较TD(0)和MC:

    ┌─────────────────┬──────────────┬─────────────┐
    │      特性       │    TD(0)     │     MC      │
    ├─────────────────┼──────────────┼─────────────┤
    │    方差         │     低       │     高      │
    │    偏差         │  初期有      │     无      │
    │    收敛速度     │     快       │     慢      │
    │  每步计算       │    O(1)      │  O(回合长)  │
    └─────────────────┴──────────────┴─────────────┘

    Example:
        >>> env = RandomWalk(n_states=19)
        >>> state, _ = env.reset()
        >>> # 获取真实价值用于验证
        >>> true_values = env.get_true_values(gamma=1.0)
        >>> print(f"中心状态价值: {true_values[10]}")  # 应为0.5
    """

    def __init__(self, n_states: int = 5) -> None:
        """
        初始化随机游走环境。

        Args:
            n_states: 非终止状态数量
                      总状态数 = n_states + 2（包括两个终止状态）

        Raises:
            ValueError: 当n_states < 1时
        """
        if n_states < 1:
            raise ValueError(f"至少需要1个非终止状态，当前: {n_states}")

        self.n_states = n_states
        self.n_total_states = n_states + 2  # 包括两个终止状态

        self.observation_space = DiscreteSpace(self.n_total_states)
        self.action_space = DiscreteSpace(2)  # 左/右（但实际上被忽略）

        # 终止状态
        self._left_terminal = 0
        self._right_terminal = self.n_total_states - 1

        # 起始状态（中间）
        self._start = (n_states + 1) // 2

        self._state = self._start

    def get_true_values(self, gamma: float = 1.0) -> np.ndarray:
        """
        计算随机策略下的真实状态价值。

        数学原理:
        --------
        对于γ=1，有解析解:
            V(i) = i / (n_states + 1)

        这是通过求解Bellman方程的线性系统得到的。

        Args:
            gamma: 折扣因子

        Returns:
            各状态的真实价值数组（包括终止状态）
        """
        if np.isclose(gamma, 1.0):
            # 解析解
            values = np.arange(self.n_total_states) / (self.n_states + 1)
        else:
            # 迭代求解（处理γ<1的情况）
            values = np.zeros(self.n_total_states)
            values[self._right_terminal] = 1.0

            for _ in range(1000):
                old_values = values.copy()
                for s in range(1, self.n_states + 1):
                    # 随机策略: 左右各50%
                    v_left = values[s - 1]
                    v_right = values[s + 1]

                    # 到达右端有奖励
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
        """
        重置到起始状态（中间位置）。

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            (初始状态索引, 信息字典)
        """
        if seed is not None:
            np.random.seed(seed)

        self._state = self._start
        return self._state, {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        执行一步随机转移。

        注意: 在标准随机游走中，动作被忽略，转移完全随机。
        保留动作接口是为了API兼容性。

        Args:
            action: 动作（0=左, 1=右），实际上被忽略

        Returns:
            (下一状态, 奖励, 是否终止, 是否截断, 信息字典)
        """
        # 50%概率左移，50%概率右移
        if np.random.random() < 0.5:
            self._state -= 1
        else:
            self._state += 1

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
        """
        渲染当前状态。

        显示一维状态链和当前位置。
        """
        # 状态标签: T, A, B, C, ..., T
        labels = ['T'] + [chr(ord('A') + i) for i in range(self.n_states)] + ['T']

        # 构建显示
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

    def get_state_labels(self) -> list:
        """获取状态标签列表。"""
        return ['T(0)'] + [chr(ord('A') + i) for i in range(self.n_states)] + ['T(1)']
