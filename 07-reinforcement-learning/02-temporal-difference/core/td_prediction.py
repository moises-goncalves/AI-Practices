"""
TD预测算法模块 (TD Prediction Algorithms)
========================================

核心思想 (Core Idea):
--------------------
TD预测算法用于策略评估(Policy Evaluation)问题：给定固定策略π，
估计该策略的状态价值函数V^π或动作价值函数Q^π。

数学原理 (Mathematical Theory):
------------------------------
策略评估的目标是求解Bellman期望方程:
    V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]

TD方法通过采样和自举来近似求解，无需知道转移概率p(s',r|s,a)。

本模块包含:
- TD(0): 最基础的单步TD预测
- (TD(λ)在advanced.py中实现)

问题背景 (Problem Statement):
----------------------------
策略评估是策略迭代算法的基础。在实际应用中，
我们常常需要评估当前策略的好坏，以便进行改进。
TD(0)提供了一种在线、增量式的评估方法。

复杂度 (Complexity):
-------------------
TD(0)预测:
- 时间: O(1) per step
- 空间: O(|S|) for V function
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Tuple, Any

from .base import BaseTDLearner, Policy, State, Action
from .config import TDConfig


class TD0ValueLearner(BaseTDLearner[State, Action]):
    """
    TD(0)状态价值学习算法。

    核心思想 (Core Idea):
    --------------------
    TD(0)是最简单的TD方法，使用单步自举来更新价值估计。
    它只看下一步的奖励和下一状态的价值估计，不等待完整回合。
    这是"用猜测更新猜测"的最直接体现。

    数学原理 (Mathematical Theory):
    ------------------------------
    更新规则 (Update Rule):
        V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]

    展开形式:
        V(S_t) ← (1-α)V(S_t) + α[R_{t+1} + γV(S_{t+1})]
                  └─旧估计─┘   └──────TD目标──────┘

    TD目标 (TD Target):
        G_t^{(1)} = R_{t+1} + γV(S_{t+1})

    TD误差 (TD Error):
        δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

    TD误差表示"新证据"与"旧信念"的差距。当V收敛到V^π时，
    TD误差的期望为0。

    收敛性定理:
        在满足以下条件时，TD(0)以概率1收敛到V^π:
        1. 策略π固定不变
        2. 学习率满足Robbins-Monro条件: Σα_t = ∞, Σα_t² < ∞
        3. 所有状态被无限次访问

    问题背景 (Problem Statement):
    ----------------------------
    给定一个固定策略π，我们想知道遵循这个策略能获得多少累积奖励。
    这是策略评估(Policy Evaluation)问题，是策略迭代算法的基础。

    与Monte Carlo相比:
    - MC需要等待回合结束，TD(0)每步都能更新
    - MC无偏但高方差，TD(0)有偏但低方差
    - TD(0)更适合在线学习和连续任务

    算法对比 (Comparison):
    ---------------------
    ┌─────────────────┬──────────────┬─────────────┬─────────────┐
    │      特性       │    TD(0)     │    MC       │   n-step    │
    ├─────────────────┼──────────────┼─────────────┼─────────────┤
    │    偏差         │    有        │    无       │   可调      │
    │    方差         │    低        │    高       │   可调      │
    │  更新时机       │   每步       │  回合结束   │   延迟n步   │
    │  数据效率       │    高        │    低       │    中       │
    │  适用场景       │  在线学习    │  短回合     │   一般      │
    └─────────────────┴──────────────┴─────────────┴─────────────┘

    复杂度 (Complexity):
    -------------------
    - 时间复杂度: O(1) per update step
    - 空间复杂度: O(|S|) for storing V function

    算法总结 (Summary):
    -----------------
    TD(0)是TD学习的基础形式，通过单步自举实现在线价值估计。
    它牺牲了无偏性换取了低方差和即时更新的能力，
    特别适合连续任务和需要快速响应的场景。

    Example:
        >>> config = TDConfig(alpha=0.1, gamma=0.99)
        >>> learner = TD0ValueLearner(config)
        >>> # 评估随机策略
        >>> metrics = learner.train(env, n_episodes=1000)
        >>> print(f"状态0的价值估计: {learner.get_value(0):.2f}")
    """

    def __init__(
        self,
        config: TDConfig,
        policy: Optional[Policy[State, Action]] = None
    ) -> None:
        """
        初始化TD(0)价值学习器。

        Args:
            config: TD学习配置，包含α、γ等超参数
            policy: 待评估的策略。如果为None，则使用ε-greedy策略
        """
        super().__init__(config)
        self._policy = policy

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行TD(0)状态价值更新。

        更新规则:
            δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
            V(S_t) ← V(S_t) + αδ_t

        当回合终止时，V(S_{t+1})视为0（终止状态无后续价值）。

        Args:
            state: 当前状态 S_t
            action: 执行的动作 A_t（本算法不直接使用）
            reward: 即时奖励 R_{t+1}
            next_state: 下一状态 S_{t+1}
            next_action: 下一动作（本算法不使用）
            done: 是否到达终止状态

        Returns:
            TD误差 δ_t，用于诊断和分析收敛性

        Note:
            TD(0)是策略评估算法，不学习Q函数。
            这里同时更新V函数，action参数主要用于接口一致性。
        """
        # 计算TD目标
        # 如果是终止状态，下一状态价值为0
        if done:
            td_target = reward
        else:
            td_target = reward + self.config.gamma * self._value_function[next_state]

        # 计算TD误差
        td_error = td_target - self._value_function[state]

        # 更新状态价值估计
        self._value_function[state] += self.config.alpha * td_error

        # 同时更新Q值（用于兼容控制任务）
        self._q_function[(state, action)] += self.config.alpha * td_error

        return td_error

    def get_state_values(self) -> Dict[State, float]:
        """
        获取所有已学习状态的价值估计。

        Returns:
            状态到价值的映射
        """
        return dict(self._value_function)

    def compute_rmse(self, true_values: Dict[State, float]) -> float:
        """
        计算估计值与真实值的均方根误差。

        RMSE = √(1/n × Σ(V̂(s) - V(s))²)

        Args:
            true_values: 真实状态价值

        Returns:
            RMSE值
        """
        common_states = set(self._value_function.keys()) & set(true_values.keys())

        if not common_states:
            return float('inf')

        squared_errors = [
            (self._value_function[s] - true_values[s]) ** 2
            for s in common_states
        ]

        return np.sqrt(np.mean(squared_errors))
