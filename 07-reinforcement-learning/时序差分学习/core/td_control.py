"""
TD控制算法模块 (TD Control Algorithms)
=====================================

核心思想 (Core Idea):
--------------------
TD控制算法解决强化学习的核心问题：找到最优策略π*。
与TD预测（评估固定策略）不同，控制算法同时学习和改进策略。

数学原理 (Mathematical Theory):
------------------------------
控制算法学习动作价值函数Q(s,a)，然后通过贪婪化得到策略:
    π(s) = argmax_a Q(s, a)

关键区别在于TD目标的计算方式：
- SARSA (On-Policy):  使用实际下一动作 Q(S', A')
- Q-Learning (Off-Policy): 使用最优下一动作 max_a Q(S', a)
- Expected SARSA: 使用期望 E_π[Q(S', A')]

本模块包含:
- SARSA: On-Policy TD控制，学习行为策略的价值
- QLearning: Off-Policy TD控制，直接学习最优价值
- ExpectedSARSA: 消除动作采样方差的SARSA变体

问题背景 (Problem Statement):
----------------------------
在实际应用中，我们的目标是找到能最大化累积奖励的策略。
TD控制算法通过迭代学习Q函数并改进策略来实现这一目标。

复杂度 (Complexity):
-------------------
所有算法:
- 时间: O(|A|) per step (需要遍历动作)
- 空间: O(|S|×|A|) for Q table
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List

from .base import BaseTDLearner, State, Action
from .config import TDConfig


class SARSA(BaseTDLearner[State, Action]):
    """
    SARSA算法实现。

    核心思想 (Core Idea):
    --------------------
    SARSA是On-Policy TD控制算法。其名称来源于更新所需的五元组:
    (State, Action, Reward, State', Action') - S.A.R.S.A.

    关键特点是使用**实际执行的下一动作A'**来计算TD目标，
    因此学习的是当前行为策略（包含探索）的真实价值。

    数学原理 (Mathematical Theory):
    ------------------------------
    更新规则 (Update Rule):
        Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

    TD目标:
        G_t^{SARSA} = R_{t+1} + γQ(S_{t+1}, A_{t+1})

    注意A_{t+1}是按当前策略实际采样的动作，不是最优动作。

    On-Policy特性:
        SARSA评估的是当前策略π_ε（包含ε概率的随机探索）的价值，
        而不是最优策略的价值。这意味着:
        - 如果ε=0.1，SARSA学习的Q值考虑了10%的随机动作
        - 在危险环境中，SARSA会学到更保守的策略

    收敛性:
        在满足GLIE(Greedy in the Limit with Infinite Exploration)条件时，
        即ε逐渐减小到0但保证所有状态-动作对被无限次访问，
        SARSA收敛到最优策略。

    问题背景 (Problem Statement):
    ----------------------------
    考虑悬崖行走(Cliff Walking)环境：智能体需要从起点走到终点，
    中间是悬崖（掉落会回到起点并受到大惩罚）。

    Q-Learning会学到沿悬崖边缘走的最短路径，因为它假设执行时是贪婪的。
    但训练时有探索，可能掉落悬崖，实际训练奖励较低。

    SARSA会学到远离悬崖的安全路径，因为它知道自己会探索（可能失足），
    所以选择更长但更安全的路径。虽然路径次优，但训练时更稳定。

    算法对比 (Comparison):
    ---------------------
    ┌────────────────┬─────────────────┬─────────────────┐
    │    特性        │     SARSA       │   Q-Learning    │
    ├────────────────┼─────────────────┼─────────────────┤
    │    类型        │   On-Policy     │   Off-Policy    │
    │  TD目标        │   Q(S', A')     │   max_a Q(S',a) │
    │  A'来源        │   实际采样      │    假设贪婪     │
    │  学习目标      │   π_ε的价值     │    π*的价值     │
    │  安全性        │     高          │       低        │
    │  最终策略      │    保守         │      激进       │
    │ 训练时奖励     │     高          │       低        │
    │ 评估时奖励     │     低          │       高        │
    └────────────────┴─────────────────┴─────────────────┘

    复杂度 (Complexity):
    -------------------
    - 时间复杂度: O(1) per update (直接索引Q表)
    - 空间复杂度: O(|S| × |A|) for Q table

    算法总结 (Summary):
    -----------------
    SARSA通过五元组(S,A,R,S',A')进行学习。它忠实地评估当前策略
    （包括探索行为）的价值，因此在需要考虑探索风险的环境中
    往往能学到更安全、更稳定的策略。代价是最终策略可能不是最优的。

    Example:
        >>> config = TDConfig(alpha=0.5, gamma=0.99, epsilon=0.1)
        >>> sarsa = SARSA(config)
        >>> metrics = sarsa.train(env, n_episodes=500)
        >>> # SARSA在Cliff Walking中学到安全路径
    """

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
        执行SARSA更新。

        Q(S_t, A_t) ← Q(S_t, A_t) + α[R + γQ(S', A') - Q(S_t, A_t)]

        Args:
            state: 当前状态 S_t
            action: 当前动作 A_t
            reward: 即时奖励 R_{t+1}
            next_state: 下一状态 S_{t+1}
            next_action: 下一动作 A_{t+1}（SARSA必需参数）
            done: 是否终止

        Returns:
            TD误差 δ_t

        Raises:
            ValueError: 当next_action为None且done为False时
        """
        current_q = self._q_function[(state, action)]

        # 计算TD目标
        if done:
            # 终止状态：无后续价值
            td_target = reward
        else:
            if next_action is None:
                raise ValueError(
                    "SARSA需要next_action参数。在非终止状态必须提供下一动作。"
                )
            td_target = reward + self.config.gamma * self._q_function[(next_state, next_action)]

        # 计算TD误差
        td_error = td_target - current_q

        # 更新Q值
        self._q_function[(state, action)] += self.config.alpha * td_error

        return td_error


class QLearning(BaseTDLearner[State, Action]):
    """
    Q-Learning算法实现。

    核心思想 (Core Idea):
    --------------------
    Q-Learning是最著名的Off-Policy TD控制算法。无论行为策略如何，
    它总是学习最优策略的Q值——使用max操作选择下一状态的最优动作，
    而不是实际执行的动作。这种"乐观主义"使其能够直接学习最优策略。

    数学原理 (Mathematical Theory):
    ------------------------------
    更新规则 (Update Rule):
        Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]

    TD目标 (Bellman最优方程的采样近似):
        G_t^{Q-Learning} = R_{t+1} + γ max_a Q(S_{t+1}, a)

    这直接对应最优Bellman方程:
        Q*(s, a) = E[R + γ max_{a'} Q*(s', a') | s, a]

    Off-Policy特性:
        Q-Learning的行为策略（用于探索）和目标策略（最优策略）分离。
        这意味着:
        - 可以用任意策略收集数据，同时学习最优策略
        - 可以从专家演示、历史数据等学习
        - 支持经验回放(Experience Replay)

    收敛性定理 (Watkins, 1989):
        在以下条件下Q-Learning以概率1收敛到Q*:
        1. 所有状态-动作对被无限次访问
        2. 学习率满足Robbins-Monro条件: Σα_t = ∞, Σα_t² < ∞

    问题背景 (Problem Statement):
    ----------------------------
    Q-Learning直接学习最优动作价值函数Q*，这是强化学习的最终目标。
    找到Q*后，最优策略就是简单的贪婪策略: π*(s) = argmax_a Q*(s,a)

    最大化偏差 (Maximization Bias):
        Q-Learning的max操作会导致系统性的过估计:
            E[max_a Q(s,a)] ≥ max_a E[Q(s,a)]

        在噪声环境中，max会偏向选择估计值偶然偏高的动作，
        导致整体Q值过于乐观。Double Q-Learning通过解耦选择和评估解决此问题。

    算法对比 (Comparison):
    ---------------------
    ┌────────────────────┬─────────────────┐
    │       特性         │   Q-Learning    │
    ├────────────────────┼─────────────────┤
    │       类型         │   Off-Policy    │
    │     TD目标         │  max_a Q(S',a)  │
    │     学习目标       │      Q*         │
    │     收敛到         │    最优策略     │
    │     可用经验回放   │       是        │
    │   潜在问题         │   最大化偏差    │
    └────────────────────┴─────────────────┘

    复杂度 (Complexity):
    -------------------
    - 时间复杂度: O(|A|) per step (需要计算max)
    - 空间复杂度: O(|S| × |A|) for Q table

    算法总结 (Summary):
    -----------------
    Q-Learning通过"假装"后续动作都是贪婪的来直接学习最优Q函数。
    这种Off-Policy特性使其可以从任何数据源学习，是深度强化学习
    (DQN)的基础。但它可能导致在危险环境中学到不安全的策略，
    以及在噪声环境中过估计Q值。

    Example:
        >>> config = TDConfig(alpha=0.5, gamma=0.99, epsilon=0.1)
        >>> q_learner = QLearning(config)
        >>> metrics = q_learner.train(env, n_episodes=500)
        >>> # Q-Learning在Cliff Walking中学到最短（但危险）路径
    """

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
        执行Q-Learning更新。

        Q(S, A) ← Q(S, A) + α[R + γ max_a Q(S', a) - Q(S, A)]

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（Q-Learning不使用，可为None）
            done: 是否终止

        Returns:
            TD误差
        """
        current_q = self._q_function[(state, action)]

        if done:
            td_target = reward
        else:
            # 关键区别：使用max而不是实际下一动作
            max_next_q = max(
                self._q_function[(next_state, a)]
                for a in self._action_space
            )
            td_target = reward + self.config.gamma * max_next_q

        td_error = td_target - current_q
        self._q_function[(state, action)] += self.config.alpha * td_error

        return td_error


class ExpectedSARSA(BaseTDLearner[State, Action]):
    """
    Expected SARSA算法实现。

    核心思想 (Core Idea):
    --------------------
    Expected SARSA是SARSA的变体，使用下一状态所有动作Q值的**期望**
    （按策略概率加权）作为TD目标，而不是单一采样动作的Q值。
    这消除了动作采样带来的方差，使学习更加稳定。

    数学原理 (Mathematical Theory):
    ------------------------------
    更新规则:
        Q(S_t, A_t) ← Q(S_t, A_t) + α[R + γE_π[Q(S', A')] - Q(S_t, A_t)]

    期望计算:
        E_π[Q(S', A')] = Σ_a π(a|S') × Q(S', a)

    对于ε-greedy策略:
        E_π[Q(S', A')] = ε/|A| × Σ_a Q(S', a) + (1-ε) × max_a Q(S', a)
                       └────探索部分────┘   └────利用部分────┘

    方差分析:
        SARSA的更新依赖于采样的下一动作A'，引入了额外方差:
            Var[Q(S', A')] > 0

        Expected SARSA使用期望值，消除了这一方差源:
            Var[E_π[Q(S', A')]] = 0 (给定S')

    与其他算法的关系:
        - 当ε=0时，E_π[Q] = max_a Q，退化为Q-Learning
        - 当只考虑单一动作时，退化为SARSA
        - 可以看作SARSA和Q-Learning的插值

    问题背景 (Problem Statement):
    ----------------------------
    SARSA的更新因为依赖采样动作而有额外方差，可能导致学习不稳定。
    Expected SARSA通过计算期望来消除这一方差，获得更稳定的学习，
    同时保持On-Policy的特性。

    算法对比 (Comparison):
    ---------------------
    ┌─────────────────┬────────────┬────────────┬────────────────┐
    │     算法        │    方差    │   偏差     │    计算成本    │
    ├─────────────────┼────────────┼────────────┼────────────────┤
    │    SARSA        │    高      │    低      │      O(1)      │
    │  Expected SARSA │    低      │    低      │     O(|A|)     │
    │   Q-Learning    │    中      │    有*     │     O(|A|)     │
    └─────────────────┴────────────┴────────────┴────────────────┘
    * Q-Learning的"偏差"指最大化偏差

    复杂度 (Complexity):
    -------------------
    - 时间复杂度: O(|A|) per step (需要遍历所有动作计算期望)
    - 空间复杂度: O(|S| × |A|) for Q table

    算法总结 (Summary):
    -----------------
    Expected SARSA通过计算策略在下一状态的期望价值，消除了SARSA中
    动作采样的方差。它在保持On-Policy特性的同时获得更稳定的更新，
    是SARSA和Q-Learning之间的优雅折中。

    Example:
        >>> config = TDConfig(alpha=0.5, gamma=0.99, epsilon=0.1)
        >>> exp_sarsa = ExpectedSARSA(config)
        >>> metrics = exp_sarsa.train(env, n_episodes=500)
    """

    def _compute_expected_q(self, state: State) -> float:
        """
        计算状态下Q值的期望。

        对于ε-greedy策略:
            E[Q(s,·)] = ε/|A| × Σ_a Q(s,a) + (1-ε) × max_a Q(s,a)

        Args:
            state: 状态

        Returns:
            期望Q值
        """
        if self._action_space is None:
            raise ValueError("未设置动作空间")

        q_values = [self._q_function[(state, a)] for a in self._action_space]
        n_actions = len(self._action_space)

        # ε-greedy策略的期望计算
        # 探索部分：每个动作概率 ε/|A|
        exploration_value = (self.config.epsilon / n_actions) * sum(q_values)

        # 利用部分：最优动作概率 (1-ε)
        exploitation_value = (1 - self.config.epsilon) * max(q_values)

        return exploration_value + exploitation_value

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
        执行Expected SARSA更新。

        Q(S, A) ← Q(S, A) + α[R + γE[Q(S', ·)] - Q(S, A)]

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（本算法不直接使用）
            done: 是否终止

        Returns:
            TD误差
        """
        current_q = self._q_function[(state, action)]

        if done:
            td_target = reward
        else:
            expected_q = self._compute_expected_q(next_state)
            td_target = reward + self.config.gamma * expected_q

        td_error = td_target - current_q
        self._q_function[(state, action)] += self.config.alpha * td_error

        return td_error
