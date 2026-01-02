"""
TD控制算法 (TD Control)
======================

核心思想:
--------
TD控制解决最优策略问题，学习Q(s,a)然后贪婪化得到策略。
关键区别在于TD目标的计算方式:
- SARSA (On-Policy): Q(S', A') - 实际下一动作
- Q-Learning (Off-Policy): max_a Q(S', a) - 最优下一动作
- Expected SARSA: E_π[Q(S', A')] - 期望下一动作

复杂度: 时间O(|A|)/步, 空间O(|S|×|A|)
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .base import BaseTDLearner, State, Action
from .config import TDConfig


class SARSA(BaseTDLearner[State, Action]):
    """
    SARSA: On-Policy TD控制。
    
    核心思想:
    --------
    名称来源: State-Action-Reward-State-Action
    使用实际下一动作A'计算TD目标，学习行为策略的真实价值。
    
    数学原理:
    --------
    Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
    
    On-Policy特性:
    - 评估当前策略π_ε（含探索）的价值
    - 在危险环境中学到更保守的策略
    - 训练时奖励高，评估时可能次优
    
    悬崖行走案例:
    - SARSA学到远离悬崖的安全路径（考虑探索风险）
    - Q-Learning学到沿悬崖边缘的最短路径（假设最优执行）
    """

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        current_q = self._q_function[(state, action)]
        if done:
            td_target = reward
        else:
            if next_action is None:
                raise ValueError("SARSA需要next_action参数")
            td_target = reward + self.config.gamma * self._q_function[(next_state, next_action)]
        td_error = td_target - current_q
        self._q_function[(state, action)] += self.config.alpha * td_error
        return td_error


class QLearning(BaseTDLearner[State, Action]):
    """
    Q-Learning: Off-Policy TD控制。
    
    核心思想:
    --------
    使用max操作选择下一状态的最优动作，直接学习最优策略Q*。
    无论行为策略如何，总是学习最优策略的价值。
    
    数学原理:
    --------
    Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
    
    对应最优Bellman方程:
    Q*(s,a) = E[R + γ max_{a'} Q*(s',a') | s,a]
    
    收敛性 (Watkins, 1989):
    在所有(s,a)被无限次访问且学习率满足Robbins-Monro条件时收敛到Q*
    
    最大化偏差:
    E[max_a Q] ≥ max_a E[Q] (Jensen不等式)
    在噪声环境中会系统性过估计，Double Q-Learning解决此问题
    """

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        current_q = self._q_function[(state, action)]
        if done:
            td_target = reward
        else:
            max_next_q = max(self._q_function[(next_state, a)] for a in self._action_space)
            td_target = reward + self.config.gamma * max_next_q
        td_error = td_target - current_q
        self._q_function[(state, action)] += self.config.alpha * td_error
        return td_error


class ExpectedSARSA(BaseTDLearner[State, Action]):
    """
    Expected SARSA: 消除动作采样方差的SARSA变体。
    
    核心思想:
    --------
    使用下一状态所有动作Q值的期望（按策略概率加权）作为TD目标，
    消除动作采样带来的方差。
    
    数学原理:
    --------
    Q(S,A) ← Q(S,A) + α[R + γE_π[Q(S',A')] - Q(S,A)]
    
    对于ε-greedy:
    E_π[Q(S',·)] = ε/|A| × ΣQ(S',a) + (1-ε) × max_a Q(S',a)
    
    特性:
    - ε=0时退化为Q-Learning
    - 比SARSA方差低，比Q-Learning偏差小
    - 计算成本O(|A|)
    """

    def _compute_expected_q(self, state: State) -> float:
        if self._action_space is None:
            raise ValueError("未设置动作空间")
        q_values = [self._q_function[(state, a)] for a in self._action_space]
        n_actions = len(self._action_space)
        exploration = (self.config.epsilon / n_actions) * sum(q_values)
        exploitation = (1 - self.config.epsilon) * max(q_values)
        return exploration + exploitation

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        current_q = self._q_function[(state, action)]
        td_target = reward if done else reward + self.config.gamma * self._compute_expected_q(next_state)
        td_error = td_target - current_q
        self._q_function[(state, action)] += self.config.alpha * td_error
        return td_error
