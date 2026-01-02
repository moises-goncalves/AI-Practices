"""
TD预测算法 (TD Prediction)
==========================

核心思想:
--------
TD预测用于策略评估：给定固定策略π，估计V^π(s)。
TD(0)是最基础的单步TD预测，每步都能更新，无需等待回合结束。

数学原理:
--------
TD(0)更新规则:
    V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]

TD目标: G_t^{(1)} = R_{t+1} + γV(S_{t+1})
TD误差: δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

收敛性: 在Robbins-Monro条件下以概率1收敛到V^π

复杂度: 时间O(1)/步, 空间O(|S|)
"""

from __future__ import annotations
from typing import Optional, Dict
import numpy as np

from .base import BaseTDLearner, Policy, State, Action
from .config import TDConfig


class TD0ValueLearner(BaseTDLearner[State, Action]):
    """
    TD(0)状态价值学习算法。
    
    核心思想:
    --------
    使用单步自举更新价值估计，是"用猜测更新猜测"的最直接体现。
    
    数学原理:
    --------
    V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
           = (1-α)V(S_t) + α·TD_target
    
    与MC对比:
    - MC: 无偏但高方差，需等待回合结束
    - TD(0): 有偏但低方差，每步更新
    
    Example:
        >>> config = TDConfig(alpha=0.1, gamma=0.99)
        >>> learner = TD0ValueLearner(config)
        >>> metrics = learner.train(env, n_episodes=1000)
    """

    def __init__(self, config: TDConfig, policy: Optional[Policy[State, Action]] = None):
        super().__init__(config)
        self._policy = policy

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        """
        TD(0)更新: δ = R + γV(S') - V(S), V(S) ← V(S) + αδ
        """
        td_target = reward if done else reward + self.config.gamma * self._value_function[next_state]
        td_error = td_target - self._value_function[state]
        self._value_function[state] += self.config.alpha * td_error
        self._q_function[(state, action)] += self.config.alpha * td_error
        return td_error

    def get_state_values(self) -> Dict[State, float]:
        return dict(self._value_function)

    def compute_rmse(self, true_values: Dict[State, float]) -> float:
        """计算RMSE = √(1/n × Σ(V̂(s) - V(s))²)"""
        common = set(self._value_function.keys()) & set(true_values.keys())
        if not common:
            return float('inf')
        errors = [(self._value_function[s] - true_values[s])**2 for s in common]
        return np.sqrt(np.mean(errors))
