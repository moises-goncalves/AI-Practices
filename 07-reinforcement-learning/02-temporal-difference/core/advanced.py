"""
高级TD算法 (Advanced TD Algorithms)
===================================

核心思想:
--------
解决基础TD算法的局限性:
- Double Q-Learning: 解决最大化偏差
- N-Step TD: 偏差-方差权衡
- TD(λ): 通过资格迹统一所有n-step方法
- SARSA(λ)/Watkins Q(λ): 资格迹的控制版本
"""

from __future__ import annotations
from collections import defaultdict
from typing import Optional, Dict, List, Tuple
import numpy as np

from .base import BaseTDLearner, State, Action
from .config import TDConfig, EligibilityTraceType


class DoubleQLearning(BaseTDLearner[State, Action]):
    """
    Double Q-Learning: 解决最大化偏差。
    
    核心思想:
    --------
    维护两个独立Q表，一个选择动作，另一个评估，解耦消除过估计。
    
    数学原理:
    --------
    最大化偏差: E[max_a Q] ≥ max_a E[Q] (Jensen不等式)
    
    解决方案 (50%概率选择更新Q_A或Q_B):
    更新Q_A: a* = argmax Q_A(S',·), Q_A ← Q_A + α[R + γQ_B(S',a*) - Q_A]
    更新Q_B: a* = argmax Q_B(S',·), Q_B ← Q_B + α[R + γQ_A(S',a*) - Q_B]
    
    复杂度: 时间O(|A|), 空间O(2×|S|×|A|)
    """

    def __init__(self, config: TDConfig) -> None:
        super().__init__(config)
        self._q_a: Dict[Tuple[State, Action], float] = defaultdict(lambda: config.initial_value)
        self._q_b: Dict[Tuple[State, Action], float] = defaultdict(lambda: config.initial_value)

    def get_q_value(self, state: State, action: Action) -> float:
        return (self._q_a[(state, action)] + self._q_b[(state, action)]) / 2

    @property
    def q_function(self) -> Dict[Tuple[State, Action], float]:
        keys = set(self._q_a.keys()) | set(self._q_b.keys())
        return {k: (self._q_a[k] + self._q_b[k]) / 2 for k in keys}

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        update_a = np.random.random() < 0.5
        q_select = self._q_a if update_a else self._q_b
        q_eval = self._q_b if update_a else self._q_a
        q_update = self._q_a if update_a else self._q_b
        
        current_q = q_update[(state, action)]
        if done:
            td_target = reward
        else:
            best_action = max(self._action_space, key=lambda a: q_select[(next_state, a)])
            td_target = reward + self.config.gamma * q_eval[(next_state, best_action)]
        
        td_error = td_target - current_q
        q_update[(state, action)] += self.config.alpha * td_error
        return td_error


class NStepTD(BaseTDLearner[State, Action]):
    """
    N-Step TD: TD(0)与MC的中间方案。
    
    数学原理:
    --------
    G_t^{(n)} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
    
    n=1: TD(0), n→∞: MC
    偏差-方差权衡: 小n高偏差低方差，大n低偏差高方差
    实践最优n通常在4-10
    """

    def __init__(self, config: TDConfig) -> None:
        super().__init__(config)
        self._buffer: List[Tuple[State, Action, float]] = []

    def _compute_n_step_return(self, rewards: List[float], final_state: State, done: bool) -> float:
        n_step_return, discount = 0.0, 1.0
        for r in rewards:
            n_step_return += discount * r
            discount *= self.config.gamma
        if not done and self._action_space:
            max_q = max(self._q_function[(final_state, a)] for a in self._action_space)
            n_step_return += discount * max_q
        return n_step_return

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        self._buffer.append((state, action, reward))
        td_error = 0.0

        if len(self._buffer) >= self.config.n_step or done:
            s, a, _ = self._buffer[0]
            rewards = [exp[2] for exp in self._buffer]
            n_step_return = self._compute_n_step_return(rewards, next_state, done)
            td_error = n_step_return - self._q_function[(s, a)]
            self._q_function[(s, a)] += self.config.alpha * td_error
            self._buffer.pop(0)

        if done:
            while self._buffer:
                s, a, _ = self._buffer[0]
                rewards = [exp[2] for exp in self._buffer]
                n_step_return = self._compute_n_step_return(rewards, next_state, True)
                td_error = n_step_return - self._q_function[(s, a)]
                self._q_function[(s, a)] += self.config.alpha * td_error
                self._buffer.pop(0)
        return td_error


class TDLambda(BaseTDLearner[State, Action]):
    """
    TD(λ): 通过资格迹统一TD(0)和MC。
    
    数学原理:
    --------
    λ-回报: G^λ = (1-λ)Σλ^{n-1}G^{(n)} (所有n-step回报的几何加权)
    
    资格迹 (后向视图的高效实现):
    累积迹: E(s) ← γλE(s) + 1
    替换迹: E(s) ← 1
    荷兰迹: E(s) ← (1-α)γλE(s) + 1
    
    更新: Q(s,a) ← Q(s,a) + αδE(s,a), ∀s,a
    
    λ=0: TD(0), λ=1: MC, 实践推荐λ=0.9
    """

    def __init__(self, config: TDConfig) -> None:
        super().__init__(config)
        self._traces: Dict[Tuple[State, Action], float] = defaultdict(float)

    def _update_traces(self, state: State, action: Action) -> None:
        gamma_lambda = self.config.gamma * self.config.lambda_
        to_remove = []
        for key in self._traces:
            self._traces[key] *= gamma_lambda
            if self._traces[key] < 1e-8:
                to_remove.append(key)
        for key in to_remove:
            del self._traces[key]

        if self.config.trace_type == EligibilityTraceType.ACCUMULATING:
            self._traces[(state, action)] += 1.0
        elif self.config.trace_type == EligibilityTraceType.REPLACING:
            self._traces[(state, action)] = 1.0
        elif self.config.trace_type == EligibilityTraceType.DUTCH:
            curr = self._traces[(state, action)]
            self._traces[(state, action)] = (1 - self.config.alpha) * gamma_lambda * curr + 1.0

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        if done:
            td_target = reward
        elif next_action is None:
            max_q = max(self._q_function[(next_state, a)] for a in self._action_space)
            td_target = reward + self.config.gamma * max_q
        else:
            td_target = reward + self.config.gamma * self._q_function[(next_state, next_action)]

        td_error = td_target - self._q_function[(state, action)]
        self._update_traces(state, action)
        
        for (s, a), trace in self._traces.items():
            self._q_function[(s, a)] += self.config.alpha * td_error * trace

        if done:
            self._traces.clear()
        return td_error


class SARSALambda(TDLambda):
    """SARSA(λ): On-Policy TD(λ)，强制使用实际下一动作。"""

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        if done:
            td_target = reward
        else:
            if next_action is None:
                raise ValueError("SARSA(λ)需要next_action")
            td_target = reward + self.config.gamma * self._q_function[(next_state, next_action)]

        td_error = td_target - self._q_function[(state, action)]
        self._update_traces(state, action)
        for (s, a), trace in self._traces.items():
            self._q_function[(s, a)] += self.config.alpha * td_error * trace
        if done:
            self._traces.clear()
        return td_error


class WatkinsQLambda(TDLambda):
    """Watkins Q(λ): Off-Policy安全的TD(λ)，探索时清零资格迹。"""

    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        if done:
            td_target = reward
        else:
            max_q = max(self._q_function[(next_state, a)] for a in self._action_space)
            td_target = reward + self.config.gamma * max_q

        td_error = td_target - self._q_function[(state, action)]
        self._update_traces(state, action)
        for (s, a), trace in self._traces.items():
            self._q_function[(s, a)] += self.config.alpha * td_error * trace

        # 探索时清零资格迹
        if not done and next_action is not None:
            max_q = max(self._q_function[(next_state, a)] for a in self._action_space)
            greedy = [a for a in self._action_space if np.isclose(self._q_function[(next_state, a)], max_q)]
            if next_action not in greedy:
                self._traces.clear()
        if done:
            self._traces.clear()
        return td_error
