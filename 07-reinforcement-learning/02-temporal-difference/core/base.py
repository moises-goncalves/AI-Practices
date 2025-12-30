"""
TD学习基类模块 (TD Learning Base Classes)
=========================================

核心思想:
--------
提供TD学习算法的通用框架，包括价值函数管理、策略实现和训练循环。
采用模板方法模式，子类只需实现特定的更新规则。

数学原理:
--------
TD学习的核心更新:
    V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
              └─旧估计─┘   └────────TD目标────────┘

TD误差 δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t) 是学习的驱动力。
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, TypeVar, Generic, Protocol
import numpy as np
import logging

from .config import TDConfig, TrainingMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

State = TypeVar('State')
Action = TypeVar('Action')


class Environment(Protocol[State, Action]):
    """环境协议，兼容Gymnasium API。"""
    def reset(self) -> Tuple[State, Dict[str, Any]]: ...
    def step(self, action: Action) -> Tuple[State, float, bool, bool, Dict[str, Any]]: ...
    @property
    def action_space(self) -> Any: ...
    @property
    def observation_space(self) -> Any: ...


class Policy(Protocol[State, Action]):
    """策略协议。"""
    def __call__(self, state: State) -> Action: ...
    def action_probabilities(self, state: State) -> Dict[Action, float]: ...


class BaseTDLearner(ABC, Generic[State, Action]):
    """
    时序差分学习算法基类。
    
    核心思想:
    --------
    TD学习结合了MC的采样和DP的自举，无需等待回合结束即可更新。
    
    数学原理:
    --------
    收敛性保证 (Robbins-Monro条件):
        Σα_t = ∞ 且 Σα_t² < ∞
    
    在满足条件时，TD(0)以概率1收敛到真实价值函数。
    """

    def __init__(self, config: TDConfig) -> None:
        self.config = config
        self._value_function: Dict[State, float] = defaultdict(lambda: config.initial_value)
        self._q_function: Dict[Tuple[State, Action], float] = defaultdict(lambda: config.initial_value)
        self.metrics = TrainingMetrics()
        self._action_space: Optional[List[Action]] = None

    @property
    def value_function(self) -> Dict[State, float]:
        return dict(self._value_function)

    @property
    def q_function(self) -> Dict[Tuple[State, Action], float]:
        return dict(self._q_function)

    def get_value(self, state: State) -> float:
        return self._value_function[state]

    def get_q_value(self, state: State, action: Action) -> float:
        return self._q_function[(state, action)]

    def set_action_space(self, actions: List[Action]) -> None:
        self._action_space = actions

    def epsilon_greedy_action(self, state: State) -> Action:
        """
        ε-greedy策略: π(a|s) = ε/|A| + (1-ε)·𝟙(a = argmax Q)
        """
        if self._action_space is None:
            raise ValueError("未设置动作空间")
        if np.random.random() < self.config.epsilon:
            return np.random.choice(self._action_space)
        q_values = [self.get_q_value(state, a) for a in self._action_space]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self._action_space, q_values) if np.isclose(q, max_q)]
        return np.random.choice(best_actions)

    def greedy_action(self, state: State) -> Action:
        if self._action_space is None:
            raise ValueError("未设置动作空间")
        q_values = [self.get_q_value(state, a) for a in self._action_space]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self._action_space, q_values) if np.isclose(q, max_q)]
        return np.random.choice(best_actions)

    @abstractmethod
    def update(self, state: State, action: Action, reward: float,
               next_state: State, next_action: Optional[Action], done: bool) -> float:
        """执行TD更新，返回TD误差。"""
        pass

    def train_episode(self, env: Environment[State, Action], max_steps: int = 10000) -> Tuple[float, int]:
        state, _ = env.reset()
        action = self.epsilon_greedy_action(state)
        total_reward, td_errors = 0.0, []

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = None if done else self.epsilon_greedy_action(next_state)
            td_error = self.update(state, action, reward, next_state, next_action, done)
            td_errors.append(abs(td_error))
            total_reward += reward
            if done:
                break
            state, action = next_state, next_action

        self.metrics.add_episode(total_reward, step + 1, np.mean(td_errors) if td_errors else 0.0)
        return total_reward, step + 1

    def train(self, env: Environment[State, Action], n_episodes: int = 1000,
              max_steps_per_episode: int = 10000, log_interval: int = 100,
              early_stop_reward: Optional[float] = None) -> TrainingMetrics:
        if self._action_space is None:
            if hasattr(env.action_space, 'n'):
                self.set_action_space(list(range(env.action_space.n)))
            else:
                raise ValueError("无法自动推断动作空间")

        for episode in range(n_episodes):
            reward, steps = self.train_episode(env, max_steps_per_episode)
            if (episode + 1) % log_interval == 0:
                avg = np.mean(self.metrics.episode_rewards[-log_interval:])
                logger.info(f"Episode {episode+1}/{n_episodes} | Avg: {avg:.2f} | Last: {reward:.2f}")
            if early_stop_reward and len(self.metrics.episode_rewards) >= 100:
                if np.mean(self.metrics.episode_rewards[-100:]) >= early_stop_reward:
                    logger.info(f"早停: 平均奖励达到 {early_stop_reward}")
                    break
        return self.metrics

    def evaluate(self, env: Environment[State, Action], n_episodes: int = 100,
                 max_steps: int = 10000) -> Tuple[float, float]:
        rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                action = self.greedy_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)
        return np.mean(rewards), np.std(rewards)
