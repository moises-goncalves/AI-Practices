"""
TD学习配置模块 (TD Learning Configuration)
==========================================

核心思想:
--------
封装所有TD算法的超参数，便于实验管理和复现。
提供参数验证和训练指标记录功能。

数学原理:
--------
关键超参数的作用:
- α (学习率): 控制新信息对估计值的影响，V ← V + α·δ
- γ (折扣因子): 决定未来奖励的重要性，G = Σγ^t·R_t
- λ (资格迹衰减): 控制TD(0)与MC的权衡，G^λ = (1-λ)Σλ^{n-1}G^{(n)}
- ε (探索率): ε-greedy策略的随机动作概率
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List
import numpy as np


class EligibilityTraceType(Enum):
    """
    资格迹类型枚举。
    
    累积迹: E(s) ← γλE(s) + 1，经典方法
    替换迹: E(s) ← 1，访问时重置，避免过度累积
    荷兰迹: E(s) ← (1-α)γλE(s) + 1，函数逼近下更稳定
    """
    ACCUMULATING = auto()
    REPLACING = auto()
    DUTCH = auto()


@dataclass
class TDConfig:
    """
    时序差分学习配置类。
    
    Attributes:
        alpha: 学习率 ∈ (0, 1]，典型值0.01-0.5
        gamma: 折扣因子 ∈ [0, 1]，γ=0只看即时奖励，γ=1长远同等重要
        lambda_: TD(λ)参数 ∈ [0, 1]，λ=0为TD(0)，λ=1为MC
        epsilon: ε-greedy探索率 ∈ [0, 1]
        n_step: n-step TD的步数
        trace_type: 资格迹类型
        initial_value: Q/V函数初始值，乐观初始化可促进探索
    """
    alpha: float = 0.1
    gamma: float = 0.99
    lambda_: float = 0.9
    epsilon: float = 0.1
    n_step: int = 1
    trace_type: EligibilityTraceType = EligibilityTraceType.ACCUMULATING
    initial_value: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha必须在(0,1]范围内，当前: {self.alpha}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma必须在[0,1]范围内，当前: {self.gamma}")
        if not 0 <= self.lambda_ <= 1:
            raise ValueError(f"lambda_必须在[0,1]范围内，当前: {self.lambda_}")
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"epsilon必须在[0,1]范围内，当前: {self.epsilon}")
        if self.n_step < 1:
            raise ValueError(f"n_step必须>=1，当前: {self.n_step}")


@dataclass
class TrainingMetrics:
    """训练指标记录类。"""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)
    value_changes: List[float] = field(default_factory=list)

    def add_episode(
        self,
        reward: float,
        length: int,
        avg_td_error: float = 0.0,
        avg_value_change: float = 0.0
    ) -> None:
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.td_errors.append(avg_td_error)
        self.value_changes.append(avg_value_change)

    def get_moving_average(self, window: int = 100) -> tuple:
        if len(self.episode_rewards) < window:
            return np.array(self.episode_rewards), np.array(self.episode_lengths)
        rewards = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        lengths = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
        return rewards, lengths
