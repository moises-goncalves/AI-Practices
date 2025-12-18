"""
时序差分学习配置模块 (Configuration Module)
==========================================

核心思想 (Core Idea):
--------------------
提供统一的配置管理和超参数封装，确保实验可复现性和参数验证。

数学原理 (Mathematical Theory):
------------------------------
TD学习的关键超参数:
- α (学习率): 控制新信息对估计值的影响，满足Robbins-Monro条件时保证收敛
  收敛条件: Σα_t = ∞ 且 Σα_t² < ∞
- γ (折扣因子): 权衡即时奖励与未来奖励，γ∈[0,1]
  V^π(s) = E[Σ_{k=0}^∞ γ^k R_{t+k+1}]
- λ (资格迹参数): 控制TD(λ)中多步回报的权重分配
  G_t^λ = (1-λ)Σ_{n=1}^∞ λ^{n-1} G_t^{(n)}
- ε (探索率): ε-greedy策略中随机探索的概率

问题背景 (Problem Statement):
----------------------------
超参数选择直接影响算法的收敛速度、稳定性和最终性能。
本模块提供参数验证、默认值建议和实验配置管理功能。

复杂度 (Complexity):
-------------------
- 配置创建: O(1)
- 参数验证: O(1)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional


class EligibilityTraceType(Enum):
    """
    资格迹类型枚举。

    资格迹 (Eligibility Traces):
    --------------------------
    资格迹是TD(λ)的核心机制，用于追踪哪些状态-动作对"有资格"
    接收当前TD误差的更新。它实现了从TD(0)到Monte Carlo的平滑过渡。

    类型说明:
    --------
    ACCUMULATING (累积迹):
        E_t(s) = γλE_{t-1}(s) + 𝟙(S_t = s)
        - 每次访问状态时，迹值累加
        - 可能导致某些状态迹值过大，引起不稳定
        - 适用于状态访问频率较低的环境

    REPLACING (替换迹):
        E_t(s) = γλE_{t-1}(s) if s ≠ S_t
        E_t(S_t) = 1
        - 访问状态时，迹值重置为1
        - 避免了累积迹的发散问题
        - 适用于状态频繁重访的环境

    DUTCH (荷兰迹):
        E_t(s) = γλE_{t-1}(s) + (1 - αγλE_{t-1}(s))𝟙(S_t = s)
        - 结合了累积迹和替换迹的优点
        - 在线性函数逼近下有更好的理论保证
        - 推荐用于大规模状态空间
    """
    ACCUMULATING = auto()
    REPLACING = auto()
    DUTCH = auto()


@dataclass
class TDConfig:
    """
    时序差分学习超参数配置。

    核心思想 (Core Idea):
    --------------------
    封装所有TD算法共享的超参数，提供参数验证和默认值。
    遵循"约定优于配置"原则，同时支持高度定制。

    数学原理 (Mathematical Theory):
    ------------------------------
    超参数的理论约束:
    - α ∈ (0, 1]: 学习率，过大导致震荡，过小收敛慢
    - γ ∈ [0, 1]: 折扣因子，影响价值函数的时间尺度
    - λ ∈ [0, 1]: TD(λ)参数，λ=0即TD(0)，λ=1即Monte Carlo
    - ε ∈ [0, 1]: 探索率，平衡探索与利用

    属性说明:
    --------
    alpha: 学习率
        - 控制单次更新的步长
        - 典型范围: [0.01, 0.5]
        - 建议: 从0.1开始，根据收敛曲线调整

    gamma: 折扣因子
        - 决定未来奖励的重要性
        - γ→0: 只关心即时奖励（近视）
        - γ→1: 平等看待所有未来奖励（远视）
        - 建议: 大多数任务用0.99

    lambda_: TD(λ)参数
        - 控制自举程度和多步回报权重
        - λ=0: 纯TD(0)，低方差高偏差
        - λ=1: 纯Monte Carlo，无偏高方差
        - 建议: 从0.9开始

    epsilon: ε-greedy探索率
        - ε概率随机动作，1-ε概率贪婪动作
        - 过低可能陷入局部最优
        - 过高学习效率低
        - 建议: 从0.1开始，可递减

    n_step: N-Step TD的步数
        - 使用n步实际奖励后自举
        - n=1等价于TD(0)
        - 建议范围: [3, 10]

    trace_type: 资格迹类型
        - 决定TD(λ)中资格迹的更新方式
        - 见EligibilityTraceType说明

    initial_value: Q/V函数初始值
        - 乐观初始化可促进探索
        - 建议: 稍高于预期平均值

    Example:
        >>> config = TDConfig(alpha=0.1, gamma=0.99, epsilon=0.1)
        >>> config = TDConfig(lambda_=0.9, trace_type=EligibilityTraceType.DUTCH)
    """
    alpha: float = 0.1
    gamma: float = 0.99
    lambda_: float = 0.9
    epsilon: float = 0.1
    n_step: int = 1
    trace_type: EligibilityTraceType = EligibilityTraceType.ACCUMULATING
    initial_value: float = 0.0

    def __post_init__(self) -> None:
        """
        参数验证。

        Raises:
            ValueError: 当参数超出有效范围时
        """
        if not 0 < self.alpha <= 1:
            raise ValueError(
                f"学习率alpha必须在(0, 1]范围内，当前值: {self.alpha}。"
                f"建议值: 0.01-0.5，从0.1开始尝试。"
            )
        if not 0 <= self.gamma <= 1:
            raise ValueError(
                f"折扣因子gamma必须在[0, 1]范围内，当前值: {self.gamma}。"
                f"大多数任务建议使用0.99。"
            )
        if not 0 <= self.lambda_ <= 1:
            raise ValueError(
                f"λ参数必须在[0, 1]范围内，当前值: {self.lambda_}。"
                f"λ=0为TD(0)，λ=1为Monte Carlo。"
            )
        if not 0 <= self.epsilon <= 1:
            raise ValueError(
                f"探索率epsilon必须在[0, 1]范围内，当前值: {self.epsilon}。"
                f"建议从0.1开始。"
            )
        if self.n_step < 1:
            raise ValueError(
                f"n_step必须至少为1，当前值: {self.n_step}。"
                f"n=1等价于TD(0)，建议范围: 3-10。"
            )

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典。"""
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda_": self.lambda_,
            "epsilon": self.epsilon,
            "n_step": self.n_step,
            "trace_type": self.trace_type.name,
            "initial_value": self.initial_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TDConfig":
        """从字典反序列化。"""
        if "trace_type" in data and isinstance(data["trace_type"], str):
            data["trace_type"] = EligibilityTraceType[data["trace_type"]]
        return cls(**data)

    def with_updates(self, **kwargs) -> "TDConfig":
        """创建更新部分参数后的新配置。"""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        if "trace_type" in config_dict and isinstance(config_dict["trace_type"], str):
            config_dict["trace_type"] = EligibilityTraceType[config_dict["trace_type"]]
        return TDConfig(**config_dict)


@dataclass
class TrainingMetrics:
    """
    训练指标记录类。

    核心思想 (Core Idea):
    --------------------
    追踪和分析训练过程中的关键指标，支持性能监控、
    超参数调优和算法比较。

    记录指标:
    --------
    - episode_rewards: 每回合累积奖励
    - episode_lengths: 每回合步数
    - td_errors: TD误差（用于收敛性分析）
    - value_changes: 价值函数变化量

    分析功能:
    --------
    - 移动平均计算
    - 收敛性检测
    - 性能统计

    Example:
        >>> metrics = TrainingMetrics()
        >>> metrics.add_episode(reward=-13.0, length=13, avg_td_error=0.5)
        >>> mean_rewards, mean_lengths = metrics.get_moving_average(window=100)
    """
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
        """
        记录一个回合的指标。

        Args:
            reward: 回合累积奖励
            length: 回合步数
            avg_td_error: 平均TD误差绝对值
            avg_value_change: 平均价值变化量
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.td_errors.append(avg_td_error)
        self.value_changes.append(avg_value_change)

    def get_moving_average(
        self,
        window: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        计算奖励和回合长度的移动平均。

        移动平均能平滑噪声，更清晰地展示学习趋势。

        Args:
            window: 移动平均窗口大小

        Returns:
            (平滑后的奖励序列, 平滑后的步数序列)
        """
        if len(self.episode_rewards) < window:
            return (
                np.array(self.episode_rewards),
                np.array(self.episode_lengths)
            )

        kernel = np.ones(window) / window
        rewards = np.convolve(self.episode_rewards, kernel, mode='valid')
        lengths = np.convolve(self.episode_lengths, kernel, mode='valid')
        return rewards, lengths

    def get_statistics(
        self,
        last_n: int = 100
    ) -> Dict[str, float]:
        """
        获取最近n回合的统计信息。

        Args:
            last_n: 统计最近多少回合

        Returns:
            统计字典，包含均值、标准差、最大最小值
        """
        if len(self.episode_rewards) == 0:
            return {}

        recent_rewards = self.episode_rewards[-last_n:]
        recent_lengths = self.episode_lengths[-last_n:]

        return {
            "reward_mean": np.mean(recent_rewards),
            "reward_std": np.std(recent_rewards),
            "reward_max": np.max(recent_rewards),
            "reward_min": np.min(recent_rewards),
            "length_mean": np.mean(recent_lengths),
            "length_std": np.std(recent_lengths),
        }

    def is_converged(
        self,
        window: int = 100,
        threshold: float = 0.01
    ) -> bool:
        """
        检测训练是否收敛。

        通过比较最近两个窗口的平均奖励变化率判断。

        Args:
            window: 检测窗口大小
            threshold: 收敛阈值（相对变化率）

        Returns:
            是否收敛
        """
        if len(self.episode_rewards) < 2 * window:
            return False

        recent = np.mean(self.episode_rewards[-window:])
        previous = np.mean(self.episode_rewards[-2*window:-window])

        if abs(previous) < 1e-8:
            change_rate = abs(recent - previous)
        else:
            change_rate = abs(recent - previous) / abs(previous)

        return change_rate < threshold

    def clear(self) -> None:
        """清空所有指标。"""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.td_errors.clear()
        self.value_changes.clear()
