#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI Gymnasium 强化学习环境接口指南

================================================================================
核心思想 (Core Idea)
================================================================================
Gymnasium (原 OpenAI Gym) 提供了强化学习研究的标准化接口，定义了智能体与环境
交互的统一 API。通过标准化的 observation-action-reward 循环，研究者可以在
不同环境间无缝切换算法，实现算法的可复用性和可比较性。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
强化学习建模为马尔可夫决策过程 (MDP):

    MDP = (S, A, P, R, γ)

其中:
    - S: 状态空间 (State Space)
    - A: 动作空间 (Action Space)
    - P(s'|s,a): 状态转移概率 (Transition Probability)
    - R(s,a,s'): 奖励函数 (Reward Function)
    - γ ∈ [0,1]: 折扣因子 (Discount Factor)

智能体目标是最大化期望累积折扣奖励:

    $$G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}$$

Gymnasium 环境封装了 MDP 的 P 和 R，暴露 S 和 A 给智能体。

================================================================================
问题背景 (Problem Statement)
================================================================================
强化学习研究面临的挑战:
1. 环境实现不统一，算法难以跨环境测试
2. 环境接口各异，增加了代码复用成本
3. 缺乏标准化的 benchmark 环境
4. 调试和可视化工具分散

Gymnasium 通过统一的 API 设计解决了这些问题。

================================================================================
接口对比 (API Comparison)
================================================================================
| 特性            | Gymnasium (新)      | Gym (旧)           |
|-----------------|--------------------|--------------------|
| 接口风格        | 函数式             | 类方法             |
| reset() 返回值  | (obs, info)        | obs                |
| step() 返回值   | (obs,r,term,trunc,i) | (obs,r,done,info) |
| 终止信号        | terminated/truncated | done             |
| 向量化环境      | 内置支持           | 需要额外库         |
| 类型提示        | 完整支持           | 部分支持           |

================================================================================
时间/空间复杂度 (Complexity)
================================================================================
- reset(): O(1) 至 O(n) 取决于环境初始化复杂度
- step(): O(1) 至 O(n) 取决于物理模拟复杂度
- render(): O(w×h) 其中 w,h 为渲染分辨率

================================================================================
算法总结 (Summary)
================================================================================
Gymnasium 是强化学习研究的基础设施，提供:
1. 统一的环境接口 (env.reset(), env.step())
2. 标准化的空间定义 (Box, Discrete, MultiDiscrete, etc.)
3. 丰富的内置环境 (Classic Control, Atari, MuJoCo, etc.)
4. 可扩展的环境包装器 (Wrappers)
5. 向量化环境支持 (Vectorized Environments)

运行环境:
    Python >= 3.8
    gymnasium >= 0.29.0
    numpy >= 1.21.0
    matplotlib >= 3.5.0 (可视化)

Author: Ziming Ding
Date: 2024
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional,
    Protocol, Sequence, Tuple, TypeVar, Union
)

import numpy as np

# 条件导入
try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.wrappers import (
        RecordEpisodeStatistics,
        TimeLimit,
        ClipAction,
    )
    # 某些包装器可能不在所有版本中
    try:
        from gymnasium.wrappers import NormalizeObservation, NormalizeReward, FrameStack
    except ImportError:
        NormalizeObservation = None
        NormalizeReward = None
        FrameStack = None
    HAS_GYMNASIUM = True
except ImportError as e:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None
    warnings.warn(
        "gymnasium 未安装。请执行: pip install gymnasium[classic-control]",
        ImportWarning
    )

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# ============================================================================
#                           类型定义与协议
# ============================================================================

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


class EnvironmentProtocol(Protocol[ObsType, ActType]):
    """
    强化学习环境协议定义

    任何符合此协议的环境都可以与标准 RL 算法交互。
    这是 Gymnasium 环境接口的抽象表示。
    """

    observation_space: spaces.Space
    action_space: spaces.Space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """重置环境到初始状态"""
        ...

    def step(
        self,
        action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """执行动作并返回转移结果"""
        ...

    def render(self) -> Optional[np.ndarray]:
        """渲染环境当前状态"""
        ...

    def close(self) -> None:
        """释放环境资源"""
        ...


class SpaceType(Enum):
    """Gymnasium 空间类型枚举"""
    DISCRETE = auto()      # 离散空间: {0, 1, ..., n-1}
    BOX = auto()           # 连续空间: [low, high]^n
    MULTI_DISCRETE = auto()  # 多离散空间
    MULTI_BINARY = auto()  # 多二值空间
    TUPLE = auto()         # 元组空间
    DICT = auto()          # 字典空间


# ============================================================================
#                           空间分析工具
# ============================================================================

@dataclass(frozen=True)
class SpaceInfo:
    """
    空间信息数据类

    封装 Gymnasium 空间的关键属性，便于算法选择合适的策略网络。

    Attributes:
        space_type: 空间类型枚举
        shape: 空间形状 (用于 Box 和 MultiDiscrete)
        n: 离散空间大小 (仅 Discrete)
        low: 下界 (仅 Box)
        high: 上界 (仅 Box)
        dtype: 数据类型
        is_bounded: 是否有界
        flat_dim: 展平后的维度
    """
    space_type: SpaceType
    shape: Optional[Tuple[int, ...]] = None
    n: Optional[int] = None
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None
    dtype: Optional[np.dtype] = None
    is_bounded: bool = True
    flat_dim: int = 0


def analyze_space(space: spaces.Space) -> SpaceInfo:
    """
    分析 Gymnasium 空间属性

    根据空间类型提取关键信息，用于构建兼容的神经网络架构。

    Parameters
    ----------
    space : spaces.Space
        Gymnasium 空间对象

    Returns
    -------
    SpaceInfo
        包含空间关键属性的数据对象

    Examples
    --------
    >>> env = gym.make("CartPole-v1")
    >>> obs_info = analyze_space(env.observation_space)
    >>> print(f"观测维度: {obs_info.shape}, 类型: {obs_info.space_type}")
    观测维度: (4,), 类型: SpaceType.BOX

    >>> act_info = analyze_space(env.action_space)
    >>> print(f"动作数量: {act_info.n}")
    动作数量: 2
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium 未安装")

    if isinstance(space, spaces.Discrete):
        return SpaceInfo(
            space_type=SpaceType.DISCRETE,
            n=int(space.n),
            dtype=space.dtype,
            flat_dim=int(space.n)
        )

    elif isinstance(space, spaces.Box):
        is_bounded = (
            np.all(np.isfinite(space.low)) and
            np.all(np.isfinite(space.high))
        )
        return SpaceInfo(
            space_type=SpaceType.BOX,
            shape=space.shape,
            low=space.low,
            high=space.high,
            dtype=space.dtype,
            is_bounded=is_bounded,
            flat_dim=int(np.prod(space.shape))
        )

    elif isinstance(space, spaces.MultiDiscrete):
        return SpaceInfo(
            space_type=SpaceType.MULTI_DISCRETE,
            shape=(len(space.nvec),),
            dtype=space.dtype,
            flat_dim=int(np.sum(space.nvec))
        )

    elif isinstance(space, spaces.MultiBinary):
        return SpaceInfo(
            space_type=SpaceType.MULTI_BINARY,
            shape=(space.n,) if isinstance(space.n, int) else space.n,
            dtype=space.dtype,
            flat_dim=int(np.prod(space.n) if hasattr(space.n, '__len__') else space.n)
        )

    elif isinstance(space, spaces.Tuple):
        total_dim = sum(
            analyze_space(s).flat_dim for s in space.spaces
        )
        return SpaceInfo(
            space_type=SpaceType.TUPLE,
            flat_dim=total_dim
        )

    elif isinstance(space, spaces.Dict):
        total_dim = sum(
            analyze_space(s).flat_dim for s in space.spaces.values()
        )
        return SpaceInfo(
            space_type=SpaceType.DICT,
            flat_dim=total_dim
        )

    else:
        raise ValueError(f"不支持的空间类型: {type(space)}")


def get_action_dim(space: spaces.Space) -> int:
    """
    获取动作空间维度

    对于离散空间返回动作数量，对于连续空间返回动作向量维度。

    Parameters
    ----------
    space : spaces.Space
        动作空间

    Returns
    -------
    int
        动作维度
    """
    info = analyze_space(space)

    if info.space_type == SpaceType.DISCRETE:
        return info.n
    elif info.space_type == SpaceType.BOX:
        return info.flat_dim
    else:
        return info.flat_dim


def get_obs_dim(space: spaces.Space) -> int:
    """
    获取观测空间维度

    返回观测向量的展平维度，用于确定网络输入层大小。

    Parameters
    ----------
    space : spaces.Space
        观测空间

    Returns
    -------
    int
        观测维度
    """
    return analyze_space(space).flat_dim


# ============================================================================
#                           环境信息收集
# ============================================================================

@dataclass
class EnvironmentSpec:
    """
    环境规格完整描述

    收集并组织环境的所有关键信息，便于算法适配。

    Attributes:
        env_id: 环境标识符
        obs_info: 观测空间信息
        action_info: 动作空间信息
        reward_range: 奖励值范围
        max_episode_steps: 最大回合步数
        is_discrete_action: 是否为离散动作空间
        is_image_obs: 是否为图像观测
        metadata: 环境元数据
    """
    env_id: str
    obs_info: SpaceInfo
    action_info: SpaceInfo
    reward_range: Tuple[float, float] = (-float('inf'), float('inf'))
    max_episode_steps: Optional[int] = None
    is_discrete_action: bool = True
    is_image_obs: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """后初始化处理"""
        self.is_discrete_action = (
            self.action_info.space_type == SpaceType.DISCRETE
        )
        if self.obs_info.shape is not None:
            self.is_image_obs = len(self.obs_info.shape) >= 3


def get_env_spec(env: gym.Env) -> EnvironmentSpec:
    """
    提取环境完整规格

    Parameters
    ----------
    env : gym.Env
        Gymnasium 环境实例

    Returns
    -------
    EnvironmentSpec
        环境规格数据对象

    Examples
    --------
    >>> env = gym.make("CartPole-v1")
    >>> spec = get_env_spec(env)
    >>> print(f"离散动作: {spec.is_discrete_action}")
    离散动作: True
    >>> print(f"观测维度: {spec.obs_info.flat_dim}")
    观测维度: 4
    """
    env_id = env.spec.id if env.spec else "unknown"
    max_steps = None

    if env.spec is not None:
        max_steps = env.spec.max_episode_steps

    reward_range = getattr(env, 'reward_range', (-float('inf'), float('inf')))
    metadata = getattr(env, 'metadata', {})

    return EnvironmentSpec(
        env_id=env_id,
        obs_info=analyze_space(env.observation_space),
        action_info=analyze_space(env.action_space),
        reward_range=reward_range,
        max_episode_steps=max_steps,
        metadata=metadata
    )


# ============================================================================
#                           环境交互工具
# ============================================================================

@dataclass
class StepResult:
    """
    单步交互结果

    封装 env.step() 的返回值，提供更清晰的访问接口。

    Attributes:
        observation: 新状态观测
        reward: 即时奖励
        terminated: 是否因任务完成而终止
        truncated: 是否因超时而截断
        info: 附加信息字典
    """
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

    @property
    def done(self) -> bool:
        """回合是否结束 (terminated or truncated)"""
        return self.terminated or self.truncated


@dataclass
class EpisodeResult:
    """
    完整回合结果

    记录一个完整回合的所有交互数据和统计信息。

    Attributes:
        observations: 状态序列
        actions: 动作序列
        rewards: 奖励序列
        total_reward: 累积奖励
        length: 回合长度
        terminated: 是否正常终止
        truncated: 是否被截断
        info: 最后一步的附加信息
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0
    terminated: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def append(
        self,
        obs: np.ndarray,
        action: Any,
        reward: float
    ) -> None:
        """添加一步数据"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_reward += reward
        self.length += 1


def run_episode(
    env: gym.Env,
    policy: Callable[[np.ndarray], Any],
    max_steps: Optional[int] = None,
    render: bool = False,
    seed: Optional[int] = None
) -> EpisodeResult:
    """
    执行一个完整回合

    使用给定策略在环境中执行交互，直到回合结束。

    Parameters
    ----------
    env : gym.Env
        Gymnasium 环境
    policy : Callable
        策略函数，接收观测返回动作
    max_steps : int, optional
        最大步数限制
    render : bool
        是否渲染
    seed : int, optional
        随机种子

    Returns
    -------
    EpisodeResult
        回合结果数据

    Examples
    --------
    >>> env = gym.make("CartPole-v1")
    >>> result = run_episode(env, lambda obs: env.action_space.sample())
    >>> print(f"回合奖励: {result.total_reward}, 长度: {result.length}")
    """
    result = EpisodeResult()

    obs, info = env.reset(seed=seed)
    result.observations.append(obs)

    steps = 0
    max_steps = max_steps or float('inf')

    while steps < max_steps:
        if render:
            env.render()

        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        result.append(obs, action, reward)

        if terminated or truncated:
            result.terminated = terminated
            result.truncated = truncated
            result.info = info
            break

        steps += 1

    return result


def evaluate_policy(
    env: gym.Env,
    policy: Callable[[np.ndarray], Any],
    n_episodes: int = 10,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    评估策略性能

    在多个回合中运行策略，统计性能指标。

    Parameters
    ----------
    env : gym.Env
        评估环境
    policy : Callable
        待评估策略
    n_episodes : int
        评估回合数
    seed : int, optional
        基础随机种子
    verbose : bool
        是否打印进度

    Returns
    -------
    dict
        包含 mean_reward, std_reward, mean_length 等统计量

    Examples
    --------
    >>> env = gym.make("CartPole-v1")
    >>> stats = evaluate_policy(env, lambda obs: 1 if obs[2] > 0 else 0, n_episodes=10)
    >>> print(f"平均奖励: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    """
    rewards = []
    lengths = []

    for i in range(n_episodes):
        episode_seed = seed + i if seed is not None else None
        result = run_episode(env, policy, seed=episode_seed)
        rewards.append(result.total_reward)
        lengths.append(result.length)

        if verbose:
            print(f"回合 {i+1}/{n_episodes}: 奖励={result.total_reward:.2f}, 长度={result.length}")

    stats = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'n_episodes': n_episodes
    }

    if verbose:
        print(f"\n评估结果 ({n_episodes} 回合):")
        print(f"  奖励: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  长度: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")

    return stats


# ============================================================================
#                           环境包装器
# ============================================================================

class ObservationNormalizer:
    """
    在线观测归一化器

    使用 Welford 算法在线计算均值和方差，实现观测的标准化。
    这对于许多 RL 算法的稳定训练至关重要。

    数学原理:
        x_normalized = (x - μ) / (σ + ε)

    其中 μ 和 σ 通过在线更新估计:
        μ_n = μ_{n-1} + (x_n - μ_{n-1}) / n
        M_n = M_{n-1} + (x_n - μ_{n-1}) * (x_n - μ_n)
        σ² = M_n / n

    Attributes:
        mean: 当前均值估计
        var: 当前方差估计
        count: 已处理样本数
        epsilon: 防止除零的小常数
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        epsilon: float = 1e-8,
        clip: float = 10.0
    ):
        """
        初始化归一化器

        Parameters
        ----------
        shape : tuple
            观测形状
        epsilon : float
            数值稳定性常数
        clip : float
            归一化后的裁剪范围
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.epsilon = epsilon
        self.clip = clip

    def update(self, x: np.ndarray) -> None:
        """
        使用新样本更新统计量 (Welford 算法)

        Parameters
        ----------
        x : np.ndarray
            新观测值 (可以是单个样本或批量)
        """
        if x.ndim == len(self.mean.shape):
            # 单个样本
            batch = np.expand_dims(x, 0)
        else:
            batch = x

        batch_count = batch.shape[0]
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)

        # Parallel Welford 算法
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count

        # 更新方差
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.var = M2 / total_count if total_count > 0 else np.ones_like(self.var)
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        归一化观测值

        Parameters
        ----------
        x : np.ndarray
            原始观测

        Returns
        -------
        np.ndarray
            归一化后的观测
        """
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """反归一化"""
        return x * np.sqrt(self.var + self.epsilon) + self.mean


class RewardScaler:
    """
    奖励缩放器

    使用指数移动平均估计奖励的标准差，用于缩放奖励。
    这有助于稳定 value function 的学习。

    实现基于 OpenAI Baselines 的方法:
        r_scaled = r / σ

    其中 σ 通过对累积折扣奖励的方差进行估计。

    Attributes:
        gamma: 折扣因子
        epsilon: 数值稳定性常数
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        """
        初始化奖励缩放器

        Parameters
        ----------
        gamma : float
            折扣因子
        epsilon : float
            数值稳定性常数
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms = None  # Running mean/std of returns
        self.ret = 0.0  # Discounted return accumulator

        # 在线统计量
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, reward: float, done: bool) -> None:
        """
        更新统计量

        Parameters
        ----------
        reward : float
            当前奖励
        done : bool
            回合是否结束
        """
        self.ret = reward + self.gamma * self.ret * (1 - done)

        # Welford 更新
        self.count += 1
        delta = self.ret - self.mean
        self.mean += delta / self.count
        delta2 = self.ret - self.mean
        self.var += delta * delta2

        if done:
            self.ret = 0.0

    def scale(self, reward: float) -> float:
        """
        缩放奖励

        Parameters
        ----------
        reward : float
            原始奖励

        Returns
        -------
        float
            缩放后的奖励
        """
        std = np.sqrt(self.var / max(1, self.count) + self.epsilon)
        return reward / std


# ============================================================================
#                           常见环境示例
# ============================================================================

def demo_cartpole():
    """
    CartPole 环境演示

    CartPole-v1 是经典控制任务:
    - 状态: [位置, 速度, 角度, 角速度]
    - 动作: 0 (向左推) 或 1 (向右推)
    - 目标: 保持杆子直立尽可能长的时间
    - 最大步数: 500
    - 成功阈值: 平均奖励 >= 475
    """
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，无法运行演示")
        return

    print("=" * 60)
    print("CartPole-v1 环境演示")
    print("=" * 60)

    # 创建环境
    env = gym.make("CartPole-v1")

    # 获取环境规格
    spec = get_env_spec(env)
    print(f"\n环境ID: {spec.env_id}")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"奖励范围: {spec.reward_range}")
    print(f"最大步数: {spec.max_episode_steps}")

    # 简单策略: 根据杆子角度决定推力方向
    def simple_policy(obs: np.ndarray) -> int:
        """基于规则的简单策略"""
        pole_angle = obs[2]
        return 1 if pole_angle > 0 else 0

    # 随机策略
    def random_policy(obs: np.ndarray) -> int:
        return env.action_space.sample()

    # 评估两种策略
    print("\n评估随机策略:")
    random_stats = evaluate_policy(env, random_policy, n_episodes=10, seed=42)

    print("\n评估简单规则策略:")
    simple_stats = evaluate_policy(env, simple_policy, n_episodes=10, seed=42)

    # 对比
    print("\n策略对比:")
    print(f"  随机策略: {random_stats['mean_reward']:.1f} ± {random_stats['std_reward']:.1f}")
    print(f"  规则策略: {simple_stats['mean_reward']:.1f} ± {simple_stats['std_reward']:.1f}")

    env.close()

    return spec


def demo_mountain_car():
    """
    MountainCar 环境演示

    MountainCar-v0 是经典的探索困难任务:
    - 状态: [位置, 速度]
    - 动作: 0 (向左加速), 1 (不动), 2 (向右加速)
    - 目标: 到达山顶 (position >= 0.5)
    - 奖励: 每步 -1，到达目标 0
    - 挑战: 随机策略几乎无法成功，需要学会利用动量
    """
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，无法运行演示")
        return

    print("=" * 60)
    print("MountainCar-v0 环境演示")
    print("=" * 60)

    env = gym.make("MountainCar-v0")
    spec = get_env_spec(env)

    print(f"\n环境ID: {spec.env_id}")
    print(f"观测空间: {env.observation_space}")
    print(f"  位置范围: [{env.observation_space.low[0]:.2f}, {env.observation_space.high[0]:.2f}]")
    print(f"  速度范围: [{env.observation_space.low[1]:.3f}, {env.observation_space.high[1]:.3f}]")
    print(f"动作空间: {env.action_space} (左/不动/右)")
    print(f"最大步数: {spec.max_episode_steps}")

    # 能量策略: 利用动量摆动
    def momentum_policy(obs: np.ndarray) -> int:
        """基于动量的策略"""
        position, velocity = obs
        # 如果向右移动则加速向右，反之亦然
        if velocity > 0:
            return 2  # 向右
        else:
            return 0  # 向左

    print("\n评估动量策略:")
    stats = evaluate_policy(env, momentum_policy, n_episodes=5, seed=42)

    env.close()

    return spec


def demo_continuous_action():
    """
    连续动作空间环境演示

    Pendulum-v1 是连续控制任务:
    - 状态: [cos(θ), sin(θ), θ̇] 其中 θ 是摆的角度
    - 动作: [-2, 2] 连续扭矩
    - 目标: 将摆从下垂位置摆到直立位置
    """
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，无法运行演示")
        return

    print("=" * 60)
    print("Pendulum-v1 (连续动作空间) 环境演示")
    print("=" * 60)

    env = gym.make("Pendulum-v1")
    spec = get_env_spec(env)

    print(f"\n环境ID: {spec.env_id}")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"  动作范围: [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")
    print(f"是否连续动作: {not spec.is_discrete_action}")

    # PD 控制器策略
    def pd_controller(obs: np.ndarray) -> np.ndarray:
        """简单 PD 控制器"""
        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)

        # PD 控制: u = -Kp * θ - Kd * θ̇
        Kp, Kd = 10.0, 2.0
        torque = -Kp * theta - Kd * theta_dot

        # 裁剪到有效范围
        return np.clip([torque], -2.0, 2.0)

    print("\n评估 PD 控制策略:")
    stats = evaluate_policy(env, pd_controller, n_episodes=5, seed=42)

    env.close()

    return spec


def demo_wrappers():
    """
    环境包装器演示

    展示 Gymnasium 内置包装器的使用方法。
    """
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，无法运行演示")
        return

    print("=" * 60)
    print("环境包装器 (Wrappers) 演示")
    print("=" * 60)

    # 基础环境
    base_env = gym.make("CartPole-v1")

    # 应用包装器
    # 1. 记录回合统计
    env = RecordEpisodeStatistics(base_env)

    print("\n1. RecordEpisodeStatistics 包装器:")
    print("   自动记录每个回合的奖励和长度")

    obs, info = env.reset(seed=42)
    total_reward = 0

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            if 'episode' in info:
                print(f"   回合统计: 奖励={info['episode']['r']:.1f}, 长度={info['episode']['l']}")
            break

    env.close()

    # 2. 时间限制包装器
    print("\n2. TimeLimit 包装器:")
    print("   限制回合最大步数")

    env = gym.make("CartPole-v1")
    env = TimeLimit(env, max_episode_steps=50)

    obs, _ = env.reset()
    steps = 0

    while True:
        obs, _, terminated, truncated, _ = env.step(1)
        steps += 1
        if terminated or truncated:
            print(f"   回合在 {steps} 步后结束 (truncated={truncated})")
            break

    env.close()

    print("\n其他常用包装器:")
    print("   - NormalizeObservation: 自动归一化观测")
    print("   - NormalizeReward: 自动归一化奖励")
    print("   - ClipAction: 裁剪连续动作到有效范围")
    print("   - FrameStack: 堆叠多帧观测 (用于部分可观测环境)")


def demo_vectorized_env():
    """
    向量化环境演示

    展示如何使用向量化环境并行采样。
    """
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，无法运行演示")
        return

    print("=" * 60)
    print("向量化环境 (Vectorized Environments) 演示")
    print("=" * 60)

    from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

    # 创建多个环境实例
    n_envs = 4

    def make_env(seed: int):
        def thunk():
            env = gym.make("CartPole-v1")
            return env
        return thunk

    # 同步向量化环境
    print(f"\n创建 {n_envs} 个并行环境 (同步模式)...")
    vec_env = SyncVectorEnv([make_env(i) for i in range(n_envs)])

    print(f"观测空间: {vec_env.observation_space}")
    print(f"动作空间: {vec_env.action_space}")
    print(f"环境数量: {vec_env.num_envs}")

    # 并行采样
    obs, info = vec_env.reset()
    print(f"\n重置后观测形状: {obs.shape}")  # (n_envs, obs_dim)

    # 执行一步
    actions = vec_env.action_space.sample()  # 自动采样 n_envs 个动作
    print(f"采样动作: {actions}")

    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    print(f"奖励: {rewards}")
    print(f"终止状态: {terminateds}")

    vec_env.close()

    print("\n向量化环境优势:")
    print("   - 批量采样，提高数据收集效率")
    print("   - 更好地利用 CPU 多核")
    print("   - 与批量神经网络推理配合良好")


# ============================================================================
#                           可视化工具
# ============================================================================

def plot_learning_curve(
    rewards: List[float],
    window: int = 100,
    title: str = "Learning Curve",
    save_path: Optional[str] = None
) -> None:
    """
    绘制学习曲线

    Parameters
    ----------
    rewards : list
        每回合奖励列表
    window : int
        滑动平均窗口大小
    title : str
        图表标题
    save_path : str, optional
        保存路径
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装，无法绘图")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # 原始曲线
    ax.plot(rewards, alpha=0.3, color='blue', label='Raw')

    # 滑动平均
    if len(rewards) >= window:
        smoothed = np.convolve(
            rewards,
            np.ones(window) / window,
            mode='valid'
        )
        ax.plot(
            range(window - 1, len(rewards)),
            smoothed,
            color='blue',
            linewidth=2,
            label=f'{window}-episode average'
        )

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")

    plt.close()


def visualize_observations(
    observations: List[np.ndarray],
    labels: Optional[List[str]] = None,
    title: str = "Observation Distribution"
) -> None:
    """
    可视化观测分布

    Parameters
    ----------
    observations : list
        观测序列
    labels : list, optional
        各维度标签
    title : str
        图表标题
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装，无法绘图")
        return

    obs_array = np.array(observations)
    n_dims = obs_array.shape[1]

    if labels is None:
        labels = [f'dim_{i}' for i in range(n_dims)]

    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4))

    if n_dims == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.hist(obs_array[:, i], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.close()


# ============================================================================
#                           单元测试
# ============================================================================

def run_tests() -> bool:
    """
    运行单元测试

    验证所有核心功能的正确性。

    Returns
    -------
    bool
        所有测试是否通过
    """
    print("\n" + "=" * 60)
    print("运行单元测试")
    print("=" * 60)

    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    all_passed = True

    # 测试 1: 空间分析
    print("\n[测试 1] 空间分析...")
    try:
        env = gym.make("CartPole-v1")

        obs_info = analyze_space(env.observation_space)
        assert obs_info.space_type == SpaceType.BOX
        assert obs_info.shape == (4,)
        assert obs_info.flat_dim == 4

        act_info = analyze_space(env.action_space)
        assert act_info.space_type == SpaceType.DISCRETE
        assert act_info.n == 2

        env.close()
        print("  [通过] 空间分析正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: 环境规格提取
    print("\n[测试 2] 环境规格提取...")
    try:
        env = gym.make("CartPole-v1")
        spec = get_env_spec(env)

        assert spec.env_id == "CartPole-v1"
        assert spec.is_discrete_action is True
        assert spec.max_episode_steps == 500

        env.close()
        print("  [通过] 环境规格提取正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: 回合执行
    print("\n[测试 3] 回合执行...")
    try:
        env = gym.make("CartPole-v1")

        result = run_episode(
            env,
            lambda obs: env.action_space.sample(),
            seed=42
        )

        assert result.length > 0
        assert len(result.observations) == result.length + 1
        assert len(result.actions) == result.length
        assert len(result.rewards) == result.length

        env.close()
        print("  [通过] 回合执行正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: 策略评估
    print("\n[测试 4] 策略评估...")
    try:
        env = gym.make("CartPole-v1")

        stats = evaluate_policy(
            env,
            lambda obs: 1 if obs[2] > 0 else 0,
            n_episodes=3,
            seed=42,
            verbose=False
        )

        assert 'mean_reward' in stats
        assert 'std_reward' in stats
        assert stats['n_episodes'] == 3

        env.close()
        print("  [通过] 策略评估正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 5: 观测归一化器
    print("\n[测试 5] 观测归一化器...")
    try:
        normalizer = ObservationNormalizer(shape=(4,))

        # 更新统计量
        for _ in range(100):
            obs = np.random.randn(4)
            normalizer.update(obs)

        # 检查均值接近 0
        test_obs = np.random.randn(4)
        normalized = normalizer.normalize(test_obs)

        assert normalized.shape == (4,)
        assert np.all(np.abs(normalized) <= 10)  # 检查裁剪

        print("  [通过] 观测归一化器正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 6: 奖励缩放器
    print("\n[测试 6] 奖励缩放器...")
    try:
        scaler = RewardScaler(gamma=0.99)

        for _ in range(100):
            scaler.update(np.random.randn(), done=False)
        scaler.update(np.random.randn(), done=True)

        scaled = scaler.scale(1.0)
        assert np.isfinite(scaled)

        print("  [通过] 奖励缩放器正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 7: 连续动作空间
    print("\n[测试 7] 连续动作空间...")
    try:
        env = gym.make("Pendulum-v1")
        spec = get_env_spec(env)

        assert spec.is_discrete_action is False
        assert spec.action_info.space_type == SpaceType.BOX

        # 测试动作采样
        action = env.action_space.sample()
        assert action.shape == (1,)

        env.close()
        print("  [通过] 连续动作空间正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 8: 向量化环境
    print("\n[测试 8] 向量化环境...")
    try:
        from gymnasium.vector import SyncVectorEnv

        n_envs = 2
        vec_env = SyncVectorEnv([
            lambda: gym.make("CartPole-v1") for _ in range(n_envs)
        ])

        obs, _ = vec_env.reset()
        assert obs.shape == (n_envs, 4)

        actions = vec_env.action_space.sample()
        obs, rewards, _, _, _ = vec_env.step(actions)

        assert obs.shape == (n_envs, 4)
        assert rewards.shape == (n_envs,)

        vec_env.close()
        print("  [通过] 向量化环境正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 9: 工具函数
    print("\n[测试 9] 工具函数...")
    try:
        env = gym.make("CartPole-v1")

        obs_dim = get_obs_dim(env.observation_space)
        act_dim = get_action_dim(env.action_space)

        assert obs_dim == 4
        assert act_dim == 2

        env.close()
        print("  [通过] 工具函数正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 10: EpisodeResult 数据类
    print("\n[测试 10] EpisodeResult 数据类...")
    try:
        result = EpisodeResult()

        for i in range(10):
            result.append(
                obs=np.zeros(4),
                action=i % 2,
                reward=1.0
            )

        assert result.length == 10
        assert result.total_reward == 10.0
        assert len(result.observations) == 10

        print("  [通过] EpisodeResult 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败，请检查。")
    print("=" * 60)

    return all_passed


# ============================================================================
#                           主程序
# ============================================================================

def main():
    """主程序入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenAI Gymnasium 强化学习环境接口指南",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python gym_introduction.py --demo cartpole
    python gym_introduction.py --demo all
    python gym_introduction.py --test
        """
    )

    parser.add_argument(
        '--demo',
        type=str,
        choices=['cartpole', 'mountaincar', 'pendulum', 'wrappers', 'vectorized', 'all'],
        default=None,
        help='运行指定演示'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='运行单元测试'
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if args.demo is None:
        # 默认运行所有演示
        args.demo = 'all'

    demos = {
        'cartpole': demo_cartpole,
        'mountaincar': demo_mountain_car,
        'pendulum': demo_continuous_action,
        'wrappers': demo_wrappers,
        'vectorized': demo_vectorized_env,
    }

    if args.demo == 'all':
        for name, demo_func in demos.items():
            print(f"\n\n{'#' * 70}")
            print(f"# {name.upper()}")
            print(f"{'#' * 70}")
            demo_func()
    else:
        demos[args.demo]()


if __name__ == "__main__":
    main()
