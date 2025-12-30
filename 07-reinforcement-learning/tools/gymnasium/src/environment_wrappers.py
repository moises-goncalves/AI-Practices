#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium 环境包装器与工具集

================================================================================
核心思想 (Core Idea)
================================================================================
环境包装器 (Wrappers) 是装饰器模式在强化学习环境中的应用。通过包装器，可以在
不修改原始环境代码的情况下，对观测、动作、奖励进行预处理和后处理。这种设计
实现了关注点分离，使得数据预处理逻辑与环境逻辑解耦。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
包装器实现了 MDP 的变换:

**观测归一化**:
    $$\\tilde{s} = \\frac{s - \\mu}{\\sigma + \\epsilon}$$

    其中 μ 和 σ 通过在线 Welford 算法估计:
    $$\\mu_n = \\mu_{n-1} + \\frac{x_n - \\mu_{n-1}}{n}$$
    $$M_n = M_{n-1} + (x_n - \\mu_{n-1})(x_n - \\mu_n)$$
    $$\\sigma^2 = \\frac{M_n}{n}$$

**奖励缩放**:
    $$\\tilde{r} = \\frac{r}{\\sigma_G}$$

    其中 σ_G 是累积折扣回报的标准差估计。

**帧堆叠** (Frame Stacking):
    将连续 k 帧堆叠为单个观测:
    $$s_t^{stacked} = [s_{t-k+1}, s_{t-k+2}, ..., s_t]$$

    这为部分可观测环境提供时序信息。

================================================================================
问题背景 (Problem Statement)
================================================================================
在实际应用中，原始环境输出往往需要额外处理:

1. **数值稳定性**: 原始观测和奖励的尺度可能导致训练不稳定
2. **部分可观测性**: 单帧观测可能不包含足够的状态信息
3. **动作约束**: 策略输出可能超出有效范围
4. **调试需求**: 需要记录详细的训练统计信息
5. **性能优化**: 需要并行化采样以提高效率

================================================================================
包装器对比 (Comparison)
================================================================================
| 包装器类型        | 作用                | 常用场景            |
|------------------|---------------------|---------------------|
| NormalizeObs     | 归一化观测          | 连续状态空间        |
| NormalizeReward  | 归一化奖励          | 奖励尺度差异大      |
| ClipAction       | 裁剪动作            | 连续动作空间        |
| FrameStack       | 堆叠帧              | 图像/部分可观测     |
| TimeLimit        | 限制步数            | 防止无限循环        |
| RecordVideo      | 录制视频            | 可视化/调试         |
| Statistics       | 统计信息            | 训练监控            |

================================================================================
复杂度 (Complexity)
================================================================================
- 大多数包装器: O(n) 空间和时间，n 为观测/动作维度
- FrameStack: O(k×n) 空间，k 为堆叠帧数
- 统计包装器: O(1) 额外开销

================================================================================
算法总结 (Summary)
================================================================================
包装器是构建健壮 RL 系统的关键组件:
1. 通过组合简单包装器实现复杂预处理流水线
2. 保持环境接口不变，算法代码无需修改
3. 便于实验对比不同预处理策略的效果
4. 提供训练监控和调试支持

Author: Ziming Ding
Date: 2024
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Deque, Dict, Generic, List, Optional,
    Sequence, Tuple, TypeVar, Union
)

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.core import Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None
    Wrapper = object
    ObservationWrapper = object
    ActionWrapper = object
    RewardWrapper = object


# ============================================================================
#                           类型定义
# ============================================================================

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


# ============================================================================
#                           运行时统计工具
# ============================================================================

@dataclass
class RunningStatistics:
    """
    在线统计量计算器 (Welford 算法)

    使用 Welford 算法在线计算均值和方差，数值稳定且内存高效。
    适用于流式数据处理场景。

    数学原理:
        增量更新公式:
        μ_n = μ_{n-1} + (x_n - μ_{n-1}) / n
        M_n = M_{n-1} + (x_n - μ_{n-1}) * (x_n - μ_n)
        σ² = M_n / n

    Attributes:
        shape: 数据形状
        mean: 当前均值估计
        var: 当前方差估计 (使用 M_n / n)
        count: 已处理样本数
    """
    shape: Tuple[int, ...]
    mean: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)
    count: int = field(default=0, init=False)
    _m2: np.ndarray = field(init=False)  # 用于计算方差

    def __post_init__(self):
        """初始化统计量"""
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self._m2 = np.zeros(self.shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """
        使用新样本更新统计量

        Parameters
        ----------
        x : np.ndarray
            新数据点，可以是单个样本或批量样本
        """
        # 处理批量数据
        if x.ndim == len(self.shape):
            batch = np.expand_dims(x, 0)
        else:
            batch = x

        batch_count = batch.shape[0]
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)

        # 合并批次统计量 (Chan et al., 1979)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        # 更新均值
        self.mean = self.mean + delta * batch_count / total_count

        # 更新方差 (使用平行算法)
        self._m2 = (
            self._m2 +
            batch_var * batch_count +
            delta**2 * self.count * batch_count / total_count
        )

        self.count = total_count

        # 从 M2 计算方差
        if self.count > 1:
            self.var = self._m2 / self.count

    def normalize(
        self,
        x: np.ndarray,
        epsilon: float = 1e-8,
        clip: float = 10.0
    ) -> np.ndarray:
        """
        标准化数据

        Parameters
        ----------
        x : np.ndarray
            待标准化数据
        epsilon : float
            数值稳定性常数
        clip : float
            裁剪范围

        Returns
        -------
        np.ndarray
            标准化后的数据
        """
        normalized = (x - self.mean) / np.sqrt(self.var + epsilon)
        return np.clip(normalized, -clip, clip)

    def denormalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """反标准化数据"""
        return x * np.sqrt(self.var + epsilon) + self.mean

    def reset(self) -> None:
        """重置统计量"""
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self._m2 = np.zeros(self.shape, dtype=np.float64)
        self.count = 0


@dataclass
class ExponentialMovingAverage:
    """
    指数移动平均计算器

    用于平滑时间序列数据，对近期数据赋予更高权重。

    数学原理:
        EMA_t = α * x_t + (1 - α) * EMA_{t-1}

    其中 α = 1 - decay 是平滑系数。

    Attributes:
        decay: 衰减系数，越大则历史权重越高
        value: 当前 EMA 值
        initialized: 是否已初始化
    """
    decay: float = 0.99
    value: float = field(default=0.0, init=False)
    initialized: bool = field(default=False, init=False)

    def update(self, x: float) -> float:
        """
        更新 EMA

        Parameters
        ----------
        x : float
            新数据点

        Returns
        -------
        float
            更新后的 EMA 值
        """
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
        return self.value

    def reset(self) -> None:
        """重置"""
        self.value = 0.0
        self.initialized = False


# ============================================================================
#                           观测包装器
# ============================================================================

class NormalizeObservationWrapper(ObservationWrapper):
    """
    观测归一化包装器

    在线估计观测的均值和方差，并将观测标准化到均值为 0、方差为 1 的分布。
    这对于使用神经网络的 RL 算法至关重要。

    注意事项:
        - 仅在训练时更新统计量
        - 评估时使用固定的统计量
        - 可以保存/加载统计量用于部署

    Parameters
    ----------
    env : gym.Env
        原始环境
    epsilon : float
        数值稳定性常数
    clip : float
        归一化后的裁剪范围

    Example
    -------
    >>> env = gym.make("Pendulum-v1")
    >>> env = NormalizeObservationWrapper(env)
    >>> obs, _ = env.reset()
    >>> # obs 现在是归一化的
    """

    def __init__(
        self,
        env: gym.Env,
        epsilon: float = 1e-8,
        clip: float = 10.0,
        update_stats: bool = True
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.epsilon = epsilon
        self.clip = clip
        self.update_stats = update_stats

        # 获取观测形状
        obs_shape = env.observation_space.shape
        self.running_stats = RunningStatistics(shape=obs_shape)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        处理观测

        Parameters
        ----------
        observation : np.ndarray
            原始观测

        Returns
        -------
        np.ndarray
            归一化后的观测
        """
        if self.update_stats:
            self.running_stats.update(observation)

        return self.running_stats.normalize(
            observation,
            epsilon=self.epsilon,
            clip=self.clip
        ).astype(np.float32)

    def set_training_mode(self, training: bool) -> None:
        """设置训练/评估模式"""
        self.update_stats = training

    def get_statistics(self) -> Dict[str, np.ndarray]:
        """获取当前统计量"""
        return {
            'mean': self.running_stats.mean.copy(),
            'var': self.running_stats.var.copy(),
            'count': self.running_stats.count
        }

    def set_statistics(self, stats: Dict[str, np.ndarray]) -> None:
        """设置统计量 (用于加载预训练模型)"""
        self.running_stats.mean = stats['mean'].copy()
        self.running_stats.var = stats['var'].copy()
        self.running_stats.count = stats['count']


class FrameStackWrapper(ObservationWrapper):
    """
    帧堆叠包装器

    将连续 k 帧观测堆叠为单个观测，为部分可观测环境提供时序信息。
    这对于需要从观测中推断速度等动态信息的场景非常有用。

    数学原理:
        堆叠观测: s_t^{stacked} = [s_{t-k+1}, s_{t-k+2}, ..., s_t]
        观测空间: R^n → R^{k×n}

    Parameters
    ----------
    env : gym.Env
        原始环境
    n_frames : int
        堆叠帧数
    stack_axis : int
        堆叠维度

    Example
    -------
    >>> env = gym.make("CartPole-v1")
    >>> env = FrameStackWrapper(env, n_frames=4)
    >>> obs, _ = env.reset()  # shape: (4, 4)
    """

    def __init__(
        self,
        env: gym.Env,
        n_frames: int = 4,
        stack_axis: int = 0
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.n_frames = n_frames
        self.stack_axis = stack_axis

        # 初始化帧缓冲区
        self.frames: Deque[np.ndarray] = deque(maxlen=n_frames)

        # 更新观测空间
        low = np.repeat(env.observation_space.low[np.newaxis, ...], n_frames, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], n_frames, axis=0)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境并初始化帧缓冲"""
        obs, info = self.env.reset(**kwargs)

        # 用初始帧填充缓冲区
        for _ in range(self.n_frames):
            self.frames.append(obs)

        return self._get_stacked_obs(), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """处理单帧观测"""
        self.frames.append(observation)
        return self._get_stacked_obs()

    def _get_stacked_obs(self) -> np.ndarray:
        """获取堆叠后的观测"""
        return np.stack(list(self.frames), axis=self.stack_axis)


class FlattenObservationWrapper(ObservationWrapper):
    """
    观测展平包装器

    将多维观测展平为一维向量，便于全连接网络处理。

    Parameters
    ----------
    env : gym.Env
        原始环境
    """

    def __init__(self, env: gym.Env):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)

        # 计算展平后的维度
        obs_shape = env.observation_space.shape
        flat_dim = int(np.prod(obs_shape))

        self.observation_space = spaces.Box(
            low=env.observation_space.low.flatten(),
            high=env.observation_space.high.flatten(),
            dtype=env.observation_space.dtype
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """展平观测"""
        return observation.flatten()


# ============================================================================
#                           动作包装器
# ============================================================================

class ClipActionWrapper(ActionWrapper):
    """
    动作裁剪包装器

    将动作裁剪到有效范围内，防止策略输出超出环境限制。
    对于连续动作空间必不可少。

    Parameters
    ----------
    env : gym.Env
        原始环境

    Example
    -------
    >>> env = gym.make("Pendulum-v1")
    >>> env = ClipActionWrapper(env)
    >>> # 动作会被自动裁剪到 [-2, 2]
    """

    def __init__(self, env: gym.Env):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)

        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("ClipActionWrapper 仅适用于 Box 动作空间")

    def action(self, action: np.ndarray) -> np.ndarray:
        """裁剪动作"""
        return np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )


class RescaleActionWrapper(ActionWrapper):
    """
    动作重缩放包装器

    将 [-1, 1] 范围的动作重缩放到环境的实际动作范围。
    这使得策略可以统一使用 tanh 输出层。

    数学原理:
        原始范围: [-1, 1]
        目标范围: [low, high]
        映射: a' = low + (a + 1) * (high - low) / 2

    Parameters
    ----------
    env : gym.Env
        原始环境
    """

    def __init__(self, env: gym.Env):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)

        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("RescaleActionWrapper 仅适用于 Box 动作空间")

        self.low = env.action_space.low
        self.high = env.action_space.high

        # 更新动作空间为 [-1, 1]
        self.action_space = spaces.Box(
            low=-np.ones_like(self.low),
            high=np.ones_like(self.high),
            dtype=env.action_space.dtype
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """将 [-1, 1] 重缩放到实际范围"""
        return self.low + (action + 1.0) * (self.high - self.low) / 2.0

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """将实际范围重缩放到 [-1, 1]"""
        return 2.0 * (action - self.low) / (self.high - self.low) - 1.0


class StickyActionWrapper(ActionWrapper):
    """
    粘性动作包装器

    以一定概率重复上一步的动作，增加环境的随机性。
    这是 Atari 环境中常用的技巧，用于测试策略的鲁棒性。

    Parameters
    ----------
    env : gym.Env
        原始环境
    sticky_prob : float
        重复上一动作的概率，范围 [0, 1]
    """

    def __init__(self, env: gym.Env, sticky_prob: float = 0.25):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.sticky_prob = sticky_prob
        self.last_action: Optional[Any] = None

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置并清除上一动作"""
        self.last_action = None
        return self.env.reset(**kwargs)

    def action(self, action: Any) -> Any:
        """可能返回上一动作"""
        if (
            self.last_action is not None and
            np.random.random() < self.sticky_prob
        ):
            return self.last_action
        self.last_action = action
        return action


# ============================================================================
#                           奖励包装器
# ============================================================================

class NormalizeRewardWrapper(RewardWrapper):
    """
    奖励归一化包装器

    通过估计累积折扣回报的标准差来缩放奖励。
    这有助于稳定 value function 的学习。

    数学原理:
        缩放奖励: r' = r / σ_G
        其中 σ_G 是累积折扣回报的标准差估计

    Parameters
    ----------
    env : gym.Env
        原始环境
    gamma : float
        折扣因子
    epsilon : float
        数值稳定性常数
    clip : float
        奖励裁剪范围

    Note
    ----
    这里使用简化实现，基于奖励本身的统计量而非完整回报。
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        clip: float = 10.0
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip = clip

        # 累积回报估计
        self.return_running = ExponentialMovingAverage(decay=gamma)
        self.running_stats = RunningStatistics(shape=(1,))
        self.returns: float = 0.0

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行步骤并归一化奖励"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 更新累积回报估计
        self.returns = reward + self.gamma * self.returns * (1 - terminated)
        self.running_stats.update(np.array([self.returns]))

        # 归一化奖励
        normalized_reward = self.reward(reward)

        # 回合结束时重置累积回报
        if terminated or truncated:
            self.returns = 0.0

        return obs, normalized_reward, terminated, truncated, info

    def reward(self, reward: float) -> float:
        """归一化奖励"""
        std = np.sqrt(self.running_stats.var[0] + self.epsilon)
        normalized = reward / max(std, self.epsilon)
        return float(np.clip(normalized, -self.clip, self.clip))


class ClipRewardWrapper(RewardWrapper):
    """
    奖励裁剪包装器

    将奖励裁剪到指定范围，常用于 Atari 游戏。

    Parameters
    ----------
    env : gym.Env
        原始环境
    min_reward : float
        最小奖励
    max_reward : float
        最大奖励
    """

    def __init__(
        self,
        env: gym.Env,
        min_reward: float = -1.0,
        max_reward: float = 1.0
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward: float) -> float:
        """裁剪奖励"""
        return float(np.clip(reward, self.min_reward, self.max_reward))


class SignRewardWrapper(RewardWrapper):
    """
    符号奖励包装器

    将奖励转换为其符号 (-1, 0, +1)。
    这是 DQN 论文中使用的技巧。

    Parameters
    ----------
    env : gym.Env
        原始环境
    """

    def reward(self, reward: float) -> float:
        """返回奖励符号"""
        return float(np.sign(reward))


# ============================================================================
#                           通用包装器
# ============================================================================

class TimeLimitWrapper(Wrapper):
    """
    时间限制包装器

    在指定步数后强制终止回合（truncated=True）。

    Parameters
    ----------
    env : gym.Env
        原始环境
    max_steps : int
        最大步数
    """

    def __init__(self, env: gym.Env, max_steps: int):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置计数器"""
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行步骤并检查时间限制"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1

        if self.current_step >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


class EpisodeStatisticsWrapper(Wrapper):
    """
    回合统计包装器

    记录每个回合的奖励、长度等统计信息。

    Parameters
    ----------
    env : gym.Env
        原始环境
    deque_size : int
        保留的历史记录数量
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.deque_size = deque_size

        # 当前回合统计
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_start_time = time.time()

        # 历史记录
        self.reward_history: Deque[float] = deque(maxlen=deque_size)
        self.length_history: Deque[int] = deque(maxlen=deque_size)
        self.time_history: Deque[float] = deque(maxlen=deque_size)

        # 总计
        self.total_steps = 0
        self.total_episodes = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置回合统计"""
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_start_time = time.time()
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """记录统计信息"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_reward += reward
        self.episode_length += 1
        self.total_steps += 1

        if terminated or truncated:
            # 记录完成的回合
            episode_time = time.time() - self.episode_start_time

            self.reward_history.append(self.episode_reward)
            self.length_history.append(self.episode_length)
            self.time_history.append(episode_time)
            self.total_episodes += 1

            # 添加到 info
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length,
                't': episode_time
            }

        return obs, reward, terminated, truncated, info

    def get_statistics(self) -> Dict[str, float]:
        """获取当前统计摘要"""
        if not self.reward_history:
            return {}

        return {
            'mean_reward': np.mean(self.reward_history),
            'std_reward': np.std(self.reward_history),
            'min_reward': np.min(self.reward_history),
            'max_reward': np.max(self.reward_history),
            'mean_length': np.mean(self.length_history),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes
        }


class ActionRepeatWrapper(Wrapper):
    """
    动作重复包装器

    重复执行同一动作多次，累积奖励。
    用于减少决策频率，常见于连续控制任务。

    Parameters
    ----------
    env : gym.Env
        原始环境
    n_repeats : int
        动作重复次数
    """

    def __init__(self, env: gym.Env, n_repeats: int = 4):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium 未安装")

        super().__init__(env)
        self.n_repeats = n_repeats

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """重复执行动作"""
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.n_repeats):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


# ============================================================================
#                           包装器工厂
# ============================================================================

def make_wrapped_env(
    env_id: str,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    clip_action: bool = True,
    frame_stack: int = 0,
    time_limit: Optional[int] = None,
    record_stats: bool = True,
    **kwargs
) -> gym.Env:
    """
    创建带包装器的环境

    便捷函数，一次性应用常用包装器。

    Parameters
    ----------
    env_id : str
        环境 ID
    normalize_obs : bool
        是否归一化观测
    normalize_reward : bool
        是否归一化奖励
    clip_action : bool
        是否裁剪动作 (仅连续动作空间)
    frame_stack : int
        帧堆叠数量，0 表示不堆叠
    time_limit : int, optional
        时间限制
    record_stats : bool
        是否记录统计信息
    **kwargs
        传递给 gym.make 的参数

    Returns
    -------
    gym.Env
        包装后的环境

    Example
    -------
    >>> env = make_wrapped_env(
    ...     "Pendulum-v1",
    ...     normalize_obs=True,
    ...     normalize_reward=True
    ... )
    """
    if not HAS_GYMNASIUM:
        raise ImportError("gymnasium 未安装")

    env = gym.make(env_id, **kwargs)

    # 统计包装器 (最内层)
    if record_stats:
        env = EpisodeStatisticsWrapper(env)

    # 时间限制
    if time_limit is not None:
        env = TimeLimitWrapper(env, time_limit)

    # 帧堆叠
    if frame_stack > 0:
        env = FrameStackWrapper(env, n_frames=frame_stack)

    # 观测归一化
    if normalize_obs and isinstance(env.observation_space, spaces.Box):
        env = NormalizeObservationWrapper(env)

    # 动作裁剪 (仅连续动作)
    if clip_action and isinstance(env.action_space, spaces.Box):
        env = ClipActionWrapper(env)

    # 奖励归一化
    if normalize_reward:
        env = NormalizeRewardWrapper(env)

    return env


# ============================================================================
#                           单元测试
# ============================================================================

def run_tests() -> bool:
    """运行单元测试"""
    print("\n" + "=" * 60)
    print("运行包装器单元测试")
    print("=" * 60)

    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    all_passed = True

    # 测试 1: RunningStatistics
    print("\n[测试 1] RunningStatistics...")
    try:
        stats = RunningStatistics(shape=(4,))

        # 更新多个样本
        for _ in range(100):
            stats.update(np.random.randn(4))

        # 检查均值接近 0
        assert np.allclose(stats.mean, 0, atol=0.3), f"均值偏差过大: {stats.mean}"

        # 检查归一化
        x = np.random.randn(4)
        normalized = stats.normalize(x)
        assert normalized.shape == (4,), f"形状错误: {normalized.shape}"

        print("  [通过] RunningStatistics 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: NormalizeObservationWrapper
    print("\n[测试 2] NormalizeObservationWrapper...")
    try:
        env = gym.make("CartPole-v1")
        env = NormalizeObservationWrapper(env)

        obs, _ = env.reset()
        assert obs.dtype == np.float32, f"dtype 错误: {obs.dtype}"

        for _ in range(50):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.all(np.abs(obs) <= 10), f"观测超出裁剪范围: {obs}"
            if terminated or truncated:
                obs, _ = env.reset()

        env.close()
        print("  [通过] NormalizeObservationWrapper 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: FrameStackWrapper
    print("\n[测试 3] FrameStackWrapper...")
    try:
        env = gym.make("CartPole-v1")
        env = FrameStackWrapper(env, n_frames=4)

        obs, _ = env.reset()
        assert obs.shape == (4, 4), f"堆叠观测形状错误: {obs.shape}"

        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (4, 4), f"步后形状错误: {obs.shape}"

        env.close()
        print("  [通过] FrameStackWrapper 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: ClipActionWrapper
    print("\n[测试 4] ClipActionWrapper...")
    try:
        env = gym.make("Pendulum-v1")
        env = ClipActionWrapper(env)

        # 测试超出范围的动作
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(np.array([100.0]))  # 应该被裁剪
        obs, _, _, _, _ = env.step(np.array([-100.0]))

        env.close()
        print("  [通过] ClipActionWrapper 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 5: RescaleActionWrapper
    print("\n[测试 5] RescaleActionWrapper...")
    try:
        env = gym.make("Pendulum-v1")
        original_low = env.action_space.low[0]
        original_high = env.action_space.high[0]

        env = RescaleActionWrapper(env)

        # 检查动作空间更新
        assert np.allclose(env.action_space.low, -1), "动作空间下界错误"
        assert np.allclose(env.action_space.high, 1), "动作空间上界错误"

        # 测试动作转换
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(np.array([0.0]))  # 应该映射到中点

        env.close()
        print("  [通过] RescaleActionWrapper 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 6: NormalizeRewardWrapper
    print("\n[测试 6] NormalizeRewardWrapper...")
    try:
        env = gym.make("CartPole-v1")
        env = NormalizeRewardWrapper(env)

        obs, _ = env.reset()
        for _ in range(100):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), f"奖励无效: {reward}"
            if terminated or truncated:
                obs, _ = env.reset()

        env.close()
        print("  [通过] NormalizeRewardWrapper 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 7: EpisodeStatisticsWrapper
    print("\n[测试 7] EpisodeStatisticsWrapper...")
    try:
        env = gym.make("CartPole-v1")
        env = EpisodeStatisticsWrapper(env)

        obs, _ = env.reset()

        # 运行几个回合
        for _ in range(500):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                assert 'episode' in info, "缺少回合统计"
                obs, _ = env.reset()

        stats = env.get_statistics()
        assert 'mean_reward' in stats, "缺少 mean_reward"
        assert stats['total_episodes'] > 0, "回合数为 0"

        env.close()
        print("  [通过] EpisodeStatisticsWrapper 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 8: make_wrapped_env
    print("\n[测试 8] make_wrapped_env 工厂函数...")
    try:
        env = make_wrapped_env(
            "Pendulum-v1",
            normalize_obs=True,
            normalize_reward=True,
            clip_action=True
        )

        obs, _ = env.reset()
        assert obs.dtype == np.float32, f"观测 dtype 错误"

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, _, _, _ = env.step(action)

        env.close()
        print("  [通过] make_wrapped_env 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 9: ActionRepeatWrapper
    print("\n[测试 9] ActionRepeatWrapper...")
    try:
        env = gym.make("CartPole-v1")
        env = ActionRepeatWrapper(env, n_repeats=4)

        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(0)

        # 奖励应该是多步的累积
        assert reward >= 1, f"奖励应该累积: {reward}"

        env.close()
        print("  [通过] ActionRepeatWrapper 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 10: 包装器组合
    print("\n[测试 10] 包装器组合...")
    try:
        env = gym.make("CartPole-v1")
        env = EpisodeStatisticsWrapper(env)
        env = TimeLimitWrapper(env, max_steps=100)
        env = FrameStackWrapper(env, n_frames=4)
        env = NormalizeObservationWrapper(env)

        obs, _ = env.reset()
        assert obs.shape == (4, 4), f"组合后形状错误: {obs.shape}"

        for _ in range(200):
            obs, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()

        env.close()
        print("  [通过] 包装器组合正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败")
    print("=" * 60)

    return all_passed


# ============================================================================
#                           主程序
# ============================================================================

def main():
    """主程序入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Gymnasium 环境包装器与工具集"
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='运行单元测试'
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='运行包装器演示'
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if args.demo or True:  # 默认运行演示
        if not HAS_GYMNASIUM:
            print("gymnasium 未安装")
            return

        print("\n" + "=" * 60)
        print("包装器演示")
        print("=" * 60)

        # 演示 1: 观测归一化
        print("\n[演示 1] 观测归一化效果")
        env = gym.make("Pendulum-v1")
        wrapped_env = NormalizeObservationWrapper(env)

        print("原始环境:")
        raw_obs, _ = env.reset()
        print(f"  观测: {raw_obs}")
        print(f"  范围: [{env.observation_space.low}, {env.observation_space.high}]")

        print("\n归一化后:")
        norm_obs, _ = wrapped_env.reset()

        # 运行一些步以收集统计量
        for _ in range(100):
            norm_obs, _, terminated, truncated, _ = wrapped_env.step(
                wrapped_env.action_space.sample()
            )
            if terminated or truncated:
                wrapped_env.reset()

        print(f"  观测: {norm_obs}")
        print(f"  统计: mean≈{wrapped_env.running_stats.mean}")

        env.close()
        wrapped_env.close()

        # 演示 2: 组合包装器
        print("\n[演示 2] 组合包装器")
        env = make_wrapped_env(
            "CartPole-v1",
            normalize_obs=True,
            record_stats=True,
            frame_stack=4
        )

        print(f"原始观测空间: Box(4,)")
        print(f"包装后观测空间: {env.observation_space}")

        obs, _ = env.reset()
        for _ in range(200):
            obs, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                if 'episode' in info:
                    print(f"  回合完成: 奖励={info['episode']['r']:.1f}, 长度={info['episode']['l']}")
                obs, _ = env.reset()

        env.close()

        print("\n演示完成!")


if __name__ == "__main__":
    main()
