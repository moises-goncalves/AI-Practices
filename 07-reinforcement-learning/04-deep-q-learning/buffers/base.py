"""
基础经验回放缓冲区

============================================================
核心思想 (Core Idea)
============================================================
经验回放是DQN的核心创新之一，解决了两个关键问题：

1. **打破时序相关性**: 连续采样的样本高度相关，违反SGD的i.i.d.假设。
   随机采样打破时序关联，提供更稳定的梯度估计。

2. **提高数据效率**: 每个经验可被多次使用，而非"用后即弃"。
   这对于数据收集成本高的RL场景尤为重要。

============================================================
数学基础 (Mathematical Foundation)
============================================================
均匀采样概率：

.. math::
    P(i) = \\frac{1}{|\\mathcal{D}|}, \\quad \\forall i \\in \\mathcal{D}

使用回放缓冲区的期望梯度：

.. math::
    \\nabla_\\theta L = \\mathbb{E}_{(s,a,r,s') \\sim \\mathcal{U}(\\mathcal{D})}
    \\left[ (Q(s,a;\\theta) - y)^2 \\right]

============================================================
复杂度分析 (Complexity Analysis)
============================================================
- 空间: O(N × d), N = 容量, d = 状态维度
- push: O(1) 摊销（循环覆写）
- sample: O(B), B = 批次大小
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np
from numpy.typing import NDArray

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Transition, FloatArray


class ReplayBuffer:
    """
    均匀经验回放缓冲区
    
    使用deque实现O(1)的push操作和自动FIFO淘汰。
    
    Attributes
    ----------
    capacity : int
        最大缓冲区容量（只读属性）
    
    Examples
    --------
    >>> buffer = ReplayBuffer(capacity=10000)
    >>> buffer.push(state, action, reward, next_state, done)
    >>> if buffer.is_ready(min_size=1000):
    ...     states, actions, rewards, next_states, dones = buffer.sample(64)
    """
    
    __slots__ = ("_capacity", "_buffer")
    
    def __init__(self, capacity: int) -> None:
        """
        初始化均匀回放缓冲区
        
        Parameters
        ----------
        capacity : int
            最大存储容量。必须为正整数。
            当缓冲区满时，最旧的转移将被淘汰（FIFO策略）。
        
        Raises
        ------
        ValueError
            如果capacity不是正整数
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"capacity必须是正整数，得到{capacity!r} "
                f"(类型: {type(capacity).__name__})"
            )
        self._capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)
    
    @property
    def capacity(self) -> int:
        """最大缓冲区容量（只读）"""
        return self._capacity
    
    def __len__(self) -> int:
        """返回当前存储的转移数量"""
        return len(self._buffer)
    
    def push(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> None:
        """
        存储单个转移到缓冲区
        
        Parameters
        ----------
        state : FloatArray
            当前状态观测，形状 (state_dim,)
        action : int
            离散动作索引
        reward : float
            即时奖励
        next_state : FloatArray
            下一状态观测，形状 (state_dim,)
        done : bool
            此转移后回合是否终止
        
        Notes
        -----
        - O(1) 摊销时间复杂度
        - 当达到容量时自动淘汰最旧的转移（FIFO）
        """
        self._buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[FloatArray, NDArray[np.int64], FloatArray, FloatArray, FloatArray]:
        """
        均匀随机采样一个mini-batch的转移
        
        Parameters
        ----------
        batch_size : int
            要采样的转移数量
        
        Returns
        -------
        states : FloatArray
            状态批次，形状 (batch_size, state_dim)
        actions : NDArray[np.int64]
            动作批次，形状 (batch_size,)
        rewards : FloatArray
            奖励批次，形状 (batch_size,)
        next_states : FloatArray
            下一状态批次，形状 (batch_size, state_dim)
        dones : FloatArray
            终止标志批次（0.0或1.0），形状 (batch_size,)
        
        Raises
        ------
        ValueError
            如果batch_size超过当前缓冲区大小
        
        Notes
        -----
        - 每次调用内无放回采样
        - 返回NumPy数组以便高效转换为张量
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size ({batch_size}) 超过缓冲区大小 ({len(self._buffer)})。"
                f"确保缓冲区至少有 {batch_size} 个转移后再采样。"
            )
        
        batch = random.sample(list(self._buffer), batch_size)
        
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        batch_rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        
        return states, actions, batch_rewards, next_states, dones
    
    def is_ready(self, min_size: int) -> bool:
        """
        检查缓冲区是否有足够的样本用于训练
        
        Parameters
        ----------
        min_size : int
            所需的最小转移数量
        
        Returns
        -------
        bool
            如果 len(buffer) >= min_size 则为True
        """
        return len(self._buffer) >= min_size
    
    def clear(self) -> None:
        """清空所有存储的转移"""
        self._buffer.clear()
    
    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self._capacity}, size={len(self)})"
