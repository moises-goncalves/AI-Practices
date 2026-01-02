"""
优先经验回放缓冲区 (Prioritized Experience Replay)

============================================================
核心思想 (Core Idea)
============================================================
优先经验回放通过TD误差大小对样本进行优先采样，使得学习更加高效：

1. **重要性采样**: 高TD误差的样本包含更多学习信号
2. **偏差校正**: 使用重要性采样权重保持梯度无偏

============================================================
数学基础 (Mathematical Foundation)
============================================================
**优先级定义**:

.. math::
    p_i = |\\delta_i| + \\epsilon

其中 δ_i 是TD误差，ε 防止零优先级。

**采样概率**:

.. math::
    P(i) = \\frac{p_i^\\alpha}{\\sum_k p_k^\\alpha}

- α = 0: 均匀采样（忽略优先级）
- α = 1: 完全优先化

**重要性采样权重**（无偏梯度）:

.. math::
    w_i = \\left( \\frac{1}{N \\cdot P(i)} \\right)^\\beta / \\max_j w_j

β 在训练过程中从 β₀ 退火到 1 以完全校正偏差。

============================================================
算法对比 (Algorithm Comparison)
============================================================
vs. 均匀回放:
+ 关注"惊讶"样本：高TD误差 = 预测差
+ 加速学习：在Atari上约2倍加速
+ 稀疏奖励友好：罕见的成功被优先化
- 计算开销：使用Sum-Tree的O(log N)采样
- 超参数敏感：α, β需要调优
- 引入偏差：需要重要性采样校正

============================================================
复杂度分析 (Complexity Analysis)
============================================================
- 空间: O(N)
- Push: O(log N)
- Sample: O(B log N)
- 更新优先级: O(B log N)

============================================================
参考文献 (References)
============================================================
Schaul, T. et al. (2016). Prioritized Experience Replay. ICLR.
"""

from __future__ import annotations

import random
from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.types import Transition, FloatArray
from .sum_tree import SumTree


class PrioritizedReplayBuffer:
    """
    优先经验回放(PER)缓冲区
    
    Attributes
    ----------
    capacity : int
        最大缓冲区大小
    beta : float
        当前重要性采样指数
    """
    
    __slots__ = (
        "_capacity", "_alpha", "_beta", "_beta_start", "_beta_frames",
        "_epsilon", "_sum_tree", "_max_priority", "_frame",
    )
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
    ) -> None:
        """
        初始化优先回放缓冲区
        
        Parameters
        ----------
        capacity : int
            最大存储转移数量
        alpha : float, default=0.6
            优先化指数 α ∈ [0, 1]。
            0 = 均匀采样，1 = 完全优先化。
        beta_start : float, default=0.4
            初始重要性采样指数 β ∈ [0, 1]。
            在训练过程中退火到1.0。
        beta_frames : int, default=100000
            β 从 beta_start 退火到 1.0 的帧数
        epsilon : float, default=1e-6
            添加到优先级的小常数以防止零优先级
        
        Raises
        ------
        ValueError
            如果 alpha 或 beta_start 不在 [0, 1] 范围内，或 capacity 不是正数
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity必须是正整数，得到{capacity!r}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha必须在[0, 1]范围内，得到{alpha}")
        if not 0 <= beta_start <= 1:
            raise ValueError(f"beta_start必须在[0, 1]范围内，得到{beta_start}")
        if epsilon <= 0:
            raise ValueError(f"epsilon必须为正数，得到{epsilon}")
        
        self._capacity = capacity
        self._alpha = alpha
        self._beta_start = beta_start
        self._beta = beta_start
        self._beta_frames = max(1, beta_frames)
        self._epsilon = epsilon
        
        self._sum_tree = SumTree(capacity)
        self._max_priority = 1.0
        self._frame = 0
    
    @property
    def capacity(self) -> int:
        """最大缓冲区大小（只读）"""
        return self._capacity
    
    @property
    def beta(self) -> float:
        """当前重要性采样指数"""
        return self._beta
    
    def __len__(self) -> int:
        """当前存储的转移数量"""
        return len(self._sum_tree)
    
    def push(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> None:
        """
        以最大优先级存储转移
        
        新转移获得最大优先级以保证在优先级更新前至少被采样一次。
        """
        transition = Transition(state, action, reward, next_state, done)
        priority = self._max_priority ** self._alpha
        self._sum_tree.add(priority, transition)
    
    def sample(
        self,
        batch_size: int,
    ) -> Tuple[FloatArray, NDArray[np.int64], FloatArray, FloatArray, 
               FloatArray, NDArray[np.int64], FloatArray]:
        """
        以优先概率采样批次
        
        使用分层采样：将总优先级分成B个相等的段，从每个段采样一个转移。
        
        Returns
        -------
        states, actions, rewards, next_states, dones, indices, weights
            indices: 用于优先级更新的树索引
            weights: 重要性采样权重
        """
        buffer_len = len(self._sum_tree)
        if batch_size > buffer_len:
            raise ValueError(
                f"batch_size ({batch_size}) 超过缓冲区大小 ({buffer_len})"
            )
        
        self._anneal_beta()
        
        indices = np.empty(batch_size, dtype=np.int64)
        weights = np.empty(batch_size, dtype=np.float32)
        batch: List[Transition] = []
        
        total = self._sum_tree.total_priority
        segment = total / batch_size
        
        min_prob = self._sum_tree.min_priority() ** self._alpha / total
        max_weight = (buffer_len * min_prob) ** (-self._beta) if min_prob > 0 else 1.0
        
        for i in range(batch_size):
            low, high = segment * i, segment * (i + 1)
            cumsum = random.uniform(low, high)
            tree_idx, priority, data = self._sum_tree.get(cumsum)
            
            prob = priority / total
            weight = (buffer_len * prob) ** (-self._beta) / max_weight
            
            indices[i] = tree_idx
            weights[i] = weight
            batch.append(data)
        
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        batch_rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        
        return states, actions, batch_rewards, next_states, dones, indices, weights
    
    def _anneal_beta(self) -> None:
        """线性退火β从beta_start到1.0"""
        self._frame += 1
        fraction = min(1.0, self._frame / self._beta_frames)
        self._beta = self._beta_start + fraction * (1.0 - self._beta_start)
    
    def update_priorities(
        self,
        indices: NDArray[np.int64],
        td_errors: FloatArray,
    ) -> None:
        """
        基于TD误差更新优先级
        
        优先级公式: p_i = (|δ_i| + ε)^α
        """
        priorities = (np.abs(td_errors) + self._epsilon) ** self._alpha
        for idx, priority in zip(indices, priorities):
            self._sum_tree.update_priority(int(idx), float(priority))
        self._max_priority = max(
            self._max_priority,
            float(np.max(priorities)) ** (1.0 / self._alpha)
        )
    
    def is_ready(self, min_size: int) -> bool:
        """检查缓冲区是否有足够的样本"""
        return len(self._sum_tree) >= min_size
    
    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer(capacity={self._capacity}, "
            f"size={len(self)}, alpha={self._alpha}, beta={self._beta:.3f})"
        )
