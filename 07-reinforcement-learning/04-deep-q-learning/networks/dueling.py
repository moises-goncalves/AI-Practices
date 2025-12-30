"""
Dueling DQN网络架构

============================================================
核心思想 (Core Idea)
============================================================
将Q函数分解为状态价值V(s)和动作优势A(s, a)：
    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)

直觉：
- V(s): "这个状态有多好" - 与动作无关
- A(s, a): "这个动作比平均好多少"

分离这些允许：
- 状态价值从所有动作经验中学习
- 在动作选择不重要的状态中更快收敛

============================================================
数学基础 (Mathematical Foundation)
============================================================
网络结构：
    h = φ(s)                    # 共享特征提取
    V(s) = V_stream(h)          # 价值流，输出标量
    A(s, ·) = A_stream(h)       # 优势流，输出向量

聚合（均值基线）：
    Q(s, a) = V(s) + (A(s, a) - 1/|A| Σ_{a'} A(s, a'))

为什么减去均值？
- 可辨识性：V和A的分解不是唯一的
    Q(s,a) = (V(s) + c) + (A(s,a) - c) 对任意c成立
- 均值减法约束：Σ_a A(s,a) = 0，使V和A唯一

============================================================
算法对比 (Algorithm Comparison)
============================================================
vs. 标准DQN:
+ 更快收敛：V(s)从所有动作学习
+ 更稳定：价值流梯度更平滑
+ 在Atari上约20%提升
- 参数稍多：约1.5倍
- 额外计算：均值操作

============================================================
复杂度分析 (Complexity Analysis)
============================================================
参数量: O(d·h + h² + h + h·|A|) ≈ 1.5倍标准DQN
前向传播: O(|θ|)

============================================================
参考文献 (References)
============================================================
Wang, Z. et al. (2016). Dueling Network Architectures for Deep
Reinforcement Learning. ICML.
"""

from __future__ import annotations

from typing import List

import torch.nn as nn
from torch import Tensor

from .base import init_weights_orthogonal


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN网络
    
    将Q函数分解为状态价值和动作优势两个流。
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
    ) -> None:
        """
        初始化Dueling网络
        
        Parameters
        ----------
        state_dim : int
            状态空间维度
        action_dim : int
            动作数量
        hidden_dims : List[int]
            隐藏层维度
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # 确定特征层和流的维度
        if len(hidden_dims) >= 2:
            feature_dim = hidden_dims[0]
            stream_hidden_dim = hidden_dims[-1]
        else:
            feature_dim = hidden_dims[0] if hidden_dims else 128
            stream_hidden_dim = feature_dim
        
        # 共享特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, feature_dim),
            nn.ReLU(),
        )
        
        # 价值流: V(s) ∈ ℝ
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, stream_hidden_dim),
            nn.ReLU(),
            nn.Linear(stream_hidden_dim, 1),
        )
        
        # 优势流: A(s, ·) ∈ ℝ^{|A|}
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, stream_hidden_dim),
            nn.ReLU(),
            nn.Linear(stream_hidden_dim, action_dim),
        )
        
        self.apply(init_weights_orthogonal)
    
    def forward(self, state: Tensor) -> Tensor:
        """
        前向传播
        
        聚合: Q = V + (A - mean(A))
        
        Parameters
        ----------
        state : Tensor
            状态张量 (batch_size, state_dim)
        
        Returns
        -------
        Tensor
            Q值 (batch_size, action_dim)
        """
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
    def __repr__(self) -> str:
        return (
            f"DuelingDQNNetwork(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, hidden_dims={self.hidden_dims})"
        )
