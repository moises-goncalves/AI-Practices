"""
神经网络基础模块

提供DQN网络的基础架构和工具函数。

============================================================
核心思想 (Core Idea)
============================================================
神经网络作为Q函数的函数逼近器，将状态映射到动作价值：
    f_θ: ℝ^d → ℝ^{|A|}
    Q(s, a; θ) = f_θ(s)[a]

============================================================
数学基础 (Mathematical Foundation)
============================================================
前向传播：
    h₀ = s
    h_{l+1} = ReLU(W_l h_l + b_l),  l = 0, ..., L-1
    Q(s, ·) = W_L h_L + b_L

参数：
    θ = {W₀, b₀, ..., W_L, b_L}
    |θ| = d·h + (L-1)·h² + h·|A| + Σ(biases)
"""

from __future__ import annotations

import math
from typing import List, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Optional[Type[nn.Module]] = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    创建多层感知机(MLP)
    
    Parameters
    ----------
    input_dim : int
        输入维度
    output_dim : int
        输出维度
    hidden_dims : List[int]
        隐藏层维度列表
    activation : Type[nn.Module]
        隐藏层激活函数类
    output_activation : Optional[Type[nn.Module]]
        输出层激活函数（None = 无激活）
    dropout : float
        Dropout概率（0 = 无dropout）
    
    Returns
    -------
    nn.Sequential
        构建的MLP模型
    """
    layers: List[nn.Module] = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    
    return nn.Sequential(*layers)


def init_weights_orthogonal(module: nn.Module, gain: float = math.sqrt(2)) -> None:
    """
    正交初始化神经网络权重
    
    正交初始化在深度网络中保持梯度尺度，推荐用于深度RL网络。
    
    数学背景：
        对于正交矩阵W: W^T W = I
        这保持范数: ||Wx|| = ||x||
        梯度幅度在各层保持稳定。
    
    Parameters
    ----------
    module : nn.Module
        要初始化的模块
    gain : float
        缩放因子（ReLU推荐√2）
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_xavier(module: nn.Module) -> None:
    """
    Xavier (Glorot) 初始化
    
    为sigmoid/tanh激活函数保持各层方差。
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DQNNetwork(nn.Module):
    """
    标准DQN网络
    
    ============================================================
    核心思想 (Core Idea)
    ============================================================
    多层感知机将状态映射到所有动作的Q值：
        f_θ: ℝ^d → ℝ^{|A|}
        Q(s, a; θ) = f_θ(s)[a]
    
    ============================================================
    算法对比 (Algorithm Comparison)
    ============================================================
    vs. Dueling网络:
    + 简单直接
    + 参数更少
    - 不分离状态价值和动作优势
    - 所有动作信息混合在一起
    
    vs. 卷积网络（用于Atari）:
    + 适合低维状态
    + 训练更快
    - 不适合图像输入
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 128],
    ) -> None:
        """
        初始化DQN网络
        
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
        
        self.network = create_mlp(state_dim, action_dim, hidden_dims)
        self.apply(init_weights_orthogonal)
    
    def forward(self, state: Tensor) -> Tensor:
        """
        前向传播
        
        Parameters
        ----------
        state : Tensor
            状态张量，形状 (batch_size, state_dim)
        
        Returns
        -------
        Tensor
            Q值张量，形状 (batch_size, action_dim)
        """
        return self.network(state)
    
    def __repr__(self) -> str:
        return (
            f"DQNNetwork(state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"hidden_dims={self.hidden_dims})"
        )
