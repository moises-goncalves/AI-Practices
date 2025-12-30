"""
DQN配置模块

============================================================
核心思想 (Core Idea)
============================================================
集中管理DQN算法的所有超参数，提供验证和序列化支持。
使用dataclass实现清晰的属性访问和自动__repr__。

============================================================
数学背景 (Mathematical Context)
============================================================
这些超参数控制学习动态：
- γ (gamma): 贝尔曼方程中的折扣因子
- α (learning_rate): 梯度下降的步长
- ε (epsilon): ε-贪婪策略中的探索概率
- τ (soft_update_tau): 软更新时的Polyak平均系数
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .enums import NetworkType, LossType


@dataclass(frozen=False)
class DQNConfig:
    """
    DQN超参数配置类
    
    ============================================================
    参数说明 (Parameter Description)
    ============================================================
    
    环境参数:
        state_dim: 状态空间维度
        action_dim: 动作空间大小（离散动作数量）
    
    网络参数:
        hidden_dims: 隐藏层维度列表，如 [128, 128]
        learning_rate: Adam优化器学习率
    
    强化学习参数:
        gamma: 折扣因子 γ ∈ [0, 1]
        epsilon_start: 初始探索率
        epsilon_end: 最终探索率
        epsilon_decay: 线性衰减的步数
    
    经验回放参数:
        buffer_size: 经验回放缓冲区容量
        batch_size: 训练批次大小
        min_buffer_size: 开始训练前的最小缓冲区大小
    
    目标网络参数:
        target_update_freq: 目标网络同步频率（步数）
        soft_update_tau: 软更新系数（None使用硬更新）
    
    算法变体:
        double_dqn: 是否使用Double DQN
        dueling: 是否使用Dueling架构
    
    训练参数:
        loss_type: 损失函数类型
        grad_clip: 梯度裁剪阈值（None禁用裁剪）
        device: 计算设备（'auto'自动选择GPU）
        seed: 随机种子（用于可复现性）
    """
    
    # 环境参数
    state_dim: int
    action_dim: int
    
    # 网络参数
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 1e-3
    
    # 强化学习参数
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    
    # 经验回放参数
    buffer_size: int = 100000
    batch_size: int = 64
    min_buffer_size: Optional[int] = None
    
    # 目标网络参数
    target_update_freq: int = 100
    soft_update_tau: Optional[float] = None
    
    # 算法变体
    double_dqn: bool = True
    dueling: bool = False
    network_type: NetworkType = NetworkType.STANDARD
    
    # 训练参数
    loss_type: LossType = LossType.HUBER
    grad_clip: Optional[float] = 10.0
    device: str = "auto"
    seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        """验证配置参数"""
        self._validate_positive("state_dim", self.state_dim)
        self._validate_positive("action_dim", self.action_dim)
        
        if not self.hidden_dims:
            raise ValueError("hidden_dims不能为空")
        for i, dim in enumerate(self.hidden_dims):
            if dim <= 0:
                raise ValueError(f"hidden_dims[{i}]必须为正数，得到{dim}")
        
        self._validate_range("learning_rate", self.learning_rate, 0, 1, exclusive_low=True)
        self._validate_range("gamma", self.gamma, 0, 1)
        
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError(
                f"无效的epsilon范围: epsilon_end={self.epsilon_end}, "
                f"epsilon_start={self.epsilon_start}。"
                f"必须满足 0 <= epsilon_end <= epsilon_start <= 1"
            )
        
        self._validate_positive("buffer_size", self.buffer_size)
        self._validate_positive("batch_size", self.batch_size)
        
        if self.batch_size > self.buffer_size:
            raise ValueError(
                f"batch_size ({self.batch_size}) 不能超过 "
                f"buffer_size ({self.buffer_size})"
            )
        
        self._validate_positive("target_update_freq", self.target_update_freq)
        
        # 如果启用dueling，自动设置network_type
        if self.dueling:
            object.__setattr__(self, "network_type", NetworkType.DUELING)
        
        # 设置默认min_buffer_size
        if self.min_buffer_size is None:
            object.__setattr__(self, "min_buffer_size", self.batch_size)
        
        if self.soft_update_tau is not None:
            self._validate_range("soft_update_tau", self.soft_update_tau, 0, 1)
    
    def _validate_positive(self, name: str, value: int) -> None:
        """验证值为正数"""
        if value <= 0:
            raise ValueError(f"{name}必须为正数，得到{value}")
    
    def _validate_range(
        self,
        name: str,
        value: float,
        low: float,
        high: float,
        exclusive_low: bool = False,
        exclusive_high: bool = False,
    ) -> None:
        """验证值在范围内"""
        low_ok = value > low if exclusive_low else value >= low
        high_ok = value < high if exclusive_high else value <= high
        if not (low_ok and high_ok):
            low_bracket = "(" if exclusive_low else "["
            high_bracket = ")" if exclusive_high else "]"
            raise ValueError(
                f"{name}必须在{low_bracket}{low}, {high}{high_bracket}范围内，得到{value}"
            )
    
    def get_device(self) -> torch.device:
        """获取计算设备"""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于序列化"""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "double_dqn": self.double_dqn,
            "dueling": self.dueling,
            "network_type": self.network_type.value,
            "loss_type": self.loss_type.value,
            "grad_clip": self.grad_clip,
            "device": self.device,
            "seed": self.seed,
            "min_buffer_size": self.min_buffer_size,
            "soft_update_tau": self.soft_update_tau,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DQNConfig":
        """从字典创建配置"""
        data = data.copy()
        if "network_type" in data and isinstance(data["network_type"], str):
            data["network_type"] = NetworkType(data["network_type"])
        if "loss_type" in data and isinstance(data["loss_type"], str):
            data["loss_type"] = LossType(data["loss_type"])
        return cls(**data)
    
    def __repr__(self) -> str:
        return (
            f"DQNConfig(state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"double_dqn={self.double_dqn}, dueling={self.dueling})"
        )
