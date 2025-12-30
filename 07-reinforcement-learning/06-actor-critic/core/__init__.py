"""
核心模块 (Core Module)

提供策略梯度算法的基础配置、类型定义和枚举类型。

核心思想 (Core Idea):
    将配置、类型和枚举分离，实现关注点分离(Separation of Concerns)，
    提高代码的可维护性和可测试性。

模块组成:
    - config.py: 训练超参数配置
    - types.py: 数据类型定义 (Trajectory, Transition等)
    - enums.py: 枚举类型 (PolicyType, AdvantageEstimator等)
"""

from .config import PolicyGradientConfig
from .types import Trajectory, Transition, TrainingMetrics
from .enums import PolicyType, AdvantageEstimator, NetworkArchitecture

__all__ = [
    "PolicyGradientConfig",
    "Trajectory",
    "Transition",
    "TrainingMetrics",
    "PolicyType",
    "AdvantageEstimator",
    "NetworkArchitecture",
]
