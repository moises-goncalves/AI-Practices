"""
神经网络模块 (Networks Module)
=============================

本模块提供用于Deep TD学习的神经网络组件。
为传统表格TD方法向深度强化学习过渡提供基础设施。

模块结构:
--------
- base.py: 基础网络类和工具
- value_networks.py: 价值函数网络
- feature_extractors.py: 状态特征提取器

注意: 此模块为可选扩展，主模块使用表格方法。
"""

from .base import (
    TileEncoder,
    PolynomialFeatures,
    FourierBasis,
)

from .value_networks import (
    LinearValueNetwork,
    LinearQNetwork,
)

__all__ = [
    # 特征编码器
    "TileEncoder",
    "PolynomialFeatures",
    "FourierBasis",
    # 线性网络
    "LinearValueNetwork",
    "LinearQNetwork",
]
