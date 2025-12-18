"""
价值网络模块 (Value Networks Module)
===================================

核心思想 (Core Idea):
--------------------
提供用于TD学习的线性函数逼近器。
这些网络可以与特征编码器配合使用。

数学原理 (Mathematical Theory):
------------------------------
线性函数逼近:
    V̂(s; w) = w^T φ(s)
    Q̂(s,a; w) = w_a^T φ(s)

梯度下降更新:
    w ← w + α × δ × ∇_w V̂(s; w)
    w ← w + α × δ × φ(s)  (线性情况)

其中 δ 是TD误差。

问题背景 (Problem Statement):
----------------------------
表格方法需要存储每个状态的价值，在大状态空间中不可行。
线性函数逼近通过参数共享实现泛化:
- 参数数量与特征维度相关，而非状态数量
- 相似状态得到相似的价值估计
- 可以泛化到未见过的状态
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, Tuple
from .base import FeatureEncoder


class LinearValueNetwork:
    """
    线性状态价值网络。

    核心思想 (Core Idea):
    --------------------
    使用线性组合逼近状态价值函数:
        V̂(s) = w^T φ(s)

    数学原理 (Mathematical Theory):
    ------------------------------
    梯度:
        ∇_w V̂(s) = φ(s)

    TD(0) 更新:
        δ = R + γ V̂(S') - V̂(S)
        w ← w + α × δ × φ(S)

    TD(λ) 更新（资格迹）:
        e ← γλe + φ(S)
        w ← w + α × δ × e

    收敛性:
    - On-policy: 收敛到 V^π 的投影
    - 收敛速度取决于特征的表达能力

    Example:
        >>> encoder = TileEncoder([(0, 1), (0, 1)], num_tilings=8)
        >>> network = LinearValueNetwork(encoder)
        >>> value = network.predict(np.array([0.5, 0.5]))
        >>> network.update(np.array([0.5, 0.5]), td_error=1.0, alpha=0.1)
    """

    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        initial_weights: Optional[np.ndarray] = None
    ) -> None:
        """
        初始化线性价值网络。

        Args:
            feature_encoder: 特征编码器
            initial_weights: 初始权重，默认为0
        """
        self.encoder = feature_encoder
        self.feature_dim = feature_encoder.feature_dim

        if initial_weights is not None:
            if len(initial_weights) != self.feature_dim:
                raise ValueError(
                    f"权重维度 {len(initial_weights)} 与特征维度 {self.feature_dim} 不匹配"
                )
            self.weights = initial_weights.copy()
        else:
            self.weights = np.zeros(self.feature_dim)

        # 资格迹
        self.eligibility_trace = np.zeros(self.feature_dim)

    def predict(self, state: np.ndarray) -> float:
        """
        预测状态价值。

        Args:
            state: 状态向量

        Returns:
            估计的状态价值
        """
        features = self.encoder.encode(state)
        return float(np.dot(self.weights, features))

    def update(
        self,
        state: np.ndarray,
        td_error: float,
        alpha: float = 0.01
    ) -> None:
        """
        使用TD误差更新权重。

        数学原理:
        --------
        TD更新规则:
            w ← w + α × δ × φ(s)

        Args:
            state: 状态向量
            td_error: TD误差 δ = R + γV(S') - V(S)
            alpha: 学习率
        """
        features = self.encoder.encode(state)
        self.weights += alpha * td_error * features

    def update_with_trace(
        self,
        state: np.ndarray,
        td_error: float,
        alpha: float = 0.01,
        gamma: float = 0.99,
        lambda_: float = 0.9
    ) -> None:
        """
        使用资格迹更新权重。

        数学原理:
        --------
        资格迹更新:
            e ← γλe + φ(S)
            w ← w + αδe

        累积迹允许多步回传，加速学习。

        Args:
            state: 状态向量
            td_error: TD误差
            alpha: 学习率
            gamma: 折扣因子
            lambda_: 迹衰减参数
        """
        features = self.encoder.encode(state)

        # 更新资格迹
        self.eligibility_trace = gamma * lambda_ * self.eligibility_trace + features

        # 更新权重
        self.weights += alpha * td_error * self.eligibility_trace

    def reset_trace(self) -> None:
        """重置资格迹（在回合开始时调用）。"""
        self.eligibility_trace = np.zeros(self.feature_dim)

    def get_weights(self) -> np.ndarray:
        """获取权重向量副本。"""
        return self.weights.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        """设置权重向量。"""
        if len(weights) != self.feature_dim:
            raise ValueError(f"权重维度不匹配")
        self.weights = weights.copy()


class LinearQNetwork:
    """
    线性动作价值网络。

    核心思想 (Core Idea):
    --------------------
    使用线性组合逼近动作价值函数:
        Q̂(s,a) = w_a^T φ(s)

    为每个动作维护独立的权重向量。

    数学原理 (Mathematical Theory):
    ------------------------------
    多动作表示:
        Q̂(s,a) = w_a^T φ(s)  ∀a ∈ A

    或统一表示:
        Q̂(s,a) = w^T φ(s,a)

    本实现使用第一种方式，每个动作一个权重向量。

    SARSA 更新:
        δ = R + γQ̂(S',A') - Q̂(S,A)
        w_A ← w_A + αδφ(S)

    Q-Learning 更新:
        δ = R + γmax_a Q̂(S',a) - Q̂(S,A)
        w_A ← w_A + αδφ(S)

    Example:
        >>> encoder = TileEncoder([(0, 1), (0, 1)], num_tilings=8)
        >>> network = LinearQNetwork(encoder, n_actions=4)
        >>> q_values = network.predict_all(np.array([0.5, 0.5]))
        >>> best_action = np.argmax(q_values)
    """

    def __init__(
        self,
        feature_encoder: FeatureEncoder,
        n_actions: int,
        initial_weights: Optional[np.ndarray] = None
    ) -> None:
        """
        初始化线性Q网络。

        Args:
            feature_encoder: 特征编码器
            n_actions: 动作数量
            initial_weights: 初始权重 (n_actions, feature_dim)，默认为0
        """
        self.encoder = feature_encoder
        self.feature_dim = feature_encoder.feature_dim
        self.n_actions = n_actions

        if initial_weights is not None:
            if initial_weights.shape != (n_actions, self.feature_dim):
                raise ValueError(
                    f"权重形状 {initial_weights.shape} 与期望 "
                    f"({n_actions}, {self.feature_dim}) 不匹配"
                )
            self.weights = initial_weights.copy()
        else:
            self.weights = np.zeros((n_actions, self.feature_dim))

        # 每个动作的资格迹
        self.eligibility_traces = np.zeros((n_actions, self.feature_dim))

    def predict(self, state: np.ndarray, action: int) -> float:
        """
        预测特定状态-动作对的Q值。

        Args:
            state: 状态向量
            action: 动作索引

        Returns:
            估计的Q值
        """
        features = self.encoder.encode(state)
        return float(np.dot(self.weights[action], features))

    def predict_all(self, state: np.ndarray) -> np.ndarray:
        """
        预测状态的所有动作Q值。

        Args:
            state: 状态向量

        Returns:
            所有动作的Q值数组
        """
        features = self.encoder.encode(state)
        return self.weights @ features

    def update(
        self,
        state: np.ndarray,
        action: int,
        td_error: float,
        alpha: float = 0.01
    ) -> None:
        """
        使用TD误差更新特定动作的权重。

        Args:
            state: 状态向量
            action: 动作索引
            td_error: TD误差
            alpha: 学习率
        """
        features = self.encoder.encode(state)
        self.weights[action] += alpha * td_error * features

    def update_with_trace(
        self,
        state: np.ndarray,
        action: int,
        td_error: float,
        alpha: float = 0.01,
        gamma: float = 0.99,
        lambda_: float = 0.9,
        trace_type: str = 'accumulating'
    ) -> None:
        """
        使用资格迹更新权重。

        Args:
            state: 状态向量
            action: 动作索引
            td_error: TD误差
            alpha: 学习率
            gamma: 折扣因子
            lambda_: 迹衰减参数
            trace_type: 迹类型 ('accumulating', 'replacing')
        """
        features = self.encoder.encode(state)

        # 更新所有迹的衰减
        self.eligibility_traces *= gamma * lambda_

        # 更新当前动作的迹
        if trace_type == 'accumulating':
            self.eligibility_traces[action] += features
        elif trace_type == 'replacing':
            self.eligibility_traces[action] = np.maximum(
                self.eligibility_traces[action], features
            )
        else:
            self.eligibility_traces[action] += features

        # 更新权重
        self.weights += alpha * td_error * self.eligibility_traces

    def reset_trace(self) -> None:
        """重置所有资格迹。"""
        self.eligibility_traces = np.zeros((self.n_actions, self.feature_dim))

    def greedy_action(self, state: np.ndarray) -> int:
        """
        获取贪婪动作。

        Args:
            state: 状态向量

        Returns:
            Q值最高的动作索引
        """
        q_values = self.predict_all(state)
        return int(np.argmax(q_values))

    def epsilon_greedy_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.1
    ) -> int:
        """
        ε-贪婪动作选择。

        Args:
            state: 状态向量
            epsilon: 探索率

        Returns:
            选择的动作索引
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return self.greedy_action(state)

    def get_weights(self) -> np.ndarray:
        """获取权重矩阵副本。"""
        return self.weights.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        """设置权重矩阵。"""
        if weights.shape != (self.n_actions, self.feature_dim):
            raise ValueError("权重形状不匹配")
        self.weights = weights.copy()


def create_linear_learner(
    state_ranges: list,
    n_actions: int,
    encoding: str = 'tile',
    **kwargs
) -> Tuple[FeatureEncoder, LinearQNetwork]:
    """
    便捷函数：创建线性学习器。

    Args:
        state_ranges: 状态范围
        n_actions: 动作数量
        encoding: 编码类型 ('tile', 'fourier', 'polynomial')
        **kwargs: 编码器参数

    Returns:
        (编码器, Q网络) 元组

    Example:
        >>> encoder, network = create_linear_learner(
        ...     state_ranges=[(0, 1), (0, 1)],
        ...     n_actions=4,
        ...     encoding='tile',
        ...     num_tilings=8
        ... )
    """
    from .base import TileEncoder, FourierBasis, PolynomialFeatures

    if encoding == 'tile':
        encoder = TileEncoder(
            state_ranges,
            num_tilings=kwargs.get('num_tilings', 8),
            tiles_per_dim=kwargs.get('tiles_per_dim', 4)
        )
    elif encoding == 'fourier':
        encoder = FourierBasis(
            state_ranges,
            order=kwargs.get('order', 3)
        )
    elif encoding == 'polynomial':
        encoder = PolynomialFeatures(
            state_dim=len(state_ranges),
            degree=kwargs.get('degree', 2)
        )
    else:
        raise ValueError(f"未知编码类型: {encoding}")

    network = LinearQNetwork(encoder, n_actions)

    return encoder, network
