"""
基础网络组件模块 (Base Network Components)
========================================

核心思想 (Core Idea):
--------------------
提供状态特征编码器，将原始状态转换为适合线性函数逼近的特征表示。
这是从表格方法过渡到函数逼近的关键步骤。

数学原理 (Mathematical Theory):
------------------------------
函数逼近将价值函数参数化:
    V̂(s; w) = w^T φ(s)

其中 φ(s) 是状态的特征向量，w 是可学习参数。

常用特征编码:
1. Tile Coding (分块编码): 将连续空间离散化为重叠的分块
2. Polynomial Features (多项式特征): x → [1, x, x², ...]
3. Fourier Basis (傅里叶基): x → [cos(πnx) for n = 0,1,2,...]

问题背景 (Problem Statement):
----------------------------
表格方法对每个状态独立存储价值，无法泛化到未见状态。
函数逼近通过参数共享实现泛化，是处理大状态空间的关键技术。
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod


class FeatureEncoder(ABC):
    """
    特征编码器基类。

    核心思想 (Core Idea):
    --------------------
    定义特征编码的统一接口，将状态映射到特征向量。

    数学原理 (Mathematical Theory):
    ------------------------------
    特征函数:
        φ: S → R^d

    将状态从原始表示转换为d维特征向量。
    """

    @abstractmethod
    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        编码状态为特征向量。

        Args:
            state: 原始状态

        Returns:
            特征向量
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """特征维度。"""
        pass


class TileEncoder(FeatureEncoder):
    """
    分块编码器 (Tile Coding)。

    核心思想 (Core Idea):
    --------------------
    将连续状态空间划分为多个重叠的分块（tiling）。
    每个分块是一组不重叠的瓦片（tile），但分块之间有偏移。
    状态落入的瓦片被激活（置1），其他为0。

    数学原理 (Mathematical Theory):
    ------------------------------
    对于d维状态空间，使用n个分块，每个维度m个瓦片:

    特征维度: d_φ = n × m^d

    编码过程:
    1. 对每个分块施加随机偏移
    2. 将状态映射到该分块的瓦片索引
    3. 将激活的瓦片设为1

    优点:
    - 线性计算复杂度
    - 良好的泛化性质
    - 稀疏表示，计算高效

    算法对比 (Comparison):
    ---------------------
    与其他编码方式对比:

    ┌─────────────────┬──────────────┬─────────────┬────────────┐
    │      方法       │   特征稀疏性 │   计算复杂度 │   泛化能力  │
    ├─────────────────┼──────────────┼─────────────┼────────────┤
    │   Tile Coding   │     稀疏     │    O(n)     │     好     │
    │   Polynomial    │     稠密     │    O(d^k)   │    一般    │
    │   Fourier       │     稠密     │    O(d^k)   │     好     │
    │   RBF           │     稠密     │    O(k)     │     好     │
    └─────────────────┴──────────────┴─────────────┴────────────┘

    Example:
        >>> encoder = TileEncoder(
        ...     state_ranges=[(0, 1), (0, 1)],
        ...     num_tilings=8,
        ...     tiles_per_dim=4
        ... )
        >>> state = np.array([0.5, 0.5])
        >>> features = encoder.encode(state)
        >>> print(f"特征维度: {features.shape[0]}, 激活数: {np.sum(features)}")
    """

    def __init__(
        self,
        state_ranges: List[Tuple[float, float]],
        num_tilings: int = 8,
        tiles_per_dim: int = 4,
        seed: Optional[int] = None
    ) -> None:
        """
        初始化分块编码器。

        Args:
            state_ranges: 每个状态维度的范围 [(min, max), ...]
            num_tilings: 分块（tiling）数量
            tiles_per_dim: 每个维度的瓦片数量
            seed: 随机种子（用于生成偏移）
        """
        self.state_ranges = state_ranges
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.state_dim = len(state_ranges)

        # 计算每个维度的瓦片宽度
        self.tile_widths = np.array([
            (high - low) / tiles_per_dim
            for low, high in state_ranges
        ])

        # 生成每个tiling的偏移
        if seed is not None:
            np.random.seed(seed)

        # 每个tiling有一个随机偏移向量
        self.offsets = np.random.uniform(
            0, 1, (num_tilings, self.state_dim)
        ) * self.tile_widths

        # 计算总特征数
        self._feature_dim = num_tilings * (tiles_per_dim ** self.state_dim)

    @property
    def feature_dim(self) -> int:
        """特征维度。"""
        return self._feature_dim

    def _get_tile_index(
        self,
        state: np.ndarray,
        tiling_idx: int
    ) -> int:
        """
        获取状态在指定tiling中的瓦片索引。

        Args:
            state: 归一化状态
            tiling_idx: tiling索引

        Returns:
            瓦片索引
        """
        # 应用偏移
        offset_state = state - self.offsets[tiling_idx]

        # 计算每个维度的瓦片索引
        tile_indices = []
        for d in range(self.state_dim):
            low, high = self.state_ranges[d]
            normalized = (offset_state[d] - low) / (high - low)
            idx = int(np.clip(normalized * self.tiles_per_dim, 0, self.tiles_per_dim - 1))
            tile_indices.append(idx)

        # 计算全局瓦片索引
        global_idx = 0
        multiplier = 1
        for idx in tile_indices:
            global_idx += idx * multiplier
            multiplier *= self.tiles_per_dim

        # 加上tiling的基础偏移
        return tiling_idx * (self.tiles_per_dim ** self.state_dim) + global_idx

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        编码状态为特征向量。

        Args:
            state: 原始状态向量

        Returns:
            稀疏特征向量（激活的位置为1，其他为0）
        """
        state = np.asarray(state).flatten()
        features = np.zeros(self._feature_dim)

        for tiling_idx in range(self.num_tilings):
            tile_idx = self._get_tile_index(state, tiling_idx)
            features[tile_idx] = 1.0

        return features

    def encode_batch(self, states: np.ndarray) -> np.ndarray:
        """
        批量编码状态。

        Args:
            states: 状态数组 (batch_size, state_dim)

        Returns:
            特征数组 (batch_size, feature_dim)
        """
        batch_size = len(states)
        features = np.zeros((batch_size, self._feature_dim))

        for i, state in enumerate(states):
            features[i] = self.encode(state)

        return features


class PolynomialFeatures(FeatureEncoder):
    """
    多项式特征编码器。

    核心思想 (Core Idea):
    --------------------
    将状态扩展为多项式基函数，捕捉非线性关系。

    数学原理 (Mathematical Theory):
    ------------------------------
    对于1维状态x，k阶多项式特征:
        φ(x) = [1, x, x², ..., x^k]

    对于多维状态，使用多维多项式:
        φ(x,y) = [1, x, y, x², xy, y², ...]

    复杂度:
        d维状态，k阶: C(d+k, k) 个特征

    Example:
        >>> encoder = PolynomialFeatures(state_dim=2, degree=3)
        >>> state = np.array([0.5, 0.5])
        >>> features = encoder.encode(state)
    """

    def __init__(
        self,
        state_dim: int,
        degree: int = 2,
        include_bias: bool = True
    ) -> None:
        """
        初始化多项式特征编码器。

        Args:
            state_dim: 状态维度
            degree: 多项式最高阶数
            include_bias: 是否包含偏置项（常数1）
        """
        self.state_dim = state_dim
        self.degree = degree
        self.include_bias = include_bias

        # 预计算特征维度
        self._feature_dim = self._compute_feature_dim()

    def _compute_feature_dim(self) -> int:
        """计算特征维度（组合数）。"""
        from math import comb
        dim = comb(self.state_dim + self.degree, self.degree)
        if not self.include_bias:
            dim -= 1
        return dim

    @property
    def feature_dim(self) -> int:
        """特征维度。"""
        return self._feature_dim

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        编码状态为多项式特征。

        Args:
            state: 原始状态向量

        Returns:
            多项式特征向量
        """
        state = np.asarray(state).flatten()
        features = []

        if self.include_bias:
            features.append(1.0)

        # 生成所有阶数的多项式项
        def generate_terms(current_powers, remaining_dim, remaining_degree):
            if remaining_dim == 0:
                if sum(current_powers) > 0 or self.include_bias:
                    term = 1.0
                    for i, power in enumerate(current_powers):
                        term *= state[i] ** power
                    features.append(term)
                return

            for power in range(remaining_degree + 1):
                generate_terms(
                    current_powers + [power],
                    remaining_dim - 1,
                    remaining_degree - power
                )

        # 从1阶开始（0阶是偏置项）
        for total_degree in range(1, self.degree + 1):
            for first_power in range(total_degree + 1):
                self._generate_polynomial_terms(
                    state, [first_power], 1, total_degree - first_power, features
                )

        return np.array(features)

    def _generate_polynomial_terms(
        self,
        state: np.ndarray,
        powers: List[int],
        dim: int,
        remaining_degree: int,
        features: List[float]
    ) -> None:
        """递归生成多项式项。"""
        if dim == self.state_dim:
            if remaining_degree == 0:
                term = 1.0
                for i, power in enumerate(powers):
                    term *= state[i] ** power
                features.append(term)
            return

        for power in range(remaining_degree + 1):
            self._generate_polynomial_terms(
                state,
                powers + [power],
                dim + 1,
                remaining_degree - power,
                features
            )


class FourierBasis(FeatureEncoder):
    """
    傅里叶基编码器。

    核心思想 (Core Idea):
    --------------------
    使用傅里叶级数的余弦基函数编码状态。
    适合周期性或光滑的价值函数。

    数学原理 (Mathematical Theory):
    ------------------------------
    对于归一化状态 s ∈ [0,1]^d，傅里叶基:
        φ_c(s) = cos(π × c^T × s)

    其中 c = (c_1, ..., c_d) 是系数向量，c_i ∈ {0, 1, ..., order}。

    特征维度: (order + 1)^d

    优点:
    - 良好的逼近性质
    - 自然处理周期性
    - 与状态范围无关（归一化后）

    Example:
        >>> encoder = FourierBasis(
        ...     state_ranges=[(0, 1), (0, 1)],
        ...     order=3
        ... )
        >>> state = np.array([0.5, 0.5])
        >>> features = encoder.encode(state)
    """

    def __init__(
        self,
        state_ranges: List[Tuple[float, float]],
        order: int = 3
    ) -> None:
        """
        初始化傅里叶基编码器。

        Args:
            state_ranges: 每个状态维度的范围 [(min, max), ...]
            order: 傅里叶级数阶数
        """
        self.state_ranges = state_ranges
        self.order = order
        self.state_dim = len(state_ranges)

        # 生成所有系数向量
        self.coefficients = self._generate_coefficients()
        self._feature_dim = len(self.coefficients)

    def _generate_coefficients(self) -> np.ndarray:
        """生成所有系数向量组合。"""
        from itertools import product

        coeffs = list(product(range(self.order + 1), repeat=self.state_dim))
        return np.array(coeffs)

    @property
    def feature_dim(self) -> int:
        """特征维度。"""
        return self._feature_dim

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        编码状态为傅里叶基特征。

        Args:
            state: 原始状态向量

        Returns:
            傅里叶基特征向量
        """
        state = np.asarray(state).flatten()

        # 归一化到[0, 1]
        normalized = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            low, high = self.state_ranges[i]
            normalized[i] = (state[i] - low) / (high - low)
            normalized[i] = np.clip(normalized[i], 0, 1)

        # 计算傅里叶基
        features = np.cos(np.pi * self.coefficients @ normalized)

        return features

    def get_learning_rates(self, base_lr: float = 0.1) -> np.ndarray:
        """
        获取每个特征建议的学习率。

        数学原理:
        --------
        高阶傅里叶基对应高频成分，通常需要较小的学习率。
        建议的学习率与系数范数成反比:
            α_c = base_lr / ||c|| (||c|| > 0)

        Args:
            base_lr: 基础学习率

        Returns:
            每个特征的建议学习率
        """
        norms = np.linalg.norm(self.coefficients, axis=1)
        norms[norms == 0] = 1  # 避免除零

        return base_lr / norms
