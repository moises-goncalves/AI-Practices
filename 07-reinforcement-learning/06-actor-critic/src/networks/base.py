"""
基础网络组件模块 (Base Network Components Module)

本模块提供构建策略梯度网络所需的基础组件。

核心思想 (Core Idea):
    将通用的网络构建块抽象出来，包括:
    1. 多层感知机 (MLP) - 最常用的函数逼近器
    2. 权重初始化 - 影响训练稳定性的关键因素
    3. 激活函数 - 引入非线性

数学原理 (Mathematical Theory):
    多层感知机 (MLP):
        h_1 = σ(W_1 x + b_1)
        h_2 = σ(W_2 h_1 + b_2)
        ...
        y = W_L h_{L-1} + b_L

    其中 σ 是激活函数，常用选择:
        - ReLU: max(0, x)，简单高效，可能导致死神经元
        - Tanh: (e^x - e^{-x})/(e^x + e^{-x})，输出有界，梯度消失
        - ELU: x if x > 0 else α(e^x - 1)，平滑，负值有梯度

    权重初始化:
        正交初始化 (Orthogonal Initialization):
            W = Q · diag(gain)，其中 Q 是正交矩阵

        Xavier/Glorot 初始化:
            W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))

        Kaiming/He 初始化 (ReLU):
            W ~ N(0, √(2/n_in))

问题背景 (Problem Statement):
    深度强化学习中，网络初始化对训练稳定性至关重要:
    1. 初始策略应接近均匀随机（探索）
    2. 初始价值估计应接近零（无先验偏见）
    3. 梯度应在合理范围内（避免爆炸/消失）

算法对比 (Comparison):
    | 初始化方法    | 适用激活函数 | 特点                    |
    |---------------|--------------|-------------------------|
    | Xavier        | Tanh, Sigmoid| 保持方差，适合饱和激活  |
    | Kaiming       | ReLU, ELU    | 考虑ReLU的非对称性      |
    | Orthogonal    | 任意         | 保持梯度范数，RL常用    |

复杂度 (Complexity):
    MLP前向传播: O(Σ_l n_l · n_{l+1})
    MLP参数量: O(Σ_l n_l · n_{l+1})

References:
    [1] Glorot & Bengio (2010). Understanding the difficulty of training DNNs.
    [2] He et al. (2015). Delving Deep into Rectifiers.
    [3] Saxe et al. (2014). Exact solutions to the nonlinear dynamics of learning.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    根据名称获取激活函数模块。

    Parameters
    ----------
    name : str
        激活函数名称，支持:
        - "relu": ReLU
        - "tanh": Tanh
        - "elu": ELU
        - "leaky_relu": LeakyReLU
        - "selu": SELU
        - "gelu": GELU
        - "swish" / "silu": Swish/SiLU

    Returns
    -------
    nn.Module
        对应的激活函数模块

    Raises
    ------
    ValueError
        当激活函数名称不支持时

    Examples
    --------
    >>> act = get_activation("relu")
    >>> x = torch.randn(10)
    >>> y = act(x)
    """
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "elu": nn.ELU(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "selu": nn.SELU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
        "softplus": nn.Softplus(),
        "identity": nn.Identity(),
    }

    name_lower = name.lower()
    if name_lower not in activations:
        raise ValueError(
            f"Unknown activation: {name}. "
            f"Supported: {list(activations.keys())}"
        )

    return activations[name_lower]


def init_weights(
    module: nn.Module,
    gain: float = 1.0,
    bias_const: float = 0.0,
    method: str = "orthogonal",
) -> None:
    """
    初始化神经网络层的权重。

    核心思想 (Core Idea):
        合适的初始化确保:
        1. 前向传播时激活值方差稳定
        2. 反向传播时梯度方差稳定
        3. 初始策略/价值函数行为合理

    数学原理 (Mathematical Theory):
        正交初始化:
            对于权重矩阵 W ∈ R^{m×n}:
            1. 生成随机矩阵 A ~ N(0, 1)
            2. QR分解: A = QR
            3. W = gain · Q[:m, :n]

            性质: W^T W ≈ I (当 m ≤ n)
            优点: 保持梯度范数，适合深层网络

        Xavier初始化:
            W ~ U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))
            或 W ~ N(0, 2/(n_in + n_out))

            推导: 保持 Var(output) = Var(input)

        Kaiming初始化:
            W ~ N(0, 2/n_in) (ReLU)
            考虑ReLU将一半激活置零

    Parameters
    ----------
    module : nn.Module
        要初始化的模块（通常是nn.Linear）

    gain : float, default=1.0
        缩放因子，用于调整初始权重范围
        - 输出层: 0.01 (小初始值，接近零)
        - 隐藏层 + ReLU: √2 ≈ 1.414
        - 隐藏层 + Tanh: 5/3 ≈ 1.667

    bias_const : float, default=0.0
        偏置初始化常数

    method : str, default="orthogonal"
        初始化方法: "orthogonal", "xavier", "kaiming"

    Examples
    --------
    >>> linear = nn.Linear(64, 32)
    >>> init_weights(linear, gain=np.sqrt(2))  # ReLU层
    >>> init_weights(linear, gain=0.01)  # 输出层

    Notes
    -----
    策略梯度中的初始化建议:
        - 隐藏层: orthogonal, gain=√2 (ReLU)
        - 策略输出层: orthogonal, gain=0.01 (小初始值)
        - 价值输出层: orthogonal, gain=1.0
    """
    if isinstance(module, nn.Linear):
        if method == "orthogonal":
            nn.init.orthogonal_(module.weight, gain=gain)
        elif method == "xavier":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        else:
            raise ValueError(f"Unknown init method: {method}")

        if module.bias is not None:
            nn.init.constant_(module.bias, bias_const)


class MLP(nn.Module):
    """
    多层感知机 (Multi-Layer Perceptron)。

    核心思想 (Core Idea):
        MLP是最基础的函数逼近器，通过堆叠线性层和非线性激活函数
        来逼近任意连续函数（万能逼近定理）。

    数学原理 (Mathematical Theory):
        前向传播:
            h_0 = x
            h_l = σ(W_l h_{l-1} + b_l),  l = 1, ..., L-1
            y = W_L h_{L-1} + b_L  (或 σ_out(W_L h_{L-1} + b_L))

        万能逼近定理 (Universal Approximation):
            单隐藏层MLP可以以任意精度逼近任意连续函数，
            但可能需要指数级宽度。深层网络更高效。

        表达能力:
            - 深度: 增加表达能力，可能导致梯度问题
            - 宽度: 增加容量，可能导致过拟合
            - 激活函数: 决定非线性类型

    Parameters
    ----------
    input_dim : int
        输入维度

    output_dim : int
        输出维度

    hidden_dims : List[int]
        隐藏层维度列表，如 [64, 64] 表示两个64维隐藏层

    activation : str or nn.Module, default="relu"
        隐藏层激活函数

    output_activation : str or nn.Module or None, default=None
        输出层激活函数，None表示线性输出

    init_method : str, default="orthogonal"
        权重初始化方法

    layer_norm : bool, default=False
        是否使用层归一化

    dropout : float, default=0.0
        Dropout概率

    Attributes
    ----------
    layers : nn.Sequential
        网络层序列

    Examples
    --------
    >>> # 策略网络 (输出logits)
    >>> policy_net = MLP(
    ...     input_dim=4,
    ...     output_dim=2,
    ...     hidden_dims=[64, 64],
    ...     activation="relu",
    ...     output_activation=None
    ... )

    >>> # 价值网络 (输出标量)
    >>> value_net = MLP(
    ...     input_dim=4,
    ...     output_dim=1,
    ...     hidden_dims=[64, 64],
    ...     activation="relu"
    ... )

    >>> # 前向传播
    >>> x = torch.randn(32, 4)  # batch of states
    >>> logits = policy_net(x)  # (32, 2)
    >>> values = value_net(x)   # (32, 1)

    Notes
    -----
    复杂度分析:
        设隐藏层维度为 h，层数为 L:
        - 参数量: O(input_dim * h + (L-1) * h² + h * output_dim)
        - 前向传播: O(batch * 参数量)
        - 内存: O(batch * max(h, input_dim, output_dim))

    架构选择指南:
        | 任务复杂度 | 推荐架构          | 参数量级    |
        |------------|-------------------|-------------|
        | 简单       | [64, 64]          | ~10K        |
        | 中等       | [256, 256]        | ~100K       |
        | 复杂       | [512, 512, 512]   | ~1M         |
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: Union[str, nn.Module] = "relu",
        output_activation: Optional[Union[str, nn.Module]] = None,
        init_method: str = "orthogonal",
        layer_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # 获取激活函数
        if isinstance(activation, str):
            activation = get_activation(activation)

        # 构建网络层
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # 线性层
            linear = nn.Linear(prev_dim, hidden_dim)
            init_weights(linear, gain=np.sqrt(2), method=init_method)
            layers.append(linear)

            # 层归一化（可选）
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            # 激活函数
            layers.append(activation)

            # Dropout（可选）
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # 输出层
        output_layer = nn.Linear(prev_dim, output_dim)
        # 输出层使用较小的初始化
        init_weights(output_layer, gain=0.01, method=init_method)
        layers.append(output_layer)

        # 输出激活函数（可选）
        if output_activation is not None:
            if isinstance(output_activation, str):
                output_activation = get_activation(output_activation)
            layers.append(output_activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状为 (batch_size, input_dim)

        Returns
        -------
        torch.Tensor
            输出张量，形状为 (batch_size, output_dim)
        """
        return self.layers(x)

    def get_num_params(self) -> int:
        """返回可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MLP(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"params={self.get_num_params():,})"
        )


class NoisyLinear(nn.Module):
    """
    噪声线性层 (Noisy Linear Layer)。

    核心思想 (Core Idea):
        在权重中添加可学习的噪声，实现参数空间探索。
        相比ε-greedy，噪声网络提供状态依赖的探索。

    数学原理 (Mathematical Theory):
        标准线性层:
            y = Wx + b

        噪声线性层:
            y = (μ_w + σ_w ⊙ ε_w)x + (μ_b + σ_b ⊙ ε_b)

        其中:
            - μ_w, μ_b: 可学习的均值参数
            - σ_w, σ_b: 可学习的标准差参数
            - ε_w, ε_b: 噪声（训练时采样，评估时为0）

        Factorized Gaussian Noise:
            ε_w = f(ε_i) ⊗ f(ε_j)
            f(x) = sign(x)√|x|

            优点: 参数量从 O(mn) 降到 O(m+n)

    Parameters
    ----------
    in_features : int
        输入特征数

    out_features : int
        输出特征数

    sigma_init : float, default=0.5
        噪声标准差初始值

    factorized : bool, default=True
        是否使用分解噪声

    References
    ----------
    [1] Fortunato et al. (2018). Noisy Networks for Exploration. ICLR.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
        factorized: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.factorized = factorized

        # 可学习参数
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # 噪声缓冲区
        if factorized:
            self.register_buffer("epsilon_i", torch.empty(in_features))
            self.register_buffer("epsilon_j", torch.empty(out_features))
        else:
            self.register_buffer("epsilon_w", torch.empty(out_features, in_features))
            self.register_buffer("epsilon_b", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """初始化参数。"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        sigma_init = self.sigma_init / np.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_init)
        self.bias_sigma.data.fill_(sigma_init)

    def reset_noise(self) -> None:
        """重新采样噪声。"""
        if self.factorized:
            epsilon_i = self._scale_noise(self.in_features)
            epsilon_j = self._scale_noise(self.out_features)
            self.epsilon_i.copy_(epsilon_i)
            self.epsilon_j.copy_(epsilon_j)
        else:
            self.epsilon_w.copy_(torch.randn(self.out_features, self.in_features))
            self.epsilon_b.copy_(torch.randn(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """生成分解噪声。"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        if self.training:
            if self.factorized:
                epsilon_w = self.epsilon_j.outer(self.epsilon_i)
                epsilon_b = self.epsilon_j
            else:
                epsilon_w = self.epsilon_w
                epsilon_b = self.epsilon_b

            weight = self.weight_mu + self.weight_sigma * epsilon_w
            bias = self.bias_mu + self.bias_sigma * epsilon_b
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(x, weight, bias)


class ResidualBlock(nn.Module):
    """
    残差块 (Residual Block)。

    核心思想 (Core Idea):
        通过跳跃连接缓解深层网络的梯度消失问题。
        y = F(x) + x

    数学原理 (Mathematical Theory):
        残差学习:
            H(x) = F(x) + x

        梯度流:
            ∂L/∂x = ∂L/∂H · (∂F/∂x + 1)

        即使 ∂F/∂x → 0，梯度仍可通过恒等映射传播。

    Parameters
    ----------
    dim : int
        输入/输出维度

    hidden_dim : int, optional
        隐藏层维度，默认等于dim

    activation : str, default="relu"
        激活函数
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "relu",
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = dim

        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            get_activation(activation),
            nn.Linear(hidden_dim, dim),
        )

        # 初始化
        for module in self.block.modules():
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return x + self.block(x)


# ==================== 模块测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Base Network Components - Unit Tests")
    print("=" * 70)

    # 测试激活函数
    print("\n[1] Testing activation functions...")
    for name in ["relu", "tanh", "elu", "gelu", "swish"]:
        act = get_activation(name)
        x = torch.randn(10)
        y = act(x)
        print(f"    {name}: input shape {x.shape} -> output shape {y.shape}")
    print("    [PASS]")

    # 测试MLP
    print("\n[2] Testing MLP...")
    mlp = MLP(
        input_dim=4,
        output_dim=2,
        hidden_dims=[64, 64],
        activation="relu",
    )
    x = torch.randn(32, 4)
    y = mlp(x)
    assert y.shape == (32, 2), f"Expected (32, 2), got {y.shape}"
    print(f"    MLP: {mlp}")
    print(f"    Input: {x.shape} -> Output: {y.shape}")
    print("    [PASS]")

    # 测试权重初始化
    print("\n[3] Testing weight initialization...")
    linear = nn.Linear(64, 32)
    init_weights(linear, gain=np.sqrt(2), method="orthogonal")
    # 检查正交性
    W = linear.weight.data
    WWT = W @ W.T
    identity = torch.eye(32) * 2  # gain^2 = 2
    error = (WWT - identity).abs().mean().item()
    print(f"    Orthogonality error: {error:.6f}")
    assert error < 0.1, "Orthogonal initialization failed"
    print("    [PASS]")

    # 测试NoisyLinear
    print("\n[4] Testing NoisyLinear...")
    noisy = NoisyLinear(64, 32, factorized=True)
    x = torch.randn(16, 64)

    # 训练模式（有噪声）
    noisy.train()
    y1 = noisy(x)
    noisy.reset_noise()
    y2 = noisy(x)
    assert not torch.allclose(y1, y2), "Noise should change output"
    print(f"    Training mode: outputs differ (noise active)")

    # 评估模式（无噪声）
    noisy.eval()
    y3 = noisy(x)
    y4 = noisy(x)
    assert torch.allclose(y3, y4), "Eval mode should be deterministic"
    print(f"    Eval mode: outputs identical (no noise)")
    print("    [PASS]")

    # 测试ResidualBlock
    print("\n[5] Testing ResidualBlock...")
    res_block = ResidualBlock(dim=64, hidden_dim=128)
    x = torch.randn(16, 64)
    y = res_block(x)
    assert y.shape == x.shape, "Residual block should preserve shape"
    print(f"    Input: {x.shape} -> Output: {y.shape}")
    print("    [PASS]")

    # 测试梯度流
    print("\n[6] Testing gradient flow...")
    mlp = MLP(input_dim=4, output_dim=1, hidden_dims=[64, 64, 64])
    x = torch.randn(8, 4, requires_grad=True)
    y = mlp(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should flow"
    grad_norm = x.grad.norm().item()
    print(f"    Gradient norm: {grad_norm:.6f}")
    print("    [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
