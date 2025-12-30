"""
策略网络模块 (Policy Networks Module)

本模块实现策略梯度算法所需的策略网络架构。

核心思想 (Core Idea):
    策略网络 π_θ(a|s) 将状态映射到动作分布，是策略梯度方法的核心组件。
    不同动作空间需要不同的分布参数化方式。

数学原理 (Mathematical Theory):
    策略梯度定理:
        ∇_θ J(θ) = E_{π_θ}[∇_θ log π_θ(a|s) · Q^π(s,a)]

    离散动作空间 - Categorical分布:
        π_θ(a|s) = softmax(f_θ(s))_a = exp(z_a) / Σ_a' exp(z_a')
        log π_θ(a|s) = z_a - log Σ_a' exp(z_a')
        H(π) = -Σ_a π(a|s) log π(a|s)

    连续动作空间 - Gaussian分布:
        π_θ(a|s) = N(a | μ_θ(s), σ_θ(s)²)
        log π_θ(a|s) = -1/2 [(a-μ)²/σ² + log(2πσ²)]
        H(π) = 1/2 log(2πeσ²)

    有界连续动作 - Squashed Gaussian:
        u ~ N(μ_θ(s), σ_θ(s)²)
        a = tanh(u) ∈ (-1, 1)
        log π(a|s) = log N(u|μ,σ²) - Σ_i log(1 - a_i²)

问题背景 (Problem Statement):
    | 动作空间类型 | 分布选择        | 网络输出        | 采样方法      |
    |--------------|-----------------|-----------------|---------------|
    | 离散有限     | Categorical     | logits          | Gumbel-max    |
    | 连续无界     | Gaussian        | μ, log σ        | 重参数化      |
    | 连续有界     | Squashed Gauss  | μ, log σ + tanh | 重参数化+变换 |

References:
    [1] Williams (1992). REINFORCE algorithm.
    [2] Haarnoja et al. (2018). SAC - Squashed Gaussian policy.
    [3] Schulman et al. (2017). PPO - Policy optimization.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .base import MLP, init_weights, get_activation


class DiscretePolicy(nn.Module):
    """
    离散动作空间策略网络。

    核心思想 (Core Idea):
        输出动作logits，通过Categorical分布采样离散动作。
        Softmax将logits转换为概率，Categorical处理采样和log_prob计算。

    数学原理 (Mathematical Theory):
        Softmax策略:
            π_θ(a|s) = exp(z_a) / Σ_a' exp(z_a')

        对数概率 (数值稳定):
            log π_θ(a|s) = z_a - logsumexp(z)

        熵:
            H(π) = -Σ_a π(a|s) log π(a|s)
                 = logsumexp(z) - Σ_a π(a|s) z_a

        梯度:
            ∇_θ log π_θ(a|s) = ∇_θ z_a - E_{a'~π}[∇_θ z_a']
                             = ∇_θ z_a - Σ_a' π(a'|s) ∇_θ z_a'

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作数量（离散动作个数）
    hidden_dims : List[int], default=[64, 64]
        隐藏层维度
    activation : str, default="relu"
        激活函数

    Attributes
    ----------
    net : MLP
        状态到logits的映射网络

    Examples
    --------
    >>> policy = DiscretePolicy(state_dim=4, action_dim=2)
    >>> state = torch.randn(32, 4)
    >>> action, log_prob, entropy = policy.sample(state)
    >>> print(f"Action shape: {action.shape}")  # (32,)

    Notes
    -----
    复杂度:
        - 前向传播: O(batch × network_params)
        - 采样: O(batch × action_dim)
        - 内存: O(batch × max(hidden_dim, action_dim))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=None,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """计算动作logits。"""
        return self.net(state)

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """获取动作分布。"""
        logits = self.forward(state)
        return Categorical(logits=logits)

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作。

        Parameters
        ----------
        state : torch.Tensor
            状态，形状 (batch, state_dim)
        deterministic : bool
            是否确定性选择（argmax）

        Returns
        -------
        action : torch.Tensor
            动作，形状 (batch,)
        log_prob : torch.Tensor
            对数概率，形状 (batch,)
        entropy : torch.Tensor
            熵，形状 (batch,)
        """
        dist = self.get_distribution(state)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作的对数概率。

        用于PPO等需要重要性采样的算法。
        """
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_probs(self, state: torch.Tensor) -> torch.Tensor:
        """获取动作概率分布。"""
        return self.get_distribution(state).probs


class ContinuousPolicy(nn.Module):
    """
    连续动作空间策略网络（Gaussian分布）。

    核心思想 (Core Idea):
        输出Gaussian分布的均值和标准差，通过重参数化技巧采样。
        标准差可以是状态无关的可学习参数，或状态相关的网络输出。

    数学原理 (Mathematical Theory):
        Gaussian策略:
            π_θ(a|s) = N(a | μ_θ(s), σ_θ(s)²)
                     = (2πσ²)^{-d/2} exp(-||a-μ||²/(2σ²))

        对数概率:
            log π_θ(a|s) = -d/2 log(2π) - Σ_i [log σ_i + (a_i-μ_i)²/(2σ_i²)]

        熵:
            H(π) = d/2 (1 + log(2π)) + Σ_i log σ_i

        重参数化技巧 (Reparameterization Trick):
            a = μ + σ ⊙ ε,  ε ~ N(0, I)
            允许梯度通过采样操作反向传播

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作空间维度
    hidden_dims : List[int], default=[64, 64]
        隐藏层维度
    activation : str, default="relu"
        激活函数
    state_dependent_std : bool, default=False
        标准差是否依赖状态
    log_std_init : float, default=0.0
        log标准差初始值
    log_std_min : float, default=-20.0
        log标准差下界
    log_std_max : float, default=2.0
        log标准差上界

    Examples
    --------
    >>> policy = ContinuousPolicy(state_dim=3, action_dim=1)
    >>> state = torch.randn(32, 3)
    >>> action, log_prob, entropy = policy.sample(state)
    >>> print(f"Action shape: {action.shape}")  # (32, 1)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        state_dependent_std: bool = False,
        log_std_init: float = 0.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 特征提取网络
        self.feature_net = MLP(
            input_dim=state_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=activation,
            output_activation=activation,
        )

        # 均值输出层
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        init_weights(self.mean_layer, gain=0.01)

        # 标准差
        if state_dependent_std:
            self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
            init_weights(self.log_std_layer, gain=0.01)
        else:
            self.log_std = nn.Parameter(
                torch.full((action_dim,), log_std_init)
            )

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算分布参数。

        Returns
        -------
        mean : torch.Tensor
            均值，形状 (batch, action_dim)
        std : torch.Tensor
            标准差，形状 (batch, action_dim)
        """
        features = self.feature_net(state)
        mean = self.mean_layer(features)

        if self.state_dependent_std:
            log_std = self.log_std_layer(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        return mean, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """获取动作分布。"""
        mean, std = self.forward(state)
        return Normal(mean, std)

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样动作。"""
        mean, std = self.forward(state)

        if deterministic:
            action = mean
            log_prob = torch.zeros(state.shape[0], device=state.device)
            entropy = torch.zeros(state.shape[0], device=state.device)
        else:
            dist = Normal(mean, std)
            action = dist.rsample()  # 重参数化采样
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定动作。"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class SquashedGaussianPolicy(nn.Module):
    """
    压缩高斯策略网络（用于有界动作空间）。

    核心思想 (Core Idea):
        通过tanh将无界Gaussian采样压缩到[-1, 1]，
        需要对log_prob进行Jacobian校正。

    数学原理 (Mathematical Theory):
        变换:
            u ~ N(μ_θ(s), σ_θ(s)²)
            a = tanh(u) ∈ (-1, 1)

        Jacobian校正 (变量替换):
            p(a) = p(u) |det(∂u/∂a)|
                 = p(u) / |det(∂a/∂u)|
                 = p(u) / Π_i (1 - tanh²(u_i))

        对数概率:
            log π(a|s) = log N(u|μ,σ²) - Σ_i log(1 - tanh²(u_i))
                       = log N(u|μ,σ²) - Σ_i log(1 - a_i²)

        数值稳定形式:
            log(1 - tanh²(u)) = log(sech²(u)) = 2(log 2 - u - softplus(-2u))

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作空间维度
    hidden_dims : List[int], default=[256, 256]
        隐藏层维度
    activation : str, default="relu"
        激活函数
    log_std_min : float, default=-20.0
        log标准差下界
    log_std_max : float, default=2.0
        log标准差上界

    Examples
    --------
    >>> policy = SquashedGaussianPolicy(state_dim=3, action_dim=1)
    >>> state = torch.randn(32, 3)
    >>> action, log_prob, entropy = policy.sample(state)
    >>> assert action.min() >= -1 and action.max() <= 1

    Notes
    -----
    适用场景:
        - 连续控制任务（MuJoCo等）
        - 动作需要有界的场景
        - SAC等最大熵强化学习算法
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 共享特征网络
        self.feature_net = MLP(
            input_dim=state_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=activation,
            output_activation=activation,
        )

        # 均值和log标准差输出
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        init_weights(self.mean_layer, gain=0.01)
        init_weights(self.log_std_layer, gain=0.01)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算分布参数。"""
        features = self.feature_net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作（带tanh压缩）。

        Returns
        -------
        action : torch.Tensor
            压缩后的动作 ∈ [-1, 1]
        log_prob : torch.Tensor
            校正后的对数概率
        entropy : torch.Tensor
            Gaussian熵（未校正）
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            u = mean
            action = torch.tanh(u)
            log_prob = torch.zeros(state.shape[0], device=state.device)
            entropy = torch.zeros(state.shape[0], device=state.device)
        else:
            dist = Normal(mean, std)
            u = dist.rsample()
            action = torch.tanh(u)

            # Jacobian校正
            log_prob = dist.log_prob(u).sum(dim=-1)
            # 数值稳定的log(1 - tanh²(u))
            log_prob -= (2 * (np.log(2) - u - nn.functional.softplus(-2 * u))).sum(dim=-1)

            entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估给定动作的对数概率。

        Parameters
        ----------
        action : torch.Tensor
            压缩后的动作 ∈ [-1, 1]
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 反tanh得到u
        action_clipped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(action_clipped)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(u).sum(dim=-1)
        # Jacobian校正
        log_prob -= (2 * (np.log(2) - u - nn.functional.softplus(-2 * u))).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy

    def get_action_scale(self) -> float:
        """返回动作缩放因子（tanh输出范围）。"""
        return 1.0


# ==================== 模块测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Policy Networks - Unit Tests")
    print("=" * 70)

    # 测试离散策略
    print("\n[1] Testing DiscretePolicy...")
    policy = DiscretePolicy(state_dim=4, action_dim=3)
    state = torch.randn(32, 4)

    action, log_prob, entropy = policy.sample(state)
    assert action.shape == (32,), f"Expected (32,), got {action.shape}"
    assert log_prob.shape == (32,), f"Expected (32,), got {log_prob.shape}"
    assert entropy.shape == (32,), f"Expected (32,), got {entropy.shape}"
    assert (action >= 0).all() and (action < 3).all(), "Actions out of range"
    print(f"    Action range: [{action.min()}, {action.max()}]")
    print(f"    Log prob range: [{log_prob.min():.3f}, {log_prob.max():.3f}]")
    print(f"    Entropy mean: {entropy.mean():.3f}")

    # 测试evaluate
    log_prob2, entropy2 = policy.evaluate(state, action)
    assert torch.allclose(log_prob, log_prob2, atol=1e-5)
    print("    [PASS]")

    # 测试连续策略
    print("\n[2] Testing ContinuousPolicy...")
    policy = ContinuousPolicy(state_dim=3, action_dim=2)
    state = torch.randn(32, 3)

    action, log_prob, entropy = policy.sample(state)
    assert action.shape == (32, 2), f"Expected (32, 2), got {action.shape}"
    assert log_prob.shape == (32,), f"Expected (32,), got {log_prob.shape}"
    print(f"    Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"    Log prob range: [{log_prob.min():.3f}, {log_prob.max():.3f}]")

    # 测试确定性采样
    action_det, _, _ = policy.sample(state, deterministic=True)
    action_det2, _, _ = policy.sample(state, deterministic=True)
    assert torch.allclose(action_det, action_det2)
    print("    Deterministic sampling: consistent")
    print("    [PASS]")

    # 测试压缩高斯策略
    print("\n[3] Testing SquashedGaussianPolicy...")
    policy = SquashedGaussianPolicy(state_dim=3, action_dim=2)
    state = torch.randn(32, 3)

    action, log_prob, entropy = policy.sample(state)
    assert action.shape == (32, 2), f"Expected (32, 2), got {action.shape}"
    assert (action >= -1).all() and (action <= 1).all(), "Actions out of [-1, 1]"
    print(f"    Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"    Log prob range: [{log_prob.min():.3f}, {log_prob.max():.3f}]")

    # 测试evaluate
    log_prob2, _ = policy.evaluate(state, action)
    assert torch.allclose(log_prob, log_prob2, atol=1e-4)
    print("    Evaluate consistency: passed")
    print("    [PASS]")

    # 测试梯度流
    print("\n[4] Testing gradient flow...")
    policy = DiscretePolicy(state_dim=4, action_dim=2)
    state = torch.randn(8, 4, requires_grad=True)
    action, log_prob, entropy = policy.sample(state)
    loss = -log_prob.mean() - 0.01 * entropy.mean()
    loss.backward()
    assert state.grad is not None
    print(f"    Gradient norm: {state.grad.norm():.6f}")
    print("    [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
