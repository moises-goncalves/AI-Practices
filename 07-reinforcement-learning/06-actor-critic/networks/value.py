"""
价值网络模块 (Value Networks Module)

本模块实现策略梯度算法所需的价值函数网络。

核心思想 (Core Idea):
    价值网络估计状态或状态-动作对的期望回报，用于:
    1. 作为基线减少方差 (REINFORCE with Baseline)
    2. 作为Critic提供TD目标 (Actor-Critic)
    3. 计算优势函数 (A2C, PPO)

数学原理 (Mathematical Theory):
    状态价值函数 V(s):
        V^π(s) = E_{π}[Σ_{t=0}^∞ γ^t r_t | s_0 = s]
               = E_{a~π}[Q^π(s,a)]

    动作价值函数 Q(s,a):
        Q^π(s,a) = E_{π}[Σ_{t=0}^∞ γ^t r_t | s_0 = s, a_0 = a]
                 = r(s,a) + γ E_{s'}[V^π(s')]

    优势函数 A(s,a):
        A^π(s,a) = Q^π(s,a) - V^π(s)
                 ≈ r + γV(s') - V(s)  (TD误差近似)

    价值函数训练目标:
        MC目标: L = E[(V_θ(s) - G_t)²]
        TD目标: L = E[(V_θ(s) - (r + γV_θ(s')))²]
        GAE目标: L = E[(V_θ(s) - (A^GAE + V_θ(s)))²]

问题背景 (Problem Statement):
    方差减少的重要性:
        - 无基线: Var[∇J] ∝ E[Q²]
        - 有基线: Var[∇J] ∝ E[A²] << E[Q²]

    最优基线 b*(s) = E[Q(s,a)²] / E[Q(s,a)] ≈ V(s)

算法对比 (Comparison):
    | 网络架构      | 参数量 | 特征共享 | 适用场景        |
    |---------------|--------|----------|-----------------|
    | 独立V网络     | 1x     | 无       | REINFORCE+BL    |
    | 独立Q网络     | 1x     | 无       | DQN, SAC        |
    | 共享AC网络    | ~1.2x  | 是       | A2C, PPO        |
    | 双头AC网络    | ~1.5x  | 部分     | 复杂任务        |

References:
    [1] Sutton & Barto (2018). RL: An Introduction. Chapter 13.
    [2] Mnih et al. (2016). A3C - Asynchronous Advantage Actor-Critic.
    [3] Schulman et al. (2016). GAE - High-dimensional continuous control.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from .base import MLP, init_weights, get_activation


class ValueNetwork(nn.Module):
    """
    状态价值函数网络 V(s)。

    核心思想 (Core Idea):
        估计从状态s出发，遵循当前策略的期望累积回报。
        作为基线减少策略梯度的方差。

    数学原理 (Mathematical Theory):
        价值函数定义:
            V^π(s) = E_{π}[G_t | s_t = s]
                   = E_{π}[Σ_{k=0}^∞ γ^k r_{t+k} | s_t = s]

        Bellman方程:
            V^π(s) = E_{a~π}[r(s,a) + γ E_{s'}[V^π(s')]]

        训练目标 (MSE Loss):
            L(φ) = E[(V_φ(s) - V_target)²]

            其中 V_target 可以是:
            - MC回报: G_t = Σ_k γ^k r_{t+k}
            - TD目标: r + γV(s')
            - GAE回报: A^GAE + V(s)

    Parameters
    ----------
    state_dim : int
        状态空间维度
    hidden_dims : List[int], default=[64, 64]
        隐藏层维度
    activation : str, default="relu"
        激活函数

    Examples
    --------
    >>> value_net = ValueNetwork(state_dim=4)
    >>> state = torch.randn(32, 4)
    >>> values = value_net(state)
    >>> print(f"Values shape: {values.shape}")  # (32, 1)

    Notes
    -----
    复杂度:
        - 前向传播: O(batch × network_params)
        - 参数量: O(state_dim × h + h² × (L-1) + h)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.state_dim = state_dim

        self.net = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=None,
        )

        # 价值输出层使用较大的初始化增益
        for module in reversed(list(self.net.modules())):
            if isinstance(module, nn.Linear):
                init_weights(module, gain=1.0)
                break

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        估计状态价值。

        Parameters
        ----------
        state : torch.Tensor
            状态，形状 (batch, state_dim)

        Returns
        -------
        torch.Tensor
            价值估计，形状 (batch, 1)
        """
        return self.net(state)


class QNetwork(nn.Module):
    """
    动作价值函数网络 Q(s,a)。

    核心思想 (Core Idea):
        估计在状态s执行动作a后，遵循策略的期望累积回报。
        用于DQN、SAC等算法。

    数学原理 (Mathematical Theory):
        Q函数定义:
            Q^π(s,a) = E_{π}[G_t | s_t = s, a_t = a]

        Bellman方程:
            Q^π(s,a) = r(s,a) + γ E_{s',a'}[Q^π(s',a')]

        与V的关系:
            V^π(s) = E_{a~π}[Q^π(s,a)]
            A^π(s,a) = Q^π(s,a) - V^π(s)

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

    Examples
    --------
    >>> q_net = QNetwork(state_dim=3, action_dim=1)
    >>> state = torch.randn(32, 3)
    >>> action = torch.randn(32, 1)
    >>> q_values = q_net(state, action)
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
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 状态和动作拼接后输入
        self.net = MLP(
            input_dim=state_dim + action_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=None,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        估计Q值。

        Parameters
        ----------
        state : torch.Tensor
            状态，形状 (batch, state_dim)
        action : torch.Tensor
            动作，形状 (batch, action_dim)

        Returns
        -------
        torch.Tensor
            Q值，形状 (batch, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ActorCriticNetwork(nn.Module):
    """
    共享特征的Actor-Critic网络。

    核心思想 (Core Idea):
        Actor和Critic共享底层特征提取网络，分别输出策略和价值。
        参数共享提高效率，但可能导致任务干扰。

    数学原理 (Mathematical Theory):
        网络结构:
            features = f_shared(s)
            π(a|s) = Actor_head(features)
            V(s) = Critic_head(features)

        联合损失:
            L = L_policy + c_v · L_value - c_ent · H(π)

        其中:
            L_policy = -E[log π(a|s) · A]
            L_value = E[(V(s) - V_target)²]
            H(π) = -E[Σ_a π(a|s) log π(a|s)]

    架构图:
        ```
        state ─► [shared_net] ─► features
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                   [actor_head]           [critic_head]
                        │                       │
                        ▼                       ▼
                     policy                   value
        ```

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
    continuous : bool, default=False
        是否连续动作空间
    log_std_init : float, default=0.0
        连续动作的初始log标准差

    Examples
    --------
    >>> # 离散动作
    >>> ac_net = ActorCriticNetwork(state_dim=4, action_dim=2)
    >>> state = torch.randn(32, 4)
    >>> action, log_prob, entropy, value = ac_net.get_action_and_value(state)

    >>> # 连续动作
    >>> ac_net = ActorCriticNetwork(state_dim=3, action_dim=1, continuous=True)
    >>> action, log_prob, entropy, value = ac_net.get_action_and_value(state)

    Notes
    -----
    设计权衡:
        共享网络:
            + 参数效率高（约60%参数量）
            + 特征复用，加速学习
            - 可能存在梯度干扰
            - 需要仔细调整损失系数

        分离网络:
            + 独立优化，更稳定
            + 可以使用不同架构
            - 参数量翻倍
            - 无法共享特征
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        continuous: bool = False,
        log_std_init: float = 0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # 获取激活函数
        act_fn = get_activation(activation)

        # 共享特征网络
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            prev_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Actor头
        if continuous:
            self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
            self.actor_log_std = nn.Parameter(
                torch.full((action_dim,), log_std_init)
            )
            init_weights(self.actor_mean, gain=0.01)
        else:
            self.actor_head = nn.Linear(hidden_dims[-1], action_dim)
            init_weights(self.actor_head, gain=0.01)

        # Critic头
        self.critic_head = nn.Linear(hidden_dims[-1], 1)
        init_weights(self.critic_head, gain=1.0)

        # 初始化共享层
        for module in self.shared_net.modules():
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，返回策略参数和价值。

        Returns
        -------
        policy_output : torch.Tensor
            离散: logits (batch, action_dim)
            连续: mean (batch, action_dim)
        value : torch.Tensor
            价值估计 (batch, 1)
        """
        features = self.shared_net(state)
        value = self.critic_head(features)

        if self.continuous:
            mean = self.actor_mean(features)
            return mean, value
        else:
            logits = self.actor_head(features)
            return logits, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """仅获取价值估计。"""
        features = self.shared_net(state)
        return self.critic_head(features).squeeze(-1)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作、对数概率、熵和价值。

        Parameters
        ----------
        state : torch.Tensor
            状态，形状 (batch, state_dim)
        action : torch.Tensor, optional
            给定动作（用于评估）
        deterministic : bool
            是否确定性选择

        Returns
        -------
        action : torch.Tensor
            动作
        log_prob : torch.Tensor
            对数概率
        entropy : torch.Tensor
            策略熵
        value : torch.Tensor
            价值估计
        """
        features = self.shared_net(state)
        value = self.critic_head(features).squeeze(-1)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            dist = Normal(mean, std)

            if action is None:
                if deterministic:
                    action = mean
                else:
                    action = dist.rsample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.actor_head(features)
            dist = Categorical(logits=logits)

            if action is None:
                if deterministic:
                    action = dist.probs.argmax(dim=-1)
                else:
                    action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作（用于PPO更新）。

        Returns
        -------
        log_prob : torch.Tensor
            动作对数概率
        entropy : torch.Tensor
            策略熵
        value : torch.Tensor
            价值估计
        """
        _, log_prob, entropy, value = self.get_action_and_value(
            state, action=action
        )
        return log_prob, entropy, value


class DualHeadActorCritic(nn.Module):
    """
    双头Actor-Critic网络（部分共享）。

    核心思想 (Core Idea):
        底层特征共享，高层分离。平衡参数效率和任务独立性。

    架构:
        ```
        state ─► [shared_base] ─► base_features
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
               [actor_layers]                        [critic_layers]
                    │                                      │
               [actor_head]                          [critic_head]
                    │                                      │
                    ▼                                      ▼
                 policy                                  value
        ```

    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        动作空间维度
    shared_dims : List[int], default=[64]
        共享层维度
    actor_dims : List[int], default=[64]
        Actor专用层维度
    critic_dims : List[int], default=[64]
        Critic专用层维度
    continuous : bool, default=False
        是否连续动作空间
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        shared_dims: List[int] = None,
        actor_dims: List[int] = None,
        critic_dims: List[int] = None,
        activation: str = "relu",
        continuous: bool = False,
    ):
        super().__init__()

        if shared_dims is None:
            shared_dims = [64]
        if actor_dims is None:
            actor_dims = [64]
        if critic_dims is None:
            critic_dims = [64]

        self.continuous = continuous
        self.action_dim = action_dim

        # 共享基础网络
        self.shared_base = MLP(
            input_dim=state_dim,
            output_dim=shared_dims[-1],
            hidden_dims=shared_dims[:-1] if len(shared_dims) > 1 else [],
            activation=activation,
            output_activation=activation,
        )

        # Actor专用层
        self.actor_net = MLP(
            input_dim=shared_dims[-1],
            output_dim=actor_dims[-1],
            hidden_dims=actor_dims[:-1] if len(actor_dims) > 1 else [],
            activation=activation,
            output_activation=activation,
        )

        # Critic专用层
        self.critic_net = MLP(
            input_dim=shared_dims[-1],
            output_dim=critic_dims[-1],
            hidden_dims=critic_dims[:-1] if len(critic_dims) > 1 else [],
            activation=activation,
            output_activation=activation,
        )

        # 输出头
        if continuous:
            self.actor_mean = nn.Linear(actor_dims[-1], action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
            init_weights(self.actor_mean, gain=0.01)
        else:
            self.actor_head = nn.Linear(actor_dims[-1], action_dim)
            init_weights(self.actor_head, gain=0.01)

        self.critic_head = nn.Linear(critic_dims[-1], 1)
        init_weights(self.critic_head, gain=1.0)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """获取价值估计。"""
        base_features = self.shared_base(state)
        critic_features = self.critic_net(base_features)
        return self.critic_head(critic_features).squeeze(-1)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作和价值。"""
        base_features = self.shared_base(state)

        # Actor分支
        actor_features = self.actor_net(base_features)
        if self.continuous:
            mean = self.actor_mean(actor_features)
            std = self.actor_log_std.exp()
            dist = Normal(mean, std)

            if action is None:
                action = mean if deterministic else dist.rsample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.actor_head(actor_features)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        # Critic分支
        critic_features = self.critic_net(base_features)
        value = self.critic_head(critic_features).squeeze(-1)

        return action, log_prob, entropy, value


# ==================== 模块测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Value Networks - Unit Tests")
    print("=" * 70)

    # 测试ValueNetwork
    print("\n[1] Testing ValueNetwork...")
    value_net = ValueNetwork(state_dim=4, hidden_dims=[64, 64])
    state = torch.randn(32, 4)
    values = value_net(state)
    assert values.shape == (32, 1), f"Expected (32, 1), got {values.shape}"
    print(f"    Values shape: {values.shape}")
    print(f"    Values range: [{values.min():.3f}, {values.max():.3f}]")
    print("    [PASS]")

    # 测试QNetwork
    print("\n[2] Testing QNetwork...")
    q_net = QNetwork(state_dim=3, action_dim=1)
    state = torch.randn(32, 3)
    action = torch.randn(32, 1)
    q_values = q_net(state, action)
    assert q_values.shape == (32, 1), f"Expected (32, 1), got {q_values.shape}"
    print(f"    Q-values shape: {q_values.shape}")
    print("    [PASS]")

    # 测试ActorCriticNetwork (离散)
    print("\n[3] Testing ActorCriticNetwork (discrete)...")
    ac_net = ActorCriticNetwork(state_dim=4, action_dim=2, continuous=False)
    state = torch.randn(32, 4)

    action, log_prob, entropy, value = ac_net.get_action_and_value(state)
    assert action.shape == (32,), f"Expected (32,), got {action.shape}"
    assert log_prob.shape == (32,), f"Expected (32,), got {log_prob.shape}"
    assert value.shape == (32,), f"Expected (32,), got {value.shape}"
    print(f"    Action shape: {action.shape}")
    print(f"    Value shape: {value.shape}")

    # 测试evaluate_actions
    log_prob2, entropy2, value2 = ac_net.evaluate_actions(state, action)
    assert torch.allclose(log_prob, log_prob2, atol=1e-5)
    print("    Evaluate consistency: passed")
    print("    [PASS]")

    # 测试ActorCriticNetwork (连续)
    print("\n[4] Testing ActorCriticNetwork (continuous)...")
    ac_net = ActorCriticNetwork(state_dim=3, action_dim=2, continuous=True)
    state = torch.randn(32, 3)

    action, log_prob, entropy, value = ac_net.get_action_and_value(state)
    assert action.shape == (32, 2), f"Expected (32, 2), got {action.shape}"
    print(f"    Action shape: {action.shape}")
    print(f"    Action range: [{action.min():.3f}, {action.max():.3f}]")
    print("    [PASS]")

    # 测试DualHeadActorCritic
    print("\n[5] Testing DualHeadActorCritic...")
    dual_ac = DualHeadActorCritic(
        state_dim=4,
        action_dim=2,
        shared_dims=[64],
        actor_dims=[64],
        critic_dims=[64],
    )
    state = torch.randn(32, 4)

    action, log_prob, entropy, value = dual_ac.get_action_and_value(state)
    assert action.shape == (32,)
    assert value.shape == (32,)
    print(f"    Action shape: {action.shape}")
    print(f"    Value shape: {value.shape}")
    print("    [PASS]")

    # 测试梯度流
    print("\n[6] Testing gradient flow...")
    ac_net = ActorCriticNetwork(state_dim=4, action_dim=2)
    state = torch.randn(8, 4, requires_grad=True)
    action, log_prob, entropy, value = ac_net.get_action_and_value(state)

    # 模拟PPO损失
    advantages = torch.randn(8)
    returns = torch.randn(8)

    policy_loss = -(log_prob * advantages).mean()
    value_loss = ((value - returns) ** 2).mean()
    entropy_loss = -entropy.mean()

    total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
    total_loss.backward()

    assert state.grad is not None
    print(f"    Total loss: {total_loss.item():.4f}")
    print(f"    Gradient norm: {state.grad.norm():.6f}")
    print("    [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
