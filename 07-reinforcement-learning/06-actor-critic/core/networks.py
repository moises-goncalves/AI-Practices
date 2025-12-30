"""
Neural Network Architectures for Policy Gradient Methods.

================================================================================
核心思想 (Core Idea)
================================================================================
Policy gradient methods directly parameterize the policy π_θ(a|s) as a neural
network. The network architecture determines the expressiveness and inductive
biases of the learned policy. This module provides:

1. **MLP**: Flexible feedforward backbone for feature extraction
2. **DiscretePolicy**: Categorical distribution over finite action sets
3. **ContinuousPolicy**: Gaussian distribution for continuous control
4. **ValueNetwork**: State value function for baseline/critic
5. **ActorCriticNetwork**: Shared-backbone architecture for efficiency

================================================================================
数学原理 (Mathematical Theory)
================================================================================
Policy Parameterization:
    For state s ∈ S, action a ∈ A:

    Discrete Actions (Softmax):
        π_θ(a|s) = softmax(f_θ(s))_a = exp(z_a) / Σ_{a'} exp(z_{a'})

        where z = f_θ(s) are unnormalized logits.

        Log-probability (numerically stable):
            log π_θ(a|s) = z_a - log Σ_{a'} exp(z_{a'})
                        = z_a - logsumexp(z)

    Continuous Actions (Gaussian):
        π_θ(a|s) = N(a | μ_θ(s), σ_θ(s)²)

        Log-probability:
            log π_θ(a|s) = -1/2 [(a-μ)²/σ² + log(2πσ²)]

        Reparameterization Trick (Kingma & Welling, 2014):
            a = μ_θ(s) + σ_θ(s) · ε,  where ε ~ N(0, I)

            Enables gradient flow through sampling operation.

        Tanh Squashing (for bounded actions in [-1, 1]):
            u ~ N(μ, σ²)
            a = tanh(u)

            Jacobian correction:
                log π(a) = log N(u|μ,σ²) - Σ_i log(1 - a_i²)

Policy Gradient Theorem (Sutton et al., 1999):
    ∇_θ J(θ) = E_{s~ρ^π, a~π_θ}[∇_θ log π_θ(a|s) · Q^π(s,a)]

    The log-probability gradient ∇_θ log π_θ(a|s) is the "score function".

Weight Initialization - Orthogonal (Saxe et al., 2014):
    For weight matrix W ∈ R^{m×n}:
        W = U · diag(gain) · V^T  where U, V are orthogonal

    Properties:
        - Preserves gradient magnitude: ||∂L/∂W_l|| ≈ ||∂L/∂W_{l+1}||
        - Prevents vanishing/exploding gradients in deep networks

================================================================================
问题背景 (Problem Statement)
================================================================================
Challenge: How to represent π(a|s) such that:
    1. Expressive enough to represent complex policies
    2. Differentiable for gradient-based optimization
    3. Capable of both exploration (stochastic) and exploitation (deterministic)

Historical Solutions:
    - Linear function approximation: π(a|s) = softmax(θ^T φ(s))
      Limited expressiveness, requires manual feature engineering

    - Neural networks: π_θ(a|s) = NN_θ(s)
      Universal approximation, end-to-end learning
      Enabled deep reinforcement learning revolution (DQN, 2013)

================================================================================
算法对比 (Comparison)
================================================================================
| Architecture        | Parameters | Features     | Use Case              |
|---------------------|------------|--------------|----------------------|
| Separate Actor+Critic| ~2x       | Independent  | Stable, simple tasks |
| Shared AC (this)    | ~1.2x      | Transfer     | Efficient, complex   |
| CNN-based           | Large      | Spatial      | Image observations   |
| Attention-based     | Very large | Long-range   | Sequence decisions   |

| Distribution  | Action Space   | Bounded | Multimodal |
|---------------|----------------|---------|------------|
| Categorical   | Discrete       | N/A     | Yes        |
| Gaussian      | Continuous     | No      | No         |
| Squashed Gaus.| Cont. bounded  | Yes     | No         |
| GMM           | Continuous     | No      | Yes        |
| Normalizing Flow| Continuous   | Yes     | Yes        |

================================================================================
复杂度 (Complexity Analysis)
================================================================================
MLP Forward Pass:
    Time: O(B × Σ_{l} d_l × d_{l+1})  where B=batch, d_l=layer dims
    Space: O(B × max(d_l))  for activations

Sampling:
    Categorical: O(B × A) for Gumbel-max trick
    Gaussian: O(B × A) for reparameterization
    Squashed: O(B × A) additional for tanh + correction

================================================================================
算法总结 (Summary)
================================================================================
This module implements production-ready neural network policies:

1. **Orthogonal initialization** ensures stable gradient flow
2. **Logit-based categorical** avoids numerical instability
3. **Reparameterization trick** enables efficient gradient estimation
4. **Tanh squashing** handles bounded continuous action spaces
5. **Shared architecture** reduces parameters while enabling feature transfer

The design prioritizes:
- Numerical stability (log-space computations, clamping)
- Modularity (composable building blocks)
- Efficiency (minimal redundant computation)

References
----------
[1] Sutton et al. (1999). Policy Gradient Methods for RL with FA.
[2] Williams (1992). Simple Statistical Gradient-Following Algorithms.
[3] Saxe et al. (2014). Exact Solutions to Nonlinear Dynamics of Learning.
[4] Haarnoja et al. (2018). Soft Actor-Critic (SAC).
[5] Kingma & Welling (2014). Auto-Encoding Variational Bayes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """
    Apply orthogonal initialization to network weights.

    核心思想 (Core Idea):
        Initialize weight matrices to be orthogonal, preserving gradient
        magnitude during backpropagation for stable deep network training.

    数学原理 (Mathematical Theory):
        For orthogonal matrix Q scaled by gain g:
            ||Qx|| = g × ||x||  (norm preservation)

        For L-layer network:
            ||∂L/∂x_0|| ≈ g^L × ||∂L/∂x_L||

        With g = √2 for ReLU, compensates for ~50% dead neurons.

    Parameters
    ----------
    module : nn.Module
        Network module containing weights to initialize.
    gain : float, default=1.0
        Scaling factor for initialization.

        Recommended values:
            - ReLU: gain = √2 ≈ 1.414
            - Tanh: gain = 5/3 ≈ 1.667
            - Linear/Identity: gain = 1.0
            - Policy output: gain = 0.01 (small initial variance)

    Examples
    --------
    >>> layer = nn.Linear(128, 64)
    >>> init_weights(layer, gain=np.sqrt(2))
    >>>
    >>> # Verify orthogonality (approximately)
    >>> W = layer.weight.data
    >>> identity_approx = W @ W.T / (layer.weight.shape[1])
    >>> # Should be close to identity matrix scaled by gain²

    Notes
    -----
    Time Complexity: O(min(m,n)² × max(m,n)) for SVD decomposition
    Space Complexity: O(m × n) for weight matrix storage

    Alternative initializations:
        - Xavier/Glorot: Good for tanh/sigmoid, assumes linear activation
        - He/Kaiming: Designed for ReLU, preserves variance
        - Orthogonal: Best for deep nets, preserves gradient magnitude
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable architecture.

    核心思想 (Core Idea):
        A flexible feedforward network serving as the backbone for policy
        and value function approximation. Supports arbitrary depth and width
        with consistent initialization.

    数学原理 (Mathematical Theory):
        Forward computation:
            h_0 = x
            h_l = σ(W_l h_{l-1} + b_l)  for l = 1, ..., L-1
            y = W_L h_{L-1} + b_L  (or σ_out(W_L h_{L-1} + b_L))

        Universal Approximation (Cybenko, 1989):
            Single hidden layer with sufficient width can approximate
            any continuous function on compact domain to arbitrary precision.

        Gradient flow with ReLU:
            ∂h_l/∂h_{l-1} = diag(h_{l-1} > 0) × W_l

            ~50% of gradients blocked per layer → use gain=√2 to compensate.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Output dimension.
    hidden_dims : List[int], default=[256, 256]
        Dimensions of hidden layers.
    activation : nn.Module, default=nn.ReLU()
        Activation function for hidden layers.
    output_activation : Optional[nn.Module], default=None
        Activation for output layer. None for linear output.

    Attributes
    ----------
    net : nn.Sequential
        Sequential container of layers and activations.

    Examples
    --------
    >>> # Policy network (logits output)
    >>> policy_net = MLP(4, 2, [128, 128])
    >>> logits = policy_net(torch.randn(32, 4))  # (32, 2)
    >>>
    >>> # Value network (scalar output)
    >>> value_net = MLP(4, 1, [128, 128])
    >>> values = value_net(torch.randn(32, 4))  # (32, 1)
    >>>
    >>> # Feature extractor
    >>> features = MLP(100, 64, [128], output_activation=nn.ReLU())

    Notes
    -----
    Complexity:
        Parameters: Σ_l (d_l × d_{l+1} + d_{l+1})
        Forward: O(batch × Σ_l d_l × d_{l+1})
        Memory: O(batch × max_l d_l) for activations

    Architecture guidelines:
        - Width: 64-512 depending on task complexity
        - Depth: 2-3 layers sufficient for most RL tasks
        - Activation: ReLU most common, Tanh for bounded outputs
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Optional[nn.Module] = None,
        output_activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]
        if activation is None:
            activation = nn.ReLU()

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                layers.append(activation)
            elif output_activation is not None:
                layers.append(output_activation)

        self.net = nn.Sequential(*layers)
        self._initialize_weights(output_activation)

    def _initialize_weights(self, output_activation: Optional[nn.Module]) -> None:
        """Apply orthogonal initialization with layer-appropriate gains."""
        for i, module in enumerate(self.net.modules()):
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))

        # Output layer: small gain for policy, unit gain for value
        for module in reversed(list(self.net.modules())):
            if isinstance(module, nn.Linear):
                gain = 0.01 if output_activation is None else 1.0
                init_weights(module, gain=gain)
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(x)

    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiscretePolicy(nn.Module):
    """
    Policy network for discrete action spaces using Softmax distribution.

    核心思想 (Core Idea):
        Output unnormalized logits and sample from a Categorical distribution.
        The log-softmax trick ensures numerical stability.

    数学原理 (Mathematical Theory):
        Softmax policy:
            π_θ(a|s) = exp(z_a) / Σ_{a'} exp(z_{a'})

        where z = f_θ(s) ∈ R^|A| are logits (unnormalized log-probabilities).

        Log-probability (numerically stable):
            log π_θ(a|s) = z_a - logsumexp(z)

        Entropy:
            H(π) = -Σ_a π(a) log π(a)
                 = logsumexp(z) - Σ_a π(a) z_a

        Gradient of log-probability:
            ∇_θ log π_θ(a|s) = ∇_θ z_a - E_{a'~π}[∇_θ z_{a'}]

            First term: increase logit of taken action
            Second term: decrease expected logit (baseline)

    问题背景 (Problem Statement):
        Discrete actions require a valid probability distribution that:
        1. Sums to 1 over all actions
        2. Is differentiable w.r.t. network parameters
        3. Allows stochastic sampling for exploration

        Solution: Softmax normalization over neural network outputs.

    算法对比 (Comparison):
        | Method           | Pros                    | Cons                   |
        |------------------|-------------------------|------------------------|
        | Softmax (this)   | Smooth, differentiable  | All actions always prob|
        | Epsilon-greedy   | Simple exploration      | Non-differentiable     |
        | Boltzmann        | Temperature control     | Same as softmax        |

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    action_dim : int
        Number of discrete actions.
    hidden_dims : List[int], default=[128, 128]
        Hidden layer dimensions.

    Examples
    --------
    >>> policy = DiscretePolicy(state_dim=4, action_dim=2)
    >>> state = torch.randn(32, 4)
    >>>
    >>> # Sample actions
    >>> action, log_prob, entropy = policy.sample(state)
    >>> print(f"Actions: {action.shape}")  # (32,)
    >>> print(f"Log probs: {log_prob.shape}")  # (32,)
    >>>
    >>> # Get probabilities
    >>> probs = policy.get_action_probs(state)  # (32, 2)
    >>> assert probs.sum(dim=-1).allclose(torch.ones(32))

    Notes
    -----
    Complexity:
        Forward: O(batch × (state_dim × h + h² + h × action_dim))
        Sampling: O(batch × action_dim) via inverse CDF
        Memory: O(batch × max(hidden_dims))

    Numerical stability:
        - Using logits avoids log(0) and precision loss
        - PyTorch Categorical handles logsumexp internally
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=nn.ReLU(),
            output_activation=None,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute action logits for given states."""
        return self.net(state)

    def get_distribution(self, state: torch.Tensor) -> Categorical:
        """Get categorical distribution over actions."""
        logits = self.forward(state)
        return Categorical(logits=logits)

    def sample(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy distribution.

        Returns
        -------
        action : torch.Tensor
            Sampled actions, shape (batch_size,).
        log_prob : torch.Tensor
            Log probabilities of actions, shape (batch_size,).
        entropy : torch.Tensor
            Entropy of distribution, shape (batch_size,).
        """
        dist = self.get_distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability of given state-action pairs."""
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (normalized)."""
        return self.get_distribution(state).probs


class ContinuousPolicy(nn.Module):
    """
    Policy network for continuous action spaces using Gaussian distribution.

    核心思想 (Core Idea):
        Output mean μ and log-std for a Gaussian distribution. Use
        reparameterization trick for differentiable sampling and tanh
        squashing for bounded action spaces.

    数学原理 (Mathematical Theory):
        Gaussian Policy:
            π_θ(a|s) = N(a | μ_θ(s), σ_θ(s)²)
                     = (2πσ²)^{-d/2} exp(-||a-μ||²/(2σ²))

        Log-probability:
            log π(a|s) = -d/2 log(2π) - d log(σ) - ||a-μ||²/(2σ²)

        Reparameterization Trick:
            Instead of: a ~ N(μ, σ²)
            Use: ε ~ N(0, I), a = μ + σ·ε

            Gradient flows through μ and σ, not through sampling.

        Tanh Squashing (for bounded actions a ∈ [-1, 1]):
            u ~ N(μ, σ²)
            a = tanh(u)

            Change of variables:
                p(a) = p(u) × |det(∂u/∂a)|
                     = p(u) × Π_i (1 - a_i²)^{-1}

            Log-probability correction:
                log π(a) = log N(u|μ,σ²) - Σ_i log(1 - a_i²)

        Entropy of Gaussian:
            H(N(μ,σ²)) = d/2 (1 + log(2π)) + Σ_i log(σ_i)

    问题背景 (Problem Statement):
        Continuous actions require:
        1. Smooth probability density for gradient computation
        2. Bounded support for physical constraints (optional)
        3. Learnable exploration via variance parameter

        Gaussian with tanh squashing satisfies all requirements.

    算法对比 (Comparison):
        | Distribution        | Bounded | Multimodal | Complexity |
        |---------------------|---------|------------|------------|
        | Gaussian            | No      | No         | Low        |
        | Squashed Gaussian   | Yes     | No         | Low        |
        | Beta                | Yes     | No         | Medium     |
        | GMM                 | No      | Yes        | High       |
        | Normalizing Flow    | Yes     | Yes        | Very High  |

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    action_dim : int
        Dimension of continuous action space.
    hidden_dims : List[int], default=[256, 256]
        Hidden layer dimensions.
    state_dependent_std : bool, default=False
        If True, std is output of network (more expressive).
        If False, std is learnable parameter (more stable).
    log_std_bounds : Tuple[float, float], default=(-20.0, 2.0)
        Bounds on log standard deviation to prevent collapse/explosion.

    Examples
    --------
    >>> policy = ContinuousPolicy(state_dim=3, action_dim=2)
    >>> state = torch.randn(32, 3)
    >>>
    >>> # Sample actions (bounded to [-1, 1])
    >>> action, log_prob, entropy = policy.sample(state)
    >>> assert action.min() >= -1 and action.max() <= 1
    >>>
    >>> # Deterministic action for evaluation
    >>> action_det, _, _ = policy.sample(state, deterministic=True)

    Notes
    -----
    Complexity:
        Forward: O(batch × (state_dim × h + h² + h × action_dim))
        Sampling: O(batch × action_dim)
        Log-prob correction: O(batch × action_dim)

    Numerical considerations:
        - log_std bounded to prevent σ→0 (collapse) or σ→∞ (explosion)
        - atanh clip prevents NaN at boundaries ±1
        - Small epsilon (1e-6) in log(1-a²) for stability
    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
        state_dependent_std: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_dependent_std = state_dependent_std

        # Feature extraction backbone
        self.feature_net = MLP(
            input_dim=state_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            activation=nn.ReLU(),
            output_activation=nn.ReLU(),
        )

        # Mean output
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        init_weights(self.mean_layer, gain=0.01)

        # Standard deviation
        if state_dependent_std:
            self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
            init_weights(self.log_std_layer, gain=0.01)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Gaussian distribution parameters.

        Returns
        -------
        mean : torch.Tensor
            Action means, shape (batch_size, action_dim).
        std : torch.Tensor
            Action standard deviations, shape (batch_size, action_dim).
        """
        features = self.feature_net(state)
        mean = self.mean_layer(features)

        if self.state_dependent_std:
            log_std = self.log_std_layer(features)
        else:
            log_std = self.log_std.expand_as(mean)

        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        return mean, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """Get Gaussian distribution over actions."""
        mean, std = self.forward(state)
        return Normal(mean, std)

    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions with tanh squashing.

        Parameters
        ----------
        state : torch.Tensor
            State observations.
        deterministic : bool, default=False
            If True, return tanh(mean) without sampling.

        Returns
        -------
        action : torch.Tensor
            Actions bounded to [-1, 1], shape (batch_size, action_dim).
        log_prob : torch.Tensor
            Log probabilities with Jacobian correction, shape (batch_size,).
        entropy : torch.Tensor
            Gaussian entropy (before squashing), shape (batch_size,).
        """
        mean, std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(state.shape[0], device=state.device)
            entropy = torch.zeros(state.shape[0], device=state.device)
        else:
            dist = Normal(mean, std)
            u = dist.rsample()  # Reparameterization trick
            action = torch.tanh(u)

            # Log probability with Jacobian correction
            log_prob = dist.log_prob(u).sum(dim=-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

            entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability of given actions."""
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        # Inverse tanh (atanh) to recover u
        action_clipped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(action_clipped)

        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class ValueNetwork(nn.Module):
    """
    State value function network V(s).

    核心思想 (Core Idea):
        Estimate expected cumulative return from a state under current policy.
        Serves as baseline for variance reduction or critic in actor-critic.

    数学原理 (Mathematical Theory):
        State Value Function:
            V^π(s) = E_{a~π, s'~P}[r(s,a) + γV^π(s')]
                   = E_π[Σ_{t=0}^∞ γ^t r_t | s_0 = s]

        Training Targets:
            Monte Carlo: L = (V_θ(s) - G_t)² where G_t = Σ_k γ^k r_{t+k}
            TD(0): L = (V_θ(s) - (r + γV_θ(s')))²
            GAE: L = (V_θ(s) - (A^GAE + V_θ(s)))²

        Role in Policy Gradient:
            Baseline for variance reduction:
                ∇J = E[∇log π(a|s) · (Q(s,a) - V(s))]
                   = E[∇log π(a|s) · A(s,a)]

            Variance reduction: Var[Q-V] << Var[Q] since E[A] = 0

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    hidden_dims : List[int], default=[128, 128]
        Hidden layer dimensions.

    Examples
    --------
    >>> value_net = ValueNetwork(state_dim=4)
    >>> states = torch.randn(32, 4)
    >>> values = value_net(states)  # (32, 1)
    >>>
    >>> # Training
    >>> returns = torch.randn(32, 1)
    >>> loss = F.mse_loss(values, returns)

    Notes
    -----
    Architecture considerations:
        - Often same structure as policy network
        - Can use larger capacity (value learning is supervised)
        - Output unbounded (values can be large)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.state_dim = state_dim

        self.net = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
            activation=nn.ReLU(),
            output_activation=None,
        )

        # Reinitialize output layer with unit gain (values can be large)
        for module in reversed(list(self.net.modules())):
            if isinstance(module, nn.Linear):
                init_weights(module, gain=1.0)
                break

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state values."""
        return self.net(state)


class ActorCriticNetwork(nn.Module):
    """
    Shared-feature Actor-Critic network architecture.

    核心思想 (Core Idea):
        Actor and Critic share a feature extraction backbone, then branch
        into separate heads. This architecture:
        1. Reduces total parameters (~60% of separate networks)
        2. Enables feature transfer between policy and value learning
        3. Provides implicit regularization through multi-task learning

    数学原理 (Mathematical Theory):
        Architecture:
            features = f_shared(s)  ∈ R^h
            π(a|s) = Actor_head(features)
            V(s) = Critic_head(features)

        Combined Loss (A2C/PPO style):
            L = L_policy + c_v · L_value - c_ent · H(π)

        Where:
            L_policy = -E[log π(a|s) · A]  (policy gradient)
            L_value = E[(V(s) - V_target)²]  (value regression)
            H(π) = -E[π log π]  (entropy bonus)

        Gradient interaction:
            ∂L/∂θ_shared = ∂L_policy/∂θ_shared + c_v · ∂L_value/∂θ_shared

            Features learned to be useful for both tasks.

    问题背景 (Problem Statement):
        Separate actor and critic networks:
        - Duplicate feature learning
        - More parameters to optimize
        - No feature transfer

        Shared architecture addresses these with potential trade-off
        of task interference (mitigated by loss coefficient tuning).

    算法对比 (Comparison):
        | Aspect           | Separate Networks | Shared Network |
        |------------------|-------------------|----------------|
        | Parameters       | ~2x               | ~1.2x          |
        | Training speed   | Slower            | Faster         |
        | Hyperparameter   | More (2 LRs)      | Fewer          |
        | Task interference| None              | Possible       |
        | Feature transfer | No                | Yes            |

    Parameters
    ----------
    state_dim : int
        Dimension of state observation space.
    action_dim : int
        Dimension of action space.
    hidden_dim : int, default=256
        Dimension of hidden layers.
    continuous : bool, default=False
        If True, use Gaussian policy for continuous actions.

    Examples
    --------
    >>> # Discrete actions (e.g., CartPole)
    >>> ac = ActorCriticNetwork(state_dim=4, action_dim=2)
    >>> state = torch.randn(32, 4)
    >>> action, log_prob, entropy, value = ac.get_action_and_value(state)
    >>>
    >>> # Continuous actions (e.g., Pendulum)
    >>> ac = ActorCriticNetwork(state_dim=3, action_dim=1, continuous=True)
    >>> action, log_prob, entropy, value = ac.get_action_and_value(state)

    Notes
    -----
    Complexity:
        Forward (both heads): O(batch × (state_dim × h + h² + h × (action_dim + 1)))
        ~40% fewer FLOPs than separate forward passes
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = False,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous

        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply appropriate initialization to all layers."""
        for module in self.shared_net.modules():
            if isinstance(module, nn.Linear):
                init_weights(module, gain=np.sqrt(2))

        if self.continuous:
            init_weights(self.actor_mean, gain=0.01)
        else:
            init_weights(self.actor_head, gain=0.01)

        init_weights(self.critic_head, gain=1.0)

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass computing policy parameters and value.

        Returns
        -------
        policy_output : torch.Tensor or Tuple
            Discrete: logits (batch, action_dim)
            Continuous: (mean, log_std) tuple
        value : torch.Tensor
            Value estimates (batch, 1)
        """
        features = self.shared_net(state)
        value = self.critic_head(features)

        if self.continuous:
            mean = self.actor_mean(features)
            return (mean, self.actor_log_std), value
        else:
            logits = self.actor_head(features)
            return logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value in single forward pass.

        Parameters
        ----------
        state : torch.Tensor
            State observations.
        action : Optional[torch.Tensor]
            If provided, evaluate this action instead of sampling.

        Returns
        -------
        action : torch.Tensor
            Sampled or provided actions.
        log_prob : torch.Tensor
            Log probabilities of actions.
        entropy : torch.Tensor
            Policy entropy.
        value : torch.Tensor
            Value estimates.
        """
        features = self.shared_net(state)
        value = self.critic_head(features).squeeze(-1)

        if self.continuous:
            mean = self.actor_mean(features)
            std = self.actor_log_std.exp()
            dist = Normal(mean, std)

            if action is None:
                u = dist.rsample()
                action = torch.tanh(u)
            else:
                action_clipped = torch.clamp(action, -1 + 1e-6, 1 - 1e-6)
                u = torch.atanh(action_clipped)

            log_prob = dist.log_prob(u).sum(dim=-1)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.actor_head(features)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value estimates only (efficient for bootstrapping)."""
        features = self.shared_net(state)
        return self.critic_head(features).squeeze(-1)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Neural Network Policy Module - Unit Tests")
    print("=" * 70)

    # Test parameters
    batch_size = 32
    state_dim = 4
    discrete_action_dim = 3
    continuous_action_dim = 2

    # Test MLP
    print("\n[1] Testing MLP...")
    mlp = MLP(state_dim, 64, [128, 128])
    x = torch.randn(batch_size, state_dim)
    out = mlp(x)
    assert out.shape == (batch_size, 64), f"MLP output shape mismatch: {out.shape}"
    print(f"    MLP parameters: {mlp.get_num_params():,}")
    print(f"    Input: {x.shape} -> Output: {out.shape}")
    print("    [PASS]")

    # Test DiscretePolicy
    print("\n[2] Testing DiscretePolicy...")
    discrete_policy = DiscretePolicy(state_dim, discrete_action_dim)
    state = torch.randn(batch_size, state_dim)

    action, log_prob, entropy = discrete_policy.sample(state)
    assert action.shape == (batch_size,), f"Action shape: {action.shape}"
    assert log_prob.shape == (batch_size,), f"Log prob shape: {log_prob.shape}"
    assert entropy.shape == (batch_size,), f"Entropy shape: {entropy.shape}"
    assert (action >= 0).all() and (action < discrete_action_dim).all()

    probs = discrete_policy.get_action_probs(state)
    assert probs.sum(dim=-1).allclose(torch.ones(batch_size), atol=1e-5)
    print(f"    Actions in [0, {discrete_action_dim-1}]: {action[:5].tolist()}")
    print(f"    Probs sum to 1: {probs.sum(dim=-1).mean():.6f}")
    print("    [PASS]")

    # Test ContinuousPolicy
    print("\n[3] Testing ContinuousPolicy...")
    continuous_policy = ContinuousPolicy(state_dim, continuous_action_dim)
    state = torch.randn(batch_size, state_dim)

    action, log_prob, entropy = continuous_policy.sample(state)
    assert action.shape == (batch_size, continuous_action_dim)
    assert action.min() >= -1 and action.max() <= 1, "Actions not bounded"

    # Test deterministic mode
    action_det, _, _ = continuous_policy.sample(state, deterministic=True)
    action_det2, _, _ = continuous_policy.sample(state, deterministic=True)
    assert torch.allclose(action_det, action_det2), "Deterministic not consistent"
    print(f"    Action range: [{action.min():.3f}, {action.max():.3f}]")
    print(f"    Deterministic consistent: True")
    print("    [PASS]")

    # Test ValueNetwork
    print("\n[4] Testing ValueNetwork...")
    value_net = ValueNetwork(state_dim)
    values = value_net(state)
    assert values.shape == (batch_size, 1)
    print(f"    Value range: [{values.min():.3f}, {values.max():.3f}]")
    print("    [PASS]")

    # Test ActorCriticNetwork (discrete)
    print("\n[5] Testing ActorCriticNetwork (discrete)...")
    ac_discrete = ActorCriticNetwork(state_dim, discrete_action_dim, continuous=False)
    action, log_prob, entropy, value = ac_discrete.get_action_and_value(state)
    assert action.shape == (batch_size,)
    assert value.shape == (batch_size,)
    print(f"    Action shape: {action.shape}, Value shape: {value.shape}")
    print("    [PASS]")

    # Test ActorCriticNetwork (continuous)
    print("\n[6] Testing ActorCriticNetwork (continuous)...")
    ac_continuous = ActorCriticNetwork(state_dim, continuous_action_dim, continuous=True)
    action, log_prob, entropy, value = ac_continuous.get_action_and_value(state)
    assert action.shape == (batch_size, continuous_action_dim)
    assert action.min() >= -1 and action.max() <= 1
    print(f"    Action range: [{action.min():.3f}, {action.max():.3f}]")
    print("    [PASS]")

    # Test gradient flow
    print("\n[7] Testing gradient flow...")
    state.requires_grad = False
    action, log_prob, entropy, value = ac_discrete.get_action_and_value(state)
    loss = -log_prob.mean() + 0.5 * value.pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()

    grad_norm = 0.0
    for p in ac_discrete.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"    Total gradient norm: {grad_norm:.4f}")
    print("    [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
