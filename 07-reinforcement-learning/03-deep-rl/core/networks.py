"""
Neural Network Architectures for Deep Reinforcement Learning

This module provides production-grade neural network implementations for value-based
and policy-based deep RL algorithms. All architectures follow established best practices
for initialization, numerical stability, and gradient flow.

Architectures:
    DQNNetwork          - Standard MLP for Q-function approximation
    DuelingDQNNetwork   - Value-Advantage decomposition architecture
    ActorCriticNetwork  - Shared-parameter policy-value network

Mathematical Framework:
    Value Function Approximation:
        Q(s, a; θ) ≈ Q*(s, a)

    Policy Function Approximation:
        π_θ(a|s) ≈ π*(a|s)

    Neural networks provide universal function approximation with compact
    representations and automatic feature learning from raw observations.

Complexity Analysis:
    Standard MLP: O(d_in × d_h + d_h² × (L-2) + d_h × d_out)
    where d_in = input dim, d_h = hidden dim, L = layers, d_out = output dim

References:
    [1] Mnih et al., "Human-level control through deep RL", Nature 2015
    [2] Wang et al., "Dueling Network Architectures", ICML 2016
    [3] Mnih et al., "Asynchronous Methods for Deep RL", ICML 2016
"""

from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def init_weights(module: nn.Module, gain: float = np.sqrt(2)) -> None:
    """
    Orthogonal weight initialization for deep RL networks.

    Core Idea:
        Orthogonal matrices preserve gradient magnitudes during backpropagation,
        preventing vanishing/exploding gradients in deep networks.

    Mathematical Principle:
        For orthogonal matrix W: ||Wx||₂ = ||x||₂
        With gain scaling: Var[y] = gain² × Var[x]

        Recommended gains:
        - ReLU: √2 (He initialization equivalent)
        - Tanh: √(5/3) ≈ 1.29
        - Linear: 1.0
        - Policy output: 0.01 (small initial actions)

    Complexity: O(fan_in × fan_out) per linear layer

    Args:
        module: Neural network module to initialize
        gain: Initialization gain for variance scaling

    References:
        [1] Saxe et al., "Exact solutions to nonlinear dynamics", ICLR 2014
        [2] He et al., "Delving Deep into Rectifiers", ICCV 2015
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DQNNetwork(nn.Module):
    """
    Standard Deep Q-Network for value function approximation.

    Core Idea:
        Multi-layer perceptron mapping states to action-value estimates.
        Universal approximation theorem guarantees convergence to Q*
        with sufficient capacity and proper training.

    Mathematical Principle:
        Q(s, a; θ) = f_θ(s)[a]

        Architecture: s → FC → ReLU → FC → ReLU → FC → Q-values

        Loss function (TD error):
            L(θ) = E[(r + γ max_{a'} Q(s', a'; θ⁻) - Q(s, a; θ))²]

    Problem Statement:
        Tabular Q-Learning requires O(|S| × |A|) memory, infeasible for
        high-dimensional state spaces. Neural networks provide compact
        representation with O(|θ|) parameters and automatic feature learning.

    Algorithm Comparison:
        vs. Linear Approximation:
            + Automatic feature extraction
            + Handles high-dimensional inputs
            - Requires more data
            - Hyperparameter tuning needed

        vs. Dueling Architecture:
            + Simpler, faster forward pass
            + Fewer parameters
            - No explicit value-advantage separation
            - Less sample efficient in certain domains

    Complexity:
        Space: O(d_s × d_h + d_h² + d_h × |A|)
        Time (forward): O(batch × (d_s × d_h + d_h² + d_h × |A|))

        where d_s = state_dim, d_h = hidden_dim, |A| = action_dim

    Theoretical Properties:
        1. Universal Approximation: Can represent any continuous Q*
        2. Convergence: Under Robbins-Monro conditions with experience replay
        3. Generalization: Shares information across similar states

    Attributes:
        net: Sequential network layers
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ) -> None:
        """
        Initialize standard DQN network.

        Args:
            state_dim: State observation dimensionality
            action_dim: Number of discrete actions (Q-value outputs)
            hidden_dim: Hidden layer width

        Raises:
            ValueError: If dimensions are non-positive
        """
        super().__init__()

        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive")

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )

        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values for all actions.

        Args:
            x: State tensor, shape (batch_size, state_dim)

        Returns:
            Q-values tensor, shape (batch_size, action_dim)
        """
        return self.net(x)


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN with value-advantage decomposition.

    Core Idea:
        Decompose Q(s,a) into state value V(s) and action advantage A(s,a).
        Enables independent learning of "how good is this state" versus
        "how good is this action relative to others."

    Mathematical Principle:
        Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]

        Decomposition:
            V(s): Expected return from state s (action-independent)
            A(s, a): Relative advantage of action a over average

        Identifiability constraint:
            Mean subtraction ensures unique decomposition:
            Σ_a A(s, a) = 0

            Without this, Q = (V + c) + (A - c) for any constant c.

    Problem Statement:
        Standard DQN learns Q(s,a) jointly. In many states, action choice
        has minimal impact (e.g., states far from critical decisions).
        Learning V(s) separately enables faster value propagation.

    Algorithm Comparison:
        vs. Standard DQN:
            + Faster convergence (V learned from all transitions)
            + More stable gradients for value learning
            + Better in states where actions similar
            - ~50% more parameters
            - Negligible computation overhead

        vs. Advantage Actor-Critic:
            + Off-policy (experience replay compatible)
            + Single network outputs both
            - Discrete actions only
            - No explicit policy output

    Complexity:
        Space: O(d_s × d_h + 2d_h² + d_h × (|A| + 1))
        Time: O(batch × (d_s × d_h + 2d_h² + d_h × |A|))

        Approximately 1.5× standard DQN parameters

    Theoretical Properties:
        1. Identifiability: Mean subtraction ensures unique V, A
        2. Faster Learning: V stream benefits from all action samples
        3. Variance Reduction: A stream has lower variance than Q

    Empirical Results (Wang et al., 2016):
        - +20% improvement over DQN on Atari
        - Particularly effective in environments with many similar-valued actions

    Attributes:
        feature_layer: Shared feature extraction
        value_stream: State value head V(s)
        advantage_stream: Action advantage head A(s,a)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ) -> None:
        """
        Initialize Dueling DQN network.

        Args:
            state_dim: State observation dimensionality
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer width for each stream
        """
        super().__init__()

        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive")

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )

        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with value-advantage aggregation.

        Aggregation: Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]

        Args:
            x: State tensor, shape (batch_size, state_dim)

        Returns:
            Q-values tensor, shape (batch_size, action_dim)
        """
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class ActorCriticNetwork(nn.Module):
    """
    Shared-parameter Actor-Critic network for policy gradient methods.

    Core Idea:
        Unified network with shared feature extraction and separate heads
        for policy (actor) and value function (critic). Enables joint
        learning of complementary representations.

    Mathematical Principle:
        Shared features:
            φ(s) = MLP(s) ∈ ℝ^{d_h}

        Policy head (Actor):
            logits(s) = W_π φ(s) + b_π ∈ ℝ^{|A|}
            π(a|s) = softmax(logits(s))

        Value head (Critic):
            V(s) = W_V φ(s) + b_V ∈ ℝ

        Joint training loss:
            L = L_policy + c_v L_value - c_e H[π]

    Problem Statement:
        Policy gradient methods require:
        1. Policy π(a|s) for action selection
        2. Value function V(s) for advantage estimation (variance reduction)
        3. Efficient parameter sharing for sample efficiency

    Algorithm Comparison:
        vs. Separate Networks:
            + ~50% fewer parameters
            + Shared gradients improve learning
            + Single forward pass for both outputs
            - Potential gradient interference
            - Value errors may affect policy

        vs. Q-function (DQN):
            + Explicit policy (handles continuous actions)
            + Stochastic policies (built-in exploration)
            + Works with on-policy algorithms
            - Requires more samples (on-policy constraint)

    Complexity:
        Space: O(d_s × d_h + d_h² + d_h × (|A| + 1))
        Time: O(batch × (d_s × d_h + d_h² + d_h × |A|))

    Theoretical Properties:
        1. Policy Gradient: ∇_θ J = E[∇_θ log π_θ(a|s) × A^π(s,a)]
        2. Shared Learning: φ(s) learns features useful for both π and V
        3. Entropy Bonus: H[π] encourages exploration

    Attributes:
        shared: Shared feature extraction layers
        actor: Policy output head (action logits)
        critic: Value function output head (scalar)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ) -> None:
        """
        Initialize Actor-Critic network.

        Args:
            state_dim: State observation dimensionality
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer width
        """
        super().__init__()

        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive")

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.shared.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.actor, gain=0.01)
        init_weights(self.critic, gain=1.0)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            x: State tensor, shape (batch_size, state_dim)

        Returns:
            Tuple (action_logits, value):
                - action_logits: shape (batch_size, action_dim)
                - value: shape (batch_size, 1)
        """
        features = self.shared(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute associated quantities.

        Used for:
        1. Action selection (action=None): Sample from π(·|s)
        2. Log-prob computation (action given): Compute log π(action|s)

        Args:
            state: State tensor, shape (batch_size, state_dim)
            action: Optional action tensor for log-prob computation

        Returns:
            Tuple (action, log_prob, entropy, value):
                - action: Sampled or provided actions, shape (batch_size,)
                - log_prob: Log probabilities, shape (batch_size,)
                - entropy: Policy entropy, shape (batch_size,)
                - value: State values, shape (batch_size,)
        """
        logits, value = self(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


if __name__ == "__main__":
    print("Network Architecture Tests")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    state_dim = 4
    action_dim = 2

    test_input = torch.randn(batch_size, state_dim).to(device)

    dqn = DQNNetwork(state_dim, action_dim).to(device)
    dqn_out = dqn(test_input)
    assert dqn_out.shape == (batch_size, action_dim)
    print(f"DQNNetwork: input {test_input.shape} -> output {dqn_out.shape}")

    dueling = DuelingDQNNetwork(state_dim, action_dim).to(device)
    dueling_out = dueling(test_input)
    assert dueling_out.shape == (batch_size, action_dim)
    print(f"DuelingDQN: input {test_input.shape} -> output {dueling_out.shape}")

    ac = ActorCriticNetwork(state_dim, action_dim).to(device)
    action, log_prob, entropy, value = ac.get_action_and_value(test_input)
    assert action.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert value.shape == (batch_size,)
    print(f"ActorCritic: action {action.shape}, log_prob {log_prob.shape}, value {value.shape}")

    print("\nAll tests passed!")
