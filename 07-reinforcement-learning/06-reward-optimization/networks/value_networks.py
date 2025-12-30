"""
Value and Potential Networks for Reward Learning

================================================================================
CORE IDEA
================================================================================
Value networks estimate the expected cumulative reward from states.
In reward optimization, they serve dual purposes:
1. Potential functions for PBRS (Φ(s) ≈ V(s))
2. Critic networks in actor-critic IRL/GAIL

================================================================================
MATHEMATICAL THEORY
================================================================================
State Value Function:
    V^π(s) = E_π[Σₜ γᵗ r_t | s₀ = s]

Action-Value Function:
    Q^π(s,a) = E_π[Σₜ γᵗ r_t | s₀ = s, a₀ = a]

Advantage Function:
    A^π(s,a) = Q^π(s,a) - V^π(s)

For potential-based reward shaping:
    Φ*(s) = V*(s)  (optimal potential = optimal value)

================================================================================
REFERENCES
================================================================================
[1] Wang et al. (2016). Dueling Network Architectures for Deep RL.
[2] Sutton & Barto (2018). Reinforcement Learning: An Introduction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class ValueNetwork:
    """Multi-layer perceptron for state value estimation.

    ============================================================================
    CORE IDEA
    ============================================================================
    Approximates V(s) = E[Σ γᵗ rₜ | s₀ = s] using a neural network.
    Used as:
    - Potential function in PBRS
    - Baseline in policy gradient methods
    - Critic in actor-critic algorithms

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The value network V_θ: S → ℝ is trained to minimize:

        L(θ) = E[(V_θ(s) - V^target)²]

    where V^target can be:
    - Monte Carlo return: Σₜ γᵗ rₜ
    - TD target: r + γV_θ(s')
    - GAE (Generalized Advantage Estimation)

    ============================================================================
    ARCHITECTURE
    ============================================================================
    Standard MLP: s → h₁ → h₂ → ... → V(s)
    - Input: State observation
    - Hidden: ReLU activations
    - Output: Single scalar value (unbounded)

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(Σᵢ dᵢ × dᵢ₊₁) forward pass
    - Space: O(Σᵢ dᵢ × dᵢ₊₁) parameters
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize value network.

        Args:
            state_dim: Dimensionality of state space.
            hidden_dims: Sizes of hidden layers.
            activation: Activation function ("relu", "tanh").
            learning_rate: Learning rate for updates.
        """
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.learning_rate = learning_rate

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network parameters with Xavier initialization."""
        dims = [self.state_dim] + list(self.hidden_dims) + [1]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        else:
            return x

    def forward(self, states: np.ndarray) -> np.ndarray:
        """Compute state values V(s).

        Args:
            states: State observations, shape (state_dim,) or (batch, state_dim).

        Returns:
            Value estimates, shape () or (batch,).
        """
        states = np.atleast_2d(states).astype(np.float64)
        single_input = states.shape[0] == 1

        x = states
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = self._activation_fn(x)

        values = x.flatten()
        return values[0] if single_input else values

    def update(
        self,
        states: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Update value network via gradient descent.

        Args:
            states: Batch of states.
            targets: Target values (e.g., Monte Carlo returns).

        Returns:
            MSE loss before update.
        """
        states = np.atleast_2d(states).astype(np.float64)
        targets = np.atleast_1d(targets).astype(np.float64).reshape(-1, 1)
        batch_size = len(states)

        x = states
        activations = [x.copy()]

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = self._activation_fn(x)
            activations.append(x.copy())

        loss = float(np.mean((x - targets) ** 2))

        delta = 2 * (x - targets) / batch_size

        for i in range(len(self._weights) - 1, -1, -1):
            grad_w = activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)

            self._weights[i] -= self.learning_rate * grad_w
            self._biases[i] -= self.learning_rate * grad_b

            if i > 0:
                delta = delta @ self._weights[i].T
                if self.activation == "relu":
                    delta = delta * (activations[i] > 0)
                elif self.activation == "tanh":
                    delta = delta * (1 - activations[i] ** 2)

        return loss

    def get_parameters(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get network parameters."""
        return (
            [w.copy() for w in self._weights],
            [b.copy() for b in self._biases],
        )

    def set_parameters(
        self,
        weights: List[np.ndarray],
        biases: List[np.ndarray],
    ) -> None:
        """Set network parameters."""
        self._weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self._biases = [np.asarray(b, dtype=np.float64) for b in biases]


class DuelingValueNetwork:
    """Dueling architecture separating value and advantage streams.

    ============================================================================
    CORE IDEA
    ============================================================================
    Decompose Q(s,a) into state value V(s) and advantage A(s,a):

        Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]

    This allows:
    - Better generalization: V(s) learned even for unvisited actions
    - More stable learning: V(s) changes slowly, A(s,a) captures action effects
    - Potential extraction: V(s) directly usable for reward shaping

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    The dueling architecture (Wang et al., 2016):

        Q(s,a; θ, α, β) = V(s; θ, β) + (A(s,a; θ, α) - (1/|A|)Σ_a' A(s,a'; θ, α))

    - θ: Shared feature extraction parameters
    - β: Value stream parameters
    - α: Advantage stream parameters

    The subtraction of mean advantage ensures:
        V(s) = max_a Q(s,a) ≈ E_a[Q(s,a)]

    ============================================================================
    ADVANTAGES
    ============================================================================
    - Learns V(s) implicitly even when not all actions are sampled
    - More stable gradients for value estimation
    - Directly extracts potential function for PBRS

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(shared + value_stream + advantage_stream)
    - Space: O(|θ| + |α| + |β|)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        shared_dims: Tuple[int, ...] = (64,),
        value_dims: Tuple[int, ...] = (32,),
        advantage_dims: Tuple[int, ...] = (32,),
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize dueling value network.

        Args:
            state_dim: State dimensionality.
            action_dim: Number of actions.
            shared_dims: Shared feature extraction layers.
            value_dims: Value stream hidden layers.
            advantage_dims: Advantage stream hidden layers.
            learning_rate: Learning rate for updates.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shared_dims = shared_dims
        self.value_dims = value_dims
        self.advantage_dims = advantage_dims
        self.learning_rate = learning_rate

        self._shared_weights: List[np.ndarray] = []
        self._shared_biases: List[np.ndarray] = []
        self._value_weights: List[np.ndarray] = []
        self._value_biases: List[np.ndarray] = []
        self._advantage_weights: List[np.ndarray] = []
        self._advantage_biases: List[np.ndarray] = []

        self._init_network()

    def _init_network(self) -> None:
        """Initialize all network components."""
        shared_dims = [self.state_dim] + list(self.shared_dims)
        for i in range(len(shared_dims) - 1):
            fan_in, fan_out = shared_dims[i], shared_dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._shared_weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._shared_biases.append(np.zeros(fan_out, dtype=np.float64))

        shared_out = shared_dims[-1]

        value_dims = [shared_out] + list(self.value_dims) + [1]
        for i in range(len(value_dims) - 1):
            fan_in, fan_out = value_dims[i], value_dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._value_weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._value_biases.append(np.zeros(fan_out, dtype=np.float64))

        adv_dims = [shared_out] + list(self.advantage_dims) + [self.action_dim]
        for i in range(len(adv_dims) - 1):
            fan_in, fan_out = adv_dims[i], adv_dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._advantage_weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._advantage_biases.append(np.zeros(fan_out, dtype=np.float64))

    def _forward_shared(self, states: np.ndarray) -> np.ndarray:
        """Forward through shared feature layers."""
        x = states
        for i, (w, b) in enumerate(zip(self._shared_weights, self._shared_biases)):
            x = x @ w + b
            x = np.maximum(0, x)
        return x

    def _forward_value(self, features: np.ndarray) -> np.ndarray:
        """Forward through value stream."""
        x = features
        for i, (w, b) in enumerate(zip(self._value_weights, self._value_biases)):
            x = x @ w + b
            if i < len(self._value_weights) - 1:
                x = np.maximum(0, x)
        return x

    def _forward_advantage(self, features: np.ndarray) -> np.ndarray:
        """Forward through advantage stream."""
        x = features
        for i, (w, b) in enumerate(
            zip(self._advantage_weights, self._advantage_biases)
        ):
            x = x @ w + b
            if i < len(self._advantage_weights) - 1:
                x = np.maximum(0, x)
        return x

    def forward(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass returning V(s), A(s,a), and Q(s,a).

        Args:
            states: State observations.

        Returns:
            Tuple of:
            - values: V(s), shape (batch, 1)
            - advantages: A(s,a), shape (batch, action_dim)
            - q_values: Q(s,a), shape (batch, action_dim)
        """
        states = np.atleast_2d(states).astype(np.float64)

        features = self._forward_shared(states)
        values = self._forward_value(features)
        advantages = self._forward_advantage(features)

        mean_advantage = np.mean(advantages, axis=1, keepdims=True)
        q_values = values + advantages - mean_advantage

        return values, advantages, q_values

    def get_value(self, states: np.ndarray) -> np.ndarray:
        """Get state value V(s) for potential-based shaping.

        Args:
            states: State observations.

        Returns:
            Value estimates V(s).
        """
        values, _, _ = self.forward(states)
        return values.flatten()

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """Get action values Q(s,a) for all actions.

        Args:
            states: State observations.

        Returns:
            Q-values for all actions, shape (batch, action_dim).
        """
        _, _, q_values = self.forward(states)
        return q_values


class PotentialNetwork:
    """Specialized network for learning potential functions in PBRS.

    ============================================================================
    CORE IDEA
    ============================================================================
    Learn a potential function Φ(s) that provides dense reward shaping
    while preserving policy invariance. The ideal potential equals the
    optimal value function: Φ*(s) = V*(s).

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Training objectives for potential learning:

    1. From demonstrations (imitation):
       L(θ) = E_D[||Φ_θ(s') - Φ_θ(s) - r||²]

    2. From value function:
       L(θ) = E[||Φ_θ(s) - V(s)||²]

    3. Self-supervised (temporal consistency):
       L(θ) = E[||Φ_θ(s') - γΦ_θ(s) - r||²]

    ============================================================================
    USAGE IN PBRS
    ============================================================================
    The shaping bonus is computed as:

        F(s, s') = γΦ(s') - Φ(s)

    A well-learned potential provides:
    - Dense gradient toward high-value states
    - Policy-invariant learning signal
    - Accelerated exploration and credit assignment

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(forward_pass)
    - Space: O(|θ|)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        output_scale: float = 1.0,
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize potential network.

        Args:
            state_dim: State dimensionality.
            hidden_dims: Hidden layer sizes.
            output_scale: Scaling factor for potential values.
                Helps control magnitude of shaping bonus.
            learning_rate: Learning rate for updates.
        """
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        self.output_scale = output_scale
        self.learning_rate = learning_rate

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network parameters."""
        dims = [self.state_dim] + list(self.hidden_dims) + [1]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def forward(self, states: np.ndarray) -> np.ndarray:
        """Compute potential values Φ(s).

        Args:
            states: State observations.

        Returns:
            Potential values, shape (batch,) or scalar.
        """
        states = np.atleast_2d(states).astype(np.float64)
        single_input = states.shape[0] == 1

        x = states
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.tanh(x)

        potentials = self.output_scale * x.flatten()
        return potentials[0] if single_input else potentials

    def compute_shaping_bonus(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        discount_factor: float = 0.99,
        done: bool = False,
    ) -> float:
        """Compute PBRS shaping bonus F(s, s') = γΦ(s') - Φ(s).

        Args:
            state: Current state.
            next_state: Next state.
            discount_factor: Discount factor γ.
            done: Whether next_state is terminal.

        Returns:
            Shaping bonus value.
        """
        phi_s = self.forward(state)
        phi_s_next = 0.0 if done else self.forward(next_state)
        return float(discount_factor * phi_s_next - phi_s)

    def update_from_value(
        self,
        states: np.ndarray,
        values: np.ndarray,
    ) -> float:
        """Update potential to match value function estimates.

        Args:
            states: Batch of states.
            values: Target value estimates V(s).

        Returns:
            MSE loss before update.
        """
        states = np.atleast_2d(states).astype(np.float64)
        values = np.atleast_1d(values).astype(np.float64).reshape(-1, 1)
        values = values / self.output_scale
        batch_size = len(states)

        x = states
        activations = [x.copy()]

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.tanh(x)
            activations.append(x.copy())

        loss = float(np.mean((x - values) ** 2))

        delta = 2 * (x - values) / batch_size

        for i in range(len(self._weights) - 1, -1, -1):
            grad_w = activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)

            self._weights[i] -= self.learning_rate * grad_w
            self._biases[i] -= self.learning_rate * grad_b

            if i > 0:
                delta = delta @ self._weights[i].T
                delta = delta * (1 - activations[i] ** 2)

        return loss

    def update_from_transitions(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        discount_factor: float = 0.99,
    ) -> float:
        """Update potential using temporal consistency.

        Minimizes: ||Φ(s') - γΦ(s) - r||²

        This encourages Φ to capture the structure of the value function.

        Args:
            states: Current states.
            next_states: Next states.
            rewards: Observed rewards.
            dones: Terminal flags.
            discount_factor: Discount factor γ.

        Returns:
            MSE loss before update.
        """
        states = np.atleast_2d(states).astype(np.float64)
        next_states = np.atleast_2d(next_states).astype(np.float64)
        rewards = np.atleast_1d(rewards).astype(np.float64)
        dones = np.atleast_1d(dones).astype(np.float64)
        batch_size = len(states)

        phi_s = self.forward(states)
        phi_s_next = self.forward(next_states) * (1 - dones)

        target = rewards + discount_factor * phi_s_next
        loss = float(np.mean((phi_s - target) ** 2))

        return loss


if __name__ == "__main__":
    print("=" * 70)
    print("Value Networks - Validation Tests")
    print("=" * 70)

    np.random.seed(42)

    state_dim = 4
    action_dim = 3

    # Test basic value network
    print("\n[Test 1] Value Network")
    value_net = ValueNetwork(state_dim)

    states = np.random.randn(16, state_dim)
    values = value_net.forward(states)
    print(f"  Value estimates shape: {values.shape}")
    assert values.shape == (16,), "Shape mismatch"

    targets = np.random.randn(16) * 10
    loss = value_net.update(states, targets)
    print(f"  Training loss: {loss:.4f}")
    print("  [PASS]")

    # Test dueling network
    print("\n[Test 2] Dueling Value Network")
    dueling = DuelingValueNetwork(state_dim, action_dim)

    v, a, q = dueling.forward(states)
    print(f"  V(s) shape: {v.shape}")
    print(f"  A(s,a) shape: {a.shape}")
    print(f"  Q(s,a) shape: {q.shape}")

    v_for_shaping = dueling.get_value(states)
    print(f"  Potential for PBRS: {v_for_shaping.shape}")
    print("  [PASS]")

    # Test potential network
    print("\n[Test 3] Potential Network")
    potential = PotentialNetwork(state_dim)

    phi = potential.forward(states[:5])
    print(f"  Φ(s) values: {phi}")

    bonus = potential.compute_shaping_bonus(states[0], states[1])
    print(f"  Shaping bonus F(s, s'): {bonus:.4f}")

    loss = potential.update_from_value(states, targets)
    print(f"  Value matching loss: {loss:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
