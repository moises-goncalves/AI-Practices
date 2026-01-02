"""
Discriminator Networks for Adversarial Imitation Learning

================================================================================
CORE IDEA
================================================================================
Discriminators distinguish between "expert" and "agent" state-action pairs.
The agent learns to fool the discriminator, thereby imitating the expert.
This approach avoids explicit reward recovery by directly matching behaviors.

================================================================================
MATHEMATICAL THEORY
================================================================================
GAIL Objective (Ho & Ermon, 2016):

    min_π max_D E_πE[log D(s,a)] + E_π[log(1-D(s,a))] - λH(π)

where:
- D(s,a): Discriminator probability that (s,a) is from expert
- π: Agent policy trying to fool D
- πE: Expert policy (demonstrations)
- H(π): Policy entropy regularization

The discriminator provides implicit reward:

    r(s,a) = -log(1 - D(s,a))  or  r(s,a) = log D(s,a) - log(1-D(s,a))

================================================================================
ALGORITHM COMPARISON
================================================================================
| Method | Reward | Policy Constraint | Sample Efficiency |
|--------|--------|-------------------|-------------------|
| GAIL   | Implicit (discriminator) | Entropy | Moderate |
| AIRL   | Explicit (recoverable)   | Entropy | Moderate |
| BC     | None (supervised)        | None    | High     |

================================================================================
REFERENCES
================================================================================
[1] Ho, J. & Ermon, S. (2016). Generative adversarial imitation learning.
[2] Fu, J. et al. (2018). Learning robust rewards with AIRL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GAILConfig:
    """Configuration for GAIL discriminator training.

    Attributes:
        learning_rate: Discriminator learning rate.
        hidden_dims: Hidden layer sizes.
        gradient_penalty_weight: Weight for gradient penalty (WGAN-GP style).
        spectral_norm: Whether to use spectral normalization.
        label_smoothing: Smooth labels to prevent overconfident discriminator.
    """

    learning_rate: float = 0.0003
    hidden_dims: Tuple[int, ...] = (64, 64)
    gradient_penalty_weight: float = 10.0
    spectral_norm: bool = False
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.gradient_penalty_weight < 0:
            raise ValueError(
                f"gradient_penalty_weight must be non-negative, "
                f"got {self.gradient_penalty_weight}"
            )
        if not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError(
                f"label_smoothing must be in [0, 0.5), got {self.label_smoothing}"
            )


class GAILDiscriminator:
    """GAIL discriminator network distinguishing expert from policy data.

    ============================================================================
    CORE IDEA
    ============================================================================
    The discriminator D: S × A → [0,1] is trained to output high values for
    expert state-action pairs and low values for agent pairs. It provides
    reward signal: D(s,a) ≈ 1 means "expert-like", rewarding the agent.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Binary cross-entropy loss:

        L_D = -E_πE[log D(s,a)] - E_π[log(1 - D(s,a))]

    Optimal discriminator (Nash equilibrium):

        D*(s,a) = ρ_E(s,a) / (ρ_E(s,a) + ρ_π(s,a))

    where ρ_E, ρ_π are occupancy measures.

    Agent reward signal:

        r(s,a) = -log(1 - D(s,a))  (original GAIL)
        r(s,a) = log D(s,a) - log(1 - D(s,a))  (logit form, more stable)

    ============================================================================
    TRAINING CONSIDERATIONS
    ============================================================================
    - Discriminator shouldn't be too strong (use few updates per policy update)
    - Gradient penalty helps with training stability
    - Label smoothing prevents overconfident predictions

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Time: O(forward + backward) per update
    - Space: O(|θ|) for discriminator parameters
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[GAILConfig] = None,
    ) -> None:
        """Initialize GAIL discriminator.

        Args:
            state_dim: Dimensionality of state space.
            action_dim: Dimensionality of action space.
            config: Discriminator configuration.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or GAILConfig()

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

        self._update_count: int = 0

    def _init_network(self) -> None:
        """Initialize discriminator network."""
        input_dim = self.state_dim + self.action_dim
        dims = [input_dim] + list(self.config.hidden_dims) + [1]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )

    def forward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Forward pass through discriminator.

        Args:
            states: State observations, shape (batch, state_dim).
            actions: Actions, shape (batch, action_dim).

        Returns:
            D(s,a) probabilities, shape (batch,).
        """
        states = np.atleast_2d(states).astype(np.float64)
        actions = np.atleast_2d(actions).astype(np.float64)

        x = np.concatenate([states, actions], axis=1)

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.tanh(x)

        return self._sigmoid(x).flatten()

    def compute_reward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        reward_type: str = "gail",
    ) -> np.ndarray:
        """Compute reward signal from discriminator output.

        Args:
            states: State observations.
            actions: Actions taken.
            reward_type: Type of reward computation:
                - "gail": -log(1 - D(s,a))
                - "airl": log D(s,a) - log(1-D(s,a))
                - "fairl": log D(s,a) - log(1-D(s,a)) / (1-D(s,a))

        Returns:
            Reward values for each state-action pair.
        """
        d = self.forward(states, actions)
        d = np.clip(d, 1e-8, 1 - 1e-8)

        if reward_type == "gail":
            return -np.log(1 - d)
        elif reward_type == "airl":
            return np.log(d) - np.log(1 - d)
        elif reward_type == "fairl":
            return np.log(d) - np.log(1 - d) / (1 - d)
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")

    def update(
        self,
        expert_states: np.ndarray,
        expert_actions: np.ndarray,
        policy_states: np.ndarray,
        policy_actions: np.ndarray,
    ) -> Dict[str, float]:
        """Update discriminator via gradient descent.

        Args:
            expert_states: Batch of expert states.
            expert_actions: Batch of expert actions.
            policy_states: Batch of policy-generated states.
            policy_actions: Batch of policy-generated actions.

        Returns:
            Training statistics dictionary.
        """
        expert_states = np.atleast_2d(expert_states).astype(np.float64)
        expert_actions = np.atleast_2d(expert_actions).astype(np.float64)
        policy_states = np.atleast_2d(policy_states).astype(np.float64)
        policy_actions = np.atleast_2d(policy_actions).astype(np.float64)

        batch_size = len(expert_states)

        expert_labels = np.ones(batch_size) * (1 - self.config.label_smoothing)
        policy_labels = np.zeros(batch_size) + self.config.label_smoothing

        expert_input = np.concatenate([expert_states, expert_actions], axis=1)
        policy_input = np.concatenate([policy_states, policy_actions], axis=1)
        all_input = np.vstack([expert_input, policy_input])
        all_labels = np.concatenate([expert_labels, policy_labels])

        activations = [all_input.copy()]
        x = all_input

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.tanh(x)
            activations.append(x.copy())

        probs = self._sigmoid(x).flatten()
        probs = np.clip(probs, 1e-8, 1 - 1e-8)

        expert_loss = -np.mean(np.log(probs[:batch_size]))
        policy_loss = -np.mean(np.log(1 - probs[batch_size:]))
        total_loss = expert_loss + policy_loss

        delta = (probs - all_labels).reshape(-1, 1) / (2 * batch_size)

        for i in range(len(self._weights) - 1, -1, -1):
            grad_w = activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)

            self._weights[i] -= self.config.learning_rate * grad_w
            self._biases[i] -= self.config.learning_rate * grad_b

            if i > 0:
                delta = delta @ self._weights[i].T
                delta = delta * (1 - activations[i] ** 2)

        self._update_count += 1

        expert_accuracy = np.mean(probs[:batch_size] > 0.5)
        policy_accuracy = np.mean(probs[batch_size:] < 0.5)

        return {
            "total_loss": float(total_loss),
            "expert_loss": float(expert_loss),
            "policy_loss": float(policy_loss),
            "expert_accuracy": float(expert_accuracy),
            "policy_accuracy": float(policy_accuracy),
            "discriminator_updates": self._update_count,
        }


class AIRLDiscriminator:
    """Adversarial Inverse Reinforcement Learning discriminator.

    ============================================================================
    CORE IDEA
    ============================================================================
    AIRL modifies GAIL to recover a disentangled reward function that is
    robust to changes in environment dynamics. The discriminator has a
    specific structure that separates state reward from shaping.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    AIRL discriminator structure:

        D(s, a, s') = exp(f(s,a,s')) / (exp(f(s,a,s')) + π(a|s))

    where f decomposes as:

        f(s, a, s') = g(s, a) + γh(s') - h(s)

    - g(s, a): Reward function (transferable)
    - h(s): Potential shaping (absorbs dynamics)

    At optimum, g(s,a) ≈ A*(s,a) (advantage function).

    ============================================================================
    ADVANTAGES OVER GAIL
    ============================================================================
    - Recovers explicit, transferable reward function
    - More robust to environment changes
    - Can be used for downstream tasks

    ============================================================================
    REFERENCES
    ============================================================================
    [1] Fu, J., Luo, K., & Levine, S. (2018). Learning robust rewards with
        adversarial inverse reinforcement learning. ICLR.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        discount_factor: float = 0.99,
        learning_rate: float = 0.0003,
    ) -> None:
        """Initialize AIRL discriminator.

        Args:
            state_dim: State dimensionality.
            action_dim: Action dimensionality.
            hidden_dims: Hidden layer sizes.
            discount_factor: Discount factor γ for shaping.
            learning_rate: Learning rate.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.gamma = discount_factor
        self.learning_rate = learning_rate

        self._reward_weights: List[np.ndarray] = []
        self._reward_biases: List[np.ndarray] = []
        self._shaping_weights: List[np.ndarray] = []
        self._shaping_biases: List[np.ndarray] = []

        self._init_networks()

    def _init_networks(self) -> None:
        """Initialize reward and shaping networks."""
        reward_dims = [self.state_dim + self.action_dim] + list(self.hidden_dims) + [1]

        for i in range(len(reward_dims) - 1):
            fan_in, fan_out = reward_dims[i], reward_dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._reward_weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._reward_biases.append(np.zeros(fan_out, dtype=np.float64))

        shaping_dims = [self.state_dim] + list(self.hidden_dims) + [1]

        for i in range(len(shaping_dims) - 1):
            fan_in, fan_out = shaping_dims[i], shaping_dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._shaping_weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._shaping_biases.append(np.zeros(fan_out, dtype=np.float64))

    def _forward_reward(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Forward through reward network g(s,a)."""
        x = np.concatenate([states, actions], axis=1)

        for i, (w, b) in enumerate(zip(self._reward_weights, self._reward_biases)):
            x = x @ w + b
            if i < len(self._reward_weights) - 1:
                x = np.tanh(x)

        return x.flatten()

    def _forward_shaping(self, states: np.ndarray) -> np.ndarray:
        """Forward through shaping network h(s)."""
        x = states.copy()

        for i, (w, b) in enumerate(zip(self._shaping_weights, self._shaping_biases)):
            x = x @ w + b
            if i < len(self._shaping_weights) - 1:
                x = np.tanh(x)

        return x.flatten()

    def compute_f(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute f(s, a, s') = g(s,a) + γh(s') - h(s).

        Args:
            states: Current states.
            actions: Actions taken.
            next_states: Next states.
            dones: Terminal flags.

        Returns:
            f values for each transition.
        """
        states = np.atleast_2d(states).astype(np.float64)
        actions = np.atleast_2d(actions).astype(np.float64)
        next_states = np.atleast_2d(next_states).astype(np.float64)
        dones = np.atleast_1d(dones).astype(np.float64)

        g = self._forward_reward(states, actions)
        h_s = self._forward_shaping(states)
        h_s_next = self._forward_shaping(next_states)

        shaping = self.gamma * h_s_next * (1 - dones) - h_s

        return g + shaping

    def compute_reward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Compute disentangled reward g(s,a).

        This is the transferable reward signal, without shaping.

        Args:
            states: State observations.
            actions: Actions taken.

        Returns:
            Reward values g(s,a).
        """
        states = np.atleast_2d(states).astype(np.float64)
        actions = np.atleast_2d(actions).astype(np.float64)
        return self._forward_reward(states, actions)


class StateActionDiscriminator:
    """Simple state-action discriminator for behavioral cloning baseline.

    ============================================================================
    CORE IDEA
    ============================================================================
    A simpler discriminator that only classifies state-action pairs without
    the GAN dynamics. Useful for:
    - Density ratio estimation
    - Behavioral cloning with discriminative loss
    - Evaluating policy similarity to expert

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Trained with binary cross-entropy:

        L = -E_πE[log D(s,a)] - E_π[log(1 - D(s,a))]

    Can estimate density ratio:

        ρ_E(s,a) / ρ_π(s,a) ≈ D(s,a) / (1 - D(s,a))
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        learning_rate: float = 0.001,
    ) -> None:
        """Initialize state-action discriminator."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._init_network()

    def _init_network(self) -> None:
        """Initialize network parameters."""
        input_dim = self.state_dim + self.action_dim
        dims = [input_dim] + list(self.hidden_dims) + [1]

        for i in range(len(dims) - 1):
            fan_in, fan_out = dims[i], dims[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self._weights.append(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std
            )
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )

    def forward(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Forward pass returning D(s,a) probabilities."""
        states = np.atleast_2d(states).astype(np.float64)
        actions = np.atleast_2d(actions).astype(np.float64)

        x = np.concatenate([states, actions], axis=1)

        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ w + b
            if i < len(self._weights) - 1:
                x = np.maximum(0, x)

        return self._sigmoid(x).flatten()

    def estimate_density_ratio(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> np.ndarray:
        """Estimate density ratio ρ_E / ρ_π.

        Args:
            states: State observations.
            actions: Actions taken.

        Returns:
            Estimated density ratios.
        """
        d = self.forward(states, actions)
        d = np.clip(d, 1e-8, 1 - 1e-8)
        return d / (1 - d)


if __name__ == "__main__":
    print("=" * 70)
    print("Discriminator Networks - Validation Tests")
    print("=" * 70)

    np.random.seed(42)

    state_dim = 4
    action_dim = 2
    batch_size = 32

    # Test GAIL discriminator
    print("\n[Test 1] GAIL Discriminator")
    gail = GAILDiscriminator(state_dim, action_dim)

    expert_s = np.random.randn(batch_size, state_dim)
    expert_a = np.random.randn(batch_size, action_dim)
    policy_s = np.random.randn(batch_size, state_dim)
    policy_a = np.random.randn(batch_size, action_dim)

    d_expert = gail.forward(expert_s, expert_a)
    print(f"  Expert D(s,a) mean: {d_expert.mean():.4f}")

    stats = gail.update(expert_s, expert_a, policy_s, policy_a)
    print(f"  Training loss: {stats['total_loss']:.4f}")

    rewards = gail.compute_reward(policy_s, policy_a, reward_type="gail")
    print(f"  GAIL rewards: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
    print("  [PASS]")

    # Test AIRL discriminator
    print("\n[Test 2] AIRL Discriminator")
    airl = AIRLDiscriminator(state_dim, action_dim)

    next_s = np.random.randn(batch_size, state_dim)
    dones = np.zeros(batch_size)

    f_values = airl.compute_f(expert_s, expert_a, next_s, dones)
    print(f"  f(s,a,s') shape: {f_values.shape}")

    g_rewards = airl.compute_reward(expert_s, expert_a)
    print(f"  Disentangled reward g(s,a): mean={g_rewards.mean():.4f}")
    print("  [PASS]")

    # Test simple discriminator
    print("\n[Test 3] State-Action Discriminator")
    simple = StateActionDiscriminator(state_dim, action_dim)

    ratios = simple.estimate_density_ratio(expert_s, expert_a)
    print(f"  Density ratios: mean={ratios.mean():.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
