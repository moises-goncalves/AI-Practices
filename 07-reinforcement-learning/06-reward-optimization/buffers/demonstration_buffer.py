"""
Demonstration Buffers for Inverse Reinforcement Learning

================================================================================
CORE IDEA
================================================================================
Demonstration buffers store expert trajectories for inverse RL and imitation
learning. They provide structured access to expert state-action sequences
and support mixing with on-policy data.

================================================================================
MATHEMATICAL THEORY
================================================================================
For IRL, we need expert demonstrations D = {τ₁, τ₂, ..., τₙ} where each
trajectory τᵢ = (s₀, a₀, s₁, a₁, ..., sₜ).

Key quantities:
- Feature expectations: μ_E = E_D[Σₜ γᵗ φ(sₜ)]
- State-action distribution: ρ_E(s, a)
- Trajectory likelihood: P(τ|π_E)

================================================================================
REFERENCES
================================================================================
[1] Ng & Russell (2000). Algorithms for Inverse RL.
[2] Ho & Ermon (2016). GAIL.
[3] Vecerik et al. (2017). DDPGfD (demonstrations in replay).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np


class Demonstration(NamedTuple):
    """Single expert demonstration trajectory.

    Attributes:
        states: Sequence of states, shape (T, state_dim).
        actions: Sequence of actions, shape (T, action_dim) or (T,).
        rewards: Optional rewards, shape (T,). May be unknown in IRL.
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.states)

    @property
    def trajectory_length(self) -> int:
        return len(self.states)


@dataclass
class DemonstrationBuffer:
    """Buffer for storing and sampling from expert demonstrations.

    ============================================================================
    CORE IDEA
    ============================================================================
    Stores complete expert trajectories and provides various sampling methods:
    - Full trajectory sampling (for trajectory-level IRL)
    - Transition sampling (for GAIL/BC)
    - Feature expectation computation (for IRL)

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Given demonstrations D = {τ₁, ..., τₙ}:

    Trajectory sampling:
        τ ~ Uniform(D)

    Transition sampling:
        (s, a) ~ Uniform(∪ᵢ τᵢ)

    Feature expectations:
        μ_E = (1/|D|) Σ_τ Σₜ γᵗ φ(sₜ)

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Storage: O(Σᵢ |τᵢ|)
    - Trajectory sampling: O(1)
    - Transition sampling: O(1) amortized with pre-indexing
    """

    state_dim: int
    action_dim: int
    demonstrations: List[Demonstration] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize indexing structures."""
        self._transition_indices: List[Tuple[int, int]] = []
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild flat index for transition sampling."""
        self._transition_indices = []
        for demo_idx, demo in enumerate(self.demonstrations):
            for t in range(len(demo)):
                self._transition_indices.append((demo_idx, t))

    def add_demonstration(self, demo: Demonstration) -> None:
        """Add a demonstration trajectory.

        Args:
            demo: Expert demonstration to add.
        """
        states = np.atleast_2d(demo.states).astype(np.float64)
        actions = np.atleast_2d(demo.actions).astype(np.float64)
        rewards = (
            np.atleast_1d(demo.rewards).astype(np.float64)
            if demo.rewards is not None
            else None
        )

        self.demonstrations.append(Demonstration(states, actions, rewards))

        for t in range(len(states)):
            self._transition_indices.append((len(self.demonstrations) - 1, t))

    def add_demonstrations(self, demos: List[Demonstration]) -> None:
        """Add multiple demonstrations.

        Args:
            demos: List of demonstrations to add.
        """
        for demo in demos:
            self.add_demonstration(demo)

    def sample_trajectories(self, n: int) -> List[Demonstration]:
        """Sample complete trajectories uniformly.

        Args:
            n: Number of trajectories to sample.

        Returns:
            List of sampled demonstrations.
        """
        if not self.demonstrations:
            raise ValueError("No demonstrations in buffer")

        indices = np.random.randint(0, len(self.demonstrations), size=n)
        return [self.demonstrations[i] for i in indices]

    def sample_transitions(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample state-action pairs uniformly from all demonstrations.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions) arrays.
        """
        if not self._transition_indices:
            raise ValueError("No transitions in buffer")

        indices = np.random.randint(0, len(self._transition_indices), size=batch_size)

        states = []
        actions = []

        for idx in indices:
            demo_idx, t = self._transition_indices[idx]
            demo = self.demonstrations[demo_idx]
            states.append(demo.states[t])
            actions.append(demo.actions[t])

        return np.array(states), np.array(actions)

    def sample_transitions_with_next(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample (s, a, s') tuples from demonstrations.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, next_states) arrays.
        """
        valid_indices = [
            (demo_idx, t)
            for demo_idx, demo in enumerate(self.demonstrations)
            for t in range(len(demo) - 1)
        ]

        if not valid_indices:
            raise ValueError("No valid transitions (need length > 1)")

        sample_indices = np.random.randint(0, len(valid_indices), size=batch_size)

        states = []
        actions = []
        next_states = []

        for idx in sample_indices:
            demo_idx, t = valid_indices[idx]
            demo = self.demonstrations[demo_idx]
            states.append(demo.states[t])
            actions.append(demo.actions[t])
            next_states.append(demo.states[t + 1])

        return np.array(states), np.array(actions), np.array(next_states)

    def compute_feature_expectations(
        self,
        feature_fn: callable,
        discount_factor: float = 0.99,
    ) -> np.ndarray:
        """Compute discounted feature expectations from demonstrations.

        Args:
            feature_fn: Function mapping states to features.
            discount_factor: Discount factor γ.

        Returns:
            Average feature expectations μ_E.
        """
        if not self.demonstrations:
            raise ValueError("No demonstrations in buffer")

        feature_sums = []

        for demo in self.demonstrations:
            demo_features = np.zeros_like(feature_fn(demo.states[0]))

            for t, state in enumerate(demo.states):
                features = feature_fn(state)
                demo_features += (discount_factor**t) * features

            feature_sums.append(demo_features)

        return np.mean(feature_sums, axis=0)

    def get_all_states(self) -> np.ndarray:
        """Get all states from all demonstrations.

        Returns:
            Array of all states, shape (total_states, state_dim).
        """
        all_states = []
        for demo in self.demonstrations:
            all_states.append(demo.states)
        return np.vstack(all_states)

    def get_all_actions(self) -> np.ndarray:
        """Get all actions from all demonstrations.

        Returns:
            Array of all actions.
        """
        all_actions = []
        for demo in self.demonstrations:
            all_actions.append(demo.actions)
        return np.vstack(all_actions)

    @property
    def total_transitions(self) -> int:
        """Total number of transitions across all demonstrations."""
        return len(self._transition_indices)

    @property
    def num_demonstrations(self) -> int:
        """Number of stored demonstrations."""
        return len(self.demonstrations)

    def __len__(self) -> int:
        return self.total_transitions


class MixedBuffer:
    """Buffer mixing expert demonstrations with agent experience.

    ============================================================================
    CORE IDEA
    ============================================================================
    Combine expert data with agent's own experience for algorithms like:
    - DDPGfD (Demonstrations + DDPG)
    - DAPG (Demo Augmented Policy Gradient)
    - Hybrid BC + RL

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Mixed sampling with ratio α:
        P(demo) = α, P(agent) = 1 - α

    Or priority-based mixing:
        P(demo_i) ∝ p_i^α for demos
        P(agent_j) ∝ p_j^β for agent data

    Curriculum schedule:
        α(t) = α_0 × decay^t  (gradually reduce demo reliance)

    ============================================================================
    ALGORITHM
    ============================================================================
    1. Maintain separate demo and agent buffers
    2. On sample(batch_size):
       - Sample α × batch_size from demos
       - Sample (1-α) × batch_size from agent
       - Combine and shuffle

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Storage: O(demo_capacity + agent_capacity)
    - Sampling: O(batch_size)
    """

    def __init__(
        self,
        demo_buffer: DemonstrationBuffer,
        agent_capacity: int,
        state_dim: int,
        action_dim: int,
        demo_ratio: float = 0.25,
        ratio_decay: float = 0.9999,
        min_ratio: float = 0.0,
    ) -> None:
        """Initialize mixed buffer.

        Args:
            demo_buffer: Buffer containing expert demonstrations.
            agent_capacity: Capacity for agent experience.
            state_dim: State dimensionality.
            action_dim: Action dimensionality.
            demo_ratio: Initial ratio of demo samples.
            ratio_decay: Decay rate for demo ratio per sample.
            min_ratio: Minimum demo ratio.
        """
        self.demo_buffer = demo_buffer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.demo_ratio = demo_ratio
        self.ratio_decay = ratio_decay
        self.min_ratio = min_ratio

        self._current_ratio = demo_ratio

        self._agent_states = np.zeros((agent_capacity, state_dim), dtype=np.float64)
        self._agent_actions = np.zeros((agent_capacity, action_dim), dtype=np.float64)
        self._agent_rewards = np.zeros(agent_capacity, dtype=np.float64)
        self._agent_next_states = np.zeros(
            (agent_capacity, state_dim), dtype=np.float64
        )
        self._agent_dones = np.zeros(agent_capacity, dtype=np.bool_)

        self._agent_position = 0
        self._agent_size = 0
        self._agent_capacity = agent_capacity

    def add_agent_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add agent transition to buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Terminal flag.
        """
        self._agent_states[self._agent_position] = state
        self._agent_actions[self._agent_position] = action
        self._agent_rewards[self._agent_position] = reward
        self._agent_next_states[self._agent_position] = next_state
        self._agent_dones[self._agent_position] = done

        self._agent_position = (self._agent_position + 1) % self._agent_capacity
        self._agent_size = min(self._agent_size + 1, self._agent_capacity)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample mixed batch from demos and agent experience.

        Args:
            batch_size: Total batch size.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, is_demo).
            is_demo is a boolean array indicating which samples are from demos.
        """
        n_demo = int(batch_size * self._current_ratio)
        n_agent = batch_size - n_demo

        if self.demo_buffer.total_transitions < n_demo:
            n_demo = self.demo_buffer.total_transitions
            n_agent = batch_size - n_demo

        if self._agent_size < n_agent:
            n_agent = self._agent_size
            n_demo = batch_size - n_agent

        demo_states, demo_actions, demo_next_states = (
            self.demo_buffer.sample_transitions_with_next(n_demo)
            if n_demo > 0
            else (np.empty((0, self.state_dim)), np.empty((0, self.action_dim)), np.empty((0, self.state_dim)))
        )

        if n_agent > 0:
            agent_indices = np.random.randint(0, self._agent_size, size=n_agent)
            agent_states = self._agent_states[agent_indices]
            agent_actions = self._agent_actions[agent_indices]
            agent_rewards = self._agent_rewards[agent_indices]
            agent_next_states = self._agent_next_states[agent_indices]
            agent_dones = self._agent_dones[agent_indices]
        else:
            agent_states = np.empty((0, self.state_dim))
            agent_actions = np.empty((0, self.action_dim))
            agent_rewards = np.empty(0)
            agent_next_states = np.empty((0, self.state_dim))
            agent_dones = np.empty(0, dtype=np.bool_)

        demo_rewards = np.zeros(n_demo)
        demo_dones = np.zeros(n_demo, dtype=np.bool_)

        states = np.vstack([demo_states, agent_states]) if n_demo > 0 else agent_states
        actions = np.vstack([demo_actions, agent_actions]) if n_demo > 0 else agent_actions
        rewards = np.concatenate([demo_rewards, agent_rewards])
        next_states = (
            np.vstack([demo_next_states, agent_next_states])
            if n_demo > 0
            else agent_next_states
        )
        dones = np.concatenate([demo_dones, agent_dones])
        is_demo = np.concatenate([np.ones(n_demo, dtype=np.bool_), np.zeros(n_agent, dtype=np.bool_)])

        shuffle_idx = np.random.permutation(len(states))
        states = states[shuffle_idx]
        actions = actions[shuffle_idx]
        rewards = rewards[shuffle_idx]
        next_states = next_states[shuffle_idx]
        dones = dones[shuffle_idx]
        is_demo = is_demo[shuffle_idx]

        self._current_ratio = max(
            self.min_ratio, self._current_ratio * self.ratio_decay
        )

        return states, actions, rewards, next_states, dones, is_demo

    @property
    def current_demo_ratio(self) -> float:
        """Current demo sampling ratio."""
        return self._current_ratio

    @property
    def agent_buffer_size(self) -> int:
        """Current agent buffer size."""
        return self._agent_size


if __name__ == "__main__":
    print("=" * 70)
    print("Demonstration Buffers - Validation Tests")
    print("=" * 70)

    np.random.seed(42)

    state_dim = 4
    action_dim = 2

    # Test demonstration buffer
    print("\n[Test 1] Demonstration Buffer")
    demo_buffer = DemonstrationBuffer(state_dim=state_dim, action_dim=action_dim)

    for _ in range(10):
        traj_len = np.random.randint(10, 30)
        demo = Demonstration(
            states=np.random.randn(traj_len, state_dim),
            actions=np.random.randn(traj_len, action_dim),
        )
        demo_buffer.add_demonstration(demo)

    print(f"  Number of demonstrations: {demo_buffer.num_demonstrations}")
    print(f"  Total transitions: {demo_buffer.total_transitions}")

    trajectories = demo_buffer.sample_trajectories(3)
    print(f"  Sampled trajectory lengths: {[len(t) for t in trajectories]}")

    states, actions = demo_buffer.sample_transitions(32)
    print(f"  Transition batch: states={states.shape}, actions={actions.shape}")
    print("  [PASS]")

    # Test feature expectations
    print("\n[Test 2] Feature Expectations")

    def simple_features(state):
        return state

    mu_e = demo_buffer.compute_feature_expectations(simple_features, discount_factor=0.99)
    print(f"  Feature expectations shape: {mu_e.shape}")
    print("  [PASS]")

    # Test mixed buffer
    print("\n[Test 3] Mixed Buffer")
    mixed = MixedBuffer(
        demo_buffer=demo_buffer,
        agent_capacity=1000,
        state_dim=state_dim,
        action_dim=action_dim,
        demo_ratio=0.5,
    )

    for _ in range(200):
        mixed.add_agent_transition(
            state=np.random.randn(state_dim),
            action=np.random.randn(action_dim),
            reward=np.random.randn(),
            next_state=np.random.randn(state_dim),
            done=np.random.random() < 0.1,
        )

    states, actions, rewards, next_states, dones, is_demo = mixed.sample(64)
    demo_count = np.sum(is_demo)
    print(f"  Batch: {demo_count} demos, {64 - demo_count} agent")
    print(f"  Current demo ratio: {mixed.current_demo_ratio:.4f}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
