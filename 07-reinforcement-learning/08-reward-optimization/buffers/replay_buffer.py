"""
Standard Replay Buffer Implementations

================================================================================
CORE IDEA
================================================================================
Experience replay breaks the temporal correlation in online RL data by
storing transitions in a buffer and sampling random minibatches for training.
This improves sample efficiency and learning stability.

================================================================================
MATHEMATICAL THEORY
================================================================================
Without replay, Q-learning updates are:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

Problem: Consecutive samples are highly correlated, causing:
- Unstable learning (gradient noise)
- Catastrophic forgetting
- Poor sample efficiency

Solution: Uniformly sample from buffer B:
    (s, a, r, s', d) ~ Uniform(B)

This provides i.i.d.-like samples from the experience distribution.

================================================================================
REFERENCES
================================================================================
[1] Lin, L.J. (1992). Self-improving reactive agents based on RL.
[2] Mnih et al. (2015). Human-level control through deep RL.
[3] Schaul et al. (2015). Prioritized Experience Replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np


class Transition(NamedTuple):
    """Single experience transition (s, a, r, s', done).

    Attributes:
        state: Current state observation.
        action: Action taken.
        reward: Reward received.
        next_state: Resulting state observation.
        done: Whether episode terminated.
    """

    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class BatchedTransitions:
    """Batched transitions for efficient processing.

    Attributes:
        states: Batch of states, shape (batch, state_dim).
        actions: Batch of actions, shape (batch,) or (batch, action_dim).
        rewards: Batch of rewards, shape (batch,).
        next_states: Batch of next states, shape (batch, state_dim).
        dones: Batch of done flags, shape (batch,).
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray

    def __len__(self) -> int:
        return len(self.states)


class ReplayBuffer:
    """Uniform experience replay buffer.

    ============================================================================
    CORE IDEA
    ============================================================================
    Store transitions in a circular buffer and sample uniformly at random.
    Each transition has equal probability of being sampled.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Sampling probability for transition i:
        P(i) = 1/N  for all i in buffer

    This approximates sampling from the empirical distribution of experiences.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    | Buffer Type | Sampling | Complexity | Use Case |
    |-------------|----------|------------|----------|
    | Uniform     | Random   | O(1)       | General  |
    | Prioritized | TD-error | O(log N)   | DQN, TD  |
    | HER         | Goal     | O(k)       | Sparse   |

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Insertion: O(1)
    - Sampling: O(batch_size)
    - Space: O(capacity × transition_size)
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        continuous_actions: bool = False,
    ) -> None:
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            state_dim: Dimensionality of state observations.
            action_dim: Dimensionality of actions.
            continuous_actions: Whether actions are continuous vectors.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions

        self._states = np.zeros((capacity, state_dim), dtype=np.float64)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float64)

        if continuous_actions:
            self._actions = np.zeros((capacity, action_dim), dtype=np.float64)
        else:
            self._actions = np.zeros(capacity, dtype=np.int64)

        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._dones = np.zeros(capacity, dtype=np.bool_)

        self._position = 0
        self._size = 0

    def add(self, transition: Transition) -> None:
        """Add a single transition to the buffer.

        Args:
            transition: Experience tuple (s, a, r, s', done).
        """
        self._states[self._position] = transition.state
        self._actions[self._position] = transition.action
        self._rewards[self._position] = transition.reward
        self._next_states[self._position] = transition.next_state
        self._dones[self._position] = transition.done

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of transitions efficiently.

        Args:
            states: Batch of states.
            actions: Batch of actions.
            rewards: Batch of rewards.
            next_states: Batch of next states.
            dones: Batch of done flags.
        """
        batch_size = len(states)

        if self._position + batch_size <= self.capacity:
            end_pos = self._position + batch_size
            self._states[self._position : end_pos] = states
            self._actions[self._position : end_pos] = actions
            self._rewards[self._position : end_pos] = rewards
            self._next_states[self._position : end_pos] = next_states
            self._dones[self._position : end_pos] = dones
            self._position = end_pos % self.capacity
        else:
            for i in range(batch_size):
                self.add(
                    Transition(
                        states[i], actions[i], rewards[i], next_states[i], dones[i]
                    )
                )

        self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> BatchedTransitions:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batched transitions.

        Raises:
            ValueError: If buffer has fewer transitions than requested.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Not enough samples in buffer: {self._size} < {batch_size}"
            )

        indices = np.random.randint(0, self._size, size=batch_size)

        return BatchedTransitions(
            states=self._states[indices].copy(),
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_states=self._next_states[indices].copy(),
            dones=self._dones[indices].copy(),
        )

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self._size == self.capacity


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer.

    ============================================================================
    CORE IDEA
    ============================================================================
    Sample transitions with higher TD-error more frequently. These are
    transitions where the agent's prediction was most wrong, indicating
    they contain the most learning signal.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Priority for transition i based on TD-error δᵢ:
        pᵢ = |δᵢ| + ε  (proportional)
        pᵢ = 1/rank(i) (rank-based)

    Sampling probability:
        P(i) = pᵢ^α / Σⱼ pⱼ^α

    where α ∈ [0, 1] controls prioritization strength:
    - α = 0: Uniform sampling
    - α = 1: Full prioritization

    Importance sampling correction:
        wᵢ = (N · P(i))^(-β)

    where β ∈ [0, 1] anneals from 0 to 1 during training.

    ============================================================================
    ALGORITHM COMPARISON
    ============================================================================
    - Higher sample efficiency than uniform replay
    - More computation per sample (tree operations)
    - Requires careful hyperparameter tuning (α, β)

    ============================================================================
    DATA STRUCTURE
    ============================================================================
    Uses a sum tree for O(log N) sampling:
    - Leaf nodes: Individual priorities
    - Internal nodes: Sum of children
    - Root: Total priority sum

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Insertion: O(log N)
    - Sampling: O(batch_size × log N)
    - Priority update: O(log N)
    - Space: O(N) for tree + O(N × transition_size)
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int = 1,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
        continuous_actions: bool = False,
    ) -> None:
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions.
            state_dim: State dimensionality.
            action_dim: Action dimensionality.
            alpha: Priority exponent (0=uniform, 1=full prioritization).
            beta_start: Initial importance sampling exponent.
            beta_frames: Frames over which to anneal beta to 1.
            epsilon: Small constant added to priorities.
            continuous_actions: Whether actions are continuous.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.continuous_actions = continuous_actions

        self._tree_capacity = 1
        while self._tree_capacity < capacity:
            self._tree_capacity *= 2

        self._sum_tree = np.zeros(2 * self._tree_capacity - 1, dtype=np.float64)
        self._min_tree = np.full(2 * self._tree_capacity - 1, np.inf, dtype=np.float64)
        self._max_priority = 1.0

        self._states = np.zeros((capacity, state_dim), dtype=np.float64)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float64)

        if continuous_actions:
            self._actions = np.zeros((capacity, action_dim), dtype=np.float64)
        else:
            self._actions = np.zeros(capacity, dtype=np.int64)

        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._dones = np.zeros(capacity, dtype=np.bool_)

        self._position = 0
        self._size = 0
        self._frame = 0

    def _get_beta(self) -> float:
        """Get current beta value (annealed toward 1)."""
        return min(
            1.0, self.beta_start + (1.0 - self.beta_start) * self._frame / self.beta_frames
        )

    def _update_tree(self, idx: int, priority: float) -> None:
        """Update sum tree with new priority."""
        tree_idx = idx + self._tree_capacity - 1
        change = priority - self._sum_tree[tree_idx]

        self._sum_tree[tree_idx] = priority
        self._min_tree[tree_idx] = priority

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            left = 2 * tree_idx + 1
            right = 2 * tree_idx + 2

            self._sum_tree[tree_idx] += change
            self._min_tree[tree_idx] = min(
                self._min_tree[left], self._min_tree[right]
            )

    def _sample_idx(self, value: float) -> int:
        """Sample index from sum tree using value in [0, total_sum]."""
        idx = 0
        while idx < self._tree_capacity - 1:
            left = 2 * idx + 1
            right = 2 * idx + 2

            if value <= self._sum_tree[left]:
                idx = left
            else:
                value -= self._sum_tree[left]
                idx = right

        data_idx = idx - (self._tree_capacity - 1)
        return data_idx

    def add(self, transition: Transition) -> None:
        """Add transition with maximum priority.

        Args:
            transition: Experience tuple.
        """
        self._states[self._position] = transition.state
        self._actions[self._position] = transition.action
        self._rewards[self._position] = transition.reward
        self._next_states[self._position] = transition.next_state
        self._dones[self._position] = transition.done

        priority = self._max_priority**self.alpha
        self._update_tree(self._position, priority)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[BatchedTransitions, np.ndarray, np.ndarray]:
        """Sample prioritized batch with importance weights.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of:
            - Batched transitions
            - Indices for priority update
            - Importance sampling weights
        """
        if self._size < batch_size:
            raise ValueError(f"Not enough samples: {self._size} < {batch_size}")

        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        total_priority = self._sum_tree[0]
        segment = total_priority / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)

            idx = self._sample_idx(value)
            idx = min(idx, self._size - 1)
            indices[i] = idx
            priorities[i] = self._sum_tree[idx + self._tree_capacity - 1]

        beta = self._get_beta()
        min_priority = self._min_tree[0]
        max_weight = (self._size * min_priority) ** (-beta)

        weights = (self._size * priorities) ** (-beta)
        weights = weights / max_weight

        self._frame += 1

        batch = BatchedTransitions(
            states=self._states[indices].copy(),
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_states=self._next_states[indices].copy(),
            dones=self._dones[indices].copy(),
        )

        return batch, indices, weights

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """Update priorities based on TD-errors.

        Args:
            indices: Indices of sampled transitions.
            td_errors: Computed TD-errors for each transition.
        """
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        for idx, priority in zip(indices, priorities):
            self._update_tree(int(idx), float(priority))
            self._max_priority = max(self._max_priority, priority ** (1.0 / self.alpha))

    def __len__(self) -> int:
        return self._size


if __name__ == "__main__":
    print("=" * 70)
    print("Replay Buffers - Validation Tests")
    print("=" * 70)

    np.random.seed(42)

    state_dim = 4
    action_dim = 1

    # Test basic replay buffer
    print("\n[Test 1] Uniform Replay Buffer")
    buffer = ReplayBuffer(capacity=1000, state_dim=state_dim)

    for i in range(500):
        t = Transition(
            state=np.random.randn(state_dim),
            action=np.random.randint(0, 4),
            reward=np.random.randn(),
            next_state=np.random.randn(state_dim),
            done=np.random.random() < 0.1,
        )
        buffer.add(t)

    print(f"  Buffer size: {len(buffer)}")
    batch = buffer.sample(32)
    print(f"  Batch shapes: states={batch.states.shape}, rewards={batch.rewards.shape}")
    print("  [PASS]")

    # Test prioritized replay
    print("\n[Test 2] Prioritized Replay Buffer")
    per = PrioritizedReplayBuffer(capacity=1000, state_dim=state_dim)

    for i in range(500):
        t = Transition(
            state=np.random.randn(state_dim),
            action=np.random.randint(0, 4),
            reward=np.random.randn(),
            next_state=np.random.randn(state_dim),
            done=np.random.random() < 0.1,
        )
        per.add(t)

    batch, indices, weights = per.sample(32)
    print(f"  Batch size: {len(batch)}")
    print(f"  Weights range: [{weights.min():.4f}, {weights.max():.4f}]")

    td_errors = np.random.randn(32)
    per.update_priorities(indices, td_errors)
    print("  Priority update successful")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
