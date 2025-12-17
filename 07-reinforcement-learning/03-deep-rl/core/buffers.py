"""
Experience Replay Buffers for Deep Reinforcement Learning

This module provides production-grade implementations of experience storage
mechanisms for both off-policy (DQN family) and on-policy (A2C, PPO) algorithms.

Buffer Types:
    ReplayBuffer            - Uniform random sampling for off-policy learning
    PrioritizedReplayBuffer - TD-error weighted sampling with importance correction
    RolloutBuffer           - Trajectory storage with GAE for on-policy methods

Mathematical Foundations:
    Off-Policy Learning:
        Experience replay enables reuse of past transitions, breaking temporal
        correlations that violate SGD's i.i.d. assumption.

        Uniform: P(τ_i) = 1/|D| for all transitions τ_i ∈ D
        Prioritized: P(i) = p_i^α / Σ_k p_k^α, p_i = |δ_i| + ε

    On-Policy Learning:
        Trajectories must come from current policy π_θ. GAE computes
        bias-variance optimal advantage estimates:

        Â_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
        δ_t = r_t + γV(s_{t+1}) - V(s_t)

References:
    [1] Mnih et al., "Human-level control through deep RL", Nature 2015
    [2] Schaul et al., "Prioritized Experience Replay", ICLR 2016
    [3] Schulman et al., "High-Dimensional Continuous Control Using GAE", ICLR 2016
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random

import numpy as np
import torch


@dataclass
class Transition:
    """
    Atomic transition data structure for experience replay.

    Mathematical Representation:
        τ = (s_t, a_t, r_t, s_{t+1}, d_t)

    where:
        s_t ∈ S: Current state observation
        a_t ∈ A: Action executed
        r_t ∈ ℝ: Immediate reward received
        s_{t+1} ∈ S: Resulting next state
        d_t ∈ {0, 1}: Episode termination indicator

    Attributes:
        state: Current state observation vector
        action: Discrete action index
        reward: Scalar immediate reward
        next_state: Subsequent state observation
        done: Boolean episode termination flag
    """
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Uniform experience replay buffer for off-policy learning.

    Core Idea:
        Store past transitions and sample uniformly at random, breaking
        temporal correlations and enabling i.i.d. assumption for SGD.

    Mathematical Principle:
        Sampling distribution: P(τ_i) = 1/|D| for all τ_i ∈ D

        This decorrelates consecutive transitions which would otherwise
        violate the i.i.d. assumptions of stochastic gradient descent.

    Problem Statement:
        Raw online learning suffers from:
        1. Catastrophic forgetting: Recent experiences overwrite old knowledge
        2. Temporal correlation: Sequential samples violate i.i.d. assumption
        3. Poor sample efficiency: Each transition used only once

        Experience replay addresses all three issues.

    Algorithm Comparison:
        vs. Online Learning:
            + Breaks temporal correlations
            + Higher data efficiency (reuse)
            + Prevents catastrophic forgetting
            - Memory overhead O(buffer_size)
            - Off-policy only

        vs. Prioritized Replay:
            + Simpler: O(1) sampling
            + Unbiased estimates
            - No focus on important transitions
            - Slower convergence in sparse reward settings

    Complexity:
        Space: O(capacity × state_dim)
        push(): O(1) amortized
        sample(N): O(N)

    Attributes:
        capacity: Maximum number of stored transitions
    """

    def __init__(self, capacity: int = 100000) -> None:
        """
        Initialize uniform replay buffer.

        Args:
            capacity: Maximum buffer size. FIFO eviction when full.

        Raises:
            ValueError: If capacity <= 0
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self._capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """Return maximum buffer capacity."""
        return self._capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store single transition.

        Complexity: O(1) amortized

        Args:
            state: Current state s_t
            action: Action taken a_t
            reward: Reward received r_t
            next_state: Next state s_{t+1}
            done: Episode termination flag
        """
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Uniformly sample mini-batch.

        Complexity: O(batch_size)

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple (states, actions, rewards, next_states, dones):
                - states: (batch_size, state_dim) float32
                - actions: (batch_size,) int64
                - rewards: (batch_size,) float32
                - next_states: (batch_size, state_dim) float32
                - dones: (batch_size,) float32

        Raises:
            ValueError: If batch_size > buffer size
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size {batch_size} exceeds buffer size {len(self._buffer)}"
            )

        batch = random.sample(self._buffer, batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has sufficient samples for training."""
        return len(self._buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay (PER) with importance sampling correction.

    Core Idea:
        Sample transitions proportional to TD error magnitude, focusing
        learning on "surprising" experiences where predictions are wrong.

    Mathematical Principle:
        Sampling probability:
            P(i) = p_i^α / Σ_k p_k^α

        Priority assignment:
            p_i = |δ_i| + ε
            δ_i = r + γ max_{a'} Q(s', a'; θ⁻) - Q(s, a; θ)

        Importance sampling weights:
            w_i = (N × P(i))^{-β} / max_j w_j

        Parameters:
            α ∈ [0, 1]: Prioritization exponent
                α = 0: uniform sampling
                α = 1: fully prioritized
            β ∈ [0, 1]: IS correction exponent
                β = 0: no correction (biased)
                β = 1: full correction (unbiased)
            ε > 0: Small constant preventing zero priority

    Problem Statement:
        Uniform sampling treats all transitions equally, ignoring that
        high TD-error transitions contain more learning signal. In sparse
        reward environments, rare successful trajectories may be under-sampled.

    Algorithm Comparison:
        vs. Uniform Replay:
            + 2× faster convergence (Atari benchmarks)
            + Better in sparse reward settings
            + Focuses on model uncertainty
            - O(log N) sampling with sum-tree
            - Hyperparameter sensitivity (α, β)

        vs. Hindsight Experience Replay:
            + Works with any reward structure
            + Directly uses TD error signal
            - No synthetic goal generation

    Complexity (this simplified implementation):
        Space: O(capacity)
        push(): O(1)
        sample(): O(capacity + batch_size)
        update_priorities(): O(batch_size)

        Note: Production implementations use sum-tree for O(log N) sampling

    Attributes:
        alpha: Prioritization exponent
        beta: Importance sampling exponent
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6
    ) -> None:
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0=uniform, 1=full)
            beta: IS correction exponent (typically annealed 0.4→1.0)
            epsilon: Small constant preventing zero priority

        Raises:
            ValueError: If α or β not in [0, 1]
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

        self._buffer: List[Transition] = []
        self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._position: int = 0
        self._max_priority: float = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition with maximum priority.

        New experiences assigned max_priority to guarantee sampling
        before priority update.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        transition = Transition(state, action, reward, next_state, done)

        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._position] = transition

        self._priorities[self._position] = self._max_priority
        self._position = (self._position + 1) % self._capacity

    def sample(
        self,
        batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """
        Sample mini-batch with importance sampling weights.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple (states, actions, rewards, next_states, dones, indices, weights):
                - indices: Buffer indices for priority updates
                - weights: Importance sampling weights

        Raises:
            ValueError: If batch_size > buffer size
        """
        buffer_len = len(self._buffer)
        if batch_size > buffer_len:
            raise ValueError(f"batch_size {batch_size} exceeds buffer size {buffer_len}")

        priorities = self._priorities[:buffer_len]
        probs = priorities ** self._alpha
        probs = probs / probs.sum()

        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)

        weights = (buffer_len * probs[indices]) ** (-self._beta)
        weights = weights / weights.max()

        batch = [self._buffer[i] for i in indices]
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return (states, actions, rewards, next_states, dones,
                indices.astype(np.int64), weights.astype(np.float32))

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        """
        Update priorities based on TD errors.

        Priority: p_i = |δ_i| + ε

        Args:
            indices: Buffer indices to update
            td_errors: Corresponding TD errors
        """
        priorities = np.abs(td_errors) + self._epsilon
        for idx, priority in zip(indices, priorities):
            self._priorities[idx] = priority
        self._max_priority = max(self._max_priority, priorities.max())

    def update_beta(self, beta: float) -> None:
        """Update IS correction exponent (typically annealed to 1.0)."""
        self._beta = min(1.0, beta)

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self._buffer) >= batch_size


class RolloutBuffer:
    """
    On-policy trajectory buffer with Generalized Advantage Estimation.

    Core Idea:
        Collect complete trajectories from current policy for on-policy
        learning, then compute GAE advantages for variance-reduced gradients.

    Mathematical Principle:
        Generalized Advantage Estimation (GAE):
            Â_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

        where TD error:
            δ_t = r_t + γ(1-d_t)V(s_{t+1}) - V(s_t)

        Recursive computation:
            Â_t = δ_t + γλ(1-d_t)Â_{t+1}

        Returns:
            R_t = Â_t + V(s_t)

        Bias-Variance Tradeoff:
            λ = 0: TD residual (low variance, high bias)
            λ = 1: Monte Carlo (high variance, low bias)
            λ ∈ (0.9, 0.99): Optimal balance

    Problem Statement:
        Policy gradient suffers from high variance. Solutions:
        1. Baseline subtraction: V(s) reduces variance without bias
        2. GAE: Exponential weighting of n-step returns
        3. On-policy constraint: Data must come from current policy

    Algorithm Comparison:
        vs. Uniform Replay (DQN):
            + On-policy correctness for policy gradient
            + Complete trajectories for GAE
            - Cannot reuse old data (except PPO's limited reuse)
            - Lower sample efficiency

        vs. Simple A = R - V:
            + Lower variance via exponential weighting
            + Bias control via λ parameter
            - Additional hyperparameter

    Complexity:
        Space: O(n_steps × state_dim)
        add(): O(1)
        compute_gae(): O(n_steps)

    Attributes:
        gamma: Discount factor γ ∈ [0, 1]
        gae_lambda: GAE parameter λ ∈ [0, 1]
    """

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> None:
        """
        Initialize rollout buffer.

        Args:
            gamma: Discount factor for future rewards
            gae_lambda: GAE bias-variance tradeoff parameter

        Raises:
            ValueError: If parameters not in valid range
        """
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not 0 <= gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self) -> None:
        """Clear buffer for new rollout."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ) -> None:
        """
        Add single timestep to buffer.

        Args:
            state: State observation s_t
            action: Action taken a_t
            log_prob: Log probability log π(a_t|s_t)
            reward: Reward received r_t
            value: Value estimate V(s_t)
            done: Episode termination flag
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(
        self,
        last_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns via dynamic programming.

        Algorithm:
            δ_t = r_t + γ(1-d_t)V(s_{t+1}) - V(s_t)
            Â_t = δ_t + γλ(1-d_t)Â_{t+1}
            R_t = Â_t + V(s_t)

        Args:
            last_value: Bootstrap value V(s_T) for incomplete episodes

        Returns:
            Tuple (returns, advantages) as torch tensors

        Complexity: O(n_steps) via single backward pass
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        n_steps = len(rewards)

        values = np.append(values, last_value)

        advantages = np.zeros(n_steps, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n_steps)):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        return torch.FloatTensor(returns), torch.FloatTensor(advantages)

    def get_batch(
        self,
        last_value: float,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Get complete batch with computed advantages.

        Args:
            last_value: Bootstrap value for final state
            device: Target device (CPU/GPU)

        Returns:
            Tuple (states, actions, log_probs, returns, advantages)
        """
        returns, advantages = self.compute_gae(last_value)

        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        log_probs = torch.FloatTensor(self.log_probs).to(device)

        return states, actions, log_probs, returns.to(device), advantages.to(device)

    def __len__(self) -> int:
        return len(self.states)


if __name__ == "__main__":
    print("Buffer Tests")
    print("=" * 50)

    buffer = ReplayBuffer(capacity=1000)
    for i in range(100):
        buffer.push(
            state=np.random.randn(4),
            action=i % 2,
            reward=float(i),
            next_state=np.random.randn(4),
            done=False
        )

    assert len(buffer) == 100
    states, actions, rewards, next_states, dones = buffer.sample(32)
    assert states.shape == (32, 4)
    print("ReplayBuffer: OK")

    per_buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.6, beta=0.4)
    for i in range(100):
        per_buffer.push(
            state=np.random.randn(4),
            action=i % 2,
            reward=float(i),
            next_state=np.random.randn(4),
            done=False
        )

    batch = per_buffer.sample(32)
    assert len(batch) == 7
    per_buffer.update_priorities(batch[5], np.random.randn(32))
    print("PrioritizedReplayBuffer: OK")

    rollout = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
    for i in range(10):
        rollout.add(
            state=np.random.randn(4),
            action=0,
            log_prob=-0.5,
            reward=1.0,
            value=0.5,
            done=(i == 9)
        )

    returns, advantages = rollout.compute_gae(last_value=0.0)
    assert returns.shape == (10,)
    assert advantages.shape == (10,)
    print("RolloutBuffer: OK")

    print("\nAll buffer tests passed!")
