"""
Experience Buffers for Policy Gradient Methods.

================================================================================
核心思想 (Core Idea)
================================================================================
Experience buffers store agent-environment interactions for training. Different
buffer types serve different algorithmic needs:

1. **EpisodeBuffer**: Dynamic storage for complete trajectories (REINFORCE)
2. **RolloutBuffer**: Fixed-size storage for on-policy algorithms (A2C, PPO)

The key insight is that policy gradient methods require complete trajectories
or rollouts to compute returns and advantages, unlike off-policy methods that
can sample arbitrary transitions.

================================================================================
数学原理 (Mathematical Theory)
================================================================================
Trajectory Storage:
    τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)

    Each transition: (s_t, a_t, r_t, s_{t+1}, done_t)

Return Computation:
    Monte Carlo Return (unbiased, high variance):
        G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}

    Recursive form:
        G_t = r_t + γ G_{t+1}
        G_T = 0 (terminal)

Generalized Advantage Estimation (GAE, Schulman et al., 2016):
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)  (TD error)

    A^{GAE(γ,λ)}_t = Σ_{k=0}^{T-t} (γλ)^k δ_{t+k}

    Special cases:
        λ = 0: A_t = δ_t (TD(0), low variance, biased)
        λ = 1: A_t = G_t - V(s_t) (MC, high variance, unbiased)

    Recursive form:
        A_t = δ_t + γλ A_{t+1}
        A_T = 0

Return from GAE:
    G_t = A_t + V(s_t)

================================================================================
问题背景 (Problem Statement)
================================================================================
Challenge: How to efficiently store and process experience for policy learning?

Requirements:
    1. Store complete trajectories for return computation
    2. Support batch processing for GPU efficiency
    3. Handle variable-length episodes
    4. Compute advantages with configurable bias-variance trade-off

Solutions:
    - EpisodeBuffer: List-based, dynamic, for episodic algorithms
    - RolloutBuffer: Array-based, fixed-size, for batched algorithms

================================================================================
算法对比 (Comparison)
================================================================================
| Buffer Type    | Memory    | Access Pattern | Use Case           |
|----------------|-----------|----------------|-------------------|
| EpisodeBuffer  | Dynamic   | Sequential     | REINFORCE, MC     |
| RolloutBuffer  | Fixed     | Batched        | A2C, PPO          |
| ReplayBuffer   | Fixed     | Random         | DQN, SAC (off-policy) |
| PrioritizedRB  | Fixed     | Weighted       | PER-DQN           |

| GAE λ Value | Bias      | Variance  | Typical Use        |
|-------------|-----------|-----------|-------------------|
| 0.0         | High      | Low       | Fast learning     |
| 0.95        | Medium    | Medium    | PPO default       |
| 1.0         | None      | High      | REINFORCE         |

================================================================================
复杂度 (Complexity Analysis)
================================================================================
EpisodeBuffer:
    Add transition: O(1) amortized
    Compute returns: O(T) where T = episode length
    Memory: O(T × transition_size)

RolloutBuffer:
    Add transition: O(1)
    Compute advantages: O(N) where N = buffer size
    Get batches: O(N / batch_size) iterations
    Memory: O(N × transition_size), pre-allocated

================================================================================
算法总结 (Summary)
================================================================================
This module provides production-ready experience storage:

1. **Transition dataclass** for type-safe experience tuples
2. **EpisodeBuffer** for variable-length trajectory storage
3. **RolloutBuffer** for efficient batched training with GAE

Key design decisions:
- NumPy arrays for CPU storage, converted to tensors on demand
- Pre-allocation in RolloutBuffer avoids memory fragmentation
- GAE computation uses efficient reverse iteration
- Batch generation with optional shuffling for SGD

References
----------
[1] Schulman et al. (2016). High-Dimensional Continuous Control Using GAE.
[2] Mnih et al. (2016). Asynchronous Methods for Deep RL (A3C).
[3] Schulman et al. (2017). Proximal Policy Optimization Algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    """
    Single environment transition tuple.

    核心思想 (Core Idea):
        Immutable record of one step of agent-environment interaction.
        Forms the atomic unit of experience for all RL algorithms.

    数学原理 (Mathematical Theory):
        Markov Decision Process transition:
            (s_t, a_t, r_t, s_{t+1}, done_t)

        Where:
            s_t ∈ S: current state
            a_t ∈ A: action taken
            r_t ∈ R: reward received
            s_{t+1} ∈ S: next state
            done_t ∈ {0, 1}: episode termination flag

    Attributes
    ----------
    state : np.ndarray
        Current state observation.
    action : np.ndarray
        Action taken (int for discrete, array for continuous).
    reward : float
        Scalar reward signal.
    next_state : np.ndarray
        Resulting state after action.
    done : bool
        True if episode terminated after this transition.
    log_prob : Optional[float]
        Log probability of action under policy (for policy gradient).
    value : Optional[float]
        Value estimate V(s_t) from critic (for actor-critic).
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: Optional[float] = None
    value: Optional[float] = None


class EpisodeBuffer:
    """
    Dynamic buffer for storing complete episodes.

    核心思想 (Core Idea):
        Accumulate transitions during episode rollout, then compute
        discounted returns for policy gradient training. Designed for
        algorithms that require complete trajectories (REINFORCE).

    数学原理 (Mathematical Theory):
        Episode trajectory:
            τ = {(s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_{T-1}, a_{T-1}, r_{T-1})}

        Discounted return at timestep t:
            G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^{T-t-1}r_{T-1}
                = Σ_{k=0}^{T-t-1} γ^k r_{t+k}

        Efficient computation (reverse iteration):
            G_{T-1} = r_{T-1}
            G_t = r_t + γ G_{t+1}

        Normalized returns (optional, for stability):
            G̃_t = (G_t - μ_G) / (σ_G + ε)

    问题背景 (Problem Statement):
        REINFORCE and Monte Carlo methods need complete episode returns.
        Challenge: Episodes have variable length, returns depend on future.

        Solution: Store transitions in list, compute returns backward.

    Parameters
    ----------
    gamma : float, default=0.99
        Discount factor for return computation.

    Examples
    --------
    >>> buffer = EpisodeBuffer(gamma=0.99)
    >>>
    >>> # Collect episode
    >>> for t in range(episode_length):
    ...     buffer.add(state, action, reward, next_state, done, log_prob)
    >>>
    >>> # Get training data
    >>> states, actions, returns, log_probs = buffer.get_training_data()
    >>> buffer.clear()

    Notes
    -----
    Complexity:
        add(): O(1) amortized
        compute_returns(): O(T)
        get_training_data(): O(T)
        Memory: O(T × state_dim)
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
        self.transitions: List[Transition] = []
        self._returns: Optional[np.ndarray] = None

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: Optional[float] = None,
        value: Optional[float] = None,
    ) -> None:
        """Add a transition to the buffer."""
        self.transitions.append(Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
        ))
        self._returns = None  # Invalidate cached returns

    def compute_returns(self, normalize: bool = True) -> np.ndarray:
        """
        Compute discounted returns for all timesteps.

        Parameters
        ----------
        normalize : bool, default=True
            If True, standardize returns to zero mean and unit variance.

        Returns
        -------
        returns : np.ndarray
            Discounted returns G_t for each timestep, shape (T,).
        """
        if self._returns is not None:
            return self._returns

        T = len(self.transitions)
        returns = np.zeros(T, dtype=np.float32)

        # Backward pass: G_t = r_t + γ G_{t+1}
        G = 0.0
        for t in reversed(range(T)):
            G = self.transitions[t].reward + self.gamma * G
            returns[t] = G

        if normalize and T > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        self._returns = returns
        return returns

    def get_training_data(
        self,
        normalize_returns: bool = True,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract training tensors from buffer.

        Returns
        -------
        states : torch.Tensor
            State observations, shape (T, state_dim).
        actions : torch.Tensor
            Actions taken, shape (T,) or (T, action_dim).
        returns : torch.Tensor
            Discounted returns, shape (T,).
        log_probs : torch.Tensor
            Log probabilities, shape (T,).
        """
        returns = self.compute_returns(normalize=normalize_returns)

        states = np.array([t.state for t in self.transitions])
        actions = np.array([t.action for t in self.transitions])
        log_probs = np.array([t.log_prob for t in self.transitions])

        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long if actions.ndim == 1 else torch.float32, device=device),
            torch.tensor(returns, dtype=torch.float32, device=device),
            torch.tensor(log_probs, dtype=torch.float32, device=device),
        )

    def clear(self) -> None:
        """Clear all stored transitions."""
        self.transitions.clear()
        self._returns = None

    def __len__(self) -> int:
        return len(self.transitions)


class RolloutBuffer:
    """
    Fixed-size buffer for on-policy rollout collection with GAE support.

    核心思想 (Core Idea):
        Pre-allocate arrays for efficient batched collection and training.
        Supports Generalized Advantage Estimation for bias-variance control.
        Designed for A2C, PPO, and similar on-policy algorithms.

    数学原理 (Mathematical Theory):
        Generalized Advantage Estimation (GAE):
            δ_t = r_t + γ V(s_{t+1}) - V(s_t)  (TD residual)

            A^{GAE}_t = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}

            Recursive computation:
                A_t = δ_t + γλ A_{t+1}

        Bias-variance trade-off controlled by λ:
            λ = 0: A_t = δ_t = r_t + γV(s') - V(s)
                   Low variance, high bias (TD(0))

            λ = 1: A_t = Σ_k γ^k r_{t+k} - V(s_t) = G_t - V(s_t)
                   High variance, no bias (Monte Carlo)

            λ ∈ (0,1): Interpolates between extremes
                       PPO default: λ = 0.95

        Return target for value function:
            G_t = A_t + V(s_t)

    问题背景 (Problem Statement):
        On-policy algorithms collect fixed-size rollouts, then update.
        Challenges:
            1. Efficient storage for GPU training
            2. Advantage computation with bootstrap values
            3. Mini-batch generation for multiple epochs

        Solution: Pre-allocated NumPy arrays with GAE computation.

    算法对比 (Comparison):
        | Feature          | EpisodeBuffer | RolloutBuffer |
        |------------------|---------------|---------------|
        | Size             | Variable      | Fixed         |
        | Memory           | Dynamic       | Pre-allocated |
        | GAE support      | No            | Yes           |
        | Multi-epoch      | No            | Yes           |
        | Use case         | REINFORCE     | A2C, PPO      |

    Parameters
    ----------
    buffer_size : int
        Maximum number of transitions to store.
    state_dim : int
        Dimension of state observations.
    action_dim : int
        Dimension of action space (1 for discrete).
    gamma : float, default=0.99
        Discount factor.
    gae_lambda : float, default=0.95
        GAE lambda parameter for bias-variance trade-off.

    Examples
    --------
    >>> buffer = RolloutBuffer(
    ...     buffer_size=2048,
    ...     state_dim=4,
    ...     action_dim=1,
    ...     gamma=0.99,
    ...     gae_lambda=0.95,
    ... )
    >>>
    >>> # Collect rollout
    >>> for _ in range(2048):
    ...     buffer.add(state, action, reward, value, log_prob, done)
    >>>
    >>> # Compute advantages with final value bootstrap
    >>> buffer.compute_advantages(last_value=critic(final_state))
    >>>
    >>> # Train for multiple epochs
    >>> for epoch in range(10):
    ...     for batch in buffer.get_batches(batch_size=64):
    ...         # Update policy and value function
    ...         pass
    >>>
    >>> buffer.reset()

    Notes
    -----
    Complexity:
        add(): O(1)
        compute_advantages(): O(N)
        get_batches(): O(N / batch_size) iterations
        Memory: O(N × (state_dim + action_dim + 5))
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        action_dim: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate arrays
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_advantages(
        self,
        last_value: float = 0.0,
        normalize: bool = True,
    ) -> None:
        """
        Compute GAE advantages and returns.

        Parameters
        ----------
        last_value : float
            Bootstrap value V(s_T) for incomplete episodes.
            Set to 0 if episode terminated, else critic estimate.
        normalize : bool, default=True
            If True, standardize advantages to zero mean, unit variance.
        """
        size = self.ptr if not self.full else self.buffer_size

        # GAE computation: A_t = δ_t + γλ A_{t+1}
        gae = 0.0
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            # TD residual: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )

            # GAE: A_t = δ_t + γλ A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        # Returns: G_t = A_t + V(s_t)
        self.returns[:size] = self.advantages[:size] + self.values[:size]

        if normalize:
            adv_mean = self.advantages[:size].mean()
            adv_std = self.advantages[:size].std() + 1e-8
            self.advantages[:size] = (self.advantages[:size] - adv_mean) / adv_std

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
        device: str = "cpu",
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate mini-batches for training.

        Parameters
        ----------
        batch_size : int
            Number of transitions per batch.
        shuffle : bool, default=True
            If True, randomize batch order each call.
        device : str, default="cpu"
            Device for output tensors.

        Yields
        ------
        batch : Tuple[torch.Tensor, ...]
            (states, actions, old_log_probs, advantages, returns)
        """
        size = self.ptr if not self.full else self.buffer_size
        indices = np.arange(size)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_indices = indices[start:end]

            yield (
                torch.tensor(self.states[batch_indices], dtype=torch.float32, device=device),
                torch.tensor(self.actions[batch_indices], dtype=torch.float32, device=device),
                torch.tensor(self.log_probs[batch_indices], dtype=torch.float32, device=device),
                torch.tensor(self.advantages[batch_indices], dtype=torch.float32, device=device),
                torch.tensor(self.returns[batch_indices], dtype=torch.float32, device=device),
            )

    def get_all(
        self,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, ...]:
        """Get all data as tensors (for small buffer sizes)."""
        size = self.ptr if not self.full else self.buffer_size

        return (
            torch.tensor(self.states[:size], dtype=torch.float32, device=device),
            torch.tensor(self.actions[:size], dtype=torch.float32, device=device),
            torch.tensor(self.log_probs[:size], dtype=torch.float32, device=device),
            torch.tensor(self.advantages[:size], dtype=torch.float32, device=device),
            torch.tensor(self.returns[:size], dtype=torch.float32, device=device),
        )

    def reset(self) -> None:
        """Reset buffer for new rollout collection."""
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        return self.ptr if not self.full else self.buffer_size


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Experience Buffer Module - Unit Tests")
    print("=" * 70)

    # Test parameters
    state_dim = 4
    action_dim = 1
    episode_length = 100
    gamma = 0.99
    gae_lambda = 0.95

    # Test EpisodeBuffer
    print("\n[1] Testing EpisodeBuffer...")
    episode_buffer = EpisodeBuffer(gamma=gamma)

    # Simulate episode
    for t in range(episode_length):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.array([np.random.randint(2)])
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = (t == episode_length - 1)
        log_prob = np.random.randn()

        episode_buffer.add(state, action, reward, next_state, done, log_prob)

    assert len(episode_buffer) == episode_length

    # Test return computation
    returns = episode_buffer.compute_returns(normalize=False)
    assert returns.shape == (episode_length,)

    # Verify return computation manually for last few steps
    manual_return = 0.0
    for t in reversed(range(episode_length)):
        manual_return = episode_buffer.transitions[t].reward + gamma * manual_return
        if t >= episode_length - 3:
            assert np.isclose(returns[t], manual_return, rtol=1e-5)

    # Test training data extraction
    states, actions, rets, log_probs = episode_buffer.get_training_data()
    assert states.shape == (episode_length, state_dim)
    assert actions.shape == (episode_length, 1)
    assert rets.shape == (episode_length,)

    episode_buffer.clear()
    assert len(episode_buffer) == 0
    print("    [PASS]")

    # Test RolloutBuffer
    print("\n[2] Testing RolloutBuffer...")
    buffer_size = 128
    rollout_buffer = RolloutBuffer(
        buffer_size=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    # Fill buffer
    for _ in range(buffer_size):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        reward = np.random.randn()
        value = np.random.randn()
        log_prob = np.random.randn()
        done = np.random.random() < 0.05

        rollout_buffer.add(state, action, reward, value, log_prob, done)

    assert len(rollout_buffer) == buffer_size

    # Test GAE computation
    last_value = np.random.randn()
    rollout_buffer.compute_advantages(last_value=last_value, normalize=True)

    # Verify advantages are normalized
    assert np.abs(rollout_buffer.advantages.mean()) < 0.1
    assert np.abs(rollout_buffer.advantages.std() - 1.0) < 0.1
    print(f"    Advantage mean: {rollout_buffer.advantages.mean():.4f}")
    print(f"    Advantage std: {rollout_buffer.advantages.std():.4f}")

    # Test batch generation
    batch_size = 32
    num_batches = 0
    total_samples = 0

    for batch in rollout_buffer.get_batches(batch_size=batch_size, shuffle=True):
        states, actions, old_log_probs, advantages, returns = batch
        assert states.shape[1] == state_dim
        assert actions.shape[1] == action_dim
        num_batches += 1
        total_samples += states.shape[0]

    assert total_samples == buffer_size
    print(f"    Generated {num_batches} batches, {total_samples} total samples")

    rollout_buffer.reset()
    assert len(rollout_buffer) == 0
    print("    [PASS]")

    # Test GAE correctness
    print("\n[3] Testing GAE computation correctness...")
    small_buffer = RolloutBuffer(
        buffer_size=5,
        state_dim=1,
        action_dim=1,
        gamma=0.9,
        gae_lambda=0.8,
    )

    # Known values for manual verification
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    values = [0.5, 1.0, 1.5, 2.0, 2.5]

    for i in range(5):
        small_buffer.add(
            state=np.array([i], dtype=np.float32),
            action=np.array([0], dtype=np.float32),
            reward=rewards[i],
            value=values[i],
            log_prob=0.0,
            done=False,
        )

    last_v = 3.0
    small_buffer.compute_advantages(last_value=last_v, normalize=False)

    # Manual GAE computation
    manual_advantages = np.zeros(5)
    gae = 0.0
    for t in reversed(range(5)):
        next_v = last_v if t == 4 else values[t + 1]
        delta = rewards[t] + 0.9 * next_v - values[t]
        gae = delta + 0.9 * 0.8 * gae
        manual_advantages[t] = gae

    assert np.allclose(small_buffer.advantages, manual_advantages, rtol=1e-5)
    print(f"    Computed advantages: {small_buffer.advantages}")
    print(f"    Manual advantages:   {manual_advantages}")
    print("    [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
