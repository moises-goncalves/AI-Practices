"""
Experience replay buffers for policy gradient algorithms.

This module provides efficient data structures for storing and sampling experience data.
"""

from typing import Tuple, Optional
import numpy as np
import torch


class ReplayBuffer:
    """
    Standard experience replay buffer for storing transitions.

    Core Idea:
        Store transitions (s, a, r, s', done) and sample mini-batches for training.
        This breaks correlations between consecutive samples and improves sample efficiency.

    Mathematical Theory:
        Experience replay reduces the variance of gradient estimates by sampling
        from a diverse set of past experiences rather than consecutive transitions.

    Problem Statement:
        In RL, consecutive transitions are highly correlated. Training on correlated
        data leads to unstable learning. Replay buffers solve this by storing and
        randomly sampling past experiences.

    Complexity:
        Time: O(1) for add, O(batch_size) for sample
        Space: O(capacity)
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        transition = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a mini-batch from the buffer.

        Args:
            batch_size: Size of mini-batch
            device: Device to place tensors on

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        transitions = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) == self.capacity


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Core Idea:
        Sample transitions with probability proportional to their TD error.
        This prioritizes learning from surprising or important transitions.

    Mathematical Theory:
        Sampling probability: P(i) ∝ |δ_i|^α

        where δ_i is the TD error and α controls how much prioritization is used.

        Importance sampling weights: w_i = (1 / (N * P(i)))^β

        where β is annealed from 0 to 1 during training.

    Problem Statement:
        Not all transitions are equally important for learning. Prioritized replay
        focuses on transitions with high TD error, leading to faster convergence.

    Advantages:
        1. Faster convergence by focusing on important transitions
        2. Better sample efficiency
        3. Reduced training time

    Complexity:
        Time: O(log capacity) for add/update, O(batch_size * log capacity) for sample
        Space: O(capacity)
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = no prioritization, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: float = 1.0
    ) -> None:
        """
        Add a transition with priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            td_error: TD error for priority (default: max priority)
        """
        transition = (state, action, reward, next_state, done)
        priority = (abs(td_error) + 1e-6) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities[len(self.buffer) - 1] = priority
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a prioritized mini-batch.

        Args:
            batch_size: Size of mini-batch
            device: Device to place tensors on

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights)
        """
        # Compute sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            batch_size,
            p=probabilities,
            replace=False
        )

        # Compute importance sampling weights
        weights = (1.0 / (len(self.buffer) * probabilities[indices])) ** self.beta
        weights = weights / weights.max()  # Normalize

        transitions = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, dones, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on new TD errors.

        Args:
            indices: Indices of transitions to update
            td_errors: New TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
