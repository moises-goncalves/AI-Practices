"""
Abstract Base Classes for Reinforcement Learning Agents

This module defines the foundational interfaces and abstract classes that all
RL agents must implement. It establishes the contract for policy learning,
value estimation, and experience collection.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class RLAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents.

    Core Idea:
        Defines the minimal interface that any RL agent must implement,
        ensuring consistency across different algorithm implementations.

    Attributes:
        state_dim (int): Dimensionality of the state space
        action_dim (int): Dimensionality of the action space
        learning_rate (float): Learning rate for parameter updates
        gamma (float): Discount factor for future rewards
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99
    ):
        """
        Initialize the RL agent.

        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            learning_rate: Learning rate for optimization
            gamma: Discount factor (0 < gamma <= 1)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        if not (0 < gamma <= 1):
            raise ValueError(f"Discount factor gamma must be in (0, 1], got {gamma}")

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action given the current state.

        Args:
            state: Current state observation

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update agent parameters using a batch of experiences.

        Args:
            batch: Dictionary containing:
                - 'states': State observations
                - 'actions': Actions taken
                - 'rewards': Rewards received
                - 'next_states': Next state observations
                - 'dones': Episode termination flags

        Returns:
            Dictionary of loss metrics for monitoring
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent parameters to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent parameters from disk."""
        pass


class PolicyNetwork(ABC):
    """
    Abstract base class for policy networks.

    Core Idea:
        Encapsulates the policy function π(a|s) that maps states to action
        distributions. Separates policy representation from learning logic.

    Mathematical Theory:
        The policy network parameterizes: π_θ(a|s) = P(a|s; θ)
        where θ are learnable parameters.
    """

    @abstractmethod
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the policy network.

        Args:
            state: State observation

        Returns:
            Tuple of (action_mean, action_std) for continuous actions
            or (action_logits,) for discrete actions
        """
        pass

    @abstractmethod
    def get_action_distribution(self, state: np.ndarray):
        """Get the action distribution for the given state."""
        pass


class ValueNetwork(ABC):
    """
    Abstract base class for value networks.

    Core Idea:
        Estimates the value function V(s) = E[∑_t γ^t r_t | s_t = s]
        which predicts the expected cumulative discounted reward from a state.

    Mathematical Theory:
        V(s) represents the baseline for advantage estimation:
        A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s)
    """

    @abstractmethod
    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Estimate the value of a state.

        Args:
            state: State observation

        Returns:
            Estimated value V(s)
        """
        pass


class ExperienceBuffer(ABC):
    """
    Abstract base class for experience buffers.

    Core Idea:
        Manages the collection and sampling of experiences (s, a, r, s', done)
        from agent-environment interactions. Different algorithms may require
        different sampling strategies (on-policy vs off-policy).
    """

    @abstractmethod
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a single experience to the buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the current number of experiences in the buffer."""
        pass
