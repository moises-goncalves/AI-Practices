"""
Base classes and interfaces for policy gradient algorithms.

This module defines the abstract interfaces that all policy gradient algorithms must implement,
ensuring consistency and modularity across different algorithm variants.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn


class BasePolicy(ABC, nn.Module):
    """
    Abstract base class for policy networks.

    Core Idea:
        A policy π(a|s) maps states to action distributions. This class provides
        the interface for both discrete and continuous action spaces.

    Mathematical Theory:
        The policy is parameterized as π_θ(a|s), where θ are learnable parameters.
        For discrete actions: π_θ(a|s) = softmax(f_θ(s))
        For continuous actions: π_θ(a|s) = N(μ_θ(s), σ_θ(s))

    Problem Statement:
        Different environments require different policy architectures. This base class
        ensures all policies implement the same interface for sampling and evaluation.

    Complexity:
        Time: O(1) for forward pass (depends on network architecture)
        Space: O(|θ|) where |θ| is the number of parameters
    """

    def __init__(self, state_dim: int, action_dim: int, action_type: str = "discrete"):
        """
        Initialize the policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_type: Either "discrete" or "continuous"
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_type = action_type

        if action_type not in ["discrete", "continuous"]:
            raise ValueError(f"action_type must be 'discrete' or 'continuous', got {action_type}")

    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            For discrete: logits of shape (batch_size, action_dim)
            For continuous: mean and log_std of shape (batch_size, action_dim)
        """
        pass

    @abstractmethod
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            action: Sampled action
            log_prob: Log probability of the sampled action
        """
        pass

    @abstractmethod
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given state-action pairs.

        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)

        Returns:
            log_prob: Log probability of the action
            entropy: Entropy of the policy distribution
        """
        pass


class BaseValueFunction(ABC, nn.Module):
    """
    Abstract base class for value function networks.

    Core Idea:
        The value function V(s) estimates the expected cumulative reward from state s.
        This is crucial for reducing variance in policy gradient estimates.

    Mathematical Theory:
        V_φ(s) ≈ E[∑_t γ^t r_t | s_0 = s]
        where φ are learnable parameters and γ is the discount factor.

    Problem Statement:
        Accurate value function estimates significantly reduce variance in policy gradients,
        leading to faster and more stable training.

    Complexity:
        Time: O(1) for forward pass
        Space: O(|φ|) where |φ| is the number of parameters
    """

    def __init__(self, state_dim: int):
        """
        Initialize the value function network.

        Args:
            state_dim: Dimension of state space
        """
        super().__init__()
        self.state_dim = state_dim

    @abstractmethod
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Value estimates of shape (batch_size, 1)
        """
        pass

    def compute_loss(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target values.

        Args:
            state: State tensor
            target: Target value tensor

        Returns:
            MSE loss
        """
        predicted = self.forward(state)
        return torch.nn.functional.mse_loss(predicted, target)


class PolicyGradientAgent(ABC):
    """
    Abstract base class for policy gradient agents.

    Core Idea:
        Policy gradient methods directly optimize the policy by computing gradients
        of the expected return with respect to policy parameters.

    Mathematical Theory:
        ∇_θ J(θ) = E[∇_θ log π_θ(a|s) Q(s,a)]
        where J(θ) is the objective function and Q(s,a) is the action-value function.

    Problem Statement:
        Policy gradient methods are on-policy, meaning they require samples from the
        current policy. This base class provides the interface for training and evaluation.

    Algorithm Comparison:
        - REINFORCE: Simple, high variance, no baseline
        - Actor-Critic: Lower variance, uses value function baseline
        - A2C/A3C: Parallel variants with better sample efficiency
        - PPO/TRPO: Trust region methods for stable updates

    Complexity:
        Time: O(T) per episode where T is episode length
        Space: O(T) for trajectory storage
    """

    def __init__(
        self,
        policy: BasePolicy,
        value_function: Optional[BaseValueFunction] = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cpu"
    ):
        """
        Initialize the policy gradient agent.

        Args:
            policy: Policy network
            value_function: Value function network (optional)
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            device: Device to run on ("cpu" or "cuda")
        """
        self.policy = policy.to(device)
        self.value_function = value_function.to(device) if value_function else None
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = device

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )

        if self.value_function:
            self.value_optimizer = torch.optim.Adam(
                self.value_function.parameters(),
                lr=learning_rate
            )

    @abstractmethod
    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the policy loss for a batch of trajectories.

        Args:
            states: State tensor of shape (batch_size, state_dim)
            actions: Action tensor of shape (batch_size, action_dim)
            returns: Return tensor of shape (batch_size,)
            advantages: Advantage tensor (optional)

        Returns:
            Policy loss (scalar)
        """
        pass

    @abstractmethod
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            states: State tensor
            actions: Action tensor
            returns: Return tensor
            advantages: Advantage tensor (optional)

        Returns:
            Dictionary with loss values and metrics
        """
        pass

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Select an action given a state.

        Args:
            state: State as numpy array

        Returns:
            action: Selected action
            log_prob: Log probability of the action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob = self.policy.sample(state_tensor)

        return action.cpu().numpy().squeeze(), log_prob.cpu().item()

    def save(self, path: str) -> None:
        """Save agent parameters to disk."""
        checkpoint = {
            "policy": self.policy.state_dict(),
            "value_function": self.value_function.state_dict() if self.value_function else None,
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict() if self.value_function else None,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load agent parameters from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        if self.value_function and checkpoint["value_function"]:
            self.value_function.load_state_dict(checkpoint["value_function"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        if self.value_function and checkpoint.get("value_optimizer"):
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
