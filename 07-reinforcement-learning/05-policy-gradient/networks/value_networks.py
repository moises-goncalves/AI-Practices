"""
Value Function Network Architectures for Policy Gradient Methods.

This module provides neural network implementations for value function estimation,
which is crucial for reducing variance in policy gradient algorithms.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base import BaseValueFunction


class ValueNetwork(BaseValueFunction):
    """
    Standard value function network for state value estimation.

    Core Idea:
        The value function V(s) estimates the expected cumulative reward from state s.
        This is implemented as a simple feedforward neural network that maps states
        to scalar value estimates.

    Mathematical Theory:
        V_φ(s) ≈ E[∑_t γ^t r_t | s_0 = s]

        The network is trained to minimize the TD error:
        L(φ) = E[(V_φ(s) - (r + γV_φ(s')))²]

        or with Monte Carlo returns:
        L(φ) = E[(V_φ(s) - G_t)²]

    Problem Statement:
        Accurate value function estimates are essential for:
        1. Reducing variance in policy gradient estimates (baseline)
        2. Computing advantages for actor-critic methods
        3. Enabling more stable and sample-efficient training

    Complexity:
        Time: O(1) for forward pass
        Space: O(|φ|) where |φ| is the number of parameters
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu"
    ):
        """
        Initialize value function network.

        Args:
            state_dim: Dimension of state space
            hidden_dims: Dimensions of hidden layers
            activation: Activation function ("relu", "tanh", "elu")
        """
        super().__init__(state_dim)

        self.activation_fn = self._get_activation(activation)

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn)
            prev_dim = hidden_dim

        # Output layer for value estimate
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Value estimates of shape (batch_size, 1)
        """
        return self.network(state)


class DuelingValueNetwork(BaseValueFunction):
    """
    Dueling value function network with separate advantage and value streams.

    Core Idea:
        The dueling architecture separates the value function into two streams:
        1. Value stream: Estimates V(s) - the baseline value of the state
        2. Advantage stream: Estimates A(s) - the advantage of being in that state

        The final value is computed as: V(s) + A(s)

        This decomposition helps the network learn more stable value estimates,
        especially in environments where many actions have similar values.

    Mathematical Theory:
        V_dueling(s) = V(s) + (A(s) - mean(A(s)))

        The subtraction of mean(A(s)) is important for identifiability:
        without it, the network could arbitrarily shift values between
        the value and advantage streams.

        This is equivalent to:
        V_dueling(s) = V(s) + A(s) - E[A(s)]

    Problem Statement:
        In many environments, the value of a state is dominated by the state itself,
        not the specific action taken. The dueling architecture allows the network
        to learn this decomposition, leading to better generalization and faster
        convergence.

    Advantages:
        1. Better value estimates through explicit decomposition
        2. Faster convergence in environments with many similar-valued actions
        3. More interpretable network outputs
        4. Improved sample efficiency

    Complexity:
        Time: O(1) for forward pass
        Space: O(|φ|) where |φ| is the number of parameters
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu"
    ):
        """
        Initialize dueling value function network.

        Args:
            state_dim: Dimension of state space
            hidden_dims: Dimensions of hidden layers
            activation: Activation function
        """
        super().__init__(state_dim)

        self.activation_fn = self._get_activation(activation)

        # Shared feature extraction layers
        shared_layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims[:-1]:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(self.activation_fn)
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            self.activation_fn,
            nn.Linear(hidden_dims[-1], 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            self.activation_fn,
            nn.Linear(hidden_dims[-1], 1)
        )

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling value network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Value estimates of shape (batch_size, 1)
        """
        # Extract shared features
        features = self.shared_network(state)

        # Compute value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: V(s) + (A(s) - mean(A(s)))
        # The mean subtraction ensures identifiability
        combined_value = value + (advantage - advantage.mean(dim=0, keepdim=True))

        return combined_value
