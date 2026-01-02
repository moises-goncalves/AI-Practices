"""
Policy Network Architectures for Policy Gradient Methods.

This module provides neural network implementations for both discrete and continuous
action spaces, supporting various policy parameterizations.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from ..core.base import BasePolicy


class DiscretePolicy(BasePolicy):
    """
    Policy network for discrete action spaces.

    Core Idea:
        For discrete actions, the policy outputs logits for each action, which are
        converted to probabilities via softmax. Actions are sampled from this
        categorical distribution.

    Mathematical Theory:
        π_θ(a|s) = softmax(f_θ(s))_a

        where f_θ(s) is the neural network output (logits).

        Sampling: a ~ Categorical(π_θ(·|s))
        Log probability: log π_θ(a|s) = log softmax(f_θ(s))_a

    Problem Statement:
        Discrete action spaces are common in many RL environments (Atari, board games).
        We need an efficient way to parameterize and sample from discrete distributions.

    Complexity:
        Time: O(|A|) where |A| is the number of actions
        Space: O(|θ|) where |θ| is the number of parameters
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu"
    ):
        """
        Initialize discrete policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: Dimensions of hidden layers
            activation: Activation function ("relu", "tanh", "elu")
        """
        super().__init__(state_dim, action_dim, action_type="discrete")

        self.activation_fn = self._get_activation(activation)

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn)
            prev_dim = hidden_dim

        # Output layer for action logits
        layers.append(nn.Linear(prev_dim, action_dim))

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
        Forward pass through the policy network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Action logits of shape (batch_size, action_dim)
        """
        return self.network(state)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)

        Returns:
            action: Sampled action of shape (batch_size,) or scalar
            log_prob: Log probability of the sampled action
        """
        # Handle both batched and unbatched inputs
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        if squeeze_output:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return action, log_prob

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given state-action pairs.

        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size,)

        Returns:
            log_prob: Log probability of the action
            entropy: Entropy of the policy distribution
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(action.long()).unsqueeze(1)
        entropy = dist.entropy()

        return log_prob, entropy


class ContinuousPolicy(BasePolicy):
    """
    Policy network for continuous action spaces with fixed variance.

    Core Idea:
        For continuous actions, the policy outputs the mean of a Gaussian distribution.
        The variance is fixed (not learned), simplifying the parameterization.

    Mathematical Theory:
        π_θ(a|s) = N(μ_θ(s), σ²)

        where μ_θ(s) is the network output and σ is a fixed standard deviation.

        Sampling: a ~ N(μ_θ(s), σ²)
        Log probability: log π_θ(a|s) = -0.5 * ((a - μ)² / σ² + log(2πσ²))

    Problem Statement:
        Continuous action spaces require different handling than discrete spaces.
        Fixed variance is simpler but may limit exploration.

    Complexity:
        Time: O(1) for forward pass
        Space: O(|θ|) where |θ| is the number of parameters
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        std: float = 0.5
    ):
        """
        Initialize continuous policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
            activation: Activation function
            std: Fixed standard deviation for Gaussian policy
        """
        super().__init__(state_dim, action_dim, action_type="continuous")

        self.std = std
        self.activation_fn = self._get_activation(activation)

        # Build network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn)
            prev_dim = hidden_dim

        # Output layer for action mean
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Bound actions to [-1, 1]

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
        Forward pass through the policy network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Action mean of shape (batch_size, action_dim)
        """
        return self.network(state)

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: State tensor

        Returns:
            action: Sampled action
            log_prob: Log probability of the sampled action
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        mean = self.forward(state)
        dist = Normal(mean, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        if squeeze_output:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return action, log_prob

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given state-action pairs.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            log_prob: Log probability of the action
            entropy: Entropy of the policy distribution
        """
        mean = self.forward(state)
        dist = Normal(mean, self.std)

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class GaussianPolicy(BasePolicy):
    """
    Policy network for continuous action spaces with learned variance.

    Core Idea:
        The policy outputs both mean and log standard deviation, allowing the
        network to learn the exploration level. This is more flexible than
        fixed variance but requires careful handling of the log_std parameter.

    Mathematical Theory:
        π_θ(a|s) = N(μ_θ(s), σ_θ(s)²)

        where both μ_θ(s) and σ_θ(s) are network outputs.

        The network outputs log_std to ensure σ > 0:
        σ = exp(log_std)

    Problem Statement:
        Learning the variance allows the policy to adapt its exploration level
        during training, potentially leading to better sample efficiency.

    Complexity:
        Time: O(1) for forward pass
        Space: O(|θ|) where |θ| includes parameters for both mean and variance
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        log_std_init: float = -0.5
    ):
        """
        Initialize Gaussian policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Dimensions of hidden layers
            activation: Activation function
            log_std_init: Initial value for log standard deviation
        """
        super().__init__(state_dim, action_dim, action_type="continuous")

        self.activation_fn = self._get_activation(activation)

        # Build shared network layers
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn)
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*layers)

        # Mean output layer
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.mean_layer.weight.data.mul_(0.1)
        self.mean_layer.bias.data.mul_(0.0)

        # Log std parameter (not dependent on state)
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
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

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Tuple of (mean, log_std) each of shape (batch_size, action_dim)
        """
        features = self.shared_network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std.expand_as(mean)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.

        Args:
            state: State tensor

        Returns:
            action: Sampled action
            log_prob: Log probability of the sampled action
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        if squeeze_output:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return action, log_prob

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given state-action pairs.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            log_prob: Log probability of the action
            entropy: Entropy of the policy distribution
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy
