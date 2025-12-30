"""
Vanilla Actor-Critic Algorithm

This module implements the foundational Actor-Critic algorithm, which combines
policy gradient methods (actor) with value-based methods (critic) to solve the
credit assignment problem.

Core Idea:
    The actor learns a policy π(a|s) to maximize expected returns, while the
    critic learns a value function V(s) to provide low-variance advantage
    estimates. The critic's TD error serves as the advantage signal for the
    actor, enabling efficient credit assignment.

Mathematical Theory:
    Policy Gradient Update:
        ∇_θ J(θ) = E[∇_θ log π_θ(a|s) * A(s,a)]

    Advantage Estimation (TD):
        A(s,a) = r + γV(s') - V(s)

    Critic Loss (MSE):
        L_critic = (r + γV(s') - V(s))^2

Problem Statement:
    The credit assignment problem asks: "How much credit should each action
    receive for the final outcome?" Actor-Critic solves this by:
    1. Using the critic's value estimate as a baseline to reduce variance
    2. Computing TD errors as advantage signals
    3. Enabling the actor to learn from immediate feedback

Comparison with Baselines:
    vs Policy Gradient (REINFORCE):
        - Lower variance through baseline subtraction
        - Faster convergence
        - Requires learning two networks

    vs Q-Learning:
        - Works with continuous action spaces
        - On-policy learning (more stable)
        - Better for high-dimensional problems

Complexity:
    Time: O(1) per update step
    Space: O(state_dim + action_dim) for network parameters
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
from .base import RLAgent, ExperienceBuffer


class SimpleBuffer(ExperienceBuffer):
    """
    Simple on-policy experience buffer for Actor-Critic.

    Stores experiences from a single episode and clears after each update.
    Suitable for on-policy algorithms like vanilla Actor-Critic.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize the buffer.

        Args:
            max_size: Maximum number of experiences to store
        """
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Add a single experience."""
        if len(self.states) >= self.max_size:
            self.clear()

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, batch_size: int = None) -> Dict[str, np.ndarray]:
        """
        Return all experiences as a batch.

        Args:
            batch_size: Ignored for on-policy buffer (returns all data)

        Returns:
            Dictionary with all stored experiences
        """
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'next_states': np.array(self.next_states),
            'dones': np.array(self.dones)
        }

    def clear(self) -> None:
        """Clear all experiences."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    @property
    def size(self) -> int:
        """Return current buffer size."""
        return len(self.states)


class ActorNetwork(nn.Module):
    """
    Neural network for the policy (actor).

    Maps states to action distributions. For continuous actions, outputs
    mean and log-std. For discrete actions, outputs action logits.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        action_type: str = 'continuous'
    ):
        """
        Initialize actor network.

        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            hidden_dim: Hidden layer dimensionality
            action_type: 'continuous' or 'discrete'
        """
        super().__init__()
        self.action_type = action_type

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if action_type == 'continuous':
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            For continuous: (mean, log_std)
            For discrete: (logits, None)
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        if self.action_type == 'continuous':
            mean = torch.tanh(self.mean(x))
            return mean, self.log_std
        else:
            logits = self.logits(x)
            return logits, None


class CriticNetwork(nn.Module):
    """
    Neural network for the value function (critic).

    Maps states to scalar value estimates V(s).
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        Initialize critic network.

        Args:
            state_dim: Dimensionality of state space
            hidden_dim: Hidden layer dimensionality
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of state.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Value estimates of shape (batch_size, 1)
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.value(x)


class VanillaActorCritic(RLAgent):
    """
    Vanilla Actor-Critic Algorithm Implementation.

    Combines policy gradient (actor) with value-based learning (critic)
    to efficiently solve the credit assignment problem.

    Algorithm Summary:
        1. Collect trajectory: (s, a, r, s', done)
        2. Compute TD error: δ = r + γV(s') - V(s)
        3. Update critic: minimize (δ)^2
        4. Update actor: maximize log π(a|s) * δ
        5. Repeat until convergence
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 128,
        action_type: str = 'continuous',
        device: str = 'cpu'
    ):
        """
        Initialize Vanilla Actor-Critic agent.

        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            learning_rate: Learning rate for both networks
            gamma: Discount factor
            hidden_dim: Hidden layer dimensionality
            action_type: 'continuous' or 'discrete'
            device: 'cpu' or 'cuda'
        """
        super().__init__(state_dim, action_dim, learning_rate, gamma)

        self.device = torch.device(device)
        self.action_type = action_type
        self.hidden_dim = hidden_dim

        # Initialize networks
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dim, action_type
        ).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dim).to(self.device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Experience buffer
        self.buffer = SimpleBuffer()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state: Current state observation

        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.action_type == 'continuous':
                mean, log_std = self.actor(state_tensor)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                return action.cpu().numpy()[0]
            else:
                logits, _ = self.actor(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                return action.cpu().numpy()[0]

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update actor and critic networks.

        Args:
            batch: Dictionary with 'states', 'actions', 'rewards', 'next_states', 'dones'

        Returns:
            Dictionary with loss metrics
        """
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)

        # Compute values for advantage estimation
        values = self.critic(states).squeeze()
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            td_target = rewards + self.gamma * next_values * (1 - dones)

        # Update critic
        critic_loss = torch.mean((values - td_target) ** 2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Recompute advantages after critic update
        with torch.no_grad():
            values = self.critic(states).squeeze()
            advantages = td_target - values

        # Update actor
        if self.action_type == 'continuous':
            mean, log_std = self.actor(states)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
        else:
            logits, _ = self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions.squeeze().long())

        actor_loss = -torch.mean(log_probs * advantages.detach())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'mean_advantage': advantages.detach().mean().item()
        }

    def save(self, path: str) -> None:
        """Save actor and critic networks."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """Load actor and critic networks."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
