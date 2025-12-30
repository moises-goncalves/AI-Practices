"""
Proximal Policy Optimization (PPO)

This module implements PPO, a state-of-the-art policy gradient algorithm that
addresses the instability of policy gradient methods through clipped objectives.

Core Idea:
    PPO constrains policy updates to a trust region by clipping the probability
    ratio between old and new policies. This prevents destructively large policy
    updates while maintaining sample efficiency and stability.

Mathematical Theory:
    Clipped Surrogate Objective:
        L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

    where:
        r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
        Â_t = advantage estimate
        ε = clipping parameter (typically 0.2)

    Generalized Advantage Estimation (GAE):
        Â_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)

Problem Statement:
    Policy gradient methods suffer from:
    1. High variance in gradient estimates
    2. Instability from large policy updates
    3. Sample inefficiency

    PPO solves these by:
    1. Using clipped objectives to limit update magnitude
    2. Employing GAE for low-variance advantage estimates
    3. Enabling multiple epochs of updates per batch

Comparison with Baselines:
    vs Vanilla Policy Gradient:
        - More stable training
        - Better sample efficiency
        - Easier hyperparameter tuning

    vs TRPO:
        - Simpler implementation (no KL divergence constraint)
        - Comparable or better performance
        - Faster computation

Complexity:
    Time: O(batch_size * num_epochs) per update
    Space: O(batch_size * trajectory_length) for storing trajectories
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List
from .base import RLAgent, ExperienceBuffer


class PPOBuffer(ExperienceBuffer):
    """
    Experience buffer optimized for PPO.

    Stores trajectories and computes advantages using GAE.
    Supports multiple epochs of updates on the same batch.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize PPO buffer.

        Args:
            max_size: Maximum trajectory length
        """
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        value: float = None,
        log_prob: float = None
    ) -> None:
        """Add experience with optional value and log_prob."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        if value is not None:
            self.values.append(value)
        if log_prob is not None:
            self.log_probs.append(log_prob)

    def compute_advantages(
        self,
        values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> None:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            values: Value estimates for all states
            gamma: Discount factor
            gae_lambda: GAE parameter (0 < lambda <= 1)
        """
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(values)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns = advantages + values

    def sample(self, batch_size: int = None) -> Dict[str, np.ndarray]:
        """
        Return all experiences as a batch.

        Args:
            batch_size: Ignored (returns all data)

        Returns:
            Dictionary with all stored experiences and computed advantages
        """
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'next_states': np.array(self.next_states),
            'dones': np.array(self.dones),
            'advantages': self.advantages,
            'returns': self.returns,
            'log_probs_old': np.array(self.log_probs) if self.log_probs else None
        }

    def clear(self) -> None:
        """Clear all experiences."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages = []
        self.returns = []

    @property
    def size(self) -> int:
        """Return current buffer size."""
        return len(self.states)


class PPOActor(nn.Module):
    """
    Actor network for PPO with policy head.

    Outputs action distribution parameters and log probabilities.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        action_type: str = 'continuous'
    ):
        """Initialize PPO actor."""
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
        """Forward pass."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        if self.action_type == 'continuous':
            mean = torch.tanh(self.mean(x))
            return mean, self.log_std
        else:
            logits = self.logits(x)
            return logits, None

    def get_distribution(self, state: torch.Tensor):
        """Get action distribution."""
        if self.action_type == 'continuous':
            mean, log_std = self.forward(state)
            std = torch.exp(log_std)
            return torch.distributions.Normal(mean, std)
        else:
            logits, _ = self.forward(state)
            return torch.distributions.Categorical(logits=logits)


class PPOCritic(nn.Module):
    """
    Critic network for PPO.

    Estimates value function V(s).
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """Initialize PPO critic."""
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.value(x)


class PPO(RLAgent):
    """
    Proximal Policy Optimization (PPO) Algorithm.

    State-of-the-art policy gradient method with clipped objectives for
    stable and sample-efficient learning.

    Algorithm Summary:
        1. Collect trajectory using current policy
        2. Compute advantages using GAE
        3. For K epochs:
            a. Sample mini-batches from trajectory
            b. Compute clipped policy loss
            c. Compute value loss
            d. Update actor and critic
        4. Repeat until convergence
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        num_epochs: int = 10,
        batch_size: int = 64,
        hidden_dim: int = 128,
        action_type: str = 'continuous',
        device: str = 'cpu'
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            learning_rate: Learning rate for optimizers
            gamma: Discount factor
            gae_lambda: GAE parameter
            clip_ratio: PPO clipping parameter (ε)
            num_epochs: Number of epochs per update
            batch_size: Mini-batch size for updates
            hidden_dim: Hidden layer dimensionality
            action_type: 'continuous' or 'discrete'
            device: 'cpu' or 'cuda'
        """
        super().__init__(state_dim, action_dim, learning_rate, gamma)

        self.device = torch.device(device)
        self.action_type = action_type
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Initialize networks
        self.actor = PPOActor(state_dim, action_dim, hidden_dim, action_type).to(self.device)
        self.critic = PPOCritic(state_dim, hidden_dim).to(self.device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Experience buffer
        self.buffer = PPOBuffer()

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Select action and return log probability and value estimate.

        Args:
            state: Current state observation

        Returns:
            Tuple of (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist = self.actor.get_distribution(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state_tensor)

            return (
                action.cpu().numpy()[0],
                log_prob.cpu().item(),
                value.cpu().item()
            )

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Update actor and critic using PPO objective.

        Args:
            batch: Dictionary with trajectory data and advantages

        Returns:
            Dictionary with loss metrics
        """
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        log_probs_old = torch.FloatTensor(batch['log_probs_old']).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0

        # Multiple epochs of updates
        for epoch in range(self.num_epochs):
            # Mini-batch updates
            indices = np.random.permutation(len(states))

            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]

                # Compute new log probabilities
                dist = self.actor.get_distribution(batch_states)
                log_probs_new = dist.log_prob(batch_actions).sum(dim=-1)

                # PPO clipped objective
                ratio = torch.exp(log_probs_new - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.mean(torch.min(surr1, surr2))

                # Critic loss
                values = self.critic(batch_states).squeeze()
                critic_loss = torch.mean((values - batch_returns) ** 2)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        num_updates = self.num_epochs * (len(states) // self.batch_size)

        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'mean_advantage': advantages.mean().item()
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
