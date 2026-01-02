"""
REINFORCE Algorithm Implementation

REINFORCE (REward Increment = Nonnegative Factor times Offset Reinforcement times Characteristic Eligibility)
is the foundational policy gradient algorithm that directly optimizes the policy using Monte Carlo returns.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from ..core.base import PolicyGradientAgent, BasePolicy, BaseValueFunction
from ..core.trajectory import Trajectory, TrajectoryBuffer


class REINFORCE(PolicyGradientAgent):
    """
    REINFORCE Algorithm - The Foundational Policy Gradient Method.

    Core Idea:
        REINFORCE directly estimates the policy gradient using Monte Carlo returns.
        It samples complete trajectories and updates the policy in the direction that
        increases the log probability of actions that led to high returns.

    Mathematical Theory:
        Policy Gradient Theorem:
        ∇_θ J(θ) = E[∇_θ log π_θ(a|s) G_t]

        where:
        - J(θ) = E[G_0] is the expected return from the start state
        - G_t = ∑_{k=0}^{T-t-1} γ^k r_{t+k} is the discounted return
        - ∇_θ log π_θ(a|s) is the score function

        Update Rule:
        θ ← θ + α ∇_θ log π_θ(a|s) G_t

    Problem Statement:
        Traditional supervised learning requires labeled data. In RL, we don't have
        labels but rewards. REINFORCE solves this by using the return as a "label"
        and the log probability gradient as the "feature". This allows us to apply
        gradient ascent directly on the expected return.

    Algorithm Comparison:
        vs. Q-Learning: On-policy vs off-policy; direct policy optimization vs value-based
        vs. Actor-Critic: Higher variance but simpler; no value function needed
        vs. PPO: No trust region; can have large policy updates

    Advantages:
        1. Simple and elegant - easy to understand and implement
        2. Theoretically sound - guaranteed convergence to local optima
        3. Works with any differentiable policy
        4. No need for value function (though one can be added for variance reduction)

    Disadvantages:
        1. High variance - requires many samples for stable estimates
        2. Sample inefficient - only uses on-policy data
        3. Slow convergence - especially in high-dimensional spaces
        4. Sensitive to reward scaling

    Complexity:
        Time: O(T) per episode where T is episode length
        Space: O(T) for storing trajectories
        Sample Complexity: O(1/ε²) for ε-optimal policy

    References:
        Williams, R. J. (1992). "Simple statistical gradient-following algorithms
        for connectionist reinforcement learning." Machine Learning, 8(3-4), 229-256.
    """

    def __init__(
        self,
        policy: BasePolicy,
        value_function: Optional[BaseValueFunction] = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize REINFORCE agent.

        Args:
            policy: Policy network π_θ(a|s)
            value_function: Optional value function for baseline (variance reduction)
            learning_rate: Learning rate for policy updates
            gamma: Discount factor
            entropy_coeff: Coefficient for entropy regularization
            device: Device to run on ("cpu" or "cuda")
        """
        super().__init__(
            policy=policy,
            value_function=value_function,
            learning_rate=learning_rate,
            gamma=gamma,
            device=device
        )
        self.entropy_coeff = entropy_coeff
        self.trajectory_buffer = TrajectoryBuffer(gamma=gamma)

    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute REINFORCE policy loss.

        Loss = -E[log π_θ(a|s) * G_t]

        The negative sign is because we use gradient descent, but we want to
        maximize the expected return.

        Args:
            states: State tensor of shape (batch_size, state_dim)
            actions: Action tensor of shape (batch_size, action_dim)
            returns: Return tensor of shape (batch_size, 1)
            advantages: Ignored in REINFORCE (uses returns directly)

        Returns:
            Policy loss (scalar)
        """
        # Compute log probabilities and entropy
        log_probs, entropy = self.policy.evaluate(states, actions)

        # REINFORCE loss: -E[log π(a|s) * G_t]
        policy_loss = -(log_probs * returns).mean()

        # Add entropy regularization to encourage exploration
        entropy_loss = -self.entropy_coeff * entropy.mean()

        total_loss = policy_loss + entropy_loss

        return total_loss

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one REINFORCE training step.

        Args:
            states: State tensor
            actions: Action tensor
            returns: Return tensor
            advantages: Ignored in REINFORCE

        Returns:
            Dictionary with loss values and metrics
        """
        # Normalize returns for stability
        returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        policy_loss = self.compute_policy_loss(states, actions, returns_normalized)

        # Policy update
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # Value function update (if available)
        value_loss = 0.0
        if self.value_function is not None:
            value_loss = self.value_function.compute_loss(states, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), max_norm=1.0)
            self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss,
        }

    def collect_trajectory(
        self,
        env,
        max_steps: int = 1000
    ) -> Trajectory:
        """
        Collect one complete trajectory from the environment.

        Args:
            env: Gym environment
            max_steps: Maximum steps per episode

        Returns:
            Trajectory object containing the episode data
        """
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []

        state, _ = env.reset()
        episode_return = 0.0

        for step in range(max_steps):
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob = self.policy.sample(state_tensor)

            action_np = action.cpu().numpy().squeeze()

            # Take environment step
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Store trajectory data
            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)

            episode_return += reward
            state = next_state

            if done:
                break

        trajectory = Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=log_probs
        )

        return trajectory, episode_return

    def train_episode(
        self,
        env,
        max_steps: int = 1000
    ) -> Dict[str, float]:
        """
        Train for one complete episode.

        Args:
            env: Gym environment
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with training metrics
        """
        # Collect trajectory
        trajectory, episode_return = self.collect_trajectory(env, max_steps)

        # Convert to tensors
        states = torch.FloatTensor(np.array(trajectory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(trajectory.actions)).to(self.device)
        log_probs = torch.FloatTensor(trajectory.log_probs).unsqueeze(1).to(self.device)

        # Compute returns
        returns = self.trajectory_buffer.compute_returns(trajectory)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

        # Training step
        metrics = self.train_step(states, actions, returns)
        metrics["episode_return"] = episode_return
        metrics["episode_length"] = len(trajectory)

        return metrics

    def train(
        self,
        env,
        num_episodes: int = 100,
        max_steps: int = 1000,
        eval_interval: int = 10
    ) -> Dict[str, list]:
        """
        Train the agent for multiple episodes.

        Args:
            env: Gym environment
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            eval_interval: Interval for evaluation

        Returns:
            Dictionary with training history
        """
        history = {
            "episode_returns": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
        }

        for episode in range(num_episodes):
            metrics = self.train_episode(env, max_steps)

            history["episode_returns"].append(metrics["episode_return"])
            history["episode_lengths"].append(metrics["episode_length"])
            history["policy_losses"].append(metrics["policy_loss"])
            history["value_losses"].append(metrics["value_loss"])

            if (episode + 1) % eval_interval == 0:
                avg_return = np.mean(history["episode_returns"][-eval_interval:])
                print(f"Episode {episode + 1}/{num_episodes} | Avg Return: {avg_return:.2f}")

        return history
