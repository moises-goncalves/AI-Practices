"""
Actor-Critic Algorithm Implementation

Actor-Critic methods combine policy gradient (actor) with value function learning (critic)
to reduce variance while maintaining the benefits of policy gradient methods.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
import torch.nn.functional as F

from ..core.base import PolicyGradientAgent, BasePolicy, BaseValueFunction
from ..core.trajectory import Trajectory, TrajectoryBuffer


class ActorCritic(PolicyGradientAgent):
    """
    Actor-Critic Algorithm - Combining Policy Gradient with Value Function Learning.

    Core Idea:
        The actor (policy) learns to select actions, while the critic (value function)
        learns to estimate the value of states. The critic provides a baseline to reduce
        variance in policy gradient estimates, leading to faster and more stable training.

    Mathematical Theory:
        Policy Gradient with Baseline:
        ∇_θ J(θ) = E[∇_θ log π_θ(a|s) (Q(s,a) - V(s))]
                  = E[∇_θ log π_θ(a|s) A(s,a)]

        where:
        - A(s,a) = Q(s,a) - V(s) is the advantage function
        - Q(s,a) ≈ r + γV(s') is the action-value estimate
        - V(s) is the state value estimate

        Temporal Difference (TD) Error:
        δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Actor Update:
        θ ← θ + α ∇_θ log π_θ(a|s) δ_t

        Critic Update:
        φ ← φ + β ∇_φ (δ_t)² = β δ_t ∇_φ V_φ(s)

    Problem Statement:
        REINFORCE has high variance because it uses Monte Carlo returns. Actor-Critic
        reduces this variance by using a learned value function as a baseline. This
        allows for more frequent updates and better sample efficiency.

    Algorithm Comparison:
        vs. REINFORCE: Lower variance, faster convergence, requires value function
        vs. Q-Learning: On-policy vs off-policy; direct policy optimization
        vs. PPO: No trust region; simpler but potentially less stable
        vs. A3C: Single-threaded vs parallel; simpler but slower

    Advantages:
        1. Lower variance than REINFORCE - faster convergence
        2. Can use TD learning - more sample efficient
        3. Works with continuous and discrete actions
        4. Theoretically sound - convergence guarantees

    Disadvantages:
        1. Requires learning two networks - more complex
        2. Critic can provide biased estimates - affects policy learning
        3. Hyperparameter tuning more critical
        4. Can be unstable if critic is poorly trained

    Complexity:
        Time: O(T) per episode where T is episode length
        Space: O(T) for storing trajectories
        Sample Complexity: O(1/ε) for ε-optimal policy (better than REINFORCE)

    References:
        Konda, V., & Tsitsiklis, J. N. (2000). "Actor-Critic Algorithms."
        SIAM Journal on Control and Optimization, 42(4), 1143-1166.
    """

    def __init__(
        self,
        policy: BasePolicy,
        value_function: BaseValueFunction,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize Actor-Critic agent.

        Args:
            policy: Policy network (actor) π_θ(a|s)
            value_function: Value function network (critic) V_φ(s)
            learning_rate: Learning rate for both actor and critic
            gamma: Discount factor
            entropy_coeff: Coefficient for entropy regularization
            device: Device to run on ("cpu" or "cuda")
        """
        if value_function is None:
            raise ValueError("Actor-Critic requires a value function (critic)")

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
        Compute Actor-Critic policy loss using advantage estimates.

        Loss = -E[log π_θ(a|s) * A(s,a)]

        where A(s,a) is the advantage function (TD error or GAE).

        Args:
            states: State tensor of shape (batch_size, state_dim)
            actions: Action tensor of shape (batch_size, action_dim)
            returns: Return tensor (used if advantages not provided)
            advantages: Advantage tensor of shape (batch_size, 1)

        Returns:
            Policy loss (scalar)
        """
        # Compute log probabilities and entropy
        log_probs, entropy = self.policy.evaluate(states, actions)

        # Use advantages if provided, otherwise compute from returns
        if advantages is None:
            with torch.no_grad():
                values = self.value_function(states)
            advantages = returns - values

        # Actor loss: -E[log π(a|s) * A(s,a)]
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Add entropy regularization
        entropy_loss = -self.entropy_coeff * entropy.mean()

        total_loss = policy_loss + entropy_loss

        return total_loss

    def compute_value_loss(
        self,
        states: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute critic (value function) loss.

        Loss = E[(V_φ(s) - G_t)²]

        Args:
            states: State tensor
            returns: Return tensor

        Returns:
            Value loss (scalar)
        """
        return self.value_function.compute_loss(states, returns)

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one Actor-Critic training step.

        Args:
            states: State tensor
            actions: Action tensor
            returns: Return tensor
            advantages: Advantage tensor (optional)

        Returns:
            Dictionary with loss values and metrics
        """
        # Normalize returns for stability
        returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Critic update
        value_loss = self.compute_value_loss(states, returns_normalized)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        # Compute advantages using updated value function
        with torch.no_grad():
            values = self.value_function(states)
            advantages = returns_normalized - values

        # Actor update
        policy_loss = self.compute_policy_loss(states, actions, returns_normalized, advantages)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def collect_trajectory(
        self,
        env,
        max_steps: int = 1000
    ) -> Tuple[Trajectory, float]:
        """
        Collect one complete trajectory from the environment.

        Args:
            env: Gym environment
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (trajectory, episode_return)
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


class A2C(PolicyGradientAgent):
    """
    Advantage Actor-Critic (A2C) - Synchronous Parallel Actor-Critic.

    Core Idea:
        A2C extends Actor-Critic by using Generalized Advantage Estimation (GAE)
        for more stable advantage estimates and batch updates across multiple
        parallel environments.

    Mathematical Theory:
        Generalized Advantage Estimation (GAE):
        A_t^GAE(γ,λ) = ∑_{l=0}^{∞} (γλ)^l δ_t^V

        where δ_t^V = r_t + γV(s_{t+1}) - V(s_t) is the TD error.

        This provides a trade-off between bias and variance:
        - λ = 0: Uses only one-step TD error (low variance, high bias)
        - λ = 1: Uses full Monte Carlo returns (high variance, low bias)

    Problem Statement:
        Standard Actor-Critic can have high variance in advantage estimates.
        GAE provides a principled way to balance bias and variance, leading to
        more stable training.

    Advantages:
        1. More stable advantage estimates via GAE
        2. Batch updates for better gradient estimates
        3. Parallel environment support for sample efficiency
        4. Better convergence properties than standard Actor-Critic

    Complexity:
        Time: O(T * N) where T is trajectory length and N is number of environments
        Space: O(T * N) for storing trajectories from all environments
    """

    def __init__(
        self,
        policy: BasePolicy,
        value_function: BaseValueFunction,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coeff: float = 0.01,
        device: str = "cpu"
    ):
        """
        Initialize A2C agent.

        Args:
            policy: Policy network (actor)
            value_function: Value function network (critic)
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            entropy_coeff: Entropy regularization coefficient
            device: Device to run on
        """
        super().__init__(
            policy=policy,
            value_function=value_function,
            learning_rate=learning_rate,
            gamma=gamma,
            device=device
        )
        self.entropy_coeff = entropy_coeff
        self.gae_lambda = gae_lambda
        self.trajectory_buffer = TrajectoryBuffer(gamma=gamma, gae_lambda=gae_lambda)

    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute A2C policy loss using GAE advantages.

        Args:
            states: State tensor
            actions: Action tensor
            returns: Return tensor
            advantages: GAE advantage tensor

        Returns:
            Policy loss
        """
        log_probs, entropy = self.policy.evaluate(states, actions)

        if advantages is None:
            with torch.no_grad():
                values = self.value_function(states)
            advantages = returns - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_loss = -self.entropy_coeff * entropy.mean()

        return policy_loss + entropy_loss

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one A2C training step with GAE.

        Args:
            states: State tensor
            actions: Action tensor
            returns: Return tensor
            advantages: GAE advantage tensor

        Returns:
            Dictionary with metrics
        """
        # Normalize advantages
        if advantages is not None:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Critic update
        value_loss = self.value_function.compute_loss(states, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), max_norm=1.0)
        self.value_optimizer.step()

        # Actor update
        policy_loss = self.compute_policy_loss(states, actions, returns, advantages)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def collect_trajectory(
        self,
        env,
        max_steps: int = 1000
    ) -> Tuple[Trajectory, float]:
        """Collect trajectory from environment."""
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []

        state, _ = env.reset()
        episode_return = 0.0

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob = self.policy.sample(state_tensor)

            action_np = action.cpu().numpy().squeeze()
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)

            episode_return += reward
            state = next_state

            if done:
                break

        return Trajectory(states, actions, rewards, dones, log_probs), episode_return

    def train_episode(
        self,
        env,
        max_steps: int = 1000
    ) -> Dict[str, float]:
        """Train for one episode."""
        trajectory, episode_return = self.collect_trajectory(env, max_steps)

        states = torch.FloatTensor(np.array(trajectory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(trajectory.actions)).to(self.device)

        # Compute returns and advantages using GAE
        returns = self.trajectory_buffer.compute_returns(trajectory)
        with torch.no_grad():
            values = self.value_function(states).cpu().numpy().squeeze()
        advantages = self.trajectory_buffer.compute_advantages(trajectory, values)

        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)

        metrics = self.train_step(states, actions, returns, advantages)
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
        """Train the agent."""
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
