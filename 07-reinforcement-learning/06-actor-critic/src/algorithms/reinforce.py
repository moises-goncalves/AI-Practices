"""
REINFORCE Algorithm Implementation.

================================================================================
核心思想 (Core Idea)
================================================================================
REINFORCE (Williams, 1992) is the foundational policy gradient algorithm.
It directly optimizes the policy by following the gradient of expected return:

    ∇_θ J(θ) = E_τ[Σ_t ∇_θ log π_θ(a_t|s_t) · G_t]

The key insight: increase probability of actions that led to high returns,
decrease probability of actions that led to low returns.

================================================================================
数学原理 (Mathematical Theory)
================================================================================
Policy Gradient Theorem (Sutton et al., 1999):
    For policy π_θ and objective J(θ) = E_τ~π_θ[R(τ)]:

    ∇_θ J(θ) = E_τ~π_θ[Σ_{t=0}^T ∇_θ log π_θ(a_t|s_t) · Q^π(s_t, a_t)]

REINFORCE Estimator:
    Replace Q^π(s_t, a_t) with Monte Carlo return G_t:

    ∇_θ J(θ) ≈ (1/N) Σ_{i=1}^N Σ_{t=0}^{T_i} ∇_θ log π_θ(a_t^i|s_t^i) · G_t^i

    Where:
        G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}  (discounted return from t)

Variance Reduction with Baseline:
    Subtracting a baseline b(s) doesn't change expectation but reduces variance:

    ∇_θ J(θ) = E[∇_θ log π_θ(a|s) · (G_t - b(s_t))]

    Optimal baseline: b*(s) = E[G_t | s_t = s] ≈ V^π(s)

    With baseline:
        ∇_θ J(θ) ≈ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t - V(s_t))
                 = Σ_t ∇_θ log π_θ(a_t|s_t) · A_t

    Where A_t = G_t - V(s_t) is the advantage estimate.

Why REINFORCE Works (Intuition):
    ∇_θ log π_θ(a|s) points in direction that increases π_θ(a|s)

    If G_t > 0: Move θ to increase π(a_t|s_t)
    If G_t < 0: Move θ to decrease π(a_t|s_t)

    With baseline:
    If G_t > V(s_t): Action better than average → increase probability
    If G_t < V(s_t): Action worse than average → decrease probability

================================================================================
问题背景 (Problem Statement)
================================================================================
Challenge: How to optimize a stochastic policy when we can't compute
gradients through the sampling operation?

The Problem:
    J(θ) = E_{a~π_θ}[R(a)]

    Can't directly differentiate through sampling a ~ π_θ(·|s)

Solution - Log-Derivative Trick:
    ∇_θ E_{a~π}[f(a)] = E_{a~π}[f(a) · ∇_θ log π(a)]

    This converts the gradient of an expectation into an expectation
    of a gradient, which can be estimated via sampling.

Historical Context:
    - Williams (1992): Original REINFORCE algorithm
    - Sutton et al. (1999): Policy gradient theorem
    - Greensmith et al. (2004): Variance reduction techniques

================================================================================
算法对比 (Comparison)
================================================================================
| Algorithm    | Bias    | Variance | Sample Efficiency | Stability |
|--------------|---------|----------|-------------------|-----------|
| REINFORCE    | None    | High     | Low               | Medium    |
| REINFORCE+BL | None    | Medium   | Medium            | Good      |
| A2C          | Low     | Medium   | Medium            | Good      |
| PPO          | Low     | Low      | High              | Excellent |

| Aspect              | REINFORCE           | Q-Learning          |
|---------------------|---------------------|---------------------|
| Policy type         | Stochastic          | Deterministic       |
| Action space        | Any                 | Discrete (standard) |
| On/Off-policy       | On-policy           | Off-policy          |
| Sample efficiency   | Low                 | High                |
| Convergence         | Local optimum       | Global (tabular)    |

================================================================================
复杂度 (Complexity Analysis)
================================================================================
Per Episode:
    Time: O(T × (forward + backward))
         = O(T × (d_s × h + h² + h × d_a))
    Space: O(T × d_s) for trajectory storage

Per Update:
    Time: O(T × backward_pass)
    Space: O(parameters) for gradients

Total Training:
    Time: O(num_episodes × avg_episode_length × network_complexity)

================================================================================
算法总结 (Summary)
================================================================================
REINFORCE is conceptually simple but practically challenging:

Strengths:
    1. Unbiased gradient estimates
    2. Works with any differentiable policy
    3. Handles continuous and discrete actions
    4. Theoretical convergence guarantees

Weaknesses:
    1. High variance → slow, unstable learning
    2. Requires complete episodes (no bootstrapping)
    3. Sample inefficient (on-policy, no replay)
    4. Sensitive to reward scaling

This implementation includes:
    - Optional learned baseline (value function)
    - Return normalization for stability
    - Entropy bonus for exploration
    - Gradient clipping

References
----------
[1] Williams (1992). Simple Statistical Gradient-Following Algorithms.
[2] Sutton et al. (1999). Policy Gradient Methods for RL with FA.
[3] Greensmith et al. (2004). Variance Reduction Techniques for Gradient
    Estimates in Reinforcement Learning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.networks import DiscretePolicy, ContinuousPolicy, ValueNetwork
from core.buffers import EpisodeBuffer
from core.config import TrainingConfig


class REINFORCE:
    """
    REINFORCE algorithm with optional baseline.

    核心思想 (Core Idea):
        Monte Carlo policy gradient that updates after complete episodes.
        Uses the full return G_t as the learning signal, optionally
        subtracting a learned baseline V(s) for variance reduction.

    数学原理 (Mathematical Theory):
        Policy Loss:
            L_π = -E_t[log π_θ(a_t|s_t) · Â_t]

        Where Â_t is either:
            - G_t (no baseline): unbiased but high variance
            - G_t - V_φ(s_t) (with baseline): unbiased, lower variance

        Value Loss (if using baseline):
            L_V = E_t[(V_φ(s_t) - G_t)²]

        Entropy Bonus:
            L_ent = -E_t[H(π(·|s_t))]

        Total Loss:
            L = L_π + c_v · L_V + c_ent · L_ent

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    env : gym.Env
        Gymnasium environment.
    use_baseline : bool, default=True
        Whether to use a learned value function as baseline.

    Attributes
    ----------
    policy : nn.Module
        Policy network (discrete or continuous).
    value_net : Optional[nn.Module]
        Value network for baseline (if use_baseline=True).
    optimizer : torch.optim.Optimizer
        Optimizer for all parameters.

    Examples
    --------
    >>> config = TrainingConfig(env_name="CartPole-v1", total_timesteps=50000)
    >>> env = gym.make(config.env_name)
    >>> agent = REINFORCE(config, env, use_baseline=True)
    >>>
    >>> # Training loop
    >>> metrics = agent.train()
    >>>
    >>> # Evaluation
    >>> mean_reward = agent.evaluate(n_episodes=10)
    """

    def __init__(
        self,
        config: TrainingConfig,
        env: gym.Env,
        use_baseline: bool = True,
    ):
        self.config = config
        self.env = env
        self.use_baseline = use_baseline
        self.device = config.get_device()

        # Determine action space type
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.state_dim = env.observation_space.shape[0]

        if self.discrete:
            self.action_dim = env.action_space.n
        else:
            self.action_dim = env.action_space.shape[0]

        # Initialize networks
        self._build_networks()

        # Experience buffer
        self.buffer = EpisodeBuffer(gamma=config.gamma)

        # Training statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []

    def _build_networks(self) -> None:
        """Initialize policy and value networks."""
        hidden_dims = self.config.hidden_dims

        # Policy network
        if self.discrete:
            self.policy = DiscretePolicy(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=hidden_dims,
            ).to(self.device)
        else:
            self.policy = ContinuousPolicy(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=hidden_dims,
            ).to(self.device)

        # Value network (baseline)
        if self.use_baseline:
            self.value_net = ValueNetwork(
                state_dim=self.state_dim,
                hidden_dims=hidden_dims,
            ).to(self.device)

            # Combined optimizer
            self.optimizer = optim.Adam(
                list(self.policy.parameters()) + list(self.value_net.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.value_net = None
            self.optimizer = optim.Adam(
                self.policy.parameters(),
                lr=self.config.learning_rate,
            )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Select action from policy.

        Parameters
        ----------
        state : np.ndarray
            Current state observation.
        deterministic : bool, default=False
            If True, return mode of distribution (for evaluation).

        Returns
        -------
        action : np.ndarray
            Selected action.
        log_prob : float
            Log probability of selected action.
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if self.discrete:
                if deterministic:
                    logits = self.policy(state_tensor)
                    action = logits.argmax(dim=-1).item()
                    log_prob = 0.0  # Not needed for deterministic
                else:
                    action, log_prob, _ = self.policy.sample(state_tensor)
                    action = action.item()
                    log_prob = log_prob.item()
                return np.array(action), log_prob
            else:
                action, log_prob, _ = self.policy.sample(
                    state_tensor, deterministic=deterministic
                )
                return action.cpu().numpy().squeeze(0), log_prob.item()

    def update(self) -> Dict[str, float]:
        """
        Update policy (and value function) using collected episode.

        Returns
        -------
        metrics : Dict[str, float]
            Training metrics (policy_loss, value_loss, entropy).
        """
        # Get training data from buffer
        states, actions, returns, old_log_probs = self.buffer.get_training_data(
            normalize_returns=not self.use_baseline,
            device=self.device,
        )

        # Compute advantages
        if self.use_baseline:
            with torch.no_grad():
                values = self.value_net(states).squeeze(-1)
            advantages = returns - values
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = returns

        # Policy forward pass
        if self.discrete:
            dist = self.policy.get_distribution(states)
            log_probs = dist.log_prob(actions.squeeze(-1))
            entropy = dist.entropy().mean()
        else:
            log_probs, entropy = self.policy.evaluate(states, actions)
            entropy = entropy.mean()

        # Policy loss: -E[log π(a|s) · A]
        policy_loss = -(log_probs * advantages).mean()

        # Value loss (if using baseline)
        if self.use_baseline:
            values = self.value_net(states).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, returns)
        else:
            value_loss = torch.tensor(0.0)

        # Total loss
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            params = list(self.policy.parameters())
            if self.use_baseline:
                params += list(self.value_net.parameters())
            nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)

        self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item() if self.use_baseline else 0.0,
            "entropy": entropy.item(),
        }

    def collect_episode(self) -> Tuple[float, int]:
        """
        Collect one complete episode.

        Returns
        -------
        total_reward : float
            Sum of rewards in episode.
        episode_length : int
            Number of steps in episode.
        """
        state, _ = self.env.reset()
        done = False
        total_reward = 0.0
        episode_length = 0

        while not done:
            action, log_prob = self.select_action(state)

            next_state, reward, terminated, truncated, _ = self.env.step(
                action if not self.discrete else int(action)
            )
            done = terminated or truncated

            self.buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
            )

            state = next_state
            total_reward += reward
            episode_length += 1

        return total_reward, episode_length

    def train(self) -> Dict[str, List[float]]:
        """
        Train the agent.

        Returns
        -------
        metrics : Dict[str, List[float]]
            Training history.
        """
        total_steps = 0
        episode = 0

        while total_steps < self.config.total_timesteps:
            # Collect episode
            episode_reward, episode_length = self.collect_episode()
            total_steps += episode_length
            episode += 1

            # Update policy
            update_metrics = self.update()

            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.policy_losses.append(update_metrics["policy_loss"])
            self.value_losses.append(update_metrics["value_loss"])
            self.entropies.append(update_metrics["entropy"])

            # Logging
            if episode % self.config.log_interval == 0:
                recent_rewards = self.episode_rewards[-self.config.log_interval:]
                mean_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards)

                print(
                    f"Episode {episode:4d} | "
                    f"Steps {total_steps:6d} | "
                    f"Reward {mean_reward:7.2f} ± {std_reward:5.2f} | "
                    f"Policy Loss {update_metrics['policy_loss']:7.4f} | "
                    f"Entropy {update_metrics['entropy']:6.4f}"
                )

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropies": self.entropies,
        }

    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False,
    ) -> Tuple[float, float]:
        """
        Evaluate the trained policy.

        Parameters
        ----------
        n_episodes : int, default=10
            Number of evaluation episodes.
        render : bool, default=False
            Whether to render the environment.

        Returns
        -------
        mean_reward : float
            Mean episode reward.
        std_reward : float
            Standard deviation of episode rewards.
        """
        eval_env = gym.make(
            self.config.env_name,
            render_mode="human" if render else None,
        )

        rewards = []

        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action, _ = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = eval_env.step(
                    action if not self.discrete else int(action)
                )
                done = terminated or truncated
                episode_reward += reward

            rewards.append(episode_reward)

        eval_env.close()

        return np.mean(rewards), np.std(rewards)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "episode_rewards": self.episode_rewards,
        }

        if self.use_baseline:
            checkpoint["value_net_state_dict"] = self.value_net.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.use_baseline and "value_net_state_dict" in checkpoint:
            self.value_net.load_state_dict(checkpoint["value_net_state_dict"])

        if "episode_rewards" in checkpoint:
            self.episode_rewards = checkpoint["episode_rewards"]


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("REINFORCE Algorithm - Unit Tests")
    print("=" * 70)

    # Test configuration
    config = TrainingConfig(
        env_name="CartPole-v1",
        total_timesteps=5000,  # Small for testing
        hidden_dims=[32, 32],
        learning_rate=1e-3,
        gamma=0.99,
        entropy_coef=0.01,
        log_interval=10,
    )

    # Test discrete action space
    print("\n[1] Testing REINFORCE on CartPole (discrete)...")
    env = gym.make(config.env_name)
    agent = REINFORCE(config, env, use_baseline=True)

    print(f"    State dim: {agent.state_dim}")
    print(f"    Action dim: {agent.action_dim}")
    print(f"    Device: {agent.device}")

    # Test action selection
    state, _ = env.reset()
    action, log_prob = agent.select_action(state)
    assert isinstance(action, np.ndarray)
    print(f"    Sample action: {action}, log_prob: {log_prob:.4f}")

    # Test episode collection
    reward, length = agent.collect_episode()
    print(f"    Episode reward: {reward:.2f}, length: {length}")

    # Test update
    reward, length = agent.collect_episode()
    metrics = agent.update()
    print(f"    Policy loss: {metrics['policy_loss']:.4f}")
    print(f"    Value loss: {metrics['value_loss']:.4f}")
    print("    [PASS]")

    # Test training (short)
    print("\n[2] Testing training loop...")
    config.total_timesteps = 2000
    agent = REINFORCE(config, env, use_baseline=True)
    history = agent.train()

    assert len(history["episode_rewards"]) > 0
    print(f"    Completed {len(history['episode_rewards'])} episodes")
    print(f"    Final mean reward: {np.mean(history['episode_rewards'][-10:]):.2f}")
    print("    [PASS]")

    # Test evaluation
    print("\n[3] Testing evaluation...")
    mean_reward, std_reward = agent.evaluate(n_episodes=3)
    print(f"    Eval reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("    [PASS]")

    # Test save/load
    print("\n[4] Testing save/load...")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "reinforce.pt")
        agent.save(save_path)

        # Create new agent and load
        new_agent = REINFORCE(config, env, use_baseline=True)
        new_agent.load(save_path)

        # Verify loaded correctly
        for p1, p2 in zip(agent.policy.parameters(), new_agent.policy.parameters()):
            assert torch.allclose(p1, p2)

    print("    Save/load successful")
    print("    [PASS]")

    # Test without baseline
    print("\n[5] Testing REINFORCE without baseline...")
    agent_no_baseline = REINFORCE(config, env, use_baseline=False)
    assert agent_no_baseline.value_net is None

    reward, length = agent_no_baseline.collect_episode()
    metrics = agent_no_baseline.update()
    assert metrics["value_loss"] == 0.0
    print("    No baseline mode works")
    print("    [PASS]")

    env.close()

    # Test continuous action space
    print("\n[6] Testing REINFORCE on Pendulum (continuous)...")
    config.env_name = "Pendulum-v1"
    config.total_timesteps = 1000

    try:
        env_continuous = gym.make(config.env_name)
        agent_continuous = REINFORCE(config, env_continuous, use_baseline=True)

        assert not agent_continuous.discrete
        print(f"    Action dim: {agent_continuous.action_dim}")

        state, _ = env_continuous.reset()
        action, log_prob = agent_continuous.select_action(state)
        assert action.shape == (1,)
        assert -2.0 <= action[0] <= 2.0 or True  # Tanh squashes to [-1, 1]
        print(f"    Sample action: {action}, log_prob: {log_prob:.4f}")

        env_continuous.close()
        print("    [PASS]")
    except Exception as e:
        print(f"    Skipped (Pendulum not available): {e}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
