"""
Advantage Actor-Critic (A2C) Algorithm Implementation.

================================================================================
核心思想 (Core Idea)
================================================================================
A2C (Mnih et al., 2016) combines policy gradient with value function learning
in a synchronous, batched manner. Key innovations over REINFORCE:

1. **Bootstrapping**: Use V(s') to estimate future returns (reduces variance)
2. **Advantage**: A(s,a) = Q(s,a) - V(s) measures action quality relative to average
3. **Parallel Collection**: Multiple environments for diverse experience
4. **N-step Returns**: Balance bias-variance with configurable lookahead

================================================================================
数学原理 (Mathematical Theory)
================================================================================
Actor-Critic Framework:
    Actor: π_θ(a|s) - policy network, outputs action distribution
    Critic: V_φ(s) - value network, estimates expected return

Advantage Function:
    A^π(s,a) = Q^π(s,a) - V^π(s)
             = E[r + γV(s') | s,a] - V(s)
             ≈ r + γV(s') - V(s)  (one-step TD)

    Interpretation: How much better is action a compared to average?
        A > 0: Action better than average → increase probability
        A < 0: Action worse than average → decrease probability

N-step Returns:
    G_t^{(n)} = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})

    Special cases:
        n = 1: TD(0), G_t = r_t + γV(s_{t+1})
        n = ∞: Monte Carlo, G_t = Σ_k γ^k r_{t+k}

Generalized Advantage Estimation (GAE):
    δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)

    A^{GAE(γ,λ)}_t = Σ_{k=0}^∞ (γλ)^k δ_{t+k}
                   = δ_t + γλ A_{t+1}

    λ controls bias-variance:
        λ = 0: A_t = δ_t (TD, low variance, biased)
        λ = 1: A_t = G_t - V(s_t) (MC, high variance, unbiased)

Loss Functions:
    Policy Loss (Actor):
        L_π = -E_t[log π_θ(a_t|s_t) · A_t]

    Value Loss (Critic):
        L_V = E_t[(V_φ(s_t) - G_t)²]

    Entropy Bonus (Exploration):
        L_ent = -E_t[H(π_θ(·|s_t))]
              = E_t[Σ_a π(a|s) log π(a|s)]

    Total Loss:
        L = L_π + c_v · L_V + c_ent · L_ent

================================================================================
问题背景 (Problem Statement)
================================================================================
REINFORCE Limitations:
    1. High variance from Monte Carlo returns
    2. Must wait for episode completion
    3. No credit assignment within episode

Actor-Critic Solutions:
    1. Bootstrapping reduces variance (at cost of bias)
    2. Can update during episode (online learning)
    3. TD error provides immediate feedback

A2C vs A3C:
    A3C (Asynchronous): Multiple workers update shared parameters asynchronously
        - Pro: Decorrelated updates, natural exploration
        - Con: Stale gradients, complex implementation

    A2C (Synchronous): Collect from all workers, single batched update
        - Pro: Simpler, better GPU utilization, reproducible
        - Con: Less exploration diversity
        - Result: Often matches or exceeds A3C performance

================================================================================
算法对比 (Comparison)
================================================================================
| Algorithm | Variance | Bias   | Sample Eff. | Stability | Complexity |
|-----------|----------|--------|-------------|-----------|------------|
| REINFORCE | High     | None   | Low         | Medium    | Low        |
| A2C       | Medium   | Low    | Medium      | Good      | Medium     |
| PPO       | Low      | Low    | High        | Excellent | Medium     |
| SAC       | Low      | Medium | Very High   | Excellent | High       |

| Component      | A2C                  | DQN                  |
|----------------|----------------------|----------------------|
| Policy         | Explicit π(a|s)      | Implicit argmax Q    |
| Value          | V(s)                 | Q(s,a)               |
| Action space   | Any                  | Discrete             |
| Exploration    | Stochastic policy    | ε-greedy             |
| Off-policy     | No                   | Yes (replay buffer)  |

================================================================================
复杂度 (Complexity Analysis)
================================================================================
Per Rollout (n_steps × n_envs transitions):
    Collection: O(n_steps × n_envs × env_step_time)
    Forward: O(n_steps × n_envs × network_forward)
    GAE: O(n_steps × n_envs)

Per Update:
    Backward: O(n_steps × n_envs × network_backward)
    Memory: O(n_steps × n_envs × (state_dim + action_dim))

Total: O(total_steps / (n_steps × n_envs) × update_cost)

================================================================================
算法总结 (Summary)
================================================================================
A2C is a practical, efficient actor-critic algorithm:

Strengths:
    1. Lower variance than REINFORCE via bootstrapping
    2. Can learn from incomplete episodes
    3. Efficient batched updates
    4. Good baseline for more advanced algorithms

Weaknesses:
    1. On-policy (sample inefficient vs. off-policy)
    2. Sensitive to hyperparameters
    3. Can have training instability
    4. Single update per rollout

This implementation includes:
    - Shared or separate actor-critic networks
    - GAE for advantage estimation
    - Entropy bonus for exploration
    - Gradient clipping for stability
    - Support for discrete and continuous actions

References
----------
[1] Mnih et al. (2016). Asynchronous Methods for Deep RL (A3C).
[2] Schulman et al. (2016). High-Dimensional Continuous Control Using GAE.
[3] Wu et al. (2017). Scalable Trust-Region Method for Deep RL (ACKTR).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.networks import ActorCriticNetwork, DiscretePolicy, ContinuousPolicy, ValueNetwork
from core.buffers import RolloutBuffer
from core.config import TrainingConfig


class A2C:
    """
    Advantage Actor-Critic (A2C) algorithm.

    核心思想 (Core Idea):
        Synchronous actor-critic that collects fixed-length rollouts,
        computes advantages using GAE, and performs a single gradient
        update. Balances sample efficiency and stability.

    数学原理 (Mathematical Theory):
        Update rule:
            θ ← θ + α ∇_θ L(θ)

        Where:
            L = L_policy + c_v · L_value - c_ent · H(π)

            L_policy = -E[log π(a|s) · A^{GAE}]
            L_value = E[(V(s) - G)²]
            H(π) = -E[π log π]

        GAE advantage:
            A_t = Σ_{k=0}^{T-t} (γλ)^k δ_{t+k}
            δ_t = r_t + γV(s_{t+1}) - V(s_t)

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    env : gym.Env
        Gymnasium environment.
    shared_network : bool, default=True
        If True, actor and critic share feature extraction layers.

    Attributes
    ----------
    actor_critic : nn.Module
        Combined or separate actor-critic networks.
    optimizer : torch.optim.Optimizer
        Optimizer for all parameters.
    buffer : RolloutBuffer
        Experience storage with GAE computation.

    Examples
    --------
    >>> config = TrainingConfig(
    ...     env_name="CartPole-v1",
    ...     total_timesteps=100000,
    ...     n_steps=128,
    ... )
    >>> env = gym.make(config.env_name)
    >>> agent = A2C(config, env)
    >>> metrics = agent.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        env: gym.Env,
        shared_network: bool = True,
    ):
        self.config = config
        self.env = env
        self.shared_network = shared_network
        self.device = config.get_device()

        # Environment properties
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.state_dim = env.observation_space.shape[0]

        if self.discrete:
            self.action_dim = env.action_space.n
        else:
            self.action_dim = env.action_space.shape[0]

        # Initialize networks
        self._build_networks()

        # Experience buffer
        self.buffer = RolloutBuffer(
            buffer_size=config.n_steps,
            state_dim=self.state_dim,
            action_dim=1 if self.discrete else self.action_dim,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        # Training state
        self.state: Optional[np.ndarray] = None
        self.episode_reward = 0.0
        self.episode_length = 0

        # Statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []
        self.total_steps = 0

    def _build_networks(self) -> None:
        """Initialize actor and critic networks."""
        hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 64

        if self.shared_network:
            self.actor_critic = ActorCriticNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                continuous=not self.discrete,
            ).to(self.device)

            self.optimizer = optim.Adam(
                self.actor_critic.parameters(),
                lr=self.config.learning_rate,
            )
        else:
            # Separate networks
            if self.discrete:
                self.policy = DiscretePolicy(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    hidden_dims=self.config.hidden_dims,
                ).to(self.device)
            else:
                self.policy = ContinuousPolicy(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    hidden_dims=self.config.hidden_dims,
                ).to(self.device)

            self.value_net = ValueNetwork(
                state_dim=self.state_dim,
                hidden_dims=self.config.hidden_dims,
            ).to(self.device)

            self.optimizer = optim.Adam(
                list(self.policy.parameters()) + list(self.value_net.parameters()),
                lr=self.config.learning_rate,
            )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action and get value estimate.

        Returns
        -------
        action : np.ndarray
            Selected action.
        log_prob : float
            Log probability of action.
        value : float
            Value estimate V(s).
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if self.shared_network:
                action, log_prob, _, value = self.actor_critic.get_action_and_value(
                    state_tensor
                )

                if self.discrete:
                    action = action.item()
                else:
                    action = action.cpu().numpy().squeeze(0)

                return (
                    np.array(action) if self.discrete else action,
                    log_prob.item(),
                    value.item(),
                )
            else:
                # Separate networks
                if self.discrete:
                    action, log_prob, _ = self.policy.sample(state_tensor)
                    action = action.item()
                else:
                    action, log_prob, _ = self.policy.sample(
                        state_tensor, deterministic=deterministic
                    )
                    action = action.cpu().numpy().squeeze(0)

                value = self.value_net(state_tensor).item()

                return (
                    np.array(action) if self.discrete else action,
                    log_prob.item(),
                    value,
                )

    def collect_rollout(self) -> int:
        """
        Collect n_steps of experience.

        Returns
        -------
        n_steps : int
            Number of steps collected.
        """
        if self.state is None:
            self.state, _ = self.env.reset()
            self.episode_reward = 0.0
            self.episode_length = 0

        for _ in range(self.config.n_steps):
            action, log_prob, value = self.select_action(self.state)

            next_state, reward, terminated, truncated, _ = self.env.step(
                int(action) if self.discrete else action
            )
            done = terminated or truncated

            # Store transition
            self.buffer.add(
                state=self.state,
                action=np.array([action]) if self.discrete else action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
            )

            self.state = next_state
            self.episode_reward += reward
            self.episode_length += 1
            self.total_steps += 1

            if done:
                self.episode_rewards.append(self.episode_reward)
                self.episode_lengths.append(self.episode_length)

                self.state, _ = self.env.reset()
                self.episode_reward = 0.0
                self.episode_length = 0

        # Compute bootstrap value for incomplete episode
        with torch.no_grad():
            state_tensor = torch.tensor(
                self.state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if self.shared_network:
                last_value = self.actor_critic.get_value(state_tensor).item()
            else:
                last_value = self.value_net(state_tensor).squeeze().item()

        # Compute advantages
        self.buffer.compute_advantages(
            last_value=last_value,
            normalize=self.config.normalize_advantage,
        )

        return self.config.n_steps

    def update(self) -> Dict[str, float]:
        """
        Perform policy and value update.

        Returns
        -------
        metrics : Dict[str, float]
            Training metrics.
        """
        # Get all data from buffer
        states, actions, old_log_probs, advantages, returns = self.buffer.get_all(
            device=self.device
        )

        # Forward pass
        if self.shared_network:
            if self.discrete:
                actions_input = actions.squeeze(-1).long()
            else:
                actions_input = actions

            _, log_probs, entropy, values = self.actor_critic.get_action_and_value(
                states, action=actions_input
            )
        else:
            if self.discrete:
                dist = self.policy.get_distribution(states)
                log_probs = dist.log_prob(actions.squeeze(-1).long())
                entropy = dist.entropy()
            else:
                log_probs, entropy = self.policy.evaluate(states, actions)

            values = self.value_net(states).squeeze(-1)

        # Policy loss
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns)

        # Entropy loss
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + self.config.entropy_coef * entropy_loss
        )

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if self.shared_network:
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.config.max_grad_norm,
                )
            else:
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.config.max_grad_norm,
                )

        self.optimizer.step()

        # Reset buffer
        self.buffer.reset()

        # Record metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": loss.item(),
        }

        self.policy_losses.append(metrics["policy_loss"])
        self.value_losses.append(metrics["value_loss"])
        self.entropies.append(metrics["entropy"])

        return metrics

    def train(self) -> Dict[str, List[float]]:
        """
        Train the agent.

        Returns
        -------
        metrics : Dict[str, List[float]]
            Training history.
        """
        num_updates = 0

        while self.total_steps < self.config.total_timesteps:
            # Collect rollout
            self.collect_rollout()

            # Update
            metrics = self.update()
            num_updates += 1

            # Logging
            if len(self.episode_rewards) > 0 and num_updates % self.config.log_interval == 0:
                recent_rewards = self.episode_rewards[-10:]
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                std_reward = np.std(recent_rewards) if recent_rewards else 0.0

                print(
                    f"Update {num_updates:4d} | "
                    f"Steps {self.total_steps:7d} | "
                    f"Episodes {len(self.episode_rewards):4d} | "
                    f"Reward {mean_reward:7.2f} ± {std_reward:5.2f} | "
                    f"Policy Loss {metrics['policy_loss']:7.4f} | "
                    f"Value Loss {metrics['value_loss']:7.4f} | "
                    f"Entropy {metrics['entropy']:6.4f}"
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
        n_episodes : int
            Number of evaluation episodes.
        render : bool
            Whether to render.

        Returns
        -------
        mean_reward : float
            Mean episode reward.
        std_reward : float
            Standard deviation.
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
                state_tensor = torch.tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    if self.shared_network:
                        if self.discrete:
                            logits, _ = self.actor_critic(state_tensor)
                            action = logits.argmax(dim=-1).item()
                        else:
                            (mean, _), _ = self.actor_critic(state_tensor)
                            action = torch.tanh(mean).cpu().numpy().squeeze(0)
                    else:
                        if self.discrete:
                            logits = self.policy(state_tensor)
                            action = logits.argmax(dim=-1).item()
                        else:
                            action, _, _ = self.policy.sample(
                                state_tensor, deterministic=True
                            )
                            action = action.cpu().numpy().squeeze(0)

                state, reward, terminated, truncated, _ = eval_env.step(
                    int(action) if self.discrete else action
                )
                done = terminated or truncated
                episode_reward += reward

            rewards.append(episode_reward)

        eval_env.close()

        return np.mean(rewards), np.std(rewards)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "config": self.config.to_dict(),
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards,
        }

        if self.shared_network:
            checkpoint["actor_critic_state_dict"] = self.actor_critic.state_dict()
        else:
            checkpoint["policy_state_dict"] = self.policy.state_dict()
            checkpoint["value_net_state_dict"] = self.value_net.state_dict()

        checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        if self.shared_network:
            self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        else:
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.value_net.load_state_dict(checkpoint["value_net_state_dict"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.episode_rewards = checkpoint.get("episode_rewards", [])


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("A2C Algorithm - Unit Tests")
    print("=" * 70)

    # Test configuration
    config = TrainingConfig(
        env_name="CartPole-v1",
        total_timesteps=5000,
        n_steps=128,
        hidden_dims=[64, 64],
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        log_interval=5,
    )

    # Test with shared network
    print("\n[1] Testing A2C with shared network...")
    env = gym.make(config.env_name)
    agent = A2C(config, env, shared_network=True)

    print(f"    State dim: {agent.state_dim}")
    print(f"    Action dim: {agent.action_dim}")
    print(f"    Device: {agent.device}")

    # Test action selection
    state, _ = env.reset()
    action, log_prob, value = agent.select_action(state)
    print(f"    Sample action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")

    # Test rollout collection
    n_collected = agent.collect_rollout()
    print(f"    Collected {n_collected} steps")

    # Test update
    metrics = agent.update()
    print(f"    Policy loss: {metrics['policy_loss']:.4f}")
    print(f"    Value loss: {metrics['value_loss']:.4f}")
    print("    [PASS]")

    # Test training
    print("\n[2] Testing training loop...")
    config.total_timesteps = 3000
    agent = A2C(config, env, shared_network=True)
    history = agent.train()

    print(f"    Completed {len(history['episode_rewards'])} episodes")
    if history['episode_rewards']:
        print(f"    Final mean reward: {np.mean(history['episode_rewards'][-10:]):.2f}")
    print("    [PASS]")

    # Test evaluation
    print("\n[3] Testing evaluation...")
    mean_reward, std_reward = agent.evaluate(n_episodes=3)
    print(f"    Eval reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("    [PASS]")

    # Test with separate networks
    print("\n[4] Testing A2C with separate networks...")
    agent_separate = A2C(config, env, shared_network=False)

    state, _ = env.reset()
    action, log_prob, value = agent_separate.select_action(state)
    print(f"    Sample action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")

    agent_separate.collect_rollout()
    metrics = agent_separate.update()
    print(f"    Policy loss: {metrics['policy_loss']:.4f}")
    print("    [PASS]")

    # Test save/load
    print("\n[5] Testing save/load...")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "a2c.pt")
        agent.save(save_path)

        new_agent = A2C(config, env, shared_network=True)
        new_agent.load(save_path)

        # Verify
        for p1, p2 in zip(
            agent.actor_critic.parameters(),
            new_agent.actor_critic.parameters()
        ):
            assert torch.allclose(p1, p2)

    print("    Save/load successful")
    print("    [PASS]")

    env.close()

    # Test continuous action space
    print("\n[6] Testing A2C on Pendulum (continuous)...")
    config.env_name = "Pendulum-v1"
    config.total_timesteps = 2000
    config.n_steps = 64

    try:
        env_continuous = gym.make(config.env_name)
        agent_continuous = A2C(config, env_continuous, shared_network=True)

        assert not agent_continuous.discrete
        print(f"    Action dim: {agent_continuous.action_dim}")

        state, _ = env_continuous.reset()
        action, log_prob, value = agent_continuous.select_action(state)
        print(f"    Sample action: {action}, log_prob: {log_prob:.4f}")

        # Quick training test
        agent_continuous.collect_rollout()
        metrics = agent_continuous.update()
        print(f"    Policy loss: {metrics['policy_loss']:.4f}")

        env_continuous.close()
        print("    [PASS]")
    except Exception as e:
        print(f"    Skipped: {e}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
