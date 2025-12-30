"""
Proximal Policy Optimization (PPO) Algorithm Implementation.

================================================================================
核心思想 (Core Idea)
================================================================================
PPO (Schulman et al., 2017) is the most widely used policy gradient algorithm,
combining the sample efficiency of TRPO with implementation simplicity. Key idea:

    Constrain policy updates to a "trust region" using a clipped objective,
    preventing destructively large updates while allowing multiple epochs
    of optimization on the same batch of data.

================================================================================
数学原理 (Mathematical Theory)
================================================================================
Motivation - The Policy Update Problem:
    Large policy updates can be catastrophic:
        π_old → π_new where π_new is very different

    This causes:
        1. Performance collapse (policy becomes bad)
        2. Poor sample efficiency (old data becomes useless)
        3. Training instability

Trust Region Policy Optimization (TRPO):
    Constrain KL divergence between old and new policy:

        max_θ E_t[r_t(θ) A_t]
        s.t. E_t[KL(π_old || π_new)] ≤ δ

    Where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) is the probability ratio.

    Problem: Requires second-order optimization (conjugate gradient + line search).

PPO-Clip Objective:
    Replace hard constraint with clipped objective:

        L^{CLIP}(θ) = E_t[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]

    Analysis:
        When A_t > 0 (good action):
            - Want to increase π(a|s), so r_t > 1
            - Clip prevents r_t from exceeding 1+ε
            - Gradient is zero when r_t > 1+ε

        When A_t < 0 (bad action):
            - Want to decrease π(a|s), so r_t < 1
            - Clip prevents r_t from going below 1-ε
            - Gradient is zero when r_t < 1-ε

    Effect: Pessimistic bound on policy improvement.

Value Function Clipping (optional):
    Similarly clip value function updates:

        L^{VF}(φ) = max[(V_φ(s) - G)², (clip(V_φ(s), V_old-ε, V_old+ε) - G)²]

    Prevents value function from changing too rapidly.

Combined Objective:
    L(θ,φ) = E_t[L^{CLIP} - c_1 L^{VF} + c_2 H(π)]

    Where:
        c_1 ≈ 0.5: Value loss coefficient
        c_2 ≈ 0.01: Entropy bonus coefficient
        H(π): Policy entropy for exploration

Multiple Epochs:
    Unlike A2C (single update), PPO reuses data for K epochs:

        for epoch in range(K):
            for batch in shuffle(rollout_data):
                update(batch)

    Clipping ensures updates remain safe even with stale data.

================================================================================
问题背景 (Problem Statement)
================================================================================
Evolution of Policy Gradient Methods:

    REINFORCE (1992):
        + Simple, unbiased
        - High variance, sample inefficient

    A2C (2016):
        + Lower variance via bootstrapping
        - Single update per rollout, can be unstable

    TRPO (2015):
        + Guaranteed monotonic improvement
        - Complex implementation, computationally expensive

    PPO (2017):
        + Simple as A2C, stable as TRPO
        + Multiple epochs per rollout
        + First-order optimization only
        = Best of both worlds

Why PPO Works:
    1. Clipping creates implicit trust region
    2. Multiple epochs improve sample efficiency
    3. Simple implementation enables scaling
    4. Robust to hyperparameter choices

================================================================================
算法对比 (Comparison)
================================================================================
| Algorithm | Sample Eff. | Stability | Complexity | Wall-clock |
|-----------|-------------|-----------|------------|------------|
| REINFORCE | Low         | Medium    | Simple     | Slow       |
| A2C       | Medium      | Good      | Simple     | Medium     |
| TRPO      | High        | Excellent | Complex    | Slow       |
| PPO       | High        | Excellent | Simple     | Fast       |

| Hyperparameter | Typical Value | Sensitivity |
|----------------|---------------|-------------|
| clip_range (ε) | 0.2           | Medium      |
| n_epochs       | 10            | Low         |
| batch_size     | 64            | Low         |
| learning_rate  | 3e-4          | High        |
| gae_lambda     | 0.95          | Low         |

================================================================================
复杂度 (Complexity Analysis)
================================================================================
Per Update Cycle:
    Rollout: O(n_steps × n_envs × env_time)
    GAE: O(n_steps × n_envs)
    Training: O(n_epochs × n_steps × n_envs / batch_size × forward_backward)

Memory:
    O(n_steps × n_envs × (state_dim + action_dim + 5))

Compared to A2C:
    ~n_epochs times more compute per rollout
    Same memory footprint
    ~n_epochs times better sample efficiency

================================================================================
算法总结 (Summary)
================================================================================
PPO is the go-to algorithm for most RL applications:

Strengths:
    1. Simple implementation (first-order only)
    2. Excellent sample efficiency (multiple epochs)
    3. Robust to hyperparameters
    4. Scales to large models and environments
    5. Works for discrete and continuous actions

Weaknesses:
    1. On-policy (less efficient than off-policy for some tasks)
    2. Sensitive to reward scaling
    3. Can plateau on hard exploration problems

This implementation includes:
    - Clipped surrogate objective
    - Optional value function clipping
    - GAE advantage estimation
    - Learning rate annealing
    - Early stopping on KL divergence
    - Gradient clipping

References
----------
[1] Schulman et al. (2017). Proximal Policy Optimization Algorithms.
[2] Schulman et al. (2015). Trust Region Policy Optimization.
[3] Engstrom et al. (2020). Implementation Matters in Deep RL.
[4] Andrychowicz et al. (2020). What Matters in On-Policy RL.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.networks import ActorCriticNetwork
from core.buffers import RolloutBuffer
from core.config import TrainingConfig


class PPO:
    """
    Proximal Policy Optimization (PPO) with clipped objective.

    核心思想 (Core Idea):
        Collect rollouts, compute GAE advantages, then perform multiple
        epochs of minibatch updates with a clipped surrogate objective
        that prevents destructively large policy changes.

    数学原理 (Mathematical Theory):
        Clipped Surrogate Objective:
            L^{CLIP} = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]

        Where:
            r_t = π_θ(a|s) / π_{θ_old}(a|s)  (probability ratio)
            A_t = GAE advantage estimate
            ε = clip_range (typically 0.2)

        The min operation selects the pessimistic bound:
            - If A > 0 and r > 1+ε: use clipped (stop increasing)
            - If A < 0 and r < 1-ε: use clipped (stop decreasing)

    Parameters
    ----------
    config : TrainingConfig
        Training configuration with PPO hyperparameters.
    env : gym.Env
        Gymnasium environment.

    Examples
    --------
    >>> config = TrainingConfig(
    ...     env_name="CartPole-v1",
    ...     total_timesteps=100000,
    ...     n_steps=2048,
    ...     n_epochs=10,
    ...     batch_size=64,
    ...     clip_range=0.2,
    ... )
    >>> env = gym.make(config.env_name)
    >>> agent = PPO(config, env)
    >>> metrics = agent.train()
    """

    def __init__(self, config: TrainingConfig, env: gym.Env):
        self.config = config
        self.env = env
        self.device = config.get_device()

        # Environment properties
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]

        # Network
        hidden_dim = config.hidden_dims[0] if config.hidden_dims else 64
        self.actor_critic = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            continuous=not self.discrete,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )

        # Learning rate scheduler
        if config.lr_schedule == "linear":
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1 - step / (config.total_timesteps // config.n_steps),
            )
        else:
            self.lr_scheduler = None

        # Buffer
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
        self.total_steps = 0

        # Statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.policy_losses: List[float] = []
        self.value_losses: List[float] = []
        self.entropies: List[float] = []
        self.clip_fractions: List[float] = []
        self.approx_kls: List[float] = []

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action and get value estimate."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = self.actor_critic.get_action_and_value(state_tensor)

            if self.discrete:
                return np.array(action.item()), log_prob.item(), value.item()
            else:
                return action.cpu().numpy().squeeze(0), log_prob.item(), value.item()

    def collect_rollout(self) -> int:
        """Collect n_steps of experience."""
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

        # Bootstrap value
        with torch.no_grad():
            state_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_value = self.actor_critic.get_value(state_tensor).item()

        self.buffer.compute_advantages(last_value=last_value, normalize=self.config.normalize_advantage)

        return self.config.n_steps

    def update(self) -> Dict[str, float]:
        """Perform PPO update with multiple epochs."""
        clip_range = self.config.clip_range
        clip_range_vf = self.config.clip_range_vf

        # Metrics accumulators
        policy_losses, value_losses, entropies = [], [], []
        clip_fractions, approx_kls = [], []

        for epoch in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(batch_size=self.config.batch_size, shuffle=True, device=self.device):
                states, actions, old_log_probs, advantages, returns = batch

                # Forward pass
                if self.discrete:
                    actions_input = actions.squeeze(-1).long()
                else:
                    actions_input = actions

                _, new_log_probs, entropy, values = self.actor_critic.get_action_and_value(
                    states, action=actions_input
                )

                # Probability ratio
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # Clipped surrogate objective
                policy_loss_1 = -advantages * ratio
                policy_loss_2 = -advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Value loss (optionally clipped)
                if clip_range_vf is not None:
                    # Get old values for clipping
                    with torch.no_grad():
                        _, _, _, old_values = self.actor_critic.get_action_and_value(states, action=actions_input)
                    values_clipped = old_values + torch.clamp(values - old_values, -clip_range_vf, clip_range_vf)
                    value_loss_1 = (values - returns) ** 2
                    value_loss_2 = (values_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

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
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    clip_fraction = ((ratio - 1).abs() > clip_range).float().mean().item()
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(-entropy_loss.item())
                clip_fractions.append(clip_fraction)
                approx_kls.append(approx_kl)

            # Early stopping on KL divergence
            if self.config.target_kl is not None:
                if np.mean(approx_kls[-len(approx_kls)//self.config.n_epochs:]) > self.config.target_kl:
                    break

        # Learning rate scheduling
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Reset buffer
        self.buffer.reset()

        # Record metrics
        metrics = {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
            "clip_fraction": np.mean(clip_fractions),
            "approx_kl": np.mean(approx_kls),
        }

        self.policy_losses.append(metrics["policy_loss"])
        self.value_losses.append(metrics["value_loss"])
        self.entropies.append(metrics["entropy"])
        self.clip_fractions.append(metrics["clip_fraction"])
        self.approx_kls.append(metrics["approx_kl"])

        return metrics

    def train(self) -> Dict[str, List[float]]:
        """Train the agent."""
        num_updates = 0

        while self.total_steps < self.config.total_timesteps:
            self.collect_rollout()
            metrics = self.update()
            num_updates += 1

            if len(self.episode_rewards) > 0 and num_updates % self.config.log_interval == 0:
                recent_rewards = self.episode_rewards[-10:]
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                std_reward = np.std(recent_rewards) if recent_rewards else 0.0

                print(
                    f"Update {num_updates:4d} | "
                    f"Steps {self.total_steps:7d} | "
                    f"Reward {mean_reward:7.2f} ± {std_reward:5.2f} | "
                    f"Policy {metrics['policy_loss']:7.4f} | "
                    f"Value {metrics['value_loss']:7.4f} | "
                    f"Clip {metrics['clip_fraction']:.3f} | "
                    f"KL {metrics['approx_kl']:.4f}"
                )

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "policy_losses": self.policy_losses,
            "value_losses": self.value_losses,
            "entropies": self.entropies,
            "clip_fractions": self.clip_fractions,
            "approx_kls": self.approx_kls,
        }

    def evaluate(self, n_episodes: int = 10, render: bool = False) -> Tuple[float, float]:
        """Evaluate the trained policy."""
        eval_env = gym.make(self.config.env_name, render_mode="human" if render else None)
        rewards = []

        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    if self.discrete:
                        logits, _ = self.actor_critic(state_tensor)
                        action = logits.argmax(dim=-1).item()
                    else:
                        (mean, _), _ = self.actor_critic(state_tensor)
                        action = torch.tanh(mean).cpu().numpy().squeeze(0)

                state, reward, terminated, truncated, _ = eval_env.step(int(action) if self.discrete else action)
                done = terminated or truncated
                episode_reward += reward

            rewards.append(episode_reward)

        eval_env.close()
        return np.mean(rewards), np.std(rewards)

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "total_steps": self.total_steps,
            "episode_rewards": self.episode_rewards,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.episode_rewards = checkpoint.get("episode_rewards", [])


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PPO Algorithm - Unit Tests")
    print("=" * 70)

    config = TrainingConfig(
        env_name="CartPole-v1",
        total_timesteps=5000,
        n_steps=128,
        n_epochs=4,
        batch_size=32,
        hidden_dims=[64, 64],
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        log_interval=5,
    )

    print("\n[1] Testing PPO initialization...")
    env = gym.make(config.env_name)
    agent = PPO(config, env)
    print(f"    State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
    print("    [PASS]")

    print("\n[2] Testing action selection...")
    state, _ = env.reset()
    action, log_prob, value = agent.select_action(state)
    print(f"    Action: {action}, Log prob: {log_prob:.4f}, Value: {value:.4f}")
    print("    [PASS]")

    print("\n[3] Testing rollout and update...")
    agent.collect_rollout()
    metrics = agent.update()
    print(f"    Policy loss: {metrics['policy_loss']:.4f}")
    print(f"    Clip fraction: {metrics['clip_fraction']:.4f}")
    print("    [PASS]")

    print("\n[4] Testing training loop...")
    config.total_timesteps = 3000
    agent = PPO(config, env)
    history = agent.train()
    print(f"    Episodes: {len(history['episode_rewards'])}")
    if history['episode_rewards']:
        print(f"    Final reward: {np.mean(history['episode_rewards'][-10:]):.2f}")
    print("    [PASS]")

    print("\n[5] Testing evaluation...")
    mean_reward, std_reward = agent.evaluate(n_episodes=3)
    print(f"    Eval: {mean_reward:.2f} ± {std_reward:.2f}")
    print("    [PASS]")

    print("\n[6] Testing save/load...")
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ppo.pt")
        agent.save(path)
        new_agent = PPO(config, env)
        new_agent.load(path)
        for p1, p2 in zip(agent.actor_critic.parameters(), new_agent.actor_critic.parameters()):
            assert torch.allclose(p1, p2)
    print("    [PASS]")

    env.close()
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
