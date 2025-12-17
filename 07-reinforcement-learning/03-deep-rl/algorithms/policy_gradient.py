#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy Gradient Algorithms: A2C and PPO

Production-grade implementations of Advantage Actor-Critic (A2C) and
Proximal Policy Optimization (PPO) algorithms.

========================================
A2C: Advantage Actor-Critic
========================================

Core Idea:
    Synchronous variant of A3C combining policy gradient with value
    function baseline for variance reduction.

Mathematical Framework:
    Policy Gradient Theorem:
        ∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) × A^π(s,a)]

    Advantage Function (reduces variance):
        A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)

    Loss Function:
        L = L_policy + c_v × L_value - c_e × H[π]

        L_policy = -E[log π(a|s) × Â]  (maximize expected return)
        L_value = E[(V(s) - R)²]       (minimize value error)
        H[π] = -E[π log π]             (maximize entropy)

Comparison with DQN:
    + Handles continuous action spaces naturally
    + Learns stochastic policies (built-in exploration)
    + On-policy (theoretically sound gradients)
    - Lower sample efficiency (data discarded after use)
    - Higher variance in gradient estimates

========================================
PPO: Proximal Policy Optimization
========================================

Core Idea:
    Constrain policy updates to prevent catastrophic forgetting while
    allowing multiple gradient steps per data batch.

Mathematical Framework:
    Policy Ratio:
        r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

    Clipped Surrogate Objective:
        L^{CLIP} = E[min(r_t × Â_t, clip(r_t, 1-ε, 1+ε) × Â_t)]

    Trust Region Interpretation:
        Clipping enforces: |π_new - π_old| ≤ ε × π_old
        This prevents drastic policy changes per update.

    GAE (Generalized Advantage Estimation):
        Â_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
        δ_t = r_t + γV(s_{t+1}) - V(s_t)

Comparison with A2C:
    + Multiple epochs per batch (better sample efficiency)
    + Stable updates via clipping (easier hyperparameter tuning)
    + State-of-the-art on many continuous control tasks
    - Slightly more complex implementation
    - Additional hyperparameters (ε, epochs)

========================================
Complexity Analysis
========================================

Space:
    O(n_steps × state_dim + hidden_dim²)

Time per update:
    A2C: O(n_steps × hidden_dim²)
    PPO: O(n_epochs × n_steps × hidden_dim²)

Sample Complexity:
    On-policy methods typically require 10x-100x more samples than
    off-policy (DQN), but scale better to complex environments.

========================================
References
========================================

[1] Mnih et al., "Asynchronous Methods for Deep RL", ICML 2016
[2] Schulman et al., "High-Dimensional Continuous Control Using GAE", ICLR 2016
[3] Schulman et al., "Proximal Policy Optimization Algorithms", 2017
[4] Andrychowicz et al., "What Matters In On-Policy RL", ICLR 2021
"""

from __future__ import annotations
import os
import warnings
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.networks import ActorCriticNetwork
from core.buffers import RolloutBuffer
from core.utils import get_device, set_seed

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


@dataclass
class A2CConfig:
    """A2C hyperparameter configuration."""
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 5
    device: str = "auto"

    def __post_init__(self):
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Dimensions must be positive")


@dataclass
class PPOConfig:
    """PPO hyperparameter configuration."""
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    n_epochs: int = 10
    mini_batch_size: int = 64
    device: str = "auto"

    def __post_init__(self):
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Dimensions must be positive")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1)")


class A2CAgent:
    """
    Advantage Actor-Critic agent.

    Algorithm Summary:
        1. Collect n_steps of experience with current policy
        2. Compute GAE advantages: Â_t = Σ (γλ)^l δ_{t+l}
        3. Update: L = -log π × Â + c_v × (V - R)² - c_e × H[π]
        4. Repeat

    Attributes:
        config: Hyperparameters
        network: Shared actor-critic network
        optimizer: Adam optimizer
        buffer: Rollout storage
    """

    def __init__(self, config: A2CConfig) -> None:
        self.config = config
        self.device = get_device(config.device)

        self.network = ActorCriticNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )

        self.buffer = RolloutBuffer(config.gamma, config.gae_lambda)

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_t)
        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, log_prob, reward, value, done):
        self.buffer.add(state, action, log_prob, reward, value, done)

    def update(self, last_value: float) -> Dict[str, float]:
        returns, advantages = self.buffer.compute_gae(last_value)

        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        _, new_log_probs, entropy, values = self.network.get_action_and_value(states, actions)

        policy_loss = -(new_log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        loss = (policy_loss +
                self.config.value_coef * value_loss +
                self.config.entropy_coef * entropy_loss)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.buffer.reset()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item()
        }


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Algorithm Summary:
        1. Collect n_steps of experience
        2. Compute GAE advantages
        3. For each epoch:
            - Shuffle and split into mini-batches
            - Compute ratio r = π_new / π_old
            - Clipped loss: min(r×Â, clip(r)×Â)
            - Update network
        4. Repeat

    Key Difference from A2C:
        - Multiple gradient updates per data batch
        - Clipping prevents destructive updates
    """

    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = get_device(config.device)

        self.network = ActorCriticNetwork(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=config.learning_rate, eps=1e-5
        )

        self.buffer = RolloutBuffer(config.gamma, config.gae_lambda)

    def get_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_t)
        return action.item(), log_prob.item(), value.item()

    def store(self, state, action, log_prob, reward, value, done):
        self.buffer.add(state, action, log_prob, reward, value, done)

    def update(self, last_value: float) -> Dict[str, float]:
        returns, advantages = self.buffer.compute_gae(last_value)

        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = len(states)
        total_losses = {"policy": 0, "value": 0, "entropy": 0}
        n_updates = 0

        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(batch_size)

            for start in range(0, batch_size, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    mb_states, mb_actions
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = (policy_loss +
                        self.config.value_coef * value_loss +
                        self.config.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                total_losses["policy"] += policy_loss.item()
                total_losses["value"] += value_loss.item()
                total_losses["entropy"] += entropy.mean().item()
                n_updates += 1

        self.buffer.reset()

        return {
            "policy_loss": total_losses["policy"] / n_updates,
            "value_loss": total_losses["value"] / n_updates,
            "entropy": total_losses["entropy"] / n_updates
        }


def train_policy_gradient(
    agent,
    env_name: str = "CartPole-v1",
    total_steps: int = 100000,
    n_steps: int = 128,
    seed: Optional[int] = None,
    algo_name: str = "Agent",
    verbose: bool = True
) -> List[float]:
    """Train A2C or PPO agent."""
    if not HAS_GYM:
        return []

    if seed is not None:
        set_seed(seed)

    env = gym.make(env_name)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training {algo_name}")
        print(f"Environment: {env_name}")
        print(f"{'=' * 60}")

    rewards_history = []
    episode_reward = 0.0
    best_avg = float("-inf")

    state, _ = env.reset(seed=seed)
    step = 0

    while step < total_steps:
        for _ in range(n_steps):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, log_prob, reward, value, done)
            episode_reward += reward
            step += 1

            if done:
                rewards_history.append(episode_reward)
                episode_reward = 0.0
                state, _ = env.reset()
            else:
                state = next_state

            if step >= total_steps:
                break

        _, _, last_value = agent.get_action(state)
        agent.update(last_value)

        if verbose and len(rewards_history) >= 20 and len(rewards_history) % 20 == 0:
            avg = np.mean(rewards_history[-20:])
            best_avg = max(best_avg, avg)
            print(f"Episodes: {len(rewards_history):4d} | Avg: {avg:7.2f} | Best: {best_avg:7.2f}")

    env.close()
    return rewards_history


def train_a2c(
    env_name: str = "CartPole-v1",
    total_steps: int = 100000,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Optional[A2CAgent], List[float]]:
    """Train A2C agent."""
    if not HAS_GYM:
        return None, []

    env = gym.make(env_name)
    config = A2CConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    env.close()

    agent = A2CAgent(config)
    rewards = train_policy_gradient(
        agent, env_name, total_steps, config.n_steps, seed, "A2C", verbose
    )
    return agent, rewards


def train_ppo(
    env_name: str = "CartPole-v1",
    total_steps: int = 100000,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Optional[PPOAgent], List[float]]:
    """Train PPO agent."""
    if not HAS_GYM:
        return None, []

    env = gym.make(env_name)
    config = PPOConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    env.close()

    agent = PPOAgent(config)
    rewards = train_policy_gradient(
        agent, env_name, total_steps, config.n_steps, seed, "PPO", verbose
    )
    return agent, rewards


def run_tests() -> bool:
    """Run unit tests."""
    print("\n" + "=" * 60)
    print("Policy Gradient Unit Tests")
    print("=" * 60)

    passed = failed = 0

    try:
        config = A2CConfig(state_dim=4, action_dim=2)
        agent = A2CAgent(config)
        state = np.random.randn(4).astype(np.float32)
        action, lp, val = agent.get_action(state)
        assert 0 <= action < 2
        for i in range(10):
            agent.store(state, 0, -0.5, 1.0, 0.5, i == 9)
        agent.update(0.0)
        print("Test 1 [A2C basic]: PASSED")
        passed += 1
    except Exception as e:
        print(f"Test 1 [A2C basic]: FAILED - {e}")
        failed += 1

    try:
        config = PPOConfig(state_dim=4, action_dim=2, mini_batch_size=32)
        agent = PPOAgent(config)
        state = np.random.randn(4).astype(np.float32)
        action, lp, val = agent.get_action(state)
        assert 0 <= action < 2
        for i in range(64):
            agent.store(state, 0, -0.5, 1.0, 0.5, i == 63)
        agent.update(0.0)
        print("Test 2 [PPO basic]: PASSED")
        passed += 1
    except Exception as e:
        print(f"Test 2 [PPO basic]: FAILED - {e}")
        failed += 1

    try:
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        for i in range(3):
            buffer.add(np.array([1.0]), 0, 0.0, 1.0, 0.5, i == 2)
        returns, adv = buffer.compute_gae(0.0)

        gamma, lam = 0.99, 0.95
        v = [0.5, 0.5, 0.5, 0.0]
        r = [1.0, 1.0, 1.0]
        d = [0, 0, 1]
        expected = np.zeros(3)
        gae = 0.0
        for t in reversed(range(3)):
            delta = r[t] + gamma * v[t + 1] * (1 - d[t]) - v[t]
            gae = delta + gamma * lam * (1 - d[t]) * gae
            expected[t] = gae
        assert np.allclose(adv.numpy(), expected, atol=1e-5)
        print("Test 3 [GAE computation]: PASSED")
        passed += 1
    except Exception as e:
        print(f"Test 3 [GAE computation]: FAILED - {e}")
        failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Policy Gradient Algorithms")
    parser.add_argument("--algo", choices=["a2c", "ppo", "test"], default="test")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.algo == "a2c":
        train_a2c(total_steps=args.steps, seed=args.seed)
    elif args.algo == "ppo":
        train_ppo(total_steps=args.steps, seed=args.seed)
    else:
        run_tests()
