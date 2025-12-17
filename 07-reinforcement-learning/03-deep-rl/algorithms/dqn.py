#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Q-Network (DQN) Algorithm Implementation

Production-grade implementation of DQN with algorithmic variants including
Double DQN, Dueling DQN, and Prioritized Experience Replay.

========================================
Algorithm Overview
========================================

DQN extends tabular Q-Learning to high-dimensional state spaces through
neural network function approximation. Key innovations:

1. Experience Replay (Lin, 1992; Mnih et al., 2013)
   - Stores transitions (s, a, r, s', done) in buffer
   - Uniformly samples mini-batches for training
   - Breaks temporal correlations, enables i.i.d. SGD assumption
   - Improves sample efficiency through data reuse

2. Target Network (Mnih et al., 2015)
   - Separate network θ⁻ for computing TD targets
   - Periodically synced with online network: θ⁻ ← θ
   - Prevents oscillations from moving targets
   - Stabilizes training convergence

3. Double Q-Learning (van Hasselt et al., 2016)
   - Decouples action selection from evaluation
   - Online network selects: a* = argmax_a Q(s', a; θ)
   - Target network evaluates: Q(s', a*; θ⁻)
   - Eliminates maximization bias: E[max Q] ≥ max E[Q]

4. Dueling Architecture (Wang et al., 2016)
   - Decomposes Q(s,a) = V(s) + A(s,a) - mean(A)
   - V(s): State value (action-independent)
   - A(s,a): Action advantage (relative to average)
   - Faster value learning when actions have similar values

5. Prioritized Experience Replay (Schaul et al., 2015)
   - Samples proportional to TD error magnitude
   - P(i) ∝ |δ_i|^α where δ_i = TD error
   - Importance sampling correction: w_i ∝ (N × P(i))^{-β}
   - Accelerates learning on high-error transitions

========================================
Mathematical Foundations
========================================

Bellman Optimality Equation:
    Q*(s, a) = E[r + γ max_{a'} Q*(s', a') | s, a]

DQN Loss (Temporal Difference):
    L(θ) = E_{(s,a,r,s')~D}[(y - Q(s, a; θ))²]
    y = r + γ max_{a'} Q(s', a'; θ⁻)

Double DQN Target:
    y^{DDQN} = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)

Convergence Conditions:
    1. Universal function approximator
    2. Experience replay (i.i.d. sampling)
    3. Target network (stable targets)
    4. Sufficient exploration (ε-greedy with decay)
    5. Robbins-Monro learning rate schedule

========================================
Complexity Analysis
========================================

Space Complexity:
    - Replay Buffer: O(buffer_size × state_dim)
    - Networks: O(hidden_dim² + hidden_dim × (state_dim + action_dim))
    - Total: O(buffer_size × state_dim + hidden_dim²)

Time Complexity (per update):
    - Sample batch: O(batch_size) uniform, O(batch_size × log buffer_size) PER
    - Forward pass: O(batch_size × hidden_dim²)
    - Backward pass: O(batch_size × hidden_dim²)
    - Total: O(batch_size × hidden_dim²)

Sample Complexity (theoretical):
    O(|S||A| / ((1-γ)⁴ε²)) for ε-optimal policy

========================================
References
========================================

[1] Mnih et al., "Playing Atari with Deep RL", NeurIPS Workshop 2013
[2] Mnih et al., "Human-level control through deep RL", Nature 2015
[3] van Hasselt et al., "Deep RL with Double Q-learning", AAAI 2016
[4] Wang et al., "Dueling Network Architectures", ICML 2016
[5] Schaul et al., "Prioritized Experience Replay", ICLR 2016
[6] Hessel et al., "Rainbow: Combining Improvements", AAAI 2018
"""

from __future__ import annotations
import os
import random
import warnings
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.networks import DQNNetwork, DuelingDQNNetwork
from core.buffers import ReplayBuffer, PrioritizedReplayBuffer
from core.utils import get_device, set_seed

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn("gymnasium not installed, environment interaction unavailable")


@dataclass
class DQNConfig:
    """
    DQN hyperparameter configuration with domain validation.

    All parameters are validated at initialization to ensure they meet
    mathematical and practical constraints.

    Mathematical Constraints:
        - state_dim, action_dim, hidden_dim ∈ ℤ⁺
        - learning_rate ∈ (0, 1]
        - gamma, epsilon_*, per_alpha, per_beta_start ∈ [0, 1]
        - buffer_size > batch_size > 0
        - target_update_freq, per_beta_frames > 0
        - grad_clip > 0

    Default Values (empirically validated):
        - learning_rate = 1e-3: Standard for Adam
        - gamma = 0.99: Long-horizon discounting
        - epsilon decay over ~200 episodes
        - buffer_size = 100000: Sufficient for most tasks
        - batch_size = 64: GPU-efficient
        - target_update = 100: Moderate stability

    Attributes:
        state_dim: State observation dimensionality
        action_dim: Number of discrete actions
        hidden_dim: Network hidden layer width
        learning_rate: Adam optimizer learning rate
        gamma: Discount factor for future rewards
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Per-episode decay multiplier
        buffer_size: Experience replay capacity
        batch_size: Mini-batch size for SGD
        target_update_freq: Steps between target network syncs
        double_dqn: Enable Double DQN
        dueling: Enable Dueling architecture
        prioritized: Enable Prioritized Experience Replay
        per_alpha: PER prioritization exponent
        per_beta_start: Initial importance sampling correction
        per_beta_frames: Frames to anneal β to 1.0
        grad_clip: Gradient norm clipping threshold
        device: Compute device ("auto", "cpu", "cuda")
    """
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 100
    double_dqn: bool = False
    dueling: bool = False
    prioritized: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100000
    grad_clip: float = 10.0
    device: str = "auto"

    def __post_init__(self) -> None:
        """Validate all hyperparameters."""
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {self.learning_rate}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0 <= self.epsilon_start <= 1:
            raise ValueError(f"epsilon_start must be in [0, 1], got {self.epsilon_start}")
        if not 0 <= self.epsilon_end <= 1:
            raise ValueError(f"epsilon_end must be in [0, 1], got {self.epsilon_end}")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError(f"epsilon_end ({self.epsilon_end}) > epsilon_start ({self.epsilon_start})")
        if not 0 < self.epsilon_decay <= 1:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}")
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.batch_size > self.buffer_size:
            raise ValueError(f"batch_size ({self.batch_size}) > buffer_size ({self.buffer_size})")
        if self.target_update_freq <= 0:
            raise ValueError(f"target_update_freq must be positive, got {self.target_update_freq}")
        if not 0 <= self.per_alpha <= 1:
            raise ValueError(f"per_alpha must be in [0, 1], got {self.per_alpha}")
        if not 0 <= self.per_beta_start <= 1:
            raise ValueError(f"per_beta_start must be in [0, 1], got {self.per_beta_start}")
        if self.per_beta_frames <= 0:
            raise ValueError(f"per_beta_frames must be positive, got {self.per_beta_frames}")
        if self.grad_clip <= 0:
            raise ValueError(f"grad_clip must be positive, got {self.grad_clip}")


class DQNAgent:
    """
    Deep Q-Network agent with algorithmic variants.

    Core Idea:
        Off-policy value-based agent learning optimal Q* through temporal
        difference learning with neural network function approximation.

    Supported Variants:
        - Standard DQN: Basic deep Q-learning
        - Double DQN: Decoupled action selection/evaluation
        - Dueling DQN: Value-advantage decomposition
        - PER DQN: Prioritized experience replay
        - Rainbow (partial): Combination of above

    Usage:
        >>> config = DQNConfig(state_dim=4, action_dim=2, double_dqn=True)
        >>> agent = DQNAgent(config)
        >>> action = agent.get_action(state)
        >>> loss = agent.update(state, action, reward, next_state, done)
        >>> agent.decay_epsilon()

    Attributes:
        config: Hyperparameter configuration
        device: Compute device (CPU/CUDA)
        q_network: Online Q-network (updated every step)
        target_network: Target Q-network (updated periodically)
        optimizer: Adam optimizer
        replay_buffer: Experience storage (uniform or prioritized)
        epsilon: Current exploration rate
        update_count: Total gradient updates performed
        frame_count: Total environment steps seen
    """

    def __init__(self, config: DQNConfig) -> None:
        """
        Initialize DQN agent.

        Args:
            config: Validated hyperparameter configuration
        """
        self.config = config
        self.device = get_device(config.device)

        NetworkClass = DuelingDQNNetwork if config.dueling else DQNNetwork
        self.q_network = NetworkClass(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.target_network = NetworkClass(
            config.state_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )

        if config.prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config.buffer_size,
                alpha=config.per_alpha,
                beta=config.per_beta_start
            )
        else:
            self.replay_buffer = ReplayBuffer(capacity=config.buffer_size)

        self.epsilon = config.epsilon_start
        self.update_count = 0
        self.frame_count = 0

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state observation
            training: If True, use ε-greedy; if False, greedy only

        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_t)
        return q_values.argmax(dim=1).item()

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """
        Store transition and perform gradient update.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag

        Returns:
            Training loss if update performed, None otherwise
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.frame_count += 1

        if not self.replay_buffer.is_ready(self.config.batch_size):
            return None

        if self.config.prioritized:
            batch = self.replay_buffer.sample(self.config.batch_size)
            states, actions, rewards, next_states, dones, indices, weights = batch
            weights_t = torch.FloatTensor(weights).to(self.device)

            fraction = min(1.0, self.frame_count / self.config.per_beta_frames)
            beta = self.config.per_beta_start + fraction * (1.0 - self.config.per_beta_start)
            self.replay_buffer.update_beta(beta)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.config.batch_size
            )
            weights_t = None

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_network(next_states_t).max(dim=1)[0]

            target_q = rewards_t + self.config.gamma * next_q * (1 - dones_t)

        td_errors = current_q - target_q

        if self.config.prioritized:
            loss = (weights_t * (td_errors ** 2)).mean()
            self.replay_buffer.update_priorities(
                indices, td_errors.detach().cpu().numpy()
            )
        else:
            loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "frame_count": self.frame_count,
            "config": self.config
        }, path)

    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]
        self.frame_count = checkpoint["frame_count"]


def train_dqn(
    env_name: str = "CartPole-v1",
    num_episodes: int = 300,
    double_dqn: bool = False,
    dueling: bool = False,
    prioritized: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[Optional[DQNAgent], List[float]]:
    """
    Train DQN agent on Gymnasium environment.

    Args:
        env_name: Gymnasium environment identifier
        num_episodes: Training episodes
        double_dqn: Enable Double DQN
        dueling: Enable Dueling architecture
        prioritized: Enable PER
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple (trained_agent, episode_rewards)
    """
    if not HAS_GYM:
        print("Error: gymnasium not installed")
        return None, []

    if seed is not None:
        set_seed(seed)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    algo_parts = []
    if double_dqn:
        algo_parts.append("Double")
    if dueling:
        algo_parts.append("Dueling")
    if prioritized:
        algo_parts.append("PER")
    algo_parts.append("DQN")
    algo_name = " ".join(algo_parts)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training {algo_name}")
        print(f"Environment: {env_name}")
        print(f"State dim: {state_dim}, Actions: {action_dim}")
        print(f"{'=' * 60}")

    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        double_dqn=double_dqn,
        dueling=dueling,
        prioritized=prioritized
    )
    agent = DQNAgent(config)

    rewards_history: List[float] = []
    best_avg = float("-inf")

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode if seed else None)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        if verbose and (episode + 1) % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            best_avg = max(best_avg, avg)
            print(
                f"Episode {episode + 1:4d} | "
                f"Avg: {avg:7.2f} | "
                f"Best: {best_avg:7.2f} | "
                f"ε: {agent.epsilon:.3f}"
            )

    env.close()

    if verbose:
        eval_rewards = evaluate_agent(agent, env_name, num_episodes=10)
        print(f"\nEvaluation: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")

    return agent, rewards_history


def evaluate_agent(
    agent: DQNAgent,
    env_name: str,
    num_episodes: int = 10
) -> List[float]:
    """
    Evaluate trained agent.

    Args:
        agent: Trained DQN agent
        env_name: Environment name
        num_episodes: Evaluation episodes

    Returns:
        List of episode rewards
    """
    if not HAS_GYM:
        return []

    env = gym.make(env_name)
    rewards: List[float] = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()
    return rewards


def run_tests() -> bool:
    """Run unit tests."""
    print("\n" + "=" * 60)
    print("DQN Unit Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    try:
        config = DQNConfig(state_dim=4, action_dim=2, batch_size=32)
        agent = DQNAgent(config)
        state = np.random.randn(4).astype(np.float32)
        action = agent.get_action(state)
        assert 0 <= action < 2
        print("Test 1 [DQNAgent basic]: PASSED")
        passed += 1
    except Exception as e:
        print(f"Test 1 [DQNAgent basic]: FAILED - {e}")
        failed += 1

    try:
        config = DQNConfig(state_dim=4, action_dim=2, batch_size=32, double_dqn=True)
        agent = DQNAgent(config)
        for _ in range(50):
            s = np.random.randn(4).astype(np.float32)
            agent.update(s, 0, 1.0, s, False)
        print("Test 2 [Double DQN]: PASSED")
        passed += 1
    except Exception as e:
        print(f"Test 2 [Double DQN]: FAILED - {e}")
        failed += 1

    try:
        config = DQNConfig(state_dim=4, action_dim=2, batch_size=32, dueling=True)
        agent = DQNAgent(config)
        for _ in range(50):
            s = np.random.randn(4).astype(np.float32)
            agent.update(s, 0, 1.0, s, False)
        print("Test 3 [Dueling DQN]: PASSED")
        passed += 1
    except Exception as e:
        print(f"Test 3 [Dueling DQN]: FAILED - {e}")
        failed += 1

    try:
        config = DQNConfig(state_dim=4, action_dim=2, batch_size=32, prioritized=True)
        agent = DQNAgent(config)
        for _ in range(50):
            s = np.random.randn(4).astype(np.float32)
            agent.update(s, 0, 1.0, s, False)
        print("Test 4 [PER DQN]: PASSED")
        passed += 1
    except Exception as e:
        print(f"Test 4 [PER DQN]: FAILED - {e}")
        failed += 1

    try:
        import tempfile
        config = DQNConfig(state_dim=4, action_dim=2)
        agent = DQNAgent(config)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        agent.save(path)
        agent2 = DQNAgent(config)
        agent2.load(path)
        os.remove(path)
        for p1, p2 in zip(agent.q_network.parameters(), agent2.q_network.parameters()):
            assert torch.allclose(p1, p2)
        print("Test 5 [Save/Load]: PASSED")
        passed += 1
    except Exception as e:
        print(f"Test 5 [Save/Load]: FAILED - {e}")
        failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN Implementation")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--per", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.train:
        train_dqn(
            num_episodes=args.episodes,
            double_dqn=args.double,
            dueling=args.dueling,
            prioritized=args.per,
            seed=args.seed
        )
    else:
        run_tests()
