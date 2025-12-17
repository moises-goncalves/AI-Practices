#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Popular Reinforcement Learning Algorithms: Comprehensive Implementation Module

================================================================================
Core Idea
================================================================================
This module provides production-grade implementations of state-of-the-art deep
reinforcement learning algorithms, designed for both academic research and
industrial deployment. The algorithms span the spectrum of RL approaches:
value-based, policy gradient, and actor-critic methods.

================================================================================
Mathematical Foundation
================================================================================

1. Markov Decision Process (MDP) Framework
------------------------------------------
All algorithms operate within the MDP framework defined by tuple (S, A, P, R, γ):

    State space:        S ⊆ ℝⁿ (continuous) or finite set (discrete)
    Action space:       A ⊆ ℝᵐ (continuous) or {0, 1, ..., k-1} (discrete)
    Transition:         P(s'|s, a) - state transition probability
    Reward function:    R: S × A × S → ℝ
    Discount factor:    γ ∈ [0, 1]

Objective: Find policy π* maximizing expected cumulative reward:

    J(π) = 𝔼_{τ~π}[∑_{t=0}^∞ γᵗ r_t]

where trajectory τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...) follows policy π.

2. Value Functions
------------------
State Value Function:
    V^π(s) = 𝔼_{π}[∑_{t=0}^∞ γᵗ r_t | s₀ = s]

Action-Value Function (Q-function):
    Q^π(s, a) = 𝔼_{π}[∑_{t=0}^∞ γᵗ r_t | s₀ = s, a₀ = a]

Advantage Function:
    A^π(s, a) = Q^π(s, a) - V^π(s)

Bellman Equations:
    V^π(s) = 𝔼_{a~π}[R(s,a) + γ 𝔼_{s'~P}[V^π(s')]]
    Q^π(s, a) = R(s,a) + γ 𝔼_{s'~P}[V^π(s')]

3. Policy Gradient Theorem
--------------------------
The gradient of the objective with respect to policy parameters θ:

    ∇_θ J(θ) = 𝔼_{τ~π_θ}[∑_{t=0}^T ∇_θ log π_θ(a_t|s_t) A^{π_θ}(s_t, a_t)]

This fundamental result enables gradient-based policy optimization.

4. Generalized Advantage Estimation (GAE)
-----------------------------------------
Exponentially-weighted average of n-step advantages:

    Â_t^{GAE(γ,λ)} = ∑_{l=0}^∞ (γλ)ˡ δ_{t+l}^V

where TD residual:
    δ_t^V = r_t + γV(s_{t+1}) - V(s_t)

Properties:
    - λ = 0: High bias, low variance (TD(0))
    - λ = 1: Low bias, high variance (Monte Carlo)
    - 0 < λ < 1: Optimal bias-variance tradeoff

================================================================================
Problem Statement
================================================================================

Modern RL faces fundamental challenges:

1. Sample Efficiency
   - Real-world interactions are expensive (robotics, autonomous driving)
   - Off-policy methods reuse data but introduce distribution shift
   - On-policy methods are stable but data-inefficient

2. Stability
   - Function approximation + bootstrapping + off-policy = "deadly triad"
   - Policy gradient has high variance
   - Large policy updates cause catastrophic performance collapse

3. Exploration vs Exploitation
   - Insufficient exploration: local optima
   - Excessive exploration: slow convergence
   - Balance requires careful hyperparameter tuning

4. Continuous Action Spaces
   - Q-learning requires max_a Q(s,a) - infeasible for continuous A
   - Policy gradient naturally handles continuous actions
   - Deterministic policies (DDPG, TD3) vs stochastic (SAC)

================================================================================
Algorithm Comparison
================================================================================

+------------+----------+----------+----------+----------+----------+----------+
| Algorithm  | Action   | On/Off   | Sample   | Variance | Stability| Hyperpar.|
|            | Space    | Policy   | Effic.   |          |          | Sensitiv.|
+============+==========+==========+==========+==========+==========+==========+
| DQN        | Discrete | Off      | High     | Low      | Medium   | Low      |
+------------+----------+----------+----------+----------+----------+----------+
| A2C        | Both     | On       | Low      | Medium   | High     | Medium   |
+------------+----------+----------+----------+----------+----------+----------+
| PPO        | Both     | On       | Low      | Low      | High     | Low      |
+------------+----------+----------+----------+----------+----------+----------+
| DDPG       | Cont.    | Off      | High     | Low      | Low      | High     |
+------------+----------+----------+----------+----------+----------+----------+
| TD3        | Cont.    | Off      | High     | Low      | Medium   | Medium   |
+------------+----------+----------+----------+----------+----------+----------+
| SAC        | Cont.    | Off      | High     | Medium   | High     | Low      |
+------------+----------+----------+----------+----------+----------+----------+

================================================================================
Complexity Analysis
================================================================================

Let:
    d_s = state dimension
    d_a = action dimension
    h = hidden layer width
    B = batch size
    N = buffer capacity
    T = episode length

Space Complexity:
    - Network parameters: O(h² + h(d_s + d_a))
    - Replay buffer: O(N × d_s)
    - Total: O(N × d_s + |θ|)

Time Complexity per update:
    - Forward pass: O(B × |θ|)
    - Backward pass: O(B × |θ|)
    - Target computation: O(B × |θ|)
    - Total: O(B × |θ|)

================================================================================
Algorithm Summary
================================================================================

1. DQN (Deep Q-Network)
   - Value-based method for discrete actions
   - Experience replay + target network for stability
   - Variants: Double DQN, Dueling DQN, Prioritized ER

2. A2C (Advantage Actor-Critic)
   - On-policy actor-critic with advantage baseline
   - Synchronous parallel data collection
   - Foundation for PPO

3. PPO (Proximal Policy Optimization)
   - Trust region method with clipped surrogate objective
   - Most popular on-policy algorithm in practice
   - Robust hyperparameters, simple implementation

4. DDPG (Deep Deterministic Policy Gradient)
   - Off-policy actor-critic for continuous actions
   - Deterministic policy with OU noise for exploration
   - Often unstable, superseded by TD3

5. TD3 (Twin Delayed DDPG)
   - Addresses DDPG instability with three techniques:
     a) Clipped double Q-learning
     b) Delayed policy updates
     c) Target policy smoothing
   - State-of-the-art for deterministic continuous control

6. SAC (Soft Actor-Critic)
   - Maximum entropy RL: max 𝔼[∑ γᵗ(r_t + αH(π(·|s_t)))]
   - Automatic temperature adjustment
   - State-of-the-art for stochastic continuous control

================================================================================
References
================================================================================
[1] Mnih et al., "Human-level control through deep RL", Nature 2015
[2] Mnih et al., "Asynchronous Methods for Deep RL", ICML 2016
[3] Schulman et al., "Proximal Policy Optimization", arXiv 2017
[4] Lillicrap et al., "Continuous control with deep RL", ICLR 2016
[5] Fujimoto et al., "Addressing Function Approximation Error", ICML 2018
[6] Haarnoja et al., "Soft Actor-Critic", ICML 2018
[7] Schulman et al., "High-Dimensional Continuous Control Using GAE", ICLR 2016
[8] Sutton & Barto, "Reinforcement Learning: An Introduction", 2nd ed. 2018

================================================================================
Requirements
================================================================================
    Python >= 3.8
    PyTorch >= 1.9
    NumPy >= 1.20
    Gymnasium >= 0.28 (optional, for training)
    matplotlib >= 3.3 (optional, for visualization)

Installation:
    pip install torch numpy gymnasium matplotlib

================================================================================
Usage
================================================================================
    # Quick start
    python popular_rl_algorithms.py --algo sac --env Pendulum-v1 --timesteps 50000

    # Run tests
    python popular_rl_algorithms.py --test

    # Compare algorithms
    python popular_rl_algorithms.py --compare

Author: AI-Practices Contributors
License: MIT
"""

from __future__ import annotations

import copy
import math
import os
import random
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn("gymnasium not installed, training functionality unavailable")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not installed, plotting functionality unavailable")


# ============================================================================
#                           Type Definitions
# ============================================================================

T = TypeVar("T")
StateType = Union[np.ndarray, Tuple[float, ...]]
ActionType = Union[int, np.ndarray]


class Transition(NamedTuple):
    """
    Single-step transition tuple for experience replay.

    Stores the fundamental MDP transition (s, a, r, s', done) with type safety.

    Mathematical Context:
        Represents one step of the Markov chain:
        s_t → a_t → r_t, s_{t+1}

    Attributes:
        state: Current state s_t ∈ S
        action: Executed action a_t ∈ A
        reward: Immediate reward r_t = R(s_t, a_t, s_{t+1})
        next_state: Successor state s_{t+1} ~ P(·|s_t, a_t)
        done: Terminal flag indicating episode end
    """
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


# ============================================================================
#                           Neural Network Utilities
# ============================================================================

def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    """
    Orthogonal weight initialization.

    Mathematical Background:
        Orthogonal matrices preserve norm: ||Wx|| = ||x||
        This maintains gradient magnitude through deep networks,
        preventing vanishing/exploding gradients.

    Recommended gains:
        - sqrt(2) for ReLU (accounts for zeroed negative half)
        - 1.0 for tanh/linear
        - 0.01 for policy output (small initial changes)

    Args:
        module: Neural network module to initialize
        gain: Scaling factor for initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: Type[nn.Module] = nn.ReLU,
    output_activation: Optional[Type[nn.Module]] = None,
) -> nn.Sequential:
    """
    Create a Multi-Layer Perceptron (MLP).

    Architecture:
        input → [Linear → Activation] × L → Linear [→ OutputActivation]

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        hidden_dims: List of hidden layer widths
        activation: Activation function class for hidden layers
        output_activation: Optional output activation class

    Returns:
        nn.Sequential containing the MLP layers
    """
    layers: List[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))

    if output_activation is not None:
        layers.append(output_activation())

    return nn.Sequential(*layers)


# ============================================================================
#                           Experience Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    Uniform Experience Replay Buffer.

    ============================================================================
    Core Idea
    ============================================================================
    Stores agent-environment interactions and enables uniform random sampling.
    Breaking temporal correlations transforms RL into pseudo-supervised learning,
    satisfying the i.i.d. assumption required by stochastic gradient descent.

    ============================================================================
    Mathematical Foundation
    ============================================================================
    Let buffer D = {τ₁, τ₂, ..., τ_N} contain N transitions.

    Uniform sampling: P(τᵢ) = 1/N for all i

    The loss function becomes:
        L(θ) = 𝔼_{τ~D}[(y - Q(s,a;θ))²]

    where y is the TD target computed from τ.

    ============================================================================
    Algorithm Comparison
    ============================================================================
    vs. Online Learning:
        + Breaks temporal correlation (consecutive states are similar)
        + Sample reuse (each transition used multiple times)
        + Lower gradient variance (batch averaging)
        - Distribution shift (old policy data)
        - Memory overhead

    vs. Prioritized Replay:
        + Simpler implementation
        + Unbiased sampling
        - Ignores sample importance
        - Inefficient for rare events

    ============================================================================
    Complexity
    ============================================================================
    Space: O(N × (d_s + d_a))
    Push: O(1) amortized
    Sample: O(B)
    """

    __slots__ = ("_capacity", "_buffer", "_position")

    def __init__(self, capacity: int) -> None:
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store

        Raises:
            ValueError: If capacity is not positive
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self._capacity = capacity
        self._buffer: List[Transition] = []
        self._position = 0

    @property
    def capacity(self) -> int:
        """Maximum buffer capacity."""
        return self._capacity

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self._buffer)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition.

        Implements circular buffer: overwrites oldest when full.

        Args:
            state: Current state
            action: Executed action
            reward: Immediate reward
            next_state: Next state
            done: Episode termination flag
        """
        transition = Transition(state, action, reward, next_state, done)

        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._position] = transition

        self._position = (self._position + 1) % self._capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random mini-batch.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each array has batch_size as first dimension

        Raises:
            ValueError: If batch_size exceeds buffer size
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({len(self._buffer)})"
            )

        batch = random.sample(self._buffer, batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.float32)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has sufficient samples for training."""
        return len(self._buffer) >= batch_size

    def clear(self) -> None:
        """Clear all stored transitions."""
        self._buffer.clear()
        self._position = 0

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self._capacity}, size={len(self)})"


# ============================================================================
#                           Actor Networks
# ============================================================================

class DeterministicActor(nn.Module):
    """
    Deterministic Policy Network for Continuous Actions.

    ============================================================================
    Core Idea
    ============================================================================
    Outputs deterministic action for given state: a = μ_θ(s)
    Used in DDPG and TD3 for continuous control tasks.

    ============================================================================
    Mathematical Foundation
    ============================================================================
    Policy: μ_θ: S → A
        μ_θ(s) = tanh(f_θ(s)) × action_scale + action_bias

    tanh bounds output to [-1, 1], then scaled to action space.

    Deterministic Policy Gradient:
        ∇_θ J = 𝔼_s[∇_a Q(s,a)|_{a=μ_θ(s)} × ∇_θ μ_θ(s)]

    ============================================================================
    Algorithm Comparison
    ============================================================================
    vs. Stochastic Actor:
        + Lower variance (no sampling)
        + Simpler gradient computation
        - Requires external exploration noise
        - May converge to local optima

    ============================================================================
    Complexity
    ============================================================================
    Parameters: O(d_s × h + h² + h × d_a)
    Forward: O(B × |θ|)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        max_action: float = 1.0,
    ) -> None:
        """
        Initialize deterministic actor.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer widths
            max_action: Action space bound (symmetric: [-max, max])
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.network = _create_mlp(state_dim, action_dim, hidden_dims)
        self.apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))

        if len(self.network) > 0:
            final_layer = self.network[-1]
            if isinstance(final_layer, nn.Linear):
                _orthogonal_init(final_layer, gain=0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute deterministic action.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Action tensor (batch_size, action_dim), bounded by max_action
        """
        return torch.tanh(self.network(state)) * self.max_action


class GaussianActor(nn.Module):
    """
    Stochastic Gaussian Policy Network for Continuous Actions.

    ============================================================================
    Core Idea
    ============================================================================
    Outputs Gaussian distribution parameters: a ~ N(μ_θ(s), σ_θ(s)²)
    Enables principled exploration through learned stochasticity.

    ============================================================================
    Mathematical Foundation
    ============================================================================
    Policy distribution:
        π_θ(a|s) = N(μ_θ(s), σ_θ(s)²)

    For bounded actions, apply tanh squashing:
        u ~ N(μ, σ²),  a = tanh(u)

    Log probability with Jacobian correction:
        log π(a|s) = log N(u|μ,σ²) - ∑ᵢ log(1 - tanh²(uᵢ))

    Entropy:
        H(π(·|s)) = ½ log(2πeσ²) (Gaussian entropy)

    ============================================================================
    Algorithm Comparison
    ============================================================================
    vs. Deterministic Actor:
        + Built-in exploration
        + Natural for entropy-regularized RL (SAC)
        + Handles multimodal optimal policies
        - Higher variance
        - More parameters (need to output σ)

    ============================================================================
    Complexity
    ============================================================================
    Parameters: O(d_s × h + h² + 2 × h × d_a) (2× for μ and log σ)
    Forward: O(B × |θ|)
    """

    LOG_STD_MIN: float = -20.0
    LOG_STD_MAX: float = 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        max_action: float = 1.0,
    ) -> None:
        """
        Initialize Gaussian actor.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer widths
            max_action: Action space bound
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.shared_net = _create_mlp(state_dim, hidden_dims[-1], hidden_dims[:-1])

        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))
        _orthogonal_init(self.mean_layer, gain=0.01)
        _orthogonal_init(self.log_std_layer, gain=0.01)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Gaussian parameters.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            (mean, log_std): Each shape (batch_size, action_dim)
        """
        features = self.shared_net(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action using reparameterization trick.

        The reparameterization trick enables gradient flow through sampling:
            a = μ + σ × ε,  where ε ~ N(0, I)

        Args:
            state: State tensor

        Returns:
            (action, log_prob, mean):
                - action: Sampled action (batch_size, action_dim)
                - log_prob: Log probability (batch_size,)
                - mean: Distribution mean (batch_size, action_dim)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        u = normal.rsample()

        action = torch.tanh(u) * self.max_action

        log_prob = normal.log_prob(u).sum(dim=-1)
        log_prob -= (
            2 * (np.log(2) - u - F.softplus(-2 * u))
        ).sum(dim=-1)
        log_prob -= np.log(self.max_action) * self.action_dim

        return action, log_prob, torch.tanh(mean) * self.max_action

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability of given action.

        Args:
            state: State tensor
            action: Action tensor (already in action space)

        Returns:
            (log_prob, entropy): Each shape (batch_size,)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        action_normalized = action / self.max_action
        action_clipped = torch.clamp(action_normalized, -1 + 1e-6, 1 - 1e-6)
        u = torch.atanh(action_clipped)

        normal = Normal(mean, std)
        log_prob = normal.log_prob(u).sum(dim=-1)
        log_prob -= (
            2 * (np.log(2) - u - F.softplus(-2 * u))
        ).sum(dim=-1)
        log_prob -= np.log(self.max_action) * self.action_dim

        entropy = normal.entropy().sum(dim=-1)

        return log_prob, entropy


# ============================================================================
#                           Critic Networks
# ============================================================================

class QNetwork(nn.Module):
    """
    Q-Value Network for Continuous Actions.

    ============================================================================
    Core Idea
    ============================================================================
    Estimates action-value function Q(s, a) by concatenating state and action
    as input to an MLP.

    ============================================================================
    Mathematical Foundation
    ============================================================================
    Function approximation:
        Q_θ(s, a) ≈ Q^π(s, a) = 𝔼_π[∑_{t=0}^∞ γᵗ r_t | s₀=s, a₀=a]

    Training objective (TD learning):
        L(θ) = 𝔼[(y - Q_θ(s, a))²]
        where y = r + γ Q_{θ'}(s', π(s'))

    ============================================================================
    Complexity
    ============================================================================
    Parameters: O((d_s + d_a) × h + h² + h)
    Forward: O(B × |θ|)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ) -> None:
        """
        Initialize Q-network.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer widths
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = _create_mlp(state_dim + action_dim, 1, hidden_dims)
        self.apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Estimate Q-value.

        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)

        Returns:
            Q-value tensor (batch_size, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class TwinQNetwork(nn.Module):
    """
    Twin Q-Networks for Double Q-Learning.

    ============================================================================
    Core Idea
    ============================================================================
    Maintains two independent Q-networks and uses minimum for target computation.
    This addresses overestimation bias in Q-learning.

    ============================================================================
    Mathematical Foundation
    ============================================================================
    Standard Q-learning overestimates:
        𝔼[max_a Q(s,a)] ≥ max_a 𝔼[Q(s,a)]

    Clipped double Q-learning:
        y = r + γ min(Q₁'(s',a'), Q₂'(s',a'))
        where a' ~ π(·|s') or a' = μ(s')

    Taking minimum provides pessimistic estimate, reducing overestimation.

    ============================================================================
    Algorithm Usage
    ============================================================================
    - TD3: Uses twin critics + target policy smoothing
    - SAC: Uses twin critics + entropy bonus

    ============================================================================
    Complexity
    ============================================================================
    Parameters: 2 × O((d_s + d_a) × h + h² + h)
    Forward: O(B × |θ|) (either Q1 or Q2)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ) -> None:
        """
        Initialize twin Q-networks.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer widths
        """
        super().__init__()

        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both Q-values.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            (Q1, Q2): Each shape (batch_size, 1)
        """
        return self.q1(state, action), self.q2(state, action)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q1 value only (for actor update)."""
        return self.q1(state, action)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute minimum of Q1 and Q2 (for conservative target)."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class ValueNetwork(nn.Module):
    """
    State Value Network V(s).

    ============================================================================
    Core Idea
    ============================================================================
    Estimates state value function: V(s) = 𝔼_{a~π}[Q(s,a)]
    Used as baseline in actor-critic methods and soft value in SAC.

    ============================================================================
    Mathematical Foundation
    ============================================================================
    State value:
        V^π(s) = 𝔼_{a~π(·|s)}[Q^π(s,a)]

    Soft value (SAC):
        V(s) = 𝔼_{a~π}[Q(s,a) - α log π(a|s)]
             = 𝔼_{a~π}[Q(s,a)] + αH(π(·|s))

    ============================================================================
    Complexity
    ============================================================================
    Parameters: O(d_s × h + h² + h)
    Forward: O(B × |θ|)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256],
    ) -> None:
        """
        Initialize value network.

        Args:
            state_dim: State space dimension
            hidden_dims: Hidden layer widths
        """
        super().__init__()

        self.network = _create_mlp(state_dim, 1, hidden_dims)
        self.apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Value tensor (batch_size, 1)
        """
        return self.network(state)


# ============================================================================
#                           DDPG (Deep Deterministic Policy Gradient)
# ============================================================================

@dataclass
class DDPGConfig:
    """
    DDPG Hyperparameter Configuration.

    ============================================================================
    Core Parameters
    ============================================================================
    - state_dim, action_dim: Environment dimensions
    - max_action: Action space bound (symmetric)
    - gamma: Discount factor (future reward decay)
    - tau: Soft update coefficient (target network momentum)
    - lr_actor, lr_critic: Learning rates

    ============================================================================
    Typical Values (from original paper)
    ============================================================================
    - lr_actor = 1e-4 (slow policy updates)
    - lr_critic = 1e-3 (faster value updates)
    - tau = 0.005 (slow target tracking)
    - gamma = 0.99 (standard for continuing tasks)
    - noise_std = 0.1 (exploration noise)
    """
    state_dim: int
    action_dim: int
    max_action: float = 1.0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1000000
    batch_size: int = 256
    noise_std: float = 0.1
    noise_clip: float = 0.5
    device: str = "auto"

    def get_device(self) -> torch.device:
        """Get compute device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent.

    ============================================================================
    Core Idea
    ============================================================================
    Extends DQN to continuous action spaces using deterministic policy gradient.
    Actor outputs action directly, critic evaluates state-action pairs.

    ============================================================================
    Mathematical Foundation
    ============================================================================
    Deterministic Policy Gradient (Silver et al., 2014):
        ∇_θ J = 𝔼_s~D[∇_a Q(s,a)|_{a=μ_θ(s)} × ∇_θ μ_θ(s)]

    Critic update (TD learning):
        L(φ) = 𝔼_{(s,a,r,s')~D}[(r + γQ_{φ'}(s', μ_{θ'}(s')) - Q_φ(s,a))²]

    Target networks (Polyak averaging):
        θ' ← τθ + (1-τ)θ'
        φ' ← τφ + (1-τ)φ'

    Exploration: Ornstein-Uhlenbeck or Gaussian noise
        a = μ_θ(s) + ε,  ε ~ N(0, σ²)

    ============================================================================
    Algorithm Flow
    ============================================================================
    1. Sample action with exploration noise: a = μ(s) + noise
    2. Execute action, observe (r, s', done)
    3. Store transition in replay buffer
    4. Sample mini-batch from buffer
    5. Compute TD target: y = r + γ Q'(s', μ'(s'))
    6. Update critic: minimize (y - Q(s,a))²
    7. Update actor: maximize Q(s, μ(s))
    8. Soft update targets: θ' ← τθ + (1-τ)θ'

    ============================================================================
    Algorithm Comparison
    ============================================================================
    vs. DQN:
        + Handles continuous actions (no max_a required)
        + Lower variance (deterministic gradient)
        - Requires separate actor network
        - Less exploration (deterministic policy)

    vs. TD3:
        - Higher overestimation bias (single Q)
        - More sensitive to hyperparameters
        - Often unstable training

    vs. SAC:
        - No entropy regularization
        - Manual exploration noise tuning
        - Often suboptimal asymptotic performance

    ============================================================================
    Complexity
    ============================================================================
    Space:
        - Networks: O(|θ_actor| + |θ_critic|) × 2 (targets)
        - Buffer: O(N × (d_s + d_a))

    Time (per update):
        - Forward: O(B × |θ|)
        - Backward: O(B × |θ|)
        - Target update: O(|θ|)

    ============================================================================
    Summary
    ============================================================================
    DDPG pioneered deep RL for continuous control by combining:
    1. Deterministic policy gradient for efficient continuous action learning
    2. Experience replay for sample efficiency
    3. Target networks for training stability

    Limitations: Brittle hyperparameter sensitivity, overestimation bias.
    Superseded by TD3 and SAC in most applications.
    """

    def __init__(self, config: DDPGConfig) -> None:
        """
        Initialize DDPG agent.

        Args:
            config: Hyperparameter configuration
        """
        self.config = config
        self.device = config.get_device()

        self._init_networks()
        self._init_optimizers()
        self._init_buffer()

        self._training_step = 0

    def _init_networks(self) -> None:
        """Initialize actor, critic, and target networks."""
        self.actor = DeterministicActor(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
            self.config.max_action,
        ).to(self.device)

        self.critic = QNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def _init_optimizers(self) -> None:
        """Initialize optimizers."""
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.lr_actor
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.lr_critic
        )

    def _init_buffer(self) -> None:
        """Initialize replay buffer."""
        self.buffer = ReplayBuffer(self.config.buffer_size)

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> np.ndarray:
        """
        Select action with optional exploration noise.

        Args:
            state: Current state
            training: Whether to add exploration noise

        Returns:
            Action array
        """
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if training:
            noise = np.random.normal(
                0, self.config.noise_std * self.config.max_action,
                size=self.config.action_dim
            )
            action = action + noise
            action = np.clip(action, -self.config.max_action, self.config.max_action)

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one training update.

        Returns:
            Dictionary of loss values, or None if buffer insufficient
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(-1)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            target_q = rewards_t + self.config.gamma * (1 - dones_t) * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update()

        self._training_step += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": current_q.mean().item(),
        }

    def _soft_update(self) -> None:
        """Soft update target networks."""
        tau = self.config.tau

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])


# ============================================================================
#                           TD3 (Twin Delayed DDPG)
# ============================================================================

@dataclass
class TD3Config:
    """
    TD3 Hyperparameter Configuration.

    ============================================================================
    Key TD3 Parameters (beyond DDPG)
    ============================================================================
    - policy_delay: Update actor every N critic updates
    - target_noise: Noise added to target actions
    - noise_clip: Clipping range for target noise

    ============================================================================
    Typical Values (from TD3 paper)
    ============================================================================
    - policy_delay = 2 (update actor half as often)
    - target_noise = 0.2 (smoothing regularization)
    - noise_clip = 0.5 (prevent extreme actions)
    """
    state_dim: int
    action_dim: int
    max_action: float = 1.0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1000000
    batch_size: int = 256
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    device: str = "auto"

    def get_device(self) -> torch.device:
        """Get compute device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.

    ============================================================================
    Core Idea
    ============================================================================
    Addresses three key issues in DDPG through:
    1. Clipped Double Q-learning (overestimation)
    2. Delayed Policy Updates (variance reduction)
    3. Target Policy Smoothing (regularization)

    ============================================================================
    Mathematical Foundation
    ============================================================================
    1. Clipped Double Q-learning:
       y = r + γ min(Q₁'(s', ã'), Q₂'(s', ã'))

       Taking minimum of two critics provides pessimistic estimate,
       counteracting overestimation bias.

    2. Delayed Policy Updates:
       Update actor every D critic updates (typically D=2).
       Allows critic to become more accurate before policy update.

    3. Target Policy Smoothing:
       ã' = μ'(s') + clip(ε, -c, c),  ε ~ N(0, σ²)

       Adds noise to target actions, smoothing the Q-function over actions.
       Acts as regularizer, preventing exploitation of Q-function errors.

    ============================================================================
    Algorithm Flow
    ============================================================================
    1. Sample action with exploration noise
    2. Execute and store transition
    3. Sample batch from buffer
    4. Compute smoothed target action: ã' = μ'(s') + clipped_noise
    5. Compute clipped target: y = r + γ min(Q₁', Q₂')
    6. Update both critics: minimize MSE loss
    7. Every D steps:
       - Update actor: maximize Q₁(s, μ(s))
       - Soft update all targets

    ============================================================================
    Algorithm Comparison
    ============================================================================
    vs. DDPG:
        + More stable training (twin Q, delayed updates)
        + Less overestimation (pessimistic Q)
        + Better asymptotic performance
        - Slightly more computation (twin networks)

    vs. SAC:
        - No automatic exploration (manual noise tuning)
        - Deterministic policy (may miss multimodal solutions)
        + Simpler implementation
        + Faster per-update computation

    ============================================================================
    Complexity
    ============================================================================
    Space:
        - Networks: 2 × O(|θ_Q|) + O(|θ_μ|) + targets
        - Buffer: O(N × (d_s + d_a))

    Time (per update):
        - Critic: O(B × |θ_Q|)
        - Actor (every D): O(B × |θ_μ|)

    ============================================================================
    Summary
    ============================================================================
    TD3 is the standard choice for deterministic continuous control:
    - Three simple techniques dramatically improve DDPG stability
    - State-of-the-art results on MuJoCo benchmarks
    - Robust hyperparameters across tasks

    Best for: Deterministic control tasks where sample efficiency matters
    Consider SAC for: Exploration-heavy tasks, maximum entropy regularization
    """

    def __init__(self, config: TD3Config) -> None:
        """
        Initialize TD3 agent.

        Args:
            config: Hyperparameter configuration
        """
        self.config = config
        self.device = config.get_device()

        self._init_networks()
        self._init_optimizers()
        self._init_buffer()

        self._training_step = 0
        self._critic_update_count = 0

    def _init_networks(self) -> None:
        """Initialize actor, twin critics, and targets."""
        self.actor = DeterministicActor(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
            self.config.max_action,
        ).to(self.device)

        self.critic = TwinQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def _init_optimizers(self) -> None:
        """Initialize optimizers."""
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.lr_actor
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.lr_critic
        )

    def _init_buffer(self) -> None:
        """Initialize replay buffer."""
        self.buffer = ReplayBuffer(self.config.buffer_size)

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> np.ndarray:
        """
        Select action with optional exploration noise.

        Args:
            state: Current state
            training: Whether to add exploration noise

        Returns:
            Action array
        """
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if training:
            noise = np.random.normal(
                0, self.config.exploration_noise * self.config.max_action,
                size=self.config.action_dim
            )
            action = action + noise
            action = np.clip(action, -self.config.max_action, self.config.max_action)

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one training update.

        Returns:
            Dictionary of loss values, or None if buffer insufficient
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(-1)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            noise = (
                torch.randn_like(actions_t) * self.config.policy_noise
            ).clamp(-self.config.noise_clip, self.config.noise_clip)

            next_actions = (
                self.actor_target(next_states_t) + noise
            ).clamp(-self.config.max_action, self.config.max_action)

            target_q1, target_q2 = self.critic_target(next_states_t, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards_t + self.config.gamma * (1 - dones_t) * target_q

        current_q1, current_q2 = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._critic_update_count += 1
        self._training_step += 1

        result = {
            "critic_loss": critic_loss.item(),
            "q1_value": current_q1.mean().item(),
            "q2_value": current_q2.mean().item(),
        }

        if self._critic_update_count % self.config.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(states_t, self.actor(states_t)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update()

            result["actor_loss"] = actor_loss.item()

        return result

    def _soft_update(self) -> None:
        """Soft update target networks."""
        tau = self.config.tau

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])


# ============================================================================
#                           SAC (Soft Actor-Critic)
# ============================================================================

@dataclass
class SACConfig:
    """
    SAC Hyperparameter Configuration.

    ============================================================================
    Key SAC Parameters
    ============================================================================
    - alpha: Temperature (entropy coefficient), or "auto" for automatic
    - target_entropy: Target entropy for automatic alpha (default: -d_a)
    - lr_alpha: Learning rate for alpha (if automatic)

    ============================================================================
    Typical Values (from SAC paper)
    ============================================================================
    - alpha = 0.2 or "auto"
    - lr_alpha = 3e-4
    - target_entropy = -action_dim
    """
    state_dim: int
    action_dim: int
    max_action: float = 1.0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_size: int = 1000000
    batch_size: int = 256
    alpha: Union[float, str] = "auto"
    target_entropy: Optional[float] = None
    device: str = "auto"

    def get_device(self) -> torch.device:
        """Get compute device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent.

    ============================================================================
    Core Idea
    ============================================================================
    Maximum entropy reinforcement learning: learn policy that maximizes both
    expected return AND entropy, encouraging exploration and robustness.

    ============================================================================
    Mathematical Foundation
    ============================================================================
    Maximum Entropy Objective:
        J(π) = ∑_t 𝔼_{(s_t,a_t)~ρ_π}[r(s_t,a_t) + α H(π(·|s_t))]

    where α (temperature) controls entropy-reward tradeoff.

    Soft Value Functions:
        V(s) = 𝔼_{a~π}[Q(s,a) - α log π(a|s)]
        Q(s,a) = r + γ 𝔼_{s'}[V(s')]

    Equivalently:
        Q(s,a) = r + γ 𝔼_{s',a'}[Q(s',a') - α log π(a'|s')]

    Policy Improvement:
        π_new = argmin_π D_KL(π(·|s) || exp(Q(s,·)/α) / Z(s))

    For Gaussian policy, this becomes gradient ascent on:
        𝔼_{s~D,a~π}[Q(s,a) - α log π(a|s)]

    Automatic Temperature Adjustment:
        J(α) = 𝔼_{a~π}[-α(log π(a|s) + H̄)]

    where H̄ is target entropy (typically -d_a).

    ============================================================================
    Algorithm Flow
    ============================================================================
    1. Sample action from stochastic policy: a ~ π(·|s)
    2. Execute and store transition
    3. Sample batch from buffer
    4. Update critics with soft Q-learning target
    5. Update actor to maximize Q - α log π
    6. Update temperature α to match target entropy
    7. Soft update target critics

    ============================================================================
    Algorithm Comparison
    ============================================================================
    vs. TD3:
        + Automatic exploration (learned stochastic policy)
        + Entropy regularization improves robustness
        + Often better asymptotic performance
        - Slightly more complex (temperature, stochastic policy)

    vs. DDPG:
        + All advantages of TD3
        + Built-in exploration
        + More stable training

    vs. PPO:
        + Higher sample efficiency (off-policy)
        + Better continuous control performance
        - Higher computation per sample
        - No discrete action support

    ============================================================================
    Complexity
    ============================================================================
    Space:
        - Networks: 2 × O(|θ_Q|) + O(|θ_π|) + targets + α
        - Buffer: O(N × (d_s + d_a))

    Time (per update):
        - Critic: O(B × |θ_Q|)
        - Actor: O(B × |θ_π|)
        - Alpha: O(B) (if automatic)

    ============================================================================
    Summary
    ============================================================================
    SAC is the state-of-the-art off-policy algorithm for continuous control:
    - Maximum entropy framework provides principled exploration
    - Automatic temperature tuning eliminates key hyperparameter
    - Robust and stable across diverse tasks
    - Excellent sample efficiency

    Best for: Continuous control with exploration requirements
    """

    def __init__(self, config: SACConfig) -> None:
        """
        Initialize SAC agent.

        Args:
            config: Hyperparameter configuration
        """
        self.config = config
        self.device = config.get_device()

        self._init_networks()
        self._init_temperature()
        self._init_optimizers()
        self._init_buffer()

        self._training_step = 0

    def _init_networks(self) -> None:
        """Initialize actor, twin critics, and targets."""
        self.actor = GaussianActor(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
            self.config.max_action,
        ).to(self.device)

        self.critic = TwinQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)

        for param in self.critic_target.parameters():
            param.requires_grad = False

    def _init_temperature(self) -> None:
        """Initialize temperature (alpha) parameter."""
        self.auto_alpha = (self.config.alpha == "auto")

        if self.auto_alpha:
            self.target_entropy = (
                self.config.target_entropy
                if self.config.target_entropy is not None
                else -self.config.action_dim
            )
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = float(self.config.alpha)
            self.log_alpha = None

    def _init_optimizers(self) -> None:
        """Initialize optimizers."""
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.lr_actor
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.lr_critic
        )

        if self.auto_alpha:
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.config.lr_alpha
            )

    def _init_buffer(self) -> None:
        """Initialize replay buffer."""
        self.buffer = ReplayBuffer(self.config.buffer_size)

    def select_action(
        self, state: np.ndarray, training: bool = True
    ) -> np.ndarray:
        """
        Select action from stochastic policy.

        Args:
            state: Current state
            training: Whether to sample or use mean

        Returns:
            Action array
        """
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            if training:
                action, _, _ = self.actor.sample(state_tensor)
            else:
                _, _, action = self.actor.sample(state_tensor)

        return action.cpu().numpy()[0]

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one training update.

        Returns:
            Dictionary of loss values, or None if buffer insufficient
        """
        if not self.buffer.is_ready(self.config.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config.batch_size
        )

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(-1)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states_t)
            target_q1, target_q2 = self.critic_target(next_states_t, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs.unsqueeze(-1)
            target_q = rewards_t + self.config.gamma * (1 - dones_t) * target_q

        current_q1, current_q2 = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probs, _ = self.actor.sample(states_t)
        q1_new, q2_new = self.critic(states_t, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs.unsqueeze(-1) - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        result = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": current_q1.mean().item(),
            "log_prob": log_probs.mean().item(),
            "alpha": self.alpha,
        }

        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            result["alpha_loss"] = alpha_loss.item()

        self._soft_update()
        self._training_step += 1

        return result

    def _soft_update(self) -> None:
        """Soft update target critic."""
        tau = self.config.tau

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config,
        }
        if self.auto_alpha:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])


# ============================================================================
#                           Training Utilities
# ============================================================================

def train_continuous_agent(
    agent: Union[DDPGAgent, TD3Agent, SACAgent],
    env_name: str = "Pendulum-v1",
    total_timesteps: int = 100000,
    start_timesteps: int = 10000,
    eval_freq: int = 5000,
    eval_episodes: int = 10,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Union[DDPGAgent, TD3Agent, SACAgent], Dict[str, List[float]]]:
    """
    Train continuous control agent.

    Args:
        agent: DDPG, TD3, or SAC agent
        env_name: Gymnasium environment name
        total_timesteps: Total training steps
        start_timesteps: Random actions before training
        eval_freq: Evaluation frequency (steps)
        eval_episodes: Number of evaluation episodes
        seed: Random seed
        verbose: Print progress

    Returns:
        (trained_agent, metrics): Agent and training history
    """
    if not HAS_GYM:
        raise ImportError("gymnasium required for training")

    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    metrics = {
        "timesteps": [],
        "eval_returns": [],
        "critic_loss": [],
        "actor_loss": [],
    }

    state, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_timesteps = 0

    if verbose:
        algo_name = agent.__class__.__name__.replace("Agent", "")
        print(f"\n{'=' * 60}")
        print(f"Training {algo_name} on {env_name}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"{'=' * 60}")

    for t in range(total_timesteps):
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, training=True)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)

        state = next_state
        episode_return += reward
        episode_timesteps += 1

        if done:
            state, _ = env.reset()
            episode_return = 0.0
            episode_timesteps = 0

        if t >= start_timesteps:
            loss_info = agent.update()
            if loss_info:
                metrics["critic_loss"].append(loss_info.get("critic_loss", 0))
                if "actor_loss" in loss_info:
                    metrics["actor_loss"].append(loss_info["actor_loss"])

        if (t + 1) % eval_freq == 0:
            eval_returns = evaluate_agent(agent, eval_env, eval_episodes)
            mean_return = np.mean(eval_returns)

            metrics["timesteps"].append(t + 1)
            metrics["eval_returns"].append(mean_return)

            if verbose:
                print(
                    f"Step {t + 1:7d} | "
                    f"Eval return: {mean_return:8.2f} ± {np.std(eval_returns):6.2f}"
                )

    env.close()
    eval_env.close()

    return agent, metrics


def evaluate_agent(
    agent: Union[DDPGAgent, TD3Agent, SACAgent],
    env,
    num_episodes: int = 10,
) -> List[float]:
    """
    Evaluate agent performance.

    Args:
        agent: Agent to evaluate
        env: Gymnasium environment
        num_episodes: Number of evaluation episodes

    Returns:
        List of episode returns
    """
    returns = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_return += reward

        returns.append(episode_return)

    return returns


def compare_algorithms(
    env_name: str = "Pendulum-v1",
    total_timesteps: int = 50000,
    seed: int = 42,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compare DDPG, TD3, and SAC on same environment.

    Args:
        env_name: Environment name
        total_timesteps: Training steps per algorithm
        seed: Random seed

    Returns:
        Dictionary mapping algorithm names to their metrics
    """
    if not HAS_GYM:
        raise ImportError("gymnasium required")

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()

    results = {}

    print("\n[1/3] Training DDPG...")
    ddpg_config = DDPGConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
    )
    ddpg_agent = DDPGAgent(ddpg_config)
    _, ddpg_metrics = train_continuous_agent(
        ddpg_agent, env_name, total_timesteps, seed=seed
    )
    results["DDPG"] = ddpg_metrics

    print("\n[2/3] Training TD3...")
    td3_config = TD3Config(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
    )
    td3_agent = TD3Agent(td3_config)
    _, td3_metrics = train_continuous_agent(
        td3_agent, env_name, total_timesteps, seed=seed
    )
    results["TD3"] = td3_metrics

    print("\n[3/3] Training SAC...")
    sac_config = SACConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
    )
    sac_agent = SACAgent(sac_config)
    _, sac_metrics = train_continuous_agent(
        sac_agent, env_name, total_timesteps, seed=seed
    )
    results["SAC"] = sac_metrics

    if HAS_MATPLOTLIB:
        plot_comparison(results, env_name)

    return results


def plot_comparison(
    results: Dict[str, Dict[str, List[float]]],
    env_name: str,
    save_path: Optional[str] = None,
) -> None:
    """Plot algorithm comparison."""
    plt.figure(figsize=(12, 6))

    colors = {"DDPG": "#1f77b4", "TD3": "#ff7f0e", "SAC": "#2ca02c"}

    for name, metrics in results.items():
        if metrics["timesteps"] and metrics["eval_returns"]:
            plt.plot(
                metrics["timesteps"],
                metrics["eval_returns"],
                label=name,
                color=colors.get(name, None),
                linewidth=2,
            )

    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Evaluation Return", fontsize=12)
    plt.title(f"Algorithm Comparison on {env_name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved: {save_path}")
    else:
        plt.savefig("algorithm_comparison.png", dpi=150)
        print("Figure saved: algorithm_comparison.png")

    plt.close()


# ============================================================================
#                           Unit Tests
# ============================================================================

def run_unit_tests() -> bool:
    """
    Run comprehensive unit tests.

    Returns:
        True if all tests pass
    """
    print("\n" + "=" * 60)
    print("Running Unit Tests")
    print("=" * 60)

    all_passed = True

    print("\n[Test 1] ReplayBuffer")
    try:
        buffer = ReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        action = np.array([0.5, -0.5], dtype=np.float32)

        for i in range(50):
            buffer.push(state, action, float(i), state, i % 10 == 0)

        assert len(buffer) == 50
        assert buffer.is_ready(32)

        batch = buffer.sample(32)
        assert len(batch) == 5
        assert batch[0].shape == (32, 4)
        assert batch[1].shape == (32, 2)

        buffer.clear()
        assert len(buffer) == 0

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n[Test 2] DeterministicActor")
    try:
        actor = DeterministicActor(state_dim=4, action_dim=2, max_action=1.0)
        state = torch.randn(32, 4)
        action = actor(state)

        assert action.shape == (32, 2)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n[Test 3] GaussianActor")
    try:
        actor = GaussianActor(state_dim=4, action_dim=2, max_action=1.0)
        state = torch.randn(32, 4)
        action, log_prob, mean = actor.sample(state)

        assert action.shape == (32, 2)
        assert log_prob.shape == (32,)
        assert mean.shape == (32, 2)
        assert not torch.isnan(log_prob).any()

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n[Test 4] QNetwork")
    try:
        critic = QNetwork(state_dim=4, action_dim=2)
        state = torch.randn(32, 4)
        action = torch.randn(32, 2)
        q_value = critic(state, action)

        assert q_value.shape == (32, 1)

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n[Test 5] TwinQNetwork")
    try:
        twin_critic = TwinQNetwork(state_dim=4, action_dim=2)
        state = torch.randn(32, 4)
        action = torch.randn(32, 2)
        q1, q2 = twin_critic(state, action)

        assert q1.shape == (32, 1)
        assert q2.shape == (32, 1)

        min_q = twin_critic.min_q(state, action)
        assert min_q.shape == (32, 1)

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n[Test 6] DDPG Agent")
    try:
        config = DDPGConfig(state_dim=4, action_dim=2, batch_size=32)
        agent = DDPGAgent(config)

        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state, training=True)
        assert action.shape == (2,)

        for _ in range(64):
            s = np.random.randn(4).astype(np.float32)
            a = np.random.randn(2).astype(np.float32)
            agent.store_transition(s, a, 1.0, s, False)

        loss_info = agent.update()
        assert loss_info is not None
        assert "critic_loss" in loss_info

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n[Test 7] TD3 Agent")
    try:
        config = TD3Config(state_dim=4, action_dim=2, batch_size=32)
        agent = TD3Agent(config)

        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state, training=True)
        assert action.shape == (2,)

        for _ in range(64):
            s = np.random.randn(4).astype(np.float32)
            a = np.random.randn(2).astype(np.float32)
            agent.store_transition(s, a, 1.0, s, False)

        for _ in range(3):
            loss_info = agent.update()

        assert loss_info is not None
        assert "critic_loss" in loss_info

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n[Test 8] SAC Agent")
    try:
        config = SACConfig(state_dim=4, action_dim=2, batch_size=32, alpha="auto")
        agent = SACAgent(config)

        state = np.random.randn(4).astype(np.float32)
        action = agent.select_action(state, training=True)
        assert action.shape == (2,)

        for _ in range(64):
            s = np.random.randn(4).astype(np.float32)
            a = np.random.randn(2).astype(np.float32)
            agent.store_transition(s, a, 1.0, s, False)

        loss_info = agent.update()
        assert loss_info is not None
        assert "critic_loss" in loss_info
        assert "actor_loss" in loss_info
        assert "alpha_loss" in loss_info

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n[Test 9] Model Save/Load")
    try:
        import tempfile

        config = SACConfig(state_dim=4, action_dim=2, batch_size=32)
        agent = SACAgent(config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        agent.save(temp_path)

        agent2 = SACAgent(config)
        agent2.load(temp_path)

        os.remove(temp_path)

        for p1, p2 in zip(
            agent.actor.parameters(),
            agent2.actor.parameters()
        ):
            assert torch.allclose(p1, p2)

        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False

    print("\n[Test 10] Environment Interaction")
    if HAS_GYM:
        try:
            env = gym.make("Pendulum-v1")
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])

            config = SACConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                batch_size=32,
            )
            agent = SACAgent(config)

            state, _ = env.reset()
            total_reward = 0.0

            for _ in range(100):
                action = agent.select_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.store_transition(state, action, reward, next_state, done)
                total_reward += reward

                if done:
                    state, _ = env.reset()
                else:
                    state = next_state

            env.close()
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            all_passed = False
    else:
        print("  SKIPPED (gymnasium not installed)")

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 60)

    return all_passed


# ============================================================================
#                           Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Popular RL Algorithms: DDPG, TD3, SAC"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["ddpg", "td3", "sac"],
        help="Algorithm to train"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Pendulum-v1",
        help="Gymnasium environment"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit tests"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all algorithms"
    )

    args = parser.parse_args()

    if args.test:
        run_unit_tests()
        return

    if args.compare:
        compare_algorithms(args.env, args.timesteps, args.seed)
        return

    if not HAS_GYM:
        print("Error: gymnasium not installed")
        return

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()

    if args.algo == "ddpg":
        config = DDPGConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
        )
        agent = DDPGAgent(config)
    elif args.algo == "td3":
        config = TD3Config(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
        )
        agent = TD3Agent(config)
    else:
        config = SACConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
        )
        agent = SACAgent(config)

    train_continuous_agent(
        agent,
        args.env,
        args.timesteps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
