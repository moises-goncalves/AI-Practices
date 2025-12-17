"""
Deep Reinforcement Learning Core Module

Production-grade implementations of foundational components for deep RL algorithms.
Provides reusable building blocks including neural network architectures, experience
replay buffers, and common utilities.

Module Structure:
    networks.py     - Neural network architectures (DQN, Dueling, Actor-Critic)
    buffers.py      - Experience replay mechanisms (Uniform, Prioritized)
    utils.py        - Common utilities (initialization, device management)

Mathematical Foundations:
    Deep RL extends tabular methods to high-dimensional spaces through function
    approximation with neural networks. Key insight: Q*(s,a) or π*(a|s) can be
    represented as parameterized functions f_θ(s) with learnable parameters θ.

    Convergence requires:
    1. Function approximator with universal approximation property
    2. Appropriate exploration strategy (ε-greedy, entropy bonus)
    3. Stabilization techniques (target networks, gradient clipping)
    4. Sample decorrelation (experience replay)

References:
    [1] Mnih et al., "Human-level control through deep RL", Nature 2015
    [2] Schulman et al., "Proximal Policy Optimization Algorithms", 2017
    [3] Haarnoja et al., "Soft Actor-Critic", ICML 2018
"""

from .networks import (
    DQNNetwork,
    DuelingDQNNetwork,
    ActorCriticNetwork,
    init_weights
)
from .buffers import (
    Transition,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    RolloutBuffer
)
from .utils import (
    get_device,
    set_seed,
    linear_schedule
)

__all__ = [
    # Networks
    'DQNNetwork',
    'DuelingDQNNetwork',
    'ActorCriticNetwork',
    'init_weights',
    # Buffers
    'Transition',
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'RolloutBuffer',
    # Utilities
    'get_device',
    'set_seed',
    'linear_schedule',
]

__version__ = '2.0.0'
