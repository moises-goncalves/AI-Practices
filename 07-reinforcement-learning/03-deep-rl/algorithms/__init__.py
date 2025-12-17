"""
Deep RL Algorithms Module

Production-grade implementations of deep reinforcement learning algorithms.
All implementations feature comprehensive documentation, type hints, and
unit tests suitable for academic publication and industrial deployment.

Algorithms:
    DQN     - Deep Q-Network with variants (Double, Dueling, PER)
    A2C     - Advantage Actor-Critic
    PPO     - Proximal Policy Optimization

Each algorithm provides:
    - Agent class with train/evaluate interface
    - Configuration dataclass with validation
    - Training/evaluation functions
    - Comprehensive unit tests
"""

from .dqn import (
    DQNConfig,
    DQNAgent,
    train_dqn,
    evaluate_agent,
)

from .policy_gradient import (
    A2CConfig,
    A2CAgent,
    PPOConfig,
    PPOAgent,
    train_a2c,
    train_ppo,
)

__all__ = [
    # DQN
    'DQNConfig',
    'DQNAgent',
    'train_dqn',
    'evaluate_agent',
    # Policy Gradient
    'A2CConfig',
    'A2CAgent',
    'PPOConfig',
    'PPOAgent',
    'train_a2c',
    'train_ppo',
]
