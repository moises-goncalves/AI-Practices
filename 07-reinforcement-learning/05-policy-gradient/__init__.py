"""
Policy Gradient Methods Module

This module implements state-of-the-art policy gradient algorithms for reinforcement learning,
including REINFORCE, Actor-Critic, and advanced variants. All implementations follow research
standards suitable for top-tier conference publications and production deployment.

Core Components:
    - core: Base classes and interfaces for policy gradient algorithms
    - algorithms: Concrete implementations (REINFORCE, A2C, A3C, PPO, TRPO)
    - networks: Neural network architectures for policy and value functions
    - buffers: Experience replay and trajectory buffers
    - utils: Utility functions for training and evaluation
    - tests: Unit tests and validation scripts
    - notebooks: Interactive tutorials and demonstrations
    - docs: Comprehensive documentation and knowledge summaries
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"

from .core.base import PolicyGradientAgent, BasePolicy, BaseValueFunction
from .algorithms.reinforce import REINFORCE
from .algorithms.actor_critic import ActorCritic, A2C

__all__ = [
    "PolicyGradientAgent",
    "BasePolicy",
    "BaseValueFunction",
    "REINFORCE",
    "ActorCritic",
    "A2C",
]
