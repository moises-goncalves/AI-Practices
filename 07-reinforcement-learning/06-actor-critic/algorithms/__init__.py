"""
Algorithms Module - Policy Gradient Implementations.

This module provides production-ready implementations of classic and modern
policy gradient algorithms for reinforcement learning.

Algorithms
----------
REINFORCE : Monte Carlo policy gradient
    - Simplest policy gradient algorithm
    - Uses complete episode returns
    - Optional learned baseline for variance reduction

A2C : Advantage Actor-Critic
    - Synchronous actor-critic with bootstrapping
    - GAE for advantage estimation
    - Single update per rollout

PPO : Proximal Policy Optimization
    - State-of-the-art on-policy algorithm
    - Clipped surrogate objective for stability
    - Multiple epochs per rollout for efficiency

Usage
-----
>>> from algorithms import REINFORCE, A2C, PPO
>>> from core.config import TrainingConfig
>>> import gymnasium as gym
>>>
>>> config = TrainingConfig(env_name="CartPole-v1")
>>> env = gym.make(config.env_name)
>>>
>>> # Choose algorithm
>>> agent = PPO(config, env)
>>> metrics = agent.train()
"""

from algorithms.reinforce import REINFORCE
from algorithms.actor_critic import A2C
from algorithms.ppo import PPO

__all__ = [
    "REINFORCE",
    "A2C",
    "PPO",
]
