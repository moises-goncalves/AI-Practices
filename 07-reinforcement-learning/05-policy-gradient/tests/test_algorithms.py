"""Unit tests for policy gradient algorithms."""

import numpy as np
import torch
import gymnasium as gym

from ..core.base import PolicyGradientAgent
from ..algorithms.reinforce import REINFORCE
from ..algorithms.actor_critic import ActorCritic, A2C
from ..networks.policy_networks import DiscretePolicy, ContinuousPolicy
from ..networks.value_networks import ValueNetwork


def test_reinforce():
    """Test REINFORCE algorithm on CartPole environment."""
    print("Testing REINFORCE algorithm...")

    # Create environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create networks
    policy = DiscretePolicy(state_dim, action_dim, hidden_dims=(32, 32))
    value_fn = ValueNetwork(state_dim, hidden_dims=(32, 32))

    # Create agent
    agent = REINFORCE(
        policy=policy,
        value_function=value_fn,
        learning_rate=1e-3,
        gamma=0.99,
        entropy_coeff=0.01,
        device="cpu"
    )

    # Train for a few episodes
    print("  Training for 5 episodes...")
    history = agent.train(env, num_episodes=5, max_steps=500, eval_interval=5)

    # Verify training
    assert len(history["episode_returns"]) == 5
    assert len(history["policy_losses"]) == 5
    assert all(isinstance(x, float) for x in history["episode_returns"])

    print("  ✓ REINFORCE test passed")
    env.close()


def test_actor_critic():
    """Test Actor-Critic algorithm on CartPole environment."""
    print("Testing Actor-Critic algorithm...")

    # Create environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create networks
    policy = DiscretePolicy(state_dim, action_dim, hidden_dims=(32, 32))
    value_fn = ValueNetwork(state_dim, hidden_dims=(32, 32))

    # Create agent
    agent = ActorCritic(
        policy=policy,
        value_function=value_fn,
        learning_rate=1e-3,
        gamma=0.99,
        entropy_coeff=0.01,
        device="cpu"
    )

    # Train for a few episodes
    print("  Training for 5 episodes...")
    history = agent.train(env, num_episodes=5, max_steps=500, eval_interval=5)

    # Verify training
    assert len(history["episode_returns"]) == 5
    assert len(history["policy_losses"]) == 5
    assert len(history["value_losses"]) == 5

    print("  ✓ Actor-Critic test passed")
    env.close()


def test_a2c():
    """Test A2C algorithm on CartPole environment."""
    print("Testing A2C algorithm...")

    # Create environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create networks
    policy = DiscretePolicy(state_dim, action_dim, hidden_dims=(32, 32))
    value_fn = ValueNetwork(state_dim, hidden_dims=(32, 32))

    # Create agent
    agent = A2C(
        policy=policy,
        value_function=value_fn,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coeff=0.01,
        device="cpu"
    )

    # Train for a few episodes
    print("  Training for 5 episodes...")
    history = agent.train(env, num_episodes=5, max_steps=500, eval_interval=5)

    # Verify training
    assert len(history["episode_returns"]) == 5
    assert len(history["policy_losses"]) == 5

    print("  ✓ A2C test passed")
    env.close()


if __name__ == "__main__":
    print("Running policy gradient algorithm tests...\n")
    test_reinforce()
    test_actor_critic()
    test_a2c()
    print("\n✓ All tests passed!")
