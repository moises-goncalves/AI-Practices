"""
Utility Functions for Training and Evaluation

This module provides helper functions for training RL agents, including
environment wrappers, trajectory collection, and performance metrics.
"""

import numpy as np
from typing import Tuple, Dict, List, Callable
import gymnasium as gym
from gymnasium import spaces


class GymEnvironmentWrapper:
    """
    Wrapper for OpenAI Gym environments.

    Provides a consistent interface for interacting with various environments
    and handles state/action space conversions.
    """

    def __init__(self, env_name: str):
        """
        Initialize environment wrapper.

        Args:
            env_name: Name of the Gym environment (e.g., 'CartPole-v1')
        """
        self.env = gym.make(env_name)
        self.state_dim = self._get_state_dim()
        self.action_dim = self._get_action_dim()
        self.action_type = self._get_action_type()

    def _get_state_dim(self) -> int:
        """Get dimensionality of state space."""
        if isinstance(self.env.observation_space, spaces.Box):
            return int(np.prod(self.env.observation_space.shape))
        elif isinstance(self.env.observation_space, spaces.Discrete):
            return self.env.observation_space.n
        else:
            raise ValueError(f"Unsupported observation space: {self.env.observation_space}")

    def _get_action_dim(self) -> int:
        """Get dimensionality of action space."""
        if isinstance(self.env.action_space, spaces.Box):
            return int(np.prod(self.env.action_space.shape))
        elif isinstance(self.env.action_space, spaces.Discrete):
            return self.env.action_space.n
        else:
            raise ValueError(f"Unsupported action space: {self.env.action_space}")

    def _get_action_type(self) -> str:
        """Determine if action space is continuous or discrete."""
        if isinstance(self.env.action_space, spaces.Box):
            return 'continuous'
        elif isinstance(self.env.action_space, spaces.Discrete):
            return 'discrete'
        else:
            return 'unknown'

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        return self._process_state(state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to execute

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.action_type == 'discrete':
            action = int(action)

        result = self.env.step(action)
        if len(result) == 4:
            next_state, reward, done, info = result
            truncated = False
        else:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated

        return self._process_state(next_state), reward, done, info

    def _process_state(self, state: np.ndarray) -> np.ndarray:
        """Process state to ensure it's a 1D array."""
        if isinstance(state, (int, float)):
            return np.array([state], dtype=np.float32)
        state = np.array(state, dtype=np.float32)
        return state.flatten()

    def render(self) -> None:
        """Render the environment."""
        self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()


def collect_trajectory(
    agent,
    env: GymEnvironmentWrapper,
    max_steps: int = 1000,
    deterministic: bool = False
) -> Dict[str, np.ndarray]:
    """
    Collect a single trajectory from the environment.

    Args:
        agent: RL agent with select_action method
        env: Environment wrapper
        max_steps: Maximum steps per episode
        deterministic: If True, use deterministic policy

    Returns:
        Dictionary with trajectory data
    """
    state = env.reset()
    trajectory = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': [],
        'episode_return': 0,
        'episode_length': 0
    }

    for step in range(max_steps):
        action = agent.select_action(state)
        if isinstance(action, tuple):
            action = action[0]

        next_state, reward, done, _ = env.step(action)

        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)
        trajectory['next_states'].append(next_state)
        trajectory['dones'].append(done)
        trajectory['episode_return'] += reward
        trajectory['episode_length'] += 1

        state = next_state

        if done:
            break

    # Convert lists to arrays
    for key in ['states', 'actions', 'rewards', 'next_states', 'dones']:
        trajectory[key] = np.array(trajectory[key])

    return trajectory


def evaluate_agent(
    agent,
    env: GymEnvironmentWrapper,
    num_episodes: int = 10,
    max_steps: int = 1000
) -> Dict[str, float]:
    """
    Evaluate agent performance over multiple episodes.

    Args:
        agent: RL agent to evaluate
        env: Environment wrapper
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode

    Returns:
        Dictionary with evaluation metrics
    """
    returns = []
    lengths = []

    for _ in range(num_episodes):
        trajectory = collect_trajectory(agent, env, max_steps)
        returns.append(trajectory['episode_return'])
        lengths.append(trajectory['episode_length'])

    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'max_return': np.max(returns),
        'min_return': np.min(returns),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths)
    }


def train_agent(
    agent,
    env: GymEnvironmentWrapper,
    num_episodes: int = 100,
    max_steps: int = 1000,
    eval_interval: int = 10,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Train an RL agent on an environment.

    Args:
        agent: RL agent to train
        env: Environment wrapper
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        eval_interval: Evaluation frequency
        verbose: Print training progress

    Returns:
        Dictionary with training history
    """
    history = {
        'episode_returns': [],
        'episode_lengths': [],
        'actor_losses': [],
        'critic_losses': [],
        'eval_returns': []
    }

    for episode in range(num_episodes):
        # Collect trajectory
        trajectory = collect_trajectory(agent, env, max_steps)

        # Update agent
        if hasattr(agent.buffer, 'compute_advantages'):
            # PPO-style update
            agent.buffer.add(
                trajectory['states'][0],
                trajectory['actions'][0],
                trajectory['rewards'][0],
                trajectory['next_states'][0],
                trajectory['dones'][0]
            )
            # For simplicity, just collect one step
            batch = trajectory
        else:
            # Actor-Critic style update
            batch = {
                'states': trajectory['states'],
                'actions': trajectory['actions'],
                'rewards': trajectory['rewards'],
                'next_states': trajectory['next_states'],
                'dones': trajectory['dones']
            }

        losses = agent.update(batch)

        history['episode_returns'].append(trajectory['episode_return'])
        history['episode_lengths'].append(trajectory['episode_length'])
        history['actor_losses'].append(losses.get('actor_loss', 0))
        history['critic_losses'].append(losses.get('critic_loss', 0))

        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_metrics = evaluate_agent(agent, env, num_episodes=5)
            history['eval_returns'].append(eval_metrics['mean_return'])

            if verbose:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Return: {trajectory['episode_return']:.2f}")
                print(f"  Eval Return: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f}")
                print(f"  Actor Loss: {losses.get('actor_loss', 0):.4f}")
                print(f"  Critic Loss: {losses.get('critic_loss', 0):.4f}")

    return history


def compute_returns(
    rewards: np.ndarray,
    gamma: float = 0.99,
    normalize: bool = False
) -> np.ndarray:
    """
    Compute discounted cumulative returns.

    Args:
        rewards: Array of rewards
        gamma: Discount factor
        normalize: If True, normalize returns to zero mean and unit variance

    Returns:
        Array of discounted returns
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    cumulative_return = 0

    for t in reversed(range(len(rewards))):
        cumulative_return = rewards[t] + gamma * cumulative_return
        returns[t] = cumulative_return

    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def compute_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> np.ndarray:
    """
    Compute advantages using Generalized Advantage Estimation (GAE).

    Args:
        rewards: Array of rewards
        values: Array of value estimates
        gamma: Discount factor
        gae_lambda: GAE parameter

    Returns:
        Array of advantage estimates
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    return advantages
