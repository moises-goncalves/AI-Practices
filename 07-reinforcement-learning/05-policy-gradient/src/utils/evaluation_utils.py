"""Evaluation utility functions for policy gradient algorithms."""

from typing import Tuple, List, Dict
import numpy as np
import torch
from ..core.trajectory import Trajectory


def evaluate_policy(
    agent,
    env,
    num_episodes: int = 10,
    max_steps: int = 1000,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate the policy on the environment.

    Args:
        agent: Policy gradient agent
        env: Gym environment
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render the environment

    Returns:
        Dictionary with evaluation metrics
    """
    episode_returns = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0.0
        episode_length = 0

        for step in range(max_steps):
            if render:
                env.render()

            # Select action (deterministic for evaluation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)

            with torch.no_grad():
                if hasattr(agent.policy, 'forward'):
                    output = agent.policy.forward(state_tensor)

                    # Handle different policy types
                    if isinstance(output, tuple):
                        # Gaussian policy returns (mean, log_std)
                        action = output[0].squeeze(0).cpu().numpy()
                    else:
                        # Discrete policy returns logits
                        action = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                else:
                    action, _ = agent.policy.sample(state_tensor)
                    action = action.cpu().numpy().squeeze()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_return += reward
            episode_length += 1
            state = next_state

            if done:
                break

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    return {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "max_return": np.max(episode_returns),
        "min_return": np.min(episode_returns),
        "mean_length": np.mean(episode_lengths),
    }


def collect_trajectories(
    agent,
    env,
    num_trajectories: int = 10,
    max_steps: int = 1000
) -> Tuple[List[Trajectory], List[float]]:
    """
    Collect multiple trajectories from the environment.

    Args:
        agent: Policy gradient agent
        env: Gym environment
        num_trajectories: Number of trajectories to collect
        max_steps: Maximum steps per trajectory

    Returns:
        Tuple of (trajectories, episode_returns)
    """
    trajectories = []
    episode_returns = []

    for _ in range(num_trajectories):
        trajectory, episode_return = agent.collect_trajectory(env, max_steps)
        trajectories.append(trajectory)
        episode_returns.append(episode_return)

    return trajectories, episode_returns


def compute_trajectory_statistics(
    trajectories: List[Trajectory]
) -> Dict[str, float]:
    """
    Compute statistics over a collection of trajectories.

    Args:
        trajectories: List of trajectory objects

    Returns:
        Dictionary with trajectory statistics
    """
    lengths = [len(traj) for traj in trajectories]
    total_rewards = [sum(traj.rewards) for traj in trajectories]

    return {
        "num_trajectories": len(trajectories),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "max_length": np.max(lengths),
        "min_length": np.min(lengths),
        "mean_return": np.mean(total_rewards),
        "std_return": np.std(total_rewards),
        "max_return": np.max(total_rewards),
        "min_return": np.min(total_rewards),
    }
