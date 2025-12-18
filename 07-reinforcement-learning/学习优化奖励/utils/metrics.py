"""
Metrics for Reward Optimization Evaluation

================================================================================
CORE IDEA
================================================================================
Provides standardized metrics for evaluating reward learning algorithms:
- Sample efficiency
- Final performance
- Learning stability
- Goal achievement rates
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_episode_statistics(
    episode_rewards: List[float],
    episode_lengths: List[int],
    window_size: int = 100,
) -> Dict[str, float]:
    """Compute comprehensive episode statistics.

    Args:
        episode_rewards: List of episode total rewards.
        episode_lengths: List of episode lengths.
        window_size: Window for computing recent statistics.

    Returns:
        Dictionary of statistics.
    """
    if not episode_rewards:
        return {}

    n = len(episode_rewards)
    window = min(window_size, n)

    recent_rewards = episode_rewards[-window:]
    recent_lengths = episode_lengths[-window:] if episode_lengths else []

    return {
        "mean_reward": float(np.mean(recent_rewards)),
        "std_reward": float(np.std(recent_rewards)),
        "max_reward": float(np.max(recent_rewards)),
        "min_reward": float(np.min(recent_rewards)),
        "median_reward": float(np.median(recent_rewards)),
        "mean_length": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
        "total_episodes": n,
        "total_steps": sum(episode_lengths) if episode_lengths else 0,
    }


def compute_success_rate(
    successes: List[bool],
    window_size: int = 100,
) -> Dict[str, float]:
    """Compute success rate metrics.

    Args:
        successes: List of success flags per episode.
        window_size: Window for recent statistics.

    Returns:
        Success rate statistics.
    """
    if not successes:
        return {"success_rate": 0.0, "recent_success_rate": 0.0}

    window = min(window_size, len(successes))
    recent = successes[-window:]

    return {
        "success_rate": float(np.mean(successes)),
        "recent_success_rate": float(np.mean(recent)),
        "total_successes": int(sum(successes)),
        "total_episodes": len(successes),
    }


def compute_reward_statistics(
    original_rewards: np.ndarray,
    shaped_rewards: np.ndarray,
    intrinsic_rewards: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute statistics comparing reward signals.

    Args:
        original_rewards: Original environment rewards.
        shaped_rewards: Shaped/augmented rewards.
        intrinsic_rewards: Optional intrinsic motivation rewards.

    Returns:
        Comparative reward statistics.
    """
    stats = {
        "original_mean": float(np.mean(original_rewards)),
        "original_std": float(np.std(original_rewards)),
        "shaped_mean": float(np.mean(shaped_rewards)),
        "shaped_std": float(np.std(shaped_rewards)),
        "shaping_contribution": float(np.mean(shaped_rewards - original_rewards)),
    }

    if np.std(original_rewards) > 1e-8 and np.std(shaped_rewards) > 1e-8:
        stats["reward_correlation"] = float(
            np.corrcoef(original_rewards.flatten(), shaped_rewards.flatten())[0, 1]
        )

    if intrinsic_rewards is not None:
        stats["intrinsic_mean"] = float(np.mean(intrinsic_rewards))
        stats["intrinsic_std"] = float(np.std(intrinsic_rewards))
        stats["intrinsic_fraction"] = float(
            np.mean(np.abs(intrinsic_rewards))
            / (np.mean(np.abs(shaped_rewards)) + 1e-8)
        )

    return stats


def compute_exploration_metrics(
    states_visited: np.ndarray,
    state_bounds: Tuple[np.ndarray, np.ndarray],
    n_bins: int = 20,
) -> Dict[str, float]:
    """Compute exploration coverage metrics.

    Args:
        states_visited: Array of visited states.
        state_bounds: (lower, upper) bounds of state space.
        n_bins: Discretization granularity.

    Returns:
        Exploration coverage statistics.
    """
    states = np.atleast_2d(states_visited)
    lower, upper = state_bounds

    normalized = (states - lower) / (upper - lower + 1e-8)
    normalized = np.clip(normalized, 0, 1 - 1e-8)
    discretized = np.floor(normalized * n_bins).astype(int)

    unique_bins = set(tuple(row) for row in discretized)
    total_bins = n_bins ** states.shape[1]

    bin_counts: Dict[Tuple, int] = {}
    for row in discretized:
        key = tuple(row)
        bin_counts[key] = bin_counts.get(key, 0) + 1

    counts = np.array(list(bin_counts.values()))
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0

    return {
        "coverage_ratio": len(unique_bins) / total_bins,
        "unique_states_visited": len(unique_bins),
        "total_possible_states": total_bins,
        "visitation_entropy": float(entropy),
        "uniformity": float(entropy / max_entropy) if max_entropy > 0 else 1.0,
        "total_visits": len(states),
    }
