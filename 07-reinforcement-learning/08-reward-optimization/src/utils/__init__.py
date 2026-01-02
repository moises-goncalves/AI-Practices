"""
Utility Functions for Reward Optimization

================================================================================
OVERVIEW
================================================================================
This module provides common utilities used across reward optimization algorithms:

- Metrics computation and logging
- Visualization helpers
- Configuration management
- Common mathematical operations

================================================================================
CONTENTS
================================================================================
- metrics: Performance metrics for reward learning
- visualization: Plotting utilities for analysis
- config: Configuration dataclasses and validation
"""

from .metrics import (
    compute_episode_statistics,
    compute_success_rate,
    compute_reward_statistics,
    compute_exploration_metrics,
)

from .visualization import (
    plot_learning_curves,
    plot_reward_distribution,
    plot_value_heatmap,
    plot_trajectory,
)

__all__ = [
    # Metrics
    "compute_episode_statistics",
    "compute_success_rate",
    "compute_reward_statistics",
    "compute_exploration_metrics",
    # Visualization
    "plot_learning_curves",
    "plot_reward_distribution",
    "plot_value_heatmap",
    "plot_trajectory",
]
