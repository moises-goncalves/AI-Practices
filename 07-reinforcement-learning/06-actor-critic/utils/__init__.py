"""
Utilities Module - Training and Visualization Tools.

This module provides essential utilities for training and analyzing
reinforcement learning agents.

Submodules
----------
training : Training utilities
    - set_seed: Reproducible experiments
    - RunningMeanStd: Online normalization
    - LearningRateScheduler: LR decay strategies
    - MetricsTracker: Training progress monitoring
    - Checkpointer: Model persistence

visualization : Plotting utilities
    - plot_training_curves: Training metrics visualization
    - plot_comparison: Algorithm comparison
    - plot_policy_distribution: Action probabilities
    - plot_value_function_1d/2d: Value function visualization
    - create_training_dashboard: Comprehensive overview
    - LivePlotter: Real-time Jupyter visualization
"""

from utils.training import (
    set_seed,
    RunningMeanStd,
    LearningRateScheduler,
    MetricsTracker,
    Checkpointer,
    compute_explained_variance,
    polyak_update,
)

from utils.visualization import (
    smooth_data,
    plot_training_curves,
    plot_comparison,
    plot_policy_distribution,
    plot_value_function_1d,
    plot_value_function_2d,
    create_training_dashboard,
    LivePlotter,
)

__all__ = [
    # Training
    "set_seed",
    "RunningMeanStd",
    "LearningRateScheduler",
    "MetricsTracker",
    "Checkpointer",
    "compute_explained_variance",
    "polyak_update",
    # Visualization
    "smooth_data",
    "plot_training_curves",
    "plot_comparison",
    "plot_policy_distribution",
    "plot_value_function_1d",
    "plot_value_function_2d",
    "create_training_dashboard",
    "LivePlotter",
]
