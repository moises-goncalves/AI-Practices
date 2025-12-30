"""
Visualization Utilities for Reward Optimization

================================================================================
CORE IDEA
================================================================================
Provides consistent, publication-quality visualizations for:
- Learning curves and training progress
- Reward distributions and shaping analysis
- Value functions and potential fields
- Agent trajectories and exploration
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def plot_learning_curves(
    data: Dict[str, List[float]],
    window_size: int = 50,
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (12, 4),
):
    """Plot smoothed learning curves.

    Args:
        data: Dictionary mapping metric names to value lists.
        window_size: Smoothing window size.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib figure and axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return None, None

    n_metrics = len(data)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, data.items()):
        values = np.array(values)

        ax.plot(values, alpha=0.3, label="Raw")

        if len(values) >= window_size:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(values, kernel, mode="valid")
            ax.plot(
                range(window_size - 1, len(values)),
                smoothed,
                label=f"Smoothed (w={window_size})",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    return fig, axes


def plot_reward_distribution(
    original: np.ndarray,
    shaped: np.ndarray,
    intrinsic: Optional[np.ndarray] = None,
    title: str = "Reward Distribution",
    figsize: Tuple[int, int] = (12, 4),
):
    """Plot reward distributions for comparison.

    Args:
        original: Original environment rewards.
        shaped: Shaped rewards.
        intrinsic: Optional intrinsic rewards.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib figure and axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return None, None

    n_plots = 2 if intrinsic is None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    axes[0].hist(original, bins=50, alpha=0.7, label="Original")
    axes[0].hist(shaped, bins=50, alpha=0.7, label="Shaped")
    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Reward Distributions")
    axes[0].legend()

    shaping_bonus = shaped - original
    axes[1].hist(shaping_bonus, bins=50, alpha=0.7, color="green")
    axes[1].axvline(x=0, color="red", linestyle="--")
    axes[1].set_xlabel("Shaping Bonus")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Shaping Bonus Distribution")

    if intrinsic is not None:
        axes[2].hist(intrinsic, bins=50, alpha=0.7, color="purple")
        axes[2].set_xlabel("Intrinsic Reward")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Intrinsic Reward Distribution")

    fig.suptitle(title)
    plt.tight_layout()

    return fig, axes


def plot_value_heatmap(
    value_fn: callable,
    bounds: Tuple[np.ndarray, np.ndarray],
    resolution: int = 50,
    title: str = "Value Function",
    figsize: Tuple[int, int] = (8, 6),
):
    """Plot 2D value function heatmap.

    Args:
        value_fn: Function mapping (x, y) states to values.
        bounds: ((x_min, y_min), (x_max, y_max)) bounds.
        resolution: Grid resolution.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib figure and axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return None, None

    lower, upper = bounds

    x = np.linspace(lower[0], upper[0], resolution)
    y = np.linspace(lower[1], upper[1], resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j]])
            Z[i, j] = value_fn(state)

    fig, ax = plt.subplots(figsize=figsize)

    contour = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(contour, ax=ax, label="Value")

    ax.set_xlabel("State Dimension 1")
    ax.set_ylabel("State Dimension 2")
    ax.set_title(title)

    return fig, ax


def plot_trajectory(
    trajectory: np.ndarray,
    goal: Optional[np.ndarray] = None,
    obstacles: Optional[List[np.ndarray]] = None,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Agent Trajectory",
    figsize: Tuple[int, int] = (8, 8),
):
    """Plot 2D agent trajectory.

    Args:
        trajectory: Array of (x, y) positions, shape (T, 2).
        goal: Optional goal position.
        obstacles: Optional list of obstacle positions.
        bounds: Optional plot bounds.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib figure and axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for visualization")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    trajectory = np.atleast_2d(trajectory)

    ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2, alpha=0.7)

    ax.plot(
        trajectory[0, 0],
        trajectory[0, 1],
        "go",
        markersize=15,
        label="Start",
    )

    ax.plot(
        trajectory[-1, 0],
        trajectory[-1, 1],
        "bs",
        markersize=12,
        label="End",
    )

    if goal is not None:
        ax.plot(goal[0], goal[1], "r*", markersize=20, label="Goal")

    if obstacles is not None:
        for obs in obstacles:
            ax.add_patch(
                plt.Circle(obs[:2], obs[2] if len(obs) > 2 else 0.5, color="gray", alpha=0.5)
            )

    if bounds is not None:
        lower, upper = bounds
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    return fig, ax
