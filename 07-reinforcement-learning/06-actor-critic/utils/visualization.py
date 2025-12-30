"""
Visualization Utilities for Policy Gradient Training.

================================================================================
核心思想 (Core Idea)
================================================================================
Effective visualization is crucial for understanding and debugging RL training.
This module provides:

1. **Training Curves**: Episode rewards, losses, and metrics over time
2. **Policy Visualization**: Action distributions and decision boundaries
3. **Value Function**: Heatmaps and surface plots
4. **Comparison Plots**: Multiple algorithms or hyperparameters

================================================================================
数学原理 (Mathematical Theory)
================================================================================
Smoothing for Visualization:

    Exponential Moving Average (EMA):
        y_smooth[t] = α × y[t] + (1-α) × y_smooth[t-1]

    Gaussian Smoothing:
        y_smooth = y * G(σ)  (convolution with Gaussian kernel)

    Rolling Mean:
        y_smooth[t] = mean(y[t-w:t+w])

Confidence Intervals:
    For n runs with mean μ and std σ:
        CI = μ ± z × σ/√n

    Where z = 1.96 for 95% confidence.

================================================================================
问题背景 (Problem Statement)
================================================================================
RL Training Visualization Challenges:
    1. High variance in episode rewards
    2. Multiple metrics to track simultaneously
    3. Comparing different algorithms/seeds
    4. Understanding policy behavior

Solutions:
    1. Smoothing and confidence bands
    2. Multi-panel figures with shared axes
    3. Aggregation across seeds with uncertainty
    4. Policy/value function visualization

================================================================================
算法总结 (Summary)
================================================================================
This module provides publication-ready visualization:

1. **plot_training_curves()**: Main training metrics
2. **plot_comparison()**: Multiple runs/algorithms
3. **plot_policy_distribution()**: Action probabilities
4. **plot_value_function()**: Value estimates
5. **create_training_dashboard()**: Comprehensive overview
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from IPython.display import display, clear_output
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def smooth_data(
    data: np.ndarray,
    window: int = 10,
    mode: str = "ema",
) -> np.ndarray:
    """
    Smooth noisy data for visualization.

    Parameters
    ----------
    data : np.ndarray
        Raw data to smooth.
    window : int, default=10
        Smoothing window size or EMA span.
    mode : str, default="ema"
        Smoothing mode: "ema", "rolling", "gaussian".

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data.
    """
    if len(data) == 0:
        return data

    data = np.asarray(data, dtype=np.float64)

    if mode == "ema":
        alpha = 2 / (window + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
        return smoothed

    elif mode == "rolling":
        kernel = np.ones(window) / window
        # Pad to handle edges
        padded = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    elif mode == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma=window / 3)

    else:
        raise ValueError(f"Unknown smoothing mode: {mode}")


def plot_training_curves(
    metrics: Dict[str, List[float]],
    title: str = "Training Progress",
    smoothing: int = 10,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot training curves with smoothing.

    核心思想 (Core Idea):
        Visualize key training metrics over time with smoothed curves
        and raw data in background for context.

    Parameters
    ----------
    metrics : Dict[str, List[float]]
        Dictionary of metric names to values.
        Expected keys: "episode_rewards", "policy_losses", "value_losses", "entropies"
    title : str
        Figure title.
    smoothing : int
        Smoothing window size.
    figsize : Tuple[int, int]
        Figure size in inches.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : Figure or None
        Matplotlib figure if available.

    Examples
    --------
    >>> metrics = {
    ...     "episode_rewards": rewards_list,
    ...     "policy_losses": policy_losses,
    ...     "value_losses": value_losses,
    ...     "entropies": entropies,
    ... }
    >>> plot_training_curves(metrics, title="PPO on CartPole")
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return None

    # Determine subplot layout
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    colors = plt.cm.tab10.colors

    for idx, (name, values) in enumerate(metrics.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        values = np.array(values)

        if len(values) == 0:
            continue

        x = np.arange(len(values))

        # Plot raw data with transparency
        ax.plot(x, values, alpha=0.3, color=colors[idx % len(colors)], linewidth=0.5)

        # Plot smoothed data
        if len(values) > smoothing:
            smoothed = smooth_data(values, window=smoothing)
            ax.plot(x, smoothed, color=colors[idx % len(colors)], linewidth=2, label=name)

        ax.set_xlabel("Episode" if "reward" in name.lower() else "Update")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_comparison(
    results: Dict[str, Dict[str, List[float]]],
    metric: str = "episode_rewards",
    title: str = "Algorithm Comparison",
    smoothing: int = 20,
    show_std: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Compare multiple algorithms or runs.

    核心思想 (Core Idea):
        Overlay training curves from different algorithms/seeds with
        confidence bands to show statistical significance.

    Parameters
    ----------
    results : Dict[str, Dict[str, List[float]]]
        Nested dict: {algorithm_name: {metric_name: values}}.
    metric : str
        Metric to compare.
    title : str
        Figure title.
    smoothing : int
        Smoothing window.
    show_std : bool
        Whether to show standard deviation bands.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : Figure or None

    Examples
    --------
    >>> results = {
    ...     "REINFORCE": {"episode_rewards": [...]},
    ...     "A2C": {"episode_rewards": [...]},
    ...     "PPO": {"episode_rewards": [...]},
    ... }
    >>> plot_comparison(results, metric="episode_rewards")
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors

    for idx, (name, metrics) in enumerate(results.items()):
        if metric not in metrics:
            continue

        values = np.array(metrics[metric])
        x = np.arange(len(values))

        # Smooth
        if len(values) > smoothing:
            smoothed = smooth_data(values, window=smoothing)
        else:
            smoothed = values

        color = colors[idx % len(colors)]
        ax.plot(x, smoothed, color=color, linewidth=2, label=name)

        # Show raw data with transparency
        ax.fill_between(
            x,
            smooth_data(values, window=smoothing * 2, mode="rolling") - np.std(values) * 0.5,
            smooth_data(values, window=smoothing * 2, mode="rolling") + np.std(values) * 0.5,
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_policy_distribution(
    policy_net,
    state_samples: np.ndarray,
    action_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Visualize policy action distribution for sample states.

    核心思想 (Core Idea):
        Show how the policy distributes probability mass across actions
        for different states, revealing learned preferences.

    Parameters
    ----------
    policy_net : nn.Module
        Policy network with get_action_probs method.
    state_samples : np.ndarray
        Sample states to visualize, shape (n_samples, state_dim).
    action_names : List[str], optional
        Names for actions.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    import torch

    n_samples = len(state_samples)
    state_tensor = torch.tensor(state_samples, dtype=torch.float32)

    with torch.no_grad():
        if hasattr(policy_net, "get_action_probs"):
            probs = policy_net.get_action_probs(state_tensor).numpy()
        else:
            logits = policy_net(state_tensor)
            probs = torch.softmax(logits, dim=-1).numpy()

    n_actions = probs.shape[1]

    if action_names is None:
        action_names = [f"Action {i}" for i in range(n_actions)]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_samples)
    width = 0.8 / n_actions

    for i in range(n_actions):
        offset = (i - n_actions / 2 + 0.5) * width
        ax.bar(x + offset, probs[:, i], width, label=action_names[i])

    ax.set_xlabel("State Sample")
    ax.set_ylabel("Action Probability")
    ax.set_title("Policy Action Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in range(n_samples)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_value_function_1d(
    value_net,
    state_range: Tuple[float, float],
    state_dim: int,
    vary_dim: int = 0,
    n_points: int = 100,
    fixed_values: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot value function along one state dimension.

    Parameters
    ----------
    value_net : nn.Module
        Value network.
    state_range : Tuple[float, float]
        Range of state values to plot.
    state_dim : int
        Total state dimension.
    vary_dim : int
        Which dimension to vary.
    n_points : int
        Number of points to evaluate.
    fixed_values : np.ndarray, optional
        Fixed values for other dimensions.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    import torch

    if fixed_values is None:
        fixed_values = np.zeros(state_dim)

    x = np.linspace(state_range[0], state_range[1], n_points)
    states = np.tile(fixed_values, (n_points, 1))
    states[:, vary_dim] = x

    state_tensor = torch.tensor(states, dtype=torch.float32)

    with torch.no_grad():
        values = value_net(state_tensor).squeeze().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, values, linewidth=2)
    ax.set_xlabel(f"State Dimension {vary_dim}")
    ax.set_ylabel("Value V(s)")
    ax.set_title("Value Function")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_value_function_2d(
    value_net,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    state_dim: int,
    x_dim: int = 0,
    y_dim: int = 1,
    n_points: int = 50,
    fixed_values: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot 2D heatmap of value function.

    Parameters
    ----------
    value_net : nn.Module
        Value network.
    x_range, y_range : Tuple[float, float]
        Ranges for x and y axes.
    state_dim : int
        Total state dimension.
    x_dim, y_dim : int
        Which dimensions to vary.
    n_points : int
        Grid resolution.
    fixed_values : np.ndarray, optional
        Fixed values for other dimensions.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    import torch

    if fixed_values is None:
        fixed_values = np.zeros(state_dim)

    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)

    states = np.tile(fixed_values, (n_points * n_points, 1))
    states[:, x_dim] = X.flatten()
    states[:, y_dim] = Y.flatten()

    state_tensor = torch.tensor(states, dtype=torch.float32)

    with torch.no_grad():
        values = value_net(state_tensor).squeeze().numpy()

    Z = values.reshape(n_points, n_points)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Value V(s)")

    ax.set_xlabel(f"State Dimension {x_dim}")
    ax.set_ylabel(f"State Dimension {y_dim}")
    ax.set_title("Value Function Heatmap")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_training_dashboard(
    metrics: Dict[str, List[float]],
    config: Optional[Dict] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Create comprehensive training dashboard.

    核心思想 (Core Idea):
        Single figure showing all important training information:
        rewards, losses, entropy, and configuration.

    Parameters
    ----------
    metrics : Dict[str, List[float]]
        Training metrics.
    config : Dict, optional
        Training configuration to display.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    fig : Figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Episode rewards
    ax1 = fig.add_subplot(gs[0, :])
    if "episode_rewards" in metrics:
        rewards = np.array(metrics["episode_rewards"])
        ax1.plot(rewards, alpha=0.3, color="blue", linewidth=0.5)
        if len(rewards) > 10:
            smoothed = smooth_data(rewards, window=20)
            ax1.plot(smoothed, color="blue", linewidth=2, label="Smoothed")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Episode Rewards")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # Policy loss
    ax2 = fig.add_subplot(gs[1, 0])
    if "policy_losses" in metrics:
        losses = np.array(metrics["policy_losses"])
        ax2.plot(losses, alpha=0.3, color="red", linewidth=0.5)
        if len(losses) > 10:
            ax2.plot(smooth_data(losses, window=10), color="red", linewidth=2)
        ax2.set_xlabel("Update")
        ax2.set_ylabel("Loss")
        ax2.set_title("Policy Loss")
        ax2.grid(True, alpha=0.3)

    # Value loss
    ax3 = fig.add_subplot(gs[1, 1])
    if "value_losses" in metrics:
        losses = np.array(metrics["value_losses"])
        ax3.plot(losses, alpha=0.3, color="green", linewidth=0.5)
        if len(losses) > 10:
            ax3.plot(smooth_data(losses, window=10), color="green", linewidth=2)
        ax3.set_xlabel("Update")
        ax3.set_ylabel("Loss")
        ax3.set_title("Value Loss")
        ax3.grid(True, alpha=0.3)

    # Entropy
    ax4 = fig.add_subplot(gs[2, 0])
    if "entropies" in metrics:
        entropies = np.array(metrics["entropies"])
        ax4.plot(entropies, alpha=0.3, color="purple", linewidth=0.5)
        if len(entropies) > 10:
            ax4.plot(smooth_data(entropies, window=10), color="purple", linewidth=2)
        ax4.set_xlabel("Update")
        ax4.set_ylabel("Entropy")
        ax4.set_title("Policy Entropy")
        ax4.grid(True, alpha=0.3)

    # Config / Stats
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    if config:
        text = "Configuration:\n" + "-" * 30 + "\n"
        for key, value in list(config.items())[:10]:
            text += f"{key}: {value}\n"
    else:
        text = "Training Statistics:\n" + "-" * 30 + "\n"
        if "episode_rewards" in metrics:
            rewards = metrics["episode_rewards"]
            text += f"Episodes: {len(rewards)}\n"
            text += f"Best Reward: {max(rewards):.2f}\n"
            text += f"Final Avg (100): {np.mean(rewards[-100:]):.2f}\n"

    ax5.text(0.1, 0.9, text, transform=ax5.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace")

    fig.suptitle("Training Dashboard", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


class LivePlotter:
    """
    Real-time training visualization for Jupyter notebooks.

    核心思想 (Core Idea):
        Update plots during training without creating new figures,
        providing live feedback on training progress.

    Examples
    --------
    >>> plotter = LivePlotter()
    >>> for episode in range(1000):
    ...     reward = train_episode()
    ...     plotter.update(reward=reward)
    ...     if episode % 10 == 0:
    ...         plotter.render()
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 4)):
        self.figsize = figsize
        self.metrics: Dict[str, List[float]] = {}
        self.fig = None
        self.axes = None

    def update(self, **kwargs) -> None:
        """Add new metric values."""
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def render(self, smoothing: int = 10) -> None:
        """Render current state of metrics."""
        if not HAS_MATPLOTLIB:
            return

        if HAS_IPYTHON:
            clear_output(wait=True)

        n_metrics = len(self.metrics)
        if n_metrics == 0:
            return

        if self.fig is None or len(self.axes) != n_metrics:
            plt.close("all")
            self.fig, self.axes = plt.subplots(1, n_metrics, figsize=self.figsize)
            if n_metrics == 1:
                self.axes = [self.axes]

        for ax, (name, values) in zip(self.axes, self.metrics.items()):
            ax.clear()
            values = np.array(values)
            ax.plot(values, alpha=0.3, linewidth=0.5)
            if len(values) > smoothing:
                ax.plot(smooth_data(values, window=smoothing), linewidth=2)
            ax.set_title(name.replace("_", " ").title())
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if HAS_IPYTHON:
            display(self.fig)
        else:
            plt.pause(0.01)

    def close(self) -> None:
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Visualization Utilities - Unit Tests")
    print("=" * 70)

    # Test smoothing
    print("\n[1] Testing smooth_data...")
    np.random.seed(42)
    noisy = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.3

    smoothed_ema = smooth_data(noisy, window=10, mode="ema")
    smoothed_rolling = smooth_data(noisy, window=10, mode="rolling")

    assert len(smoothed_ema) == len(noisy)
    assert len(smoothed_rolling) == len(noisy)
    assert np.std(smoothed_ema) < np.std(noisy)  # Should be smoother
    print(f"    Original std: {np.std(noisy):.4f}")
    print(f"    Smoothed std: {np.std(smoothed_ema):.4f}")
    print("    [PASS]")

    # Test plot functions (without display)
    print("\n[2] Testing plot functions...")

    if HAS_MATPLOTLIB:
        # Generate fake training data
        n_episodes = 200
        metrics = {
            "episode_rewards": list(np.cumsum(np.random.randn(n_episodes) * 10 + 5)),
            "policy_losses": list(np.exp(-np.linspace(0, 3, n_episodes)) + np.random.randn(n_episodes) * 0.1),
            "value_losses": list(np.exp(-np.linspace(0, 2, n_episodes)) + np.random.randn(n_episodes) * 0.05),
            "entropies": list(np.linspace(1.5, 0.5, n_episodes) + np.random.randn(n_episodes) * 0.1),
        }

        # Test training curves
        fig = plot_training_curves(metrics, title="Test Training")
        assert fig is not None
        plt.close(fig)
        print("    plot_training_curves: OK")

        # Test comparison
        results = {
            "Algorithm A": {"episode_rewards": list(np.cumsum(np.random.randn(100) * 10 + 5))},
            "Algorithm B": {"episode_rewards": list(np.cumsum(np.random.randn(100) * 10 + 7))},
        }
        fig = plot_comparison(results)
        assert fig is not None
        plt.close(fig)
        print("    plot_comparison: OK")

        # Test dashboard
        fig = create_training_dashboard(metrics, config={"lr": 3e-4, "gamma": 0.99})
        assert fig is not None
        plt.close(fig)
        print("    create_training_dashboard: OK")

        print("    [PASS]")
    else:
        print("    Matplotlib not available, skipping plot tests")
        print("    [SKIP]")

    # Test LivePlotter
    print("\n[3] Testing LivePlotter...")
    plotter = LivePlotter()
    for i in range(50):
        plotter.update(reward=i + np.random.randn() * 5)
    assert len(plotter.metrics["reward"]) == 50
    plotter.close()
    print("    [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
