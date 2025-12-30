"""Visualization and Utility Functions for Tabular RL.

This module provides visualization tools for analyzing training progress,
Q-tables, policies, and comparing algorithms.

Functions:
    - plot_learning_curves: Compare training curves across algorithms
    - visualize_q_table: Heatmap visualization of value functions
    - visualize_policy: Arrow-based policy visualization
    - extract_path: Extract greedy policy trajectory
    - compare_algorithms: Run comparative experiments
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def extract_path(
    agent: Any,
    env: Any,
    max_steps: int = 50,
) -> List[Tuple[int, int]]:
    """Extract greedy policy trajectory from trained agent.

    Core Idea:
        Follow the greedy policy π(s) = argmax_a Q(s,a) from start
        to termination, recording the state sequence.

    Args:
        agent: Trained agent with get_action(state, training=False)
        env: Environment with reset() and step()
        max_steps: Maximum trajectory length (prevents infinite loops)

    Returns:
        List of states [(row, col), ...] from start to goal/max_steps

    Example:
        >>> path = extract_path(trained_agent, env)
        >>> env.render(path=path)
    """
    result = env.reset()
    state = result[0] if isinstance(result, tuple) else result
    path = [state]

    for _ in range(max_steps):
        action = agent.get_action(state, training=False)

        result = env.step(action)
        if len(result) == 3:
            next_state, _, done = result
        else:
            next_state, _, terminated, truncated, _ = result
            done = terminated or truncated

        path.append(next_state)
        state = next_state

        if done:
            break

    return path


def plot_learning_curves(
    metrics_dict: Dict[str, Any],
    window: int = 10,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    title: str = "Learning Curves",
) -> None:
    """Plot comparative learning curves for multiple algorithms.

    Core Idea:
        Visualize training progress by plotting smoothed episode rewards
        and steps, enabling direct comparison between algorithms.

    Args:
        metrics_dict: Dict mapping algorithm name to TrainingMetrics
        window: Moving average window for smoothing
        figsize: Figure dimensions (width, height)
        save_path: If provided, saves figure to this path
        title: Plot title

    Example:
        >>> plot_learning_curves({
        ...     'Q-Learning': q_metrics,
        ...     'SARSA': sarsa_metrics
        ... })
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10.colors

    # Reward curves
    ax1 = axes[0]
    for idx, (name, metrics) in enumerate(metrics_dict.items()):
        rewards = metrics.episode_rewards
        if len(rewards) >= window:
            smoothed = np.convolve(
                rewards, np.ones(window) / window, mode="valid"
            )
            ax1.plot(
                smoothed,
                label=name,
                color=colors[idx % len(colors)],
                alpha=0.8,
                linewidth=2,
            )

    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Total Reward", fontsize=11)
    ax1.set_title("Episode Rewards", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Steps curves
    ax2 = axes[1]
    for idx, (name, metrics) in enumerate(metrics_dict.items()):
        steps = metrics.episode_lengths
        if len(steps) >= window:
            smoothed = np.convolve(
                steps, np.ones(window) / window, mode="valid"
            )
            ax2.plot(
                smoothed,
                label=name,
                color=colors[idx % len(colors)],
                alpha=0.8,
                linewidth=2,
            )

    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Steps", fontsize=11)
    ax2.set_title("Episode Length", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def visualize_q_table(
    agent: Any,
    env: Any,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
) -> None:
    """Visualize Q-table as heatmap and policy arrows.

    Core Idea:
        Provide visual insight into learned value function V(s) = max_a Q(s,a)
        and extracted greedy policy π(s) = argmax_a Q(s,a).

    Args:
        agent: Trained agent with q_table attribute
        env: Environment with height, width, cliff attributes
        figsize: Figure dimensions
        save_path: If provided, saves figure to this path

    Visualization Components:
        1. Value function heatmap: V(s) at each state
        2. Policy arrows: Direction of greedy action
        3. Q-value distribution: Histogram of max Q-values
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    height = env.height
    width = env.width

    # Compute value function and policy
    v_table = np.zeros((height, width), dtype=np.float32)
    policy_arrows = np.zeros((height, width), dtype=int)

    for i in range(height):
        for j in range(width):
            state = (i, j)
            if state in agent.q_table:
                v_table[i, j] = np.max(agent.q_table[state])
                policy_arrows[i, j] = int(np.argmax(agent.q_table[state]))

    # Value function heatmap
    ax1 = axes[0]
    im = ax1.imshow(v_table, cmap="RdYlGn", aspect="auto")
    ax1.set_title("V(s) = max_a Q(s,a)", fontweight="bold")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im, ax=ax1)

    # Mark cliff cells
    if hasattr(env, "cliff"):
        for pos in env.cliff:
            ax1.add_patch(
                plt.Rectangle(
                    (pos[1] - 0.5, pos[0] - 0.5),
                    1, 1,
                    fill=True,
                    color="black",
                    alpha=0.5,
                )
            )

    # Policy arrows
    ax2 = axes[1]
    arrow_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    for i in range(height):
        for j in range(width):
            state = (i, j)
            if hasattr(env, "cliff") and state in env.cliff:
                ax2.text(j, i, "X", ha="center", va="center", fontsize=12, color="red")
            elif hasattr(env, "goal") and state == env.goal:
                ax2.text(
                    j, i, "G", ha="center", va="center",
                    fontsize=12, color="green", fontweight="bold"
                )
            elif hasattr(env, "start") and state == env.start:
                ax2.text(
                    j, i, "S", ha="center", va="center",
                    fontsize=12, color="blue", fontweight="bold"
                )
            else:
                ax2.text(
                    j, i, arrow_map[policy_arrows[i, j]],
                    ha="center", va="center", fontsize=14
                )

    ax2.set_xlim(-0.5, width - 0.5)
    ax2.set_ylim(height - 0.5, -0.5)
    ax2.set_title("Greedy Policy π(s)", fontweight="bold")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    ax2.grid(True, alpha=0.3)

    # Q-value distribution
    ax3 = axes[2]
    q_max_values = [
        np.max(q) for q in agent.q_table.values() if np.any(q != 0)
    ]
    if q_max_values:
        ax3.hist(
            q_max_values, bins=30, edgecolor="black", alpha=0.7, color="steelblue"
        )
    ax3.set_xlabel("Max Q-Value", fontsize=11)
    ax3.set_ylabel("Frequency", fontsize=11)
    ax3.set_title("Q-Value Distribution", fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def visualize_policy(
    agent: Any,
    env: Any,
    show_values: bool = False,
) -> str:
    """Print ASCII visualization of greedy policy.

    Args:
        agent: Trained agent with q_table
        env: Environment with height, width attributes
        show_values: If True, show V(s) values instead of arrows

    Returns:
        String representation of policy
    """
    arrow_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    height = env.height
    width = env.width

    lines = []
    lines.append("Learned Policy (Greedy):")
    lines.append("┌" + "───" * width + "┐")

    for i in range(height):
        row = "│"
        for j in range(width):
            state = (i, j)

            if hasattr(env, "start") and state == env.start:
                row += " S "
            elif hasattr(env, "goal") and state == env.goal:
                row += " G "
            elif hasattr(env, "cliff") and state in env.cliff:
                row += " C "
            elif state in agent.q_table:
                if show_values:
                    v = np.max(agent.q_table[state])
                    row += f"{v:3.0f}"[:3]
                else:
                    best_action = np.argmax(agent.q_table[state])
                    row += f" {arrow_map[best_action]} "
            else:
                row += " . "

        lines.append(row + "│")

    lines.append("└" + "───" * width + "┘")

    output = "\n".join(lines)
    print(output)
    return output


def compare_algorithms(
    env_class: type,
    agent_classes: Dict[str, type],
    episodes: int = 500,
    n_runs: int = 5,
    agent_kwargs: Optional[Dict[str, dict]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run comparative experiments across multiple algorithms.

    Core Idea:
        Execute multiple training runs for each algorithm and aggregate
        statistics to enable fair comparison with confidence bounds.

    Args:
        env_class: Environment class to instantiate
        agent_classes: Dict mapping name to agent class
        episodes: Training episodes per run
        n_runs: Number of independent runs per algorithm
        agent_kwargs: Optional per-algorithm kwargs
        verbose: Whether to print progress

    Returns:
        Dict mapping algorithm name to {
            'rewards': array of shape (n_runs, episodes),
            'mean': mean reward per episode,
            'std': std reward per episode
        }
    """
    from .training import train_q_learning, train_sarsa

    results = {}
    agent_kwargs = agent_kwargs or {}

    for name, agent_class in agent_classes.items():
        if verbose:
            print(f"\nTraining {name}...")

        all_rewards = []
        kwargs = agent_kwargs.get(name, {})

        for run in range(n_runs):
            env = env_class()
            agent = agent_class(n_actions=env.n_actions, **kwargs)

            # Detect SARSA
            is_sarsa = "SARSA" in name and "Expected" not in name

            if is_sarsa:
                metrics = train_sarsa(env, agent, episodes=episodes, verbose=False)
            else:
                metrics = train_q_learning(env, agent, episodes=episodes, verbose=False)

            all_rewards.append(metrics.episode_rewards)

            if verbose:
                print(f"  Run {run + 1}/{n_runs}: "
                      f"Final reward = {np.mean(metrics.episode_rewards[-50:]):.2f}")

        rewards_array = np.array(all_rewards)
        results[name] = {
            "rewards": rewards_array,
            "mean": np.mean(rewards_array, axis=0),
            "std": np.std(rewards_array, axis=0),
        }

    return results


def plot_comparison_with_ci(
    results: Dict[str, Dict[str, np.ndarray]],
    window: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot comparison with confidence intervals.

    Args:
        results: Output from compare_algorithms
        window: Smoothing window
        figsize: Figure size
        save_path: Optional save path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors

    for idx, (name, data) in enumerate(results.items()):
        mean = data["mean"]
        std = data["std"]

        # Smooth
        if len(mean) >= window:
            mean_smooth = np.convolve(mean, np.ones(window) / window, mode="valid")
            std_smooth = np.convolve(std, np.ones(window) / window, mode="valid")
            x = np.arange(window - 1, len(mean))
        else:
            mean_smooth = mean
            std_smooth = std
            x = np.arange(len(mean))

        color = colors[idx % len(colors)]
        ax.plot(x, mean_smooth, label=name, color=color, linewidth=2)
        ax.fill_between(
            x,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total Reward", fontsize=11)
    ax.set_title("Algorithm Comparison (mean ± std)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    print("Utility Functions Unit Tests")
    print("=" * 60)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from agents import QLearningAgent
    from environments import CliffWalkingEnv
    from training import train_q_learning

    # Test 1: Extract path
    env = CliffWalkingEnv()
    agent = QLearningAgent(
        n_actions=4, learning_rate=0.5, epsilon=0.1, epsilon_decay=1.0
    )
    train_q_learning(env, agent, episodes=200, verbose=False)

    path = extract_path(agent, env)
    assert len(path) > 1, "Path should have multiple states"
    assert path[0] == env.start, "Path should start at start"
    print(f"✓ Test 1: extract_path works (length={len(path)})")

    # Test 2: Visualize policy (ASCII)
    output = visualize_policy(agent, env)
    assert "S" in output, "Should show start"
    assert "G" in output, "Should show goal"
    print("✓ Test 2: visualize_policy produces valid output")

    # Test 3: Check matplotlib availability
    try:
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False

    if has_matplotlib:
        print("✓ Test 3: matplotlib available for plotting")
    else:
        print("⚠ Test 3: matplotlib not available, plot tests skipped")

    print("=" * 60)
    print("All tests passed!")
