"""
Visualization Utilities for MDP Analysis

This module provides comprehensive visualization tools for analyzing MDPs,
including policy visualization, value function heatmaps, and convergence plots.

Core Idea:
    Effective visualization is critical for understanding MDP solutions.
    This module provides tools to visualize policies, value functions,
    and algorithm convergence behavior.

Mathematical Theory:
    Value function visualization: Heatmap of V(s) across state space
    Policy visualization: Arrow plots showing action selection
    Convergence analysis: Plots of algorithm convergence metrics

Problem Statement:
    Visual analysis helps identify issues with MDP formulations and solutions.
    Provides intuitive understanding of optimal behavior.

Complexity:
    - Rendering: O(|S|) for grid-based visualization
    - Plotting: O(k) where k is number of iterations
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ..core.mdp import State, Action, MarkovDecisionProcess
from ..core.policy import Policy, DeterministicPolicy
from ..core.value_function import StateValueFunction, ActionValueFunction


class MDPVisualizer:
    """
    Comprehensive MDP visualization toolkit.

    Core Idea:
        Provides multiple visualization methods for understanding MDP solutions.

    Methods:
        - plot_value_function: Heatmap of state values
        - plot_policy: Arrow plot of policy
        - plot_convergence: Algorithm convergence curves
        - plot_q_function: Q-function heatmap
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize

    def plot_value_function(self, mdp: MarkovDecisionProcess,
                           value_function: StateValueFunction,
                           title: str = "State Value Function",
                           cmap: str = "RdYlGn") -> plt.Figure:
        """
        Plot value function as heatmap.

        Args:
            mdp: The MDP
            value_function: State value function to visualize
            title: Plot title
            cmap: Colormap name

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get values as array
        values = value_function.to_array()

        # Reshape for grid visualization if applicable
        if hasattr(mdp, 'height') and hasattr(mdp, 'width'):
            values_grid = values.reshape((mdp.height, mdp.width))
            im = ax.imshow(values_grid, cmap=cmap, aspect='auto')

            # Add text annotations
            for i in range(mdp.height):
                for j in range(mdp.width):
                    text = ax.text(j, i, f'{values_grid[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)

            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
        else:
            # For non-grid MDPs, plot as bar chart
            ax.bar(range(len(values)), values)
            ax.set_xlabel("State")
            ax.set_ylabel("Value")

        ax.set_title(title)
        plt.colorbar(im if hasattr(mdp, 'height') else None, ax=ax)
        plt.tight_layout()

        return fig

    def plot_policy(self, mdp: MarkovDecisionProcess,
                   policy: DeterministicPolicy,
                   title: str = "Optimal Policy",
                   value_function: Optional[StateValueFunction] = None) -> plt.Figure:
        """
        Plot policy as arrow field.

        Args:
            mdp: The MDP
            policy: Deterministic policy to visualize
            title: Plot title
            value_function: Optional value function for background

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if not (hasattr(mdp, 'height') and hasattr(mdp, 'width')):
            raise ValueError("Policy visualization requires grid-based MDP")

        # Plot value function as background if provided
        if value_function:
            values = value_function.to_array()
            values_grid = values.reshape((mdp.height, mdp.width))
            im = ax.imshow(values_grid, cmap="gray", aspect='auto', alpha=0.3)

        # Plot policy arrows
        for row in range(mdp.height):
            for col in range(mdp.width):
                state = mdp._get_state(row, col)

                try:
                    action = policy.get_action(state)

                    # Determine arrow direction
                    if hasattr(mdp, 'UP'):
                        if action == mdp.UP:
                            dx, dy = 0, -0.3
                        elif action == mdp.DOWN:
                            dx, dy = 0, 0.3
                        elif action == mdp.LEFT:
                            dx, dy = -0.3, 0
                        elif action == mdp.RIGHT:
                            dx, dy = 0.3, 0
                        else:
                            continue

                        ax.arrow(col, row, dx, dy, head_width=0.15, head_length=0.1,
                                fc='red', ec='red', linewidth=2)
                except:
                    pass

        ax.set_xlim(-0.5, mdp.width - 0.5)
        ax.set_ylim(mdp.height - 0.5, -0.5)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(title)
        ax.set_xticks(range(mdp.width))
        ax.set_yticks(range(mdp.height))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_convergence(self, convergence_history: List[float],
                        title: str = "Algorithm Convergence",
                        ylabel: str = "Max Value Change",
                        yscale: str = "log") -> plt.Figure:
        """
        Plot algorithm convergence curve.

        Args:
            convergence_history: List of convergence metrics per iteration
            title: Plot title
            ylabel: Y-axis label
            yscale: Y-axis scale ('linear' or 'log')

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        iterations = range(1, len(convergence_history) + 1)
        ax.plot(iterations, convergence_history, 'b-', linewidth=2, marker='o')

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_q_function(self, mdp: MarkovDecisionProcess,
                       q_function: ActionValueFunction,
                       title: str = "Action Value Function (Q-function)") -> plt.Figure:
        """
        Plot Q-function as heatmap.

        Args:
            mdp: The MDP
            q_function: Action value function to visualize
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Get Q-values as array
        q_array = q_function.to_array()

        # Plot heatmap
        im = ax.imshow(q_array, cmap="RdYlGn", aspect='auto')

        # Add text annotations
        for i in range(q_array.shape[0]):
            for j in range(q_array.shape[1]):
                text = ax.text(j, i, f'{q_array[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        ax.set_xlabel("Action")
        ax.set_ylabel("State")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        return fig

    def plot_comparison(self, mdp: MarkovDecisionProcess,
                       value_function: StateValueFunction,
                       policy: DeterministicPolicy,
                       convergence_history: Optional[List[float]] = None) -> plt.Figure:
        """
        Plot comprehensive comparison of MDP solution.

        Args:
            mdp: The MDP
            value_function: State value function
            policy: Deterministic policy
            convergence_history: Optional convergence history

        Returns:
            Matplotlib figure with subplots
        """
        if convergence_history:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Value function
        ax = axes[0, 0] if convergence_history else axes[0]
        values = value_function.to_array()
        if hasattr(mdp, 'height') and hasattr(mdp, 'width'):
            values_grid = values.reshape((mdp.height, mdp.width))
            im = ax.imshow(values_grid, cmap="RdYlGn", aspect='auto')
            for i in range(mdp.height):
                for j in range(mdp.width):
                    ax.text(j, i, f'{values_grid[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
            plt.colorbar(im, ax=ax)

        ax.set_title("State Value Function")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Plot 2: Policy
        ax = axes[0, 1] if convergence_history else axes[1]
        if hasattr(mdp, 'height') and hasattr(mdp, 'width'):
            for row in range(mdp.height):
                for col in range(mdp.width):
                    state = mdp._get_state(row, col)
                    try:
                        action = policy.get_action(state)
                        if hasattr(mdp, 'UP'):
                            if action == mdp.UP:
                                dx, dy = 0, -0.3
                            elif action == mdp.DOWN:
                                dx, dy = 0, 0.3
                            elif action == mdp.LEFT:
                                dx, dy = -0.3, 0
                            elif action == mdp.RIGHT:
                                dx, dy = 0.3, 0
                            else:
                                continue
                            ax.arrow(col, row, dx, dy, head_width=0.15, head_length=0.1,
                                    fc='red', ec='red', linewidth=2)
                    except:
                        pass

            ax.set_xlim(-0.5, mdp.width - 0.5)
            ax.set_ylim(mdp.height - 0.5, -0.5)
            ax.set_xticks(range(mdp.width))
            ax.set_yticks(range(mdp.height))
            ax.grid(True, alpha=0.3)

        ax.set_title("Optimal Policy")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Plot 3: Convergence (if provided)
        if convergence_history:
            ax = axes[1, 0]
            iterations = range(1, len(convergence_history) + 1)
            ax.plot(iterations, convergence_history, 'b-', linewidth=2, marker='o')
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Max Value Change")
            ax.set_title("Convergence History")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

            # Plot 4: Statistics
            ax = axes[1, 1]
            ax.axis('off')
            stats_text = (
                f"MDP Statistics:\n"
                f"States: {mdp.get_state_space_size()}\n"
                f"Actions: {mdp.get_action_space_size()}\n"
                f"Discount Factor: {mdp.gamma:.3f}\n"
                f"\nValue Function Statistics:\n"
                f"Max Value: {value_function.get_max_value():.3f}\n"
                f"Min Value: {value_function.get_min_value():.3f}\n"
                f"Mean Value: {value_function.get_mean_value():.3f}\n"
                f"\nConvergence:\n"
                f"Iterations: {len(convergence_history)}\n"
                f"Final Change: {convergence_history[-1]:.2e}"
            )
            ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig
