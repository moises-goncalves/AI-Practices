"""
可视化工具 (Visualization Utilities)
===================================
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_training_curve(
    rewards: List[float],
    window: int = 100,
    title: str = "Training Curve",
    ax: Optional[Any] = None
) -> Optional[Any]:
    """绘制训练曲线（含移动平均）。"""
    if not HAS_MATPLOTLIB:
        print("matplotlib未安装，跳过绘图")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, color='blue', linewidth=2, label=f'MA({window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_value_function(
    values: Dict[int, float],
    shape: Tuple[int, int],
    title: str = "Value Function",
    ax: Optional[Any] = None
) -> Optional[Any]:
    """绘制价值函数热力图。"""
    if not HAS_MATPLOTLIB:
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    grid = np.zeros(shape)
    for state, value in values.items():
        row, col = state // shape[1], state % shape[1]
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            grid[row, col] = value
    
    im = ax.imshow(grid, cmap='RdYlGn')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax


def plot_policy(
    q_function: Dict[Tuple[int, int], float],
    shape: Tuple[int, int],
    title: str = "Policy",
    ax: Optional[Any] = None
) -> Optional[Any]:
    """绘制策略箭头图。"""
    if not HAS_MATPLOTLIB:
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    arrows = {0: (0, 0.3), 1: (0.3, 0), 2: (0, -0.3), 3: (-0.3, 0)}  # UP, RIGHT, DOWN, LEFT
    
    for row in range(shape[0]):
        for col in range(shape[1]):
            state = row * shape[1] + col
            q_vals = [q_function.get((state, a), 0.0) for a in range(4)]
            best_action = np.argmax(q_vals)
            dx, dy = arrows[best_action]
            ax.arrow(col, row, dx, dy, head_width=0.1, head_length=0.05, fc='blue', ec='blue')
    
    ax.set_xlim(-0.5, shape[1] - 0.5)
    ax.set_ylim(shape[0] - 0.5, -0.5)
    ax.set_title(title)
    ax.grid(True)
    return ax
