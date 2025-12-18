"""
可视化工具模块 (Visualization Module)
====================================

核心思想 (Core Idea):
--------------------
提供丰富的可视化工具，帮助理解TD学习的过程和结果。
支持训练曲线、价值函数热力图、策略可视化、TD误差分析等。

设计原则:
--------
- matplotlib可选依赖，无matplotlib时优雅降级
- 统一的API风格和参数命名
- 支持保存到文件或直接显示
- 高度可配置的图形参数

数学原理 (Mathematical Theory):
------------------------------
可视化帮助理解:
1. 学习曲线: E[R|π_t] 随训练进行的变化
2. 价值函数: V(s) 或 Q(s,a) 的空间分布
3. 策略: π(s) = argmax_a Q(s,a) 的决策边界
4. TD误差: δ_t = R + γV(S') - V(S) 的分布特性
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

# 条件导入matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib未安装，可视化功能不可用。请运行: pip install matplotlib")


# 类型别名
QFunction = Dict[Tuple[Any, Any], float]
ValueFunction = Dict[Any, float]


def _check_matplotlib() -> None:
    """检查matplotlib是否可用。"""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib未安装。请运行: pip install matplotlib"
        )


def plot_training_curves(
    metrics_list: List[Dict[str, List[float]]],
    labels: List[str],
    figsize: Tuple[int, int] = (12, 4),
    window: int = 50,
    title: str = "训练曲线",
    save_path: Optional[str] = None
) -> None:
    """
    绘制训练曲线比较图。

    核心思想 (Core Idea):
    --------------------
    可视化不同算法的学习效率，通过移动平均减少噪声，
    同时保留原始数据的半透明显示以展示方差。

    数学原理 (Mathematical Theory):
    ------------------------------
    移动平均:
        MA_t = (1/w) × Σ_{i=0}^{w-1} R_{t-i}

    其中w是窗口大小。移动平均减少噪声但引入滞后。

    Args:
        metrics_list: 各算法的指标字典列表
            每个字典应包含 'episode_rewards' 和 'episode_lengths'
        labels: 各算法的标签
        figsize: 图形尺寸
        window: 移动平均窗口大小
        title: 图形标题
        save_path: 保存路径，None则显示

    Example:
        >>> metrics_sarsa = {'episode_rewards': [...], 'episode_lengths': [...]}
        >>> metrics_qlearn = {'episode_rewards': [...], 'episode_lengths': [...]}
        >>> plot_training_curves(
        ...     [metrics_sarsa, metrics_qlearn],
        ...     ['SARSA', 'Q-Learning']
        ... )
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_list)))

    # 奖励曲线
    ax1 = axes[0]
    for metrics, label, color in zip(metrics_list, labels, colors):
        rewards = metrics.get('episode_rewards', [])
        if len(rewards) == 0:
            continue

        episodes = np.arange(len(rewards))

        # 原始数据（半透明）
        ax1.plot(episodes, rewards, alpha=0.2, color=color)

        # 移动平均
        if len(rewards) >= window:
            smoothed = np.convolve(
                rewards,
                np.ones(window) / window,
                mode='valid'
            )
            ax1.plot(
                np.arange(window - 1, len(rewards)),
                smoothed,
                label=label,
                color=color,
                linewidth=2
            )
        else:
            ax1.plot(episodes, rewards, label=label, color=color, linewidth=2)

    ax1.set_xlabel('回合 (Episode)')
    ax1.set_ylabel('累积奖励 (Reward)')
    ax1.set_title('回合奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 步数曲线
    ax2 = axes[1]
    for metrics, label, color in zip(metrics_list, labels, colors):
        lengths = metrics.get('episode_lengths', [])
        if len(lengths) == 0:
            continue

        episodes = np.arange(len(lengths))

        if len(lengths) >= window:
            smoothed = np.convolve(
                lengths,
                np.ones(window) / window,
                mode='valid'
            )
            ax2.plot(
                np.arange(window - 1, len(lengths)),
                smoothed,
                label=label,
                color=color,
                linewidth=2
            )
        else:
            ax2.plot(episodes, lengths, label=label, color=color, linewidth=2)

    ax2.set_xlabel('回合 (Episode)')
    ax2.set_ylabel('步数 (Steps)')
    ax2.set_title('回合长度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_value_heatmap(
    value_function: ValueFunction,
    grid_shape: Tuple[int, int],
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = 'RdYlGn',
    title: str = "状态价值函数 V(s)",
    obstacles: Optional[List[Tuple[int, int]]] = None,
    goals: Optional[List[Tuple[int, int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    绘制网格世界的价值函数热力图。

    核心思想 (Core Idea):
    --------------------
    通过颜色编码可视化状态价值的空间分布，
    帮助理解价值函数的结构和算法的学习效果。

    数学原理 (Mathematical Theory):
    ------------------------------
    状态价值函数:
        V^π(s) = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]

    热力图显示每个状态的估计价值，颜色越暖表示价值越高。

    Args:
        value_function: 状态价值函数 {state: value}
        grid_shape: 网格形状 (height, width)
        figsize: 图形尺寸
        cmap: 颜色映射名称
        title: 图形标题
        obstacles: 障碍物位置列表
        goals: 目标位置列表
        start: 起始位置
        save_path: 保存路径

    Example:
        >>> plot_value_heatmap(
        ...     learner.value_function,
        ...     (4, 12),
        ...     obstacles=[(1, 2)],
        ...     goals=[(3, 11)]
        ... )
    """
    _check_matplotlib()

    height, width = grid_shape
    values = np.zeros((height, width))

    for state, value in value_function.items():
        if isinstance(state, tuple):
            row, col = state
        else:
            row = state // width
            col = state % width

        if 0 <= row < height and 0 <= col < width:
            values[row, col] = value

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    im = ax.imshow(values, cmap=cmap, aspect='auto')

    # 添加数值标注
    value_range = values.max() - values.min()
    for i in range(height):
        for j in range(width):
            text_color = 'white' if abs(values[i, j] - values.min()) > value_range / 2 else 'black'
            ax.text(
                j, i, f'{values[i, j]:.1f}',
                ha='center', va='center',
                color=text_color, fontsize=8
            )

    # 标记特殊位置
    obstacles = obstacles or []
    goals = goals or []

    for pos in obstacles:
        ax.add_patch(plt.Rectangle(
            (pos[1] - 0.5, pos[0] - 0.5), 1, 1,
            fill=True, color='gray', alpha=0.7
        ))
        ax.text(pos[1], pos[0], 'X', ha='center', va='center',
                color='white', fontweight='bold')

    for pos in goals:
        ax.add_patch(plt.Rectangle(
            (pos[1] - 0.5, pos[0] - 0.5), 1, 1,
            fill=False, edgecolor='gold', linewidth=3
        ))
        ax.text(pos[1], pos[0], 'G', ha='center', va='center',
                color='gold', fontweight='bold')

    if start:
        ax.add_patch(plt.Rectangle(
            (start[1] - 0.5, start[0] - 0.5), 1, 1,
            fill=False, edgecolor='blue', linewidth=3
        ))

    # 设置刻度
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_xticklabels(np.arange(width))
    ax.set_yticklabels(np.arange(height))

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('价值 (Value)')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('列 (Column)')
    ax.set_ylabel('行 (Row)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_q_value_heatmap(
    q_function: QFunction,
    grid_shape: Tuple[int, int],
    n_actions: int = 4,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = 'RdYlGn',
    title: str = "动作价值函数 Q(s,a)",
    save_path: Optional[str] = None
) -> None:
    """
    绘制Q函数的分动作热力图。

    核心思想 (Core Idea):
    --------------------
    为每个动作绘制一个子图，显示该动作在各状态的Q值。
    便于分析不同动作的价值分布差异。

    数学原理 (Mathematical Theory):
    ------------------------------
    动作价值函数:
        Q^π(s,a) = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]

    Args:
        q_function: Q函数 {(state, action): value}
        grid_shape: 网格形状 (height, width)
        n_actions: 动作数量
        figsize: 图形尺寸
        cmap: 颜色映射
        title: 图形标题
        save_path: 保存路径
    """
    _check_matplotlib()

    height, width = grid_shape
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT'][:n_actions]

    n_rows = (n_actions + 1) // 2
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # 全局颜色范围
    all_values = list(q_function.values())
    vmin = min(all_values) if all_values else 0
    vmax = max(all_values) if all_values else 1

    for action in range(n_actions):
        ax = axes[action]
        q_values = np.zeros((height, width))

        for (state, a), value in q_function.items():
            if a != action:
                continue

            if isinstance(state, tuple):
                row, col = state
            else:
                row = state // width
                col = state % width

            if 0 <= row < height and 0 <= col < width:
                q_values[row, col] = value

        im = ax.imshow(q_values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Q(s, {action_names[action]})', fontsize=11)
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))

    # 隐藏多余的子图
    for i in range(n_actions, len(axes)):
        axes[i].axis('off')

    # 添加共享颜色条
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Q值')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_policy_arrows(
    q_function: QFunction,
    grid_shape: Tuple[int, int],
    n_actions: int = 4,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "策略可视化 π(s)",
    obstacles: Optional[List[Tuple[int, int]]] = None,
    goals: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    绘制策略的箭头图。

    核心思想 (Core Idea):
    --------------------
    在每个状态显示最优动作的方向箭头，
    直观展示学习到的策略。

    数学原理 (Mathematical Theory):
    ------------------------------
    贪婪策略:
        π(s) = argmax_a Q(s, a)

    箭头指向动作对应的移动方向。

    Args:
        q_function: Q函数
        grid_shape: 网格形状
        n_actions: 动作数量
        figsize: 图形尺寸
        title: 图形标题
        obstacles: 障碍物位置
        goals: 目标位置
        save_path: 保存路径
    """
    _check_matplotlib()

    height, width = grid_shape

    # 动作对应的箭头方向 (dx, dy)
    arrow_directions = {
        0: (0, 0.3),    # UP
        1: (0.3, 0),    # RIGHT
        2: (0, -0.3),   # DOWN
        3: (-0.3, 0),   # LEFT
    }

    fig, ax = plt.subplots(figsize=figsize)

    # 创建背景网格
    for i in range(height + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
    for j in range(width + 1):
        ax.axvline(j, color='gray', linewidth=0.5)

    obstacles = obstacles or []
    goals = goals or []

    # 绘制障碍物和目标
    for pos in obstacles:
        ax.add_patch(plt.Rectangle(
            (pos[1], height - pos[0] - 1), 1, 1,
            fill=True, color='gray', alpha=0.7
        ))

    for pos in goals:
        ax.add_patch(plt.Rectangle(
            (pos[1], height - pos[0] - 1), 1, 1,
            fill=True, color='gold', alpha=0.5
        ))

    # 绘制每个状态的最优动作箭头
    for state in range(height * width):
        row = state // width
        col = state % width

        if (row, col) in obstacles or (row, col) in goals:
            continue

        # 获取该状态所有动作的Q值
        q_values = []
        for action in range(n_actions):
            key = (state, action)
            if key in q_function:
                q_values.append((action, q_function[key]))
            else:
                key2 = ((row, col), action)
                if key2 in q_function:
                    q_values.append((action, q_function[key2]))

        if not q_values:
            continue

        best_action = max(q_values, key=lambda x: x[1])[0]

        # 绘制箭头
        center_x = col + 0.5
        center_y = height - row - 0.5

        if best_action in arrow_directions:
            dx, dy = arrow_directions[best_action]
            ax.arrow(
                center_x - dx/2, center_y - dy/2,
                dx, dy,
                head_width=0.15, head_length=0.1,
                fc='blue', ec='blue', alpha=0.8
            )

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(width) + 0.5)
    ax.set_yticks(np.arange(height) + 0.5)
    ax.set_xticklabels(np.arange(width))
    ax.set_yticklabels(np.arange(height)[::-1])
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_td_error_analysis(
    td_errors_list: List[List[float]],
    labels: List[str],
    figsize: Tuple[int, int] = (12, 4),
    window: int = 100,
    title: str = "TD误差分析",
    save_path: Optional[str] = None
) -> None:
    """
    绘制TD误差分析图。

    核心思想 (Core Idea):
    --------------------
    分析TD误差的收敛特性和分布，帮助诊断学习过程。

    数学原理 (Mathematical Theory):
    ------------------------------
    TD误差:
        δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

    理想情况下，随着学习进行:
    - E[δ] → 0 (无偏估计)
    - Var[δ] → 0 (收敛)

    Args:
        td_errors_list: 各算法的TD误差列表
        labels: 算法标签
        figsize: 图形尺寸
        window: 移动平均窗口
        title: 图形标题
        save_path: 保存路径
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(td_errors_list)))

    # TD误差绝对值的移动平均
    ax1 = axes[0]
    for errors, label, color in zip(td_errors_list, labels, colors):
        if len(errors) < window:
            continue

        abs_errors = np.abs(errors)
        smoothed = np.convolve(
            abs_errors,
            np.ones(window) / window,
            mode='valid'
        )
        ax1.plot(smoothed, label=label, color=color, linewidth=1.5)

    ax1.set_xlabel('步数 (Steps)')
    ax1.set_ylabel('|δ| (移动平均)')
    ax1.set_title('TD误差绝对值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # TD误差分布直方图
    ax2 = axes[1]
    for errors, label, color in zip(td_errors_list, labels, colors):
        ax2.hist(
            errors, bins=50, alpha=0.5,
            label=label, color=color, density=True
        )

    ax2.set_xlabel('TD误差 (δ)')
    ax2.set_ylabel('密度')
    ax2.set_title('TD误差分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_lambda_comparison(
    lambda_values: List[float],
    final_rmse: List[float],
    learning_curves: Optional[Dict[float, List[float]]] = None,
    figsize: Tuple[int, int] = (12, 4),
    title: str = "TD(λ) λ值比较",
    save_path: Optional[str] = None
) -> None:
    """
    绘制不同λ值的效果比较图。

    核心思想 (Core Idea):
    --------------------
    分析λ参数对TD(λ)性能的影响，找出最优λ值。

    数学原理 (Mathematical Theory):
    ------------------------------
    TD(λ)权衡了TD(0)和Monte Carlo:
    - λ=0: TD(0)，只看一步
    - λ=1: Monte Carlo，看完整回合
    - 中间值: 加权平均多步回报

    最优λ通常不在两端，而是在0.8-0.95之间。

    Args:
        lambda_values: λ值列表
        final_rmse: 各λ值对应的最终RMSE
        learning_curves: 可选的学习曲线 {λ: [rmse_per_episode]}
        figsize: 图形尺寸
        title: 图形标题
        save_path: 保存路径
    """
    _check_matplotlib()

    n_plots = 2 if learning_curves else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    # λ vs RMSE
    ax1 = axes[0]
    ax1.plot(lambda_values, final_rmse, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('λ')
    ax1.set_ylabel('RMSE')
    ax1.set_title('最终RMSE vs λ')
    ax1.grid(True, alpha=0.3)

    # 标记最优λ
    best_idx = np.argmin(final_rmse)
    ax1.axvline(lambda_values[best_idx], color='r', linestyle='--', alpha=0.5)
    ax1.annotate(
        f'最优 λ={lambda_values[best_idx]:.2f}',
        xy=(lambda_values[best_idx], final_rmse[best_idx]),
        xytext=(lambda_values[best_idx] + 0.1, final_rmse[best_idx] + 0.05),
        arrowprops=dict(arrowstyle='->', color='red')
    )

    # 学习曲线
    if learning_curves:
        ax2 = axes[1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(learning_curves)))

        for (lam, curve), color in zip(sorted(learning_curves.items()), colors):
            ax2.plot(curve, label=f'λ={lam:.2f}', color=color, alpha=0.8)

        ax2.set_xlabel('回合 (Episode)')
        ax2.set_ylabel('RMSE')
        ax2.set_title('学习曲线')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_multi_seed_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 4),
    window: int = 50,
    title: str = "多种子实验比较",
    show_std: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    绘制多种子实验的比较图（带置信区间）。

    核心思想 (Core Idea):
    --------------------
    通过多随机种子实验评估算法的稳定性，
    显示均值和标准差区间。

    数学原理 (Mathematical Theory):
    ------------------------------
    给定n个种子的结果 {R_1, R_2, ..., R_n}:
    - 均值: μ = (1/n) × Σ R_i
    - 标准差: σ = √((1/n) × Σ(R_i - μ)²)

    置信区间 [μ - σ, μ + σ] 包含约68%的数据。

    Args:
        results_dict: {算法名: run_multi_seed_experiment返回的结果}
        figsize: 图形尺寸
        window: 移动平均窗口
        title: 图形标题
        show_std: 是否显示标准差区间
        save_path: 保存路径
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for (name, results), color in zip(results_dict.items(), colors):
        mean_reward = results['mean_reward']
        std_reward = results['std_reward']
        mean_length = results['mean_length']
        std_length = results['std_length']

        # 平滑处理
        if len(mean_reward) >= window:
            smooth_reward = np.convolve(mean_reward, np.ones(window)/window, mode='valid')
            smooth_std = np.convolve(std_reward, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(mean_reward))
        else:
            smooth_reward = mean_reward
            smooth_std = std_reward
            x = np.arange(len(mean_reward))

        # 奖励曲线
        axes[0].plot(x, smooth_reward, label=name, color=color, linewidth=2)
        if show_std:
            axes[0].fill_between(
                x,
                smooth_reward - smooth_std,
                smooth_reward + smooth_std,
                alpha=0.2, color=color
            )

        # 步数曲线
        if len(mean_length) >= window:
            smooth_length = np.convolve(mean_length, np.ones(window)/window, mode='valid')
            smooth_std_len = np.convolve(std_length, np.ones(window)/window, mode='valid')
            x = np.arange(window-1, len(mean_length))
        else:
            smooth_length = mean_length
            smooth_std_len = std_length
            x = np.arange(len(mean_length))

        axes[1].plot(x, smooth_length, label=name, color=color, linewidth=2)
        if show_std:
            axes[1].fill_between(
                x,
                smooth_length - smooth_std_len,
                smooth_length + smooth_std_len,
                alpha=0.2, color=color
            )

    axes[0].set_xlabel('回合')
    axes[0].set_ylabel('累积奖励')
    axes[0].set_title('奖励曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('回合')
    axes[1].set_ylabel('步数')
    axes[1].set_title('回合长度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_cliff_walking_path(
    q_function: QFunction,
    height: int = 4,
    width: int = 12,
    figsize: Tuple[int, int] = (14, 4),
    title: str = "Cliff Walking 学习路径",
    save_path: Optional[str] = None
) -> None:
    """
    可视化Cliff Walking环境中学习到的路径。

    核心思想 (Core Idea):
    --------------------
    展示从Q函数提取的贪婪策略在悬崖行走环境中的实际路径，
    帮助理解SARSA和Q-Learning的行为差异。

    Args:
        q_function: Q函数
        height: 网格高度
        width: 网格宽度
        figsize: 图形尺寸
        title: 图形标题
        save_path: 保存路径
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制网格
    for i in range(height + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
    for j in range(width + 1):
        ax.axvline(j, color='gray', linewidth=0.5)

    # 绘制悬崖
    for c in range(1, 11):
        ax.add_patch(plt.Rectangle(
            (c, 0), 1, 1,
            fill=True, color='red', alpha=0.5
        ))
        ax.text(c + 0.5, 0.5, 'C', ha='center', va='center', fontsize=8)

    # 起点和终点
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color='green', alpha=0.5))
    ax.text(0.5, 0.5, 'S', ha='center', va='center', fontweight='bold')

    ax.add_patch(plt.Rectangle((11, 0), 1, 1, fill=True, color='gold', alpha=0.5))
    ax.text(11.5, 0.5, 'G', ha='center', va='center', fontweight='bold')

    # 根据Q函数模拟最优路径
    state = (3, 0)
    path = [state]

    for _ in range(50):
        state_idx = state[0] * width + state[1]

        q_values = []
        for a in range(4):
            key = (state_idx, a)
            if key in q_function:
                q_values.append((a, q_function[key]))

        if not q_values:
            break

        best_action = max(q_values, key=lambda x: x[1])[0]

        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        delta = deltas[best_action]
        new_state = (
            int(np.clip(state[0] + delta[0], 0, height - 1)),
            int(np.clip(state[1] + delta[1], 0, width - 1))
        )

        if new_state == (3, 11):
            path.append(new_state)
            break

        if new_state[0] == 3 and 1 <= new_state[1] <= 10:
            path.append(new_state)
            break

        state = new_state
        path.append(state)

    # 绘制路径
    for i, pos in enumerate(path):
        x = pos[1] + 0.5
        y = height - pos[0] - 0.5

        if i == 0:
            continue

        ax.plot(x, y, 'bo', markersize=12, alpha=0.7)
        ax.text(x, y, str(i), ha='center', va='center', color='white', fontsize=8)

    # 绘制路径连线
    if len(path) > 1:
        xs = [p[1] + 0.5 for p in path]
        ys = [height - p[0] - 0.5 for p in path]
        ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.5)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('列')
    ax.set_ylabel('行')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
