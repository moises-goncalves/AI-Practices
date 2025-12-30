"""
时序差分学习工具模块 (Temporal Difference Learning Utilities)
============================================================

核心功能:
--------
1. 可视化: 训练曲线、价值函数热力图、策略箭头图、学习对比图
2. 实验管理: 多种子实验、超参数搜索、结果统计
3. 分析工具: 收敛性检测、策略提取、性能评估
4. 序列化: 模型保存/加载、实验配置导出

设计原则:
--------
- 无依赖单独使用: 仅依赖numpy和matplotlib
- 与主模块解耦: 通过字典接口传递数据
- 高度可配置: 支持各种自定义参数
"""

from __future__ import annotations

import numpy as np
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import (
    Dict, List, Optional, Tuple, Any, Callable,
    Union, TypeVar, Generic
)
from collections import defaultdict
import warnings

# 条件导入matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib未安装，可视化功能不可用")


# =============================================================================
# 类型定义
# =============================================================================

State = TypeVar('State')
Action = TypeVar('Action')

# Q函数类型: {(state, action): value}
QFunction = Dict[Tuple[Any, Any], float]
# 价值函数类型: {state: value}
ValueFunction = Dict[Any, float]


# =============================================================================
# 可视化工具
# =============================================================================

def check_matplotlib() -> None:
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

    Args:
        metrics_list: 各算法的指标字典列表
            每个字典应包含'episode_rewards'和'episode_lengths'
        labels: 各算法的标签
        figsize: 图形尺寸
        window: 移动平均窗口大小
        title: 图形标题
        save_path: 保存路径，None则显示

    Example:
        >>> metrics_sarsa = sarsa_learner.metrics.__dict__
        >>> metrics_qlearn = qlearn_learner.metrics.__dict__
        >>> plot_training_curves(
        ...     [metrics_sarsa, metrics_qlearn],
        ...     ['SARSA', 'Q-Learning']
        ... )
    """
    check_matplotlib()

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

        # 移动平均
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
    title: str = "状态价值函数",
    obstacles: Optional[List[Tuple[int, int]]] = None,
    goals: Optional[List[Tuple[int, int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    绘制网格世界的价值函数热力图。

    Args:
        value_function: 状态价值函数 {state: value}
            state可以是整数索引或(row, col)元组
        grid_shape: 网格形状 (height, width)
        figsize: 图形尺寸
        cmap: 颜色映射
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
    check_matplotlib()

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
    for i in range(height):
        for j in range(width):
            text_color = 'white' if abs(values[i, j]) > (values.max() - values.min()) / 2 else 'black'
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
        ax.text(pos[1], pos[0], 'X', ha='center', va='center', color='white', fontweight='bold')

    for pos in goals:
        ax.add_patch(plt.Rectangle(
            (pos[1] - 0.5, pos[0] - 0.5), 1, 1,
            fill=False, edgecolor='gold', linewidth=3
        ))
        ax.text(pos[1], pos[0], 'G', ha='center', va='center', color='gold', fontweight='bold')

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

    为每个动作绘制一个子图，显示该动作在各状态的Q值。

    Args:
        q_function: Q函数 {(state, action): value}
        grid_shape: 网格形状 (height, width)
        n_actions: 动作数量
        figsize: 图形尺寸
        cmap: 颜色映射
        title: 图形标题
        save_path: 保存路径
    """
    check_matplotlib()

    height, width = grid_shape
    action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT'][:n_actions]

    # 计算子图布局
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
    title: str = "策略可视化",
    obstacles: Optional[List[Tuple[int, int]]] = None,
    goals: Optional[List[Tuple[int, int]]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    绘制策略的箭头图。

    在每个状态显示最优动作的方向箭头。

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
    check_matplotlib()

    height, width = grid_shape

    # 动作对应的箭头方向 (dx, dy)
    # 注意: matplotlib的箭头y轴向上，所以UP是+y
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

        # 跳过障碍物和目标
        if (row, col) in obstacles or (row, col) in goals:
            continue

        # 获取该状态所有动作的Q值
        q_values = []
        for action in range(n_actions):
            key = (state, action)
            if key in q_function:
                q_values.append((action, q_function[key]))
            else:
                # 也尝试元组形式的状态
                key2 = ((row, col), action)
                if key2 in q_function:
                    q_values.append((action, q_function[key2]))

        if not q_values:
            continue

        # 找最优动作
        best_action = max(q_values, key=lambda x: x[1])[0]

        # 绘制箭头
        # 中心位置 (matplotlib坐标系y向上)
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

    Args:
        td_errors_list: 各算法的TD误差列表
        labels: 算法标签
        figsize: 图形尺寸
        window: 移动平均窗口
        title: 图形标题
        save_path: 保存路径
    """
    check_matplotlib()

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

    Args:
        lambda_values: λ值列表
        final_rmse: 各λ值对应的最终RMSE
        learning_curves: 可选的学习曲线 {λ: [rmse_per_episode]}
        figsize: 图形尺寸
        title: 图形标题
        save_path: 保存路径
    """
    check_matplotlib()

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


# =============================================================================
# 实验管理工具
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    实验配置类。

    用于管理和记录实验的所有超参数。
    """
    algorithm: str
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    lambda_: float = 0.0
    n_step: int = 1
    n_episodes: int = 1000
    n_seeds: int = 5
    env_name: str = "GridWorld"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """从字典创建。"""
        return cls(**data)


@dataclass
class ExperimentResult:
    """
    实验结果类。

    存储单次实验的完整结果。
    """
    config: ExperimentConfig
    metrics: Dict[str, Any]
    q_function: Optional[QFunction] = None
    value_function: Optional[ValueFunction] = None
    training_time: float = 0.0
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典。"""
        return {
            'config': self.config.to_dict(),
            'metrics': {
                k: list(v) if isinstance(v, np.ndarray) else v
                for k, v in self.metrics.items()
            },
            'training_time': self.training_time,
            'seed': self.seed,
        }


def run_multi_seed_experiment(
    learner_factory: Callable[[], Any],
    env_factory: Callable[[], Any],
    n_episodes: int = 1000,
    n_seeds: int = 5,
    max_steps_per_episode: int = 10000,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    运行多种子实验。

    Args:
        learner_factory: 创建学习器的工厂函数
        env_factory: 创建环境的工厂函数
        n_episodes: 每次实验的回合数
        n_seeds: 随机种子数量
        max_steps_per_episode: 每回合最大步数
        verbose: 是否打印进度

    Returns:
        包含所有种子结果的字典:
        {
            'rewards': List[List[float]],  # 每个种子的奖励曲线
            'lengths': List[List[float]],  # 每个种子的步数曲线
            'mean_reward': np.ndarray,     # 平均奖励曲线
            'std_reward': np.ndarray,      # 奖励标准差
            'mean_length': np.ndarray,     # 平均步数曲线
            'std_length': np.ndarray,      # 步数标准差
        }

    Example:
        >>> results = run_multi_seed_experiment(
        ...     lambda: SARSA(TDConfig(alpha=0.1)),
        ...     lambda: gym.make('CliffWalking-v0'),
        ...     n_episodes=500,
        ...     n_seeds=10
        ... )
    """
    all_rewards = []
    all_lengths = []

    for seed in range(n_seeds):
        if verbose:
            print(f"运行种子 {seed + 1}/{n_seeds}...")

        np.random.seed(seed)

        learner = learner_factory()
        env = env_factory()

        # 训练
        metrics = learner.train(
            env,
            n_episodes=n_episodes,
            max_steps_per_episode=max_steps_per_episode,
            log_interval=n_episodes + 1  # 禁止日志
        )

        all_rewards.append(metrics.episode_rewards)
        all_lengths.append(metrics.episode_lengths)

        if hasattr(env, 'close'):
            env.close()

    # 计算统计量
    rewards_array = np.array(all_rewards)
    lengths_array = np.array(all_lengths)

    return {
        'rewards': all_rewards,
        'lengths': all_lengths,
        'mean_reward': np.mean(rewards_array, axis=0),
        'std_reward': np.std(rewards_array, axis=0),
        'mean_length': np.mean(lengths_array, axis=0),
        'std_length': np.std(lengths_array, axis=0),
    }


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

    Args:
        results_dict: {算法名: run_multi_seed_experiment返回的结果}
        figsize: 图形尺寸
        window: 移动平均窗口
        title: 图形标题
        show_std: 是否显示标准差区间
        save_path: 保存路径
    """
    check_matplotlib()

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


# =============================================================================
# 分析工具
# =============================================================================

def compute_rmse(
    estimated: Union[ValueFunction, QFunction],
    true_values: Union[ValueFunction, QFunction]
) -> float:
    """
    计算估计值与真实值的RMSE。

    RMSE = √(1/n × Σ(V̂(s) - V(s))²)

    Args:
        estimated: 估计的价值函数
        true_values: 真实价值函数

    Returns:
        均方根误差

    Example:
        >>> rmse = compute_rmse(learner.value_function, env.get_optimal_value())
    """
    common_keys = set(estimated.keys()) & set(true_values.keys())

    if len(common_keys) == 0:
        return float('inf')

    squared_errors = [
        (estimated[k] - true_values[k]) ** 2
        for k in common_keys
    ]

    return np.sqrt(np.mean(squared_errors))


def extract_greedy_policy(
    q_function: QFunction,
    n_states: int,
    n_actions: int
) -> Dict[int, int]:
    """
    从Q函数提取贪婪策略。

    π(s) = argmax_a Q(s, a)

    Args:
        q_function: Q函数
        n_states: 状态数量
        n_actions: 动作数量

    Returns:
        策略字典 {state: action}
    """
    policy = {}

    for state in range(n_states):
        q_values = []
        for action in range(n_actions):
            key = (state, action)
            if key in q_function:
                q_values.append((action, q_function[key]))

        if q_values:
            best_action = max(q_values, key=lambda x: x[1])[0]
            policy[state] = best_action

    return policy


def compute_state_visitation(
    episode_trajectories: List[List[int]]
) -> Dict[int, int]:
    """
    计算状态访问频率。

    Args:
        episode_trajectories: 回合轨迹列表，每个轨迹是状态序列

    Returns:
        状态访问次数字典
    """
    visitation = defaultdict(int)

    for trajectory in episode_trajectories:
        for state in trajectory:
            visitation[state] += 1

    return dict(visitation)


def detect_convergence(
    rewards: List[float],
    window: int = 100,
    threshold: float = 0.01
) -> Tuple[bool, int]:
    """
    检测训练是否收敛。

    通过检查最近窗口内奖励的变化率来判断。

    Args:
        rewards: 奖励序列
        window: 检测窗口大小
        threshold: 变化率阈值

    Returns:
        (是否收敛, 收敛回合数)
    """
    if len(rewards) < 2 * window:
        return False, -1

    for i in range(window, len(rewards) - window):
        recent = rewards[i:i+window]
        previous = rewards[i-window:i]

        mean_recent = np.mean(recent)
        mean_previous = np.mean(previous)

        # 计算相对变化
        if abs(mean_previous) > 1e-8:
            change_rate = abs(mean_recent - mean_previous) / abs(mean_previous)
        else:
            change_rate = abs(mean_recent - mean_previous)

        if change_rate < threshold:
            return True, i

    return False, -1


# =============================================================================
# 序列化工具
# =============================================================================

def save_q_function(
    q_function: QFunction,
    filepath: str
) -> None:
    """
    保存Q函数到文件。

    Args:
        q_function: Q函数字典
        filepath: 文件路径
    """
    # 转换键为字符串以支持JSON
    serializable = {
        str(k): v for k, v in q_function.items()
    }

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(q_function, f)


def load_q_function(filepath: str) -> QFunction:
    """
    从文件加载Q函数。

    Args:
        filepath: 文件路径

    Returns:
        Q函数字典
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        # 转换键回元组
        return {
            eval(k): v for k, v in data.items()
        }
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def save_experiment_results(
    results: Dict[str, Any],
    filepath: str
) -> None:
    """
    保存实验结果。

    Args:
        results: 实验结果字典
        filepath: 文件路径
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 转换numpy数组为列表
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    serializable = convert(results)

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)


# =============================================================================
# 环境可视化辅助
# =============================================================================

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

    Args:
        q_function: Q函数
        height: 网格高度
        width: 网格宽度
        figsize: 图形尺寸
        title: 图形标题
        save_path: 保存路径
    """
    check_matplotlib()

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
    state = (3, 0)  # 起点
    path = [state]

    for _ in range(50):  # 防止无限循环
        state_idx = state[0] * width + state[1]

        # 找最优动作
        q_values = []
        for a in range(4):
            key = (state_idx, a)
            if key in q_function:
                q_values.append((a, q_function[key]))

        if not q_values:
            break

        best_action = max(q_values, key=lambda x: x[1])[0]

        # 计算下一状态
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        delta = deltas[best_action]
        new_state = (
            np.clip(state[0] + delta[0], 0, height - 1),
            np.clip(state[1] + delta[1], 0, width - 1)
        )

        # 检查是否到达终点
        if new_state == (3, 11):
            path.append(new_state)
            break

        # 检查是否掉入悬崖
        if new_state[0] == 3 and 1 <= new_state[1] <= 10:
            path.append(new_state)
            break

        state = new_state
        path.append(state)

    # 绘制路径
    for i, pos in enumerate(path):
        # 转换坐标（y轴翻转）
        x = pos[1] + 0.5
        y = height - pos[0] - 0.5

        if i == 0:
            continue  # 跳过起点（已标记）

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


# =============================================================================
# 单元测试
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("工具模块测试")
    print("=" * 60)

    # 测试数据
    np.random.seed(42)

    # 创建模拟指标
    mock_metrics = {
        'episode_rewards': list(np.random.randn(500).cumsum()),
        'episode_lengths': list(np.random.randint(10, 100, 500)),
        'td_errors': list(np.random.randn(10000) * 0.5),
    }

    mock_metrics_2 = {
        'episode_rewards': list(np.random.randn(500).cumsum() + 10),
        'episode_lengths': list(np.random.randint(8, 80, 500)),
        'td_errors': list(np.random.randn(10000) * 0.3),
    }

    # 测试训练曲线绘制
    print("\n1. 测试训练曲线绘制...")
    if MATPLOTLIB_AVAILABLE:
        plot_training_curves(
            [mock_metrics, mock_metrics_2],
            ['算法A', '算法B'],
            save_path='/tmp/test_training_curves.png'
        )
        print("   ✓ 训练曲线保存到 /tmp/test_training_curves.png")
    else:
        print("   ⊘ matplotlib不可用，跳过可视化测试")

    # 测试Q函数热力图
    print("\n2. 测试Q函数热力图...")
    mock_q = {
        (i, a): np.random.randn()
        for i in range(48)
        for a in range(4)
    }

    if MATPLOTLIB_AVAILABLE:
        plot_q_value_heatmap(
            mock_q,
            (4, 12),
            save_path='/tmp/test_q_heatmap.png'
        )
        print("   ✓ Q函数热力图保存到 /tmp/test_q_heatmap.png")

    # 测试RMSE计算
    print("\n3. 测试RMSE计算...")
    estimated = {i: np.random.randn() for i in range(10)}
    true_values = {i: np.random.randn() for i in range(10)}
    rmse = compute_rmse(estimated, true_values)
    print(f"   RMSE = {rmse:.4f}")

    # 测试策略提取
    print("\n4. 测试策略提取...")
    policy = extract_greedy_policy(mock_q, 48, 4)
    print(f"   提取了 {len(policy)} 个状态的策略")

    # 测试收敛检测
    print("\n5. 测试收敛检测...")
    converging_rewards = list(np.random.randn(200) * 10) + [50 + np.random.randn() for _ in range(300)]
    converged, episode = detect_convergence(converging_rewards)
    print(f"   收敛: {converged}, 回合: {episode}")

    # 测试序列化
    print("\n6. 测试Q函数序列化...")
    save_q_function(mock_q, '/tmp/test_q_function.pkl')
    loaded_q = load_q_function('/tmp/test_q_function.pkl')
    print(f"   保存/加载成功，键数量: {len(loaded_q)}")

    # 测试实验配置
    print("\n7. 测试实验配置...")
    config = ExperimentConfig(
        algorithm='sarsa',
        alpha=0.1,
        gamma=0.99,
        description='测试实验'
    )
    config_dict = config.to_dict()
    restored_config = ExperimentConfig.from_dict(config_dict)
    print(f"   配置序列化/反序列化: {restored_config.algorithm}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
