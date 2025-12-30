"""
可视化工具模块

提供训练曲线绘制和算法比较可视化。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_training_curves(
    episode_rewards: List[float],
    losses: Optional[List[float]] = None,
    epsilon_history: Optional[List[float]] = None,
    eval_rewards: Optional[List[tuple]] = None,
    title: str = "DQN训练进度",
    window: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional[Figure]:
    """
    绘制训练曲线
    
    生成四个子图：
    1. 回合奖励（带平滑）
    2. 训练损失
    3. 探索率衰减
    4. 评估性能
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib未安装，无法绘图")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # 1. 奖励曲线
    episodes = list(range(1, len(episode_rewards) + 1))
    axes[0, 0].plot(episodes, episode_rewards, alpha=0.3, color="blue", label="原始")
    
    if len(episode_rewards) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(episode_rewards, kernel, mode="valid")
        smoothed_x = list(range(window, len(episodes) + 1))
        axes[0, 0].plot(smoothed_x, smoothed, color="blue", linewidth=2, label=f"MA({window})")
    
    axes[0, 0].set_xlabel("回合")
    axes[0, 0].set_ylabel("总奖励")
    axes[0, 0].set_title("回合奖励")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 损失曲线
    if losses:
        axes[0, 1].plot(losses, alpha=0.5, color="red", linewidth=0.5)
        loss_window = min(100, len(losses) // 10) if len(losses) > 10 else 1
        if loss_window > 1:
            smoothed = np.convolve(losses, np.ones(loss_window) / loss_window, mode="valid")
            axes[0, 1].plot(range(loss_window - 1, len(losses)), smoothed, 
                          color="red", linewidth=2, label=f"MA({loss_window})")
            axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, "无损失数据", ha="center", va="center", fontsize=12)
    
    axes[0, 1].set_xlabel("回合")
    axes[0, 1].set_ylabel("损失")
    axes[0, 1].set_title("训练损失")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Epsilon衰减
    if epsilon_history:
        axes[1, 0].plot(epsilon_history, color="green", linewidth=2)
        axes[1, 0].set_ylim(0, 1.05)
    else:
        axes[1, 0].text(0.5, 0.5, "无epsilon数据", ha="center", va="center", fontsize=12)
    
    axes[1, 0].set_xlabel("回合")
    axes[1, 0].set_ylabel("Epsilon")
    axes[1, 0].set_title("探索率衰减")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 评估性能
    if eval_rewards:
        eval_episodes = [e[0] for e in eval_rewards]
        eval_means = [e[1] for e in eval_rewards]
        eval_stds = [e[2] for e in eval_rewards]
        axes[1, 1].errorbar(eval_episodes, eval_means, yerr=eval_stds,
                          fmt="o-", color="purple", capsize=3, markersize=6)
    else:
        axes[1, 1].text(0.5, 0.5, "无评估数据", ha="center", va="center", fontsize=12)
    
    axes[1, 1].set_xlabel("回合")
    axes[1, 1].set_ylabel("奖励")
    axes[1, 1].set_title("评估性能")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def plot_algorithm_comparison(
    results: Dict[str, List[List[float]]],
    title: str = "算法比较",
    window: int = 10,
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    绘制多算法比较图
    
    Parameters
    ----------
    results : Dict[str, List[List[float]]]
        算法名称到奖励列表的映射（每个算法多个种子）
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors
    
    # 学习曲线
    for idx, (name, rewards_list) in enumerate(results.items()):
        all_rewards = np.array(rewards_list)
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)
        
        if len(mean_rewards) >= window:
            kernel = np.ones(window) / window
            mean_smooth = np.convolve(mean_rewards, kernel, mode="valid")
            std_smooth = np.convolve(std_rewards, kernel, mode="valid")
            x = np.arange(window - 1, len(mean_rewards))
            
            axes[0].plot(x, mean_smooth, label=name, color=colors[idx], linewidth=2)
            axes[0].fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                               color=colors[idx], alpha=0.2)
    
    axes[0].set_xlabel("回合", fontsize=11)
    axes[0].set_ylabel("总奖励", fontsize=11)
    axes[0].set_title("学习曲线", fontsize=12)
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 最终性能柱状图
    names = list(results.keys())
    final_means = []
    final_stds = []
    
    for name in names:
        all_final = [np.mean(r[-50:]) for r in results[name]]
        final_means.append(np.mean(all_final))
        final_stds.append(np.std(all_final))
    
    x_pos = np.arange(len(names))
    bars = axes[1].bar(x_pos, final_means, yerr=final_stds, capsize=5,
                      color=[colors[i] for i in range(len(names))], alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(names, rotation=15, ha="right")
    axes[1].set_ylabel("平均奖励（最后50回合）", fontsize=11)
    axes[1].set_title("最终性能比较", fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()
    return fig
