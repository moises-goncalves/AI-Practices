"""
实验管理模块 (Experiment Management Module)
==========================================

核心思想 (Core Idea):
--------------------
提供实验管理工具，包括配置管理、多种子实验、超参数搜索等功能。
确保实验的可复现性和结果的统计可靠性。

设计原则:
--------
- 配置与实现分离
- 支持多种子统计
- 完整的实验记录
- 易于扩展的超参数搜索

数学原理 (Mathematical Theory):
------------------------------
多种子实验的统计分析:
1. 均值估计: μ̂ = (1/n) × Σ_{i=1}^n X_i
2. 标准误差: SE = σ / √n
3. 置信区间: [μ̂ - z_{α/2}×SE, μ̂ + z_{α/2}×SE]

超参数搜索: 网格搜索或随机搜索
"""

from __future__ import annotations

import numpy as np
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from itertools import product


@dataclass
class ExperimentConfig:
    """
    实验配置类。

    核心思想 (Core Idea):
    --------------------
    封装实验的所有超参数，支持序列化和反序列化。
    便于实验管理和结果复现。

    Attributes:
        algorithm: 算法名称
        alpha: 学习率
        gamma: 折扣因子
        epsilon: 探索率
        lambda_: 资格迹衰减参数
        n_step: N步TD的步数
        n_episodes: 训练回合数
        n_seeds: 随机种子数量
        env_name: 环境名称
        description: 实验描述

    Example:
        >>> config = ExperimentConfig(
        ...     algorithm='sarsa',
        ...     alpha=0.1,
        ...     gamma=0.99,
        ...     description='测试SARSA在CliffWalking上的表现'
        ... )
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
    max_steps_per_episode: int = 10000

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """从字典创建配置。"""
        return cls(**data)


@dataclass
class ExperimentResult:
    """
    实验结果类。

    核心思想 (Core Idea):
    --------------------
    存储单次实验的完整结果，包括配置、指标、学习到的值函数等。

    Attributes:
        config: 实验配置
        metrics: 训练指标
        q_function: 学习到的Q函数
        value_function: 学习到的状态价值函数
        training_time: 训练耗时
        seed: 随机种子
    """
    config: ExperimentConfig
    metrics: Dict[str, Any]
    q_function: Optional[Dict[Tuple[Any, Any], float]] = None
    value_function: Optional[Dict[Any, float]] = None
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

    def get_final_reward(self, window: int = 100) -> float:
        """获取最后window个回合的平均奖励。"""
        rewards = self.metrics.get('episode_rewards', [])
        if len(rewards) < window:
            return float(np.mean(rewards)) if rewards else 0.0
        return float(np.mean(rewards[-window:]))


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

    核心思想 (Core Idea):
    --------------------
    通过多个随机种子评估算法的稳定性，计算均值和标准差。
    确保结果的统计可靠性。

    数学原理 (Mathematical Theory):
    ------------------------------
    给定n个种子的结果 {R_1, ..., R_n}:
    - 均值: μ̂ = (1/n) × Σ R_i
    - 标准差: σ̂ = √((1/(n-1)) × Σ(R_i - μ̂)²)
    - 95%置信区间: μ̂ ± 1.96 × σ̂/√n

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
            'total_time': float,           # 总耗时
        }

    Example:
        >>> results = run_multi_seed_experiment(
        ...     lambda: SARSA(TDConfig(alpha=0.1)),
        ...     lambda: CliffWalkingEnv(),
        ...     n_episodes=500,
        ...     n_seeds=10
        ... )
        >>> print(f"平均最终奖励: {np.mean(results['mean_reward'][-100:]):.2f}")
    """
    all_rewards = []
    all_lengths = []
    start_time = time.time()

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

    total_time = time.time() - start_time

    return {
        'rewards': all_rewards,
        'lengths': all_lengths,
        'mean_reward': np.mean(rewards_array, axis=0),
        'std_reward': np.std(rewards_array, axis=0),
        'mean_length': np.mean(lengths_array, axis=0),
        'std_length': np.std(lengths_array, axis=0),
        'total_time': total_time,
    }


def run_hyperparameter_search(
    learner_class: type,
    env_factory: Callable[[], Any],
    param_grid: Dict[str, List[Any]],
    n_episodes: int = 500,
    n_seeds: int = 3,
    metric: str = 'final_reward',
    higher_is_better: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    运行超参数网格搜索。

    核心思想 (Core Idea):
    --------------------
    遍历参数组合，找出最优超参数配置。
    每个配置运行多个种子以确保结果稳定。

    数学原理 (Mathematical Theory):
    ------------------------------
    网格搜索穷举所有参数组合:
        Θ* = argmax_{θ∈Θ} (1/n) × Σ_{i=1}^n Performance(θ, seed_i)

    总实验数 = |param_1| × |param_2| × ... × |param_k| × n_seeds

    Args:
        learner_class: 学习器类
        env_factory: 环境工厂函数
        param_grid: 参数网格 {'param_name': [value1, value2, ...]}
        n_episodes: 每次实验的回合数
        n_seeds: 每个配置的种子数
        metric: 评估指标 ('final_reward', 'final_length', 'convergence_speed')
        higher_is_better: 指标越高越好
        verbose: 是否打印进度

    Returns:
        {
            'best_params': Dict[str, Any],
            'best_score': float,
            'all_results': List[Dict],
            'total_time': float,
        }

    Example:
        >>> param_grid = {
        ...     'alpha': [0.1, 0.3, 0.5],
        ...     'epsilon': [0.05, 0.1, 0.2],
        ... }
        >>> results = run_hyperparameter_search(
        ...     SARSA,
        ...     lambda: CliffWalkingEnv(),
        ...     param_grid,
        ...     n_episodes=300,
        ...     n_seeds=3
        ... )
        >>> print(f"最优参数: {results['best_params']}")
    """
    from ..core import TDConfig  # 避免循环导入

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    all_results = []
    best_score = float('-inf') if higher_is_better else float('inf')
    best_params = None
    start_time = time.time()

    total_combinations = len(all_combinations)

    for idx, combination in enumerate(all_combinations):
        params = dict(zip(param_names, combination))

        if verbose:
            print(f"\n[{idx + 1}/{total_combinations}] 测试参数: {params}")

        # 创建配置
        config_params = {
            'alpha': params.get('alpha', 0.1),
            'gamma': params.get('gamma', 0.99),
            'epsilon': params.get('epsilon', 0.1),
            'lambda_': params.get('lambda_', 0.0),
        }
        config = TDConfig(**{k: v for k, v in config_params.items()
                             if k in config_params})

        # 多种子运行
        seed_scores = []
        for seed in range(n_seeds):
            np.random.seed(seed)

            learner = learner_class(config)
            env = env_factory()

            metrics = learner.train(
                env,
                n_episodes=n_episodes,
                max_steps_per_episode=10000,
                log_interval=n_episodes + 1
            )

            # 计算得分
            if metric == 'final_reward':
                score = float(np.mean(metrics.episode_rewards[-100:]))
            elif metric == 'final_length':
                score = float(np.mean(metrics.episode_lengths[-100:]))
            elif metric == 'convergence_speed':
                # 找到首次达到目标的回合
                target = np.mean(metrics.episode_rewards[-50:]) * 0.9
                converged_idx = n_episodes
                for i, r in enumerate(metrics.episode_rewards):
                    if np.mean(metrics.episode_rewards[max(0, i-20):i+1]) >= target:
                        converged_idx = i
                        break
                score = -converged_idx  # 越早越好
            else:
                score = float(np.mean(metrics.episode_rewards[-100:]))

            seed_scores.append(score)

            if hasattr(env, 'close'):
                env.close()

        mean_score = np.mean(seed_scores)
        std_score = np.std(seed_scores)

        result = {
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'seed_scores': seed_scores,
        }
        all_results.append(result)

        if verbose:
            print(f"   得分: {mean_score:.2f} ± {std_score:.2f}")

        # 更新最优
        if higher_is_better:
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
        else:
            if mean_score < best_score:
                best_score = mean_score
                best_params = params.copy()

    total_time = time.time() - start_time

    if verbose:
        print(f"\n搜索完成! 总耗时: {total_time:.1f}秒")
        print(f"最优参数: {best_params}")
        print(f"最优得分: {best_score:.2f}")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results,
        'total_time': total_time,
    }
