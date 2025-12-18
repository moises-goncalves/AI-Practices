"""
分析工具模块 (Analysis Module)
=============================

核心思想 (Core Idea):
--------------------
提供算法分析工具，包括收敛检测、误差计算、策略提取等功能。
帮助评估和理解TD学习算法的性能。

数学原理 (Mathematical Theory):
------------------------------
误差度量:
- RMSE = √((1/n) × Σ(V̂(s) - V(s))²)
- MAE = (1/n) × Σ|V̂(s) - V(s)|

收敛判定:
- 基于价值变化: max_s |V_{t+1}(s) - V_t(s)| < ε
- 基于性能稳定: |E[R]_{recent} - E[R]_{previous}| / |E[R]_{previous}| < ε
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict


# 类型别名
QFunction = Dict[Tuple[Any, Any], float]
ValueFunction = Dict[Any, float]


def compute_rmse(
    estimated: Union[ValueFunction, QFunction],
    true_values: Union[ValueFunction, QFunction]
) -> float:
    """
    计算估计值与真实值的均方根误差。

    核心思想 (Core Idea):
    --------------------
    RMSE是评估价值函数估计精度的标准指标，
    衡量估计值与真实值的平均偏差。

    数学原理 (Mathematical Theory):
    ------------------------------
    均方根误差 (Root Mean Square Error):
        RMSE = √((1/n) × Σ_{s∈S}(V̂(s) - V(s))²)

    特性:
    - 对大误差敏感（平方放大）
    - 与原值同量纲
    - 总是非负

    Args:
        estimated: 估计的价值函数
        true_values: 真实价值函数

    Returns:
        均方根误差

    Example:
        >>> estimated = {0: 0.5, 1: 0.8, 2: 0.9}
        >>> true_values = {0: 0.5, 1: 0.7, 2: 1.0}
        >>> rmse = compute_rmse(estimated, true_values)
        >>> print(f"RMSE = {rmse:.4f}")
    """
    common_keys = set(estimated.keys()) & set(true_values.keys())

    if len(common_keys) == 0:
        return float('inf')

    squared_errors = [
        (estimated[k] - true_values[k]) ** 2
        for k in common_keys
    ]

    return float(np.sqrt(np.mean(squared_errors)))


def compute_mae(
    estimated: Union[ValueFunction, QFunction],
    true_values: Union[ValueFunction, QFunction]
) -> float:
    """
    计算平均绝对误差。

    数学原理 (Mathematical Theory):
    ------------------------------
    平均绝对误差 (Mean Absolute Error):
        MAE = (1/n) × Σ_{s∈S}|V̂(s) - V(s)|

    Args:
        estimated: 估计的价值函数
        true_values: 真实价值函数

    Returns:
        平均绝对误差
    """
    common_keys = set(estimated.keys()) & set(true_values.keys())

    if len(common_keys) == 0:
        return float('inf')

    absolute_errors = [
        abs(estimated[k] - true_values[k])
        for k in common_keys
    ]

    return float(np.mean(absolute_errors))


def extract_greedy_policy(
    q_function: QFunction,
    n_states: int,
    n_actions: int
) -> Dict[int, int]:
    """
    从Q函数提取贪婪策略。

    核心思想 (Core Idea):
    --------------------
    贪婪策略选择每个状态下Q值最高的动作。

    数学原理 (Mathematical Theory):
    ------------------------------
    贪婪策略:
        π(s) = argmax_a Q(s, a)

    这是确定性策略，不包含探索。

    Args:
        q_function: Q函数 {(state, action): value}
        n_states: 状态数量
        n_actions: 动作数量

    Returns:
        策略字典 {state: action}

    Example:
        >>> q_function = {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 1.5}
        >>> policy = extract_greedy_policy(q_function, n_states=2, n_actions=2)
        >>> print(policy)  # {0: 1, 1: 0}
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

    核心思想 (Core Idea):
    --------------------
    统计各状态被访问的次数，帮助理解探索的均匀性。

    数学原理 (Mathematical Theory):
    ------------------------------
    状态访问频率:
        N(s) = Σ_{τ∈Trajectories} Σ_{t} 𝟙[S_t = s]

    均匀探索应使各状态访问频率接近。

    Args:
        episode_trajectories: 回合轨迹列表，每个轨迹是状态序列

    Returns:
        状态访问次数字典 {state: count}
    """
    visitation = defaultdict(int)

    for trajectory in episode_trajectories:
        for state in trajectory:
            visitation[state] += 1

    return dict(visitation)


def detect_convergence(
    rewards: List[float],
    window: int = 100,
    threshold: float = 0.01,
    min_episodes: int = 200
) -> Tuple[bool, int]:
    """
    检测训练是否收敛。

    核心思想 (Core Idea):
    --------------------
    通过检查最近窗口内奖励的变化率来判断收敛。
    当变化率低于阈值时认为已收敛。

    数学原理 (Mathematical Theory):
    ------------------------------
    收敛判定标准:
        |E[R]_{recent} - E[R]_{previous}| / |E[R]_{previous}| < threshold

    这测量相对变化，对不同量级的奖励都适用。

    Args:
        rewards: 奖励序列
        window: 检测窗口大小
        threshold: 变化率阈值
        min_episodes: 最少需要的回合数

    Returns:
        (是否收敛, 收敛回合数)

    Example:
        >>> rewards = list(np.random.randn(200) * 10) + [50] * 300
        >>> converged, episode = detect_convergence(rewards)
        >>> print(f"收敛: {converged}, 回合: {episode}")
    """
    if len(rewards) < min_episodes:
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


def evaluate_policy(
    policy: Dict[int, int],
    env: Any,
    n_episodes: int = 100,
    max_steps: int = 1000,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    评估确定性策略的性能。

    核心思想 (Core Idea):
    --------------------
    通过多次模拟来估计策略的期望回报和方差。

    数学原理 (Mathematical Theory):
    ------------------------------
    策略评估估计:
        V̂(π) = (1/N) × Σ_{i=1}^N G_i

    其中 G_i 是第i个回合的累积回报。

    Args:
        policy: 策略字典 {state: action}
        env: 环境实例
        n_episodes: 评估回合数
        max_steps: 每回合最大步数
        seed: 随机种子

    Returns:
        (平均回报, 回报标准差, 平均步数)

    Example:
        >>> policy = extract_greedy_policy(q_function, n_states, n_actions)
        >>> mean_return, std_return, mean_steps = evaluate_policy(policy, env)
        >>> print(f"平均回报: {mean_return:.2f} ± {std_return:.2f}")
    """
    if seed is not None:
        np.random.seed(seed)

    episode_returns = []
    episode_lengths = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_return = 0.0
        steps = 0

        for _ in range(max_steps):
            action = policy.get(state, 0)  # 默认动作0
            next_state, reward, terminated, truncated, _ = env.step(action)

            total_return += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        episode_returns.append(total_return)
        episode_lengths.append(steps)

    return (
        float(np.mean(episode_returns)),
        float(np.std(episode_returns)),
        float(np.mean(episode_lengths))
    )


def compute_action_value_from_state_value(
    state_value: ValueFunction,
    env: Any,
    gamma: float = 0.99
) -> QFunction:
    """
    从状态价值函数计算动作价值函数。

    数学原理 (Mathematical Theory):
    ------------------------------
    动作价值与状态价值的关系:
        Q(s,a) = Σ_{s'} P(s'|s,a) × [R(s,a,s') + γV(s')]

    对于确定性环境简化为:
        Q(s,a) = R(s,a) + γV(s')

    Args:
        state_value: 状态价值函数
        env: 环境实例（需要提供转移信息）
        gamma: 折扣因子

    Returns:
        动作价值函数
    """
    q_function = {}

    # 获取环境参数
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    for state in range(n_states):
        for action in range(n_actions):
            # 模拟转移
            env._state = env._state_to_pos(state) if hasattr(env, '_state_to_pos') else state
            original_state = env._state

            next_state, reward, _, _, _ = env.step(action)

            # 计算Q值
            next_v = state_value.get(next_state, 0.0)
            q_function[(state, action)] = reward + gamma * next_v

            # 恢复状态
            env._state = original_state

    return q_function


def analyze_exploration_coverage(
    q_function: QFunction,
    n_states: int,
    n_actions: int
) -> Dict[str, Any]:
    """
    分析探索覆盖率。

    核心思想 (Core Idea):
    --------------------
    检查Q函数覆盖了多少状态-动作对，
    评估探索的充分性。

    Args:
        q_function: Q函数
        n_states: 状态数量
        n_actions: 动作数量

    Returns:
        分析结果字典
    """
    total_pairs = n_states * n_actions
    covered_pairs = len(q_function)

    # 统计每个状态覆盖的动作数
    state_coverage = defaultdict(int)
    for (state, action) in q_function.keys():
        state_coverage[state] += 1

    fully_covered_states = sum(1 for count in state_coverage.values()
                               if count == n_actions)
    partially_covered_states = len(state_coverage) - fully_covered_states
    uncovered_states = n_states - len(state_coverage)

    return {
        'total_pairs': total_pairs,
        'covered_pairs': covered_pairs,
        'coverage_ratio': covered_pairs / total_pairs if total_pairs > 0 else 0.0,
        'fully_covered_states': fully_covered_states,
        'partially_covered_states': partially_covered_states,
        'uncovered_states': uncovered_states,
        'state_coverage': dict(state_coverage),
    }
