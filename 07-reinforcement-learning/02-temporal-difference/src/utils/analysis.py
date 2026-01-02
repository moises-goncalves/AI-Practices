"""
分析工具 (Analysis Utilities)
============================
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np


def compute_rmse(estimated: Dict[Any, float], true_values: Dict[Any, float]) -> float:
    """计算均方根误差。"""
    common = set(estimated.keys()) & set(true_values.keys())
    if not common:
        return float('inf')
    errors = [(estimated[s] - true_values[s])**2 for s in common]
    return np.sqrt(np.mean(errors))


def extract_policy(q_function: Dict[Tuple[Any, int], float], n_actions: int = 4) -> Dict[Any, int]:
    """从Q函数提取贪婪策略。"""
    states = set(s for s, a in q_function.keys())
    policy = {}
    for state in states:
        q_vals = [q_function.get((state, a), 0.0) for a in range(n_actions)]
        policy[state] = int(np.argmax(q_vals))
    return policy


def compare_algorithms(
    results: Dict[str, List[float]],
    window: int = 100
) -> Dict[str, Dict[str, float]]:
    """比较多个算法的性能。"""
    comparison = {}
    for name, rewards in results.items():
        comparison[name] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'max': np.max(rewards),
            'final_100_mean': np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
        }
    return comparison
