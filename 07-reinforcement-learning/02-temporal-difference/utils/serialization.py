"""
序列化工具模块 (Serialization Module)
====================================

核心思想 (Core Idea):
--------------------
提供模型和实验结果的保存/加载功能，
支持JSON和Pickle两种格式。

设计原则:
--------
- JSON用于可读性和跨平台兼容性
- Pickle用于完整Python对象序列化
- 自动创建目录
- 清晰的错误处理
"""

from __future__ import annotations

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union


# 类型别名
QFunction = Dict[Tuple[Any, Any], float]
ValueFunction = Dict[Any, float]


def save_q_function(
    q_function: QFunction,
    filepath: str
) -> None:
    """
    保存Q函数到文件。

    核心思想 (Core Idea):
    --------------------
    将Q函数序列化保存，支持JSON（可读）和Pickle（完整）格式。

    Args:
        q_function: Q函数字典 {(state, action): value}
        filepath: 文件路径
            - .json: 保存为JSON格式（键转为字符串）
            - 其他: 保存为Pickle格式

    Example:
        >>> save_q_function(learner.q_table, 'models/q_function.pkl')
    """
    # 确保目录存在
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if filepath.endswith('.json'):
        # JSON格式：键必须是字符串
        serializable = {
            str(k): float(v) for k, v in q_function.items()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    else:
        # Pickle格式：保留完整类型
        with open(filepath, 'wb') as f:
            pickle.dump(q_function, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_q_function(filepath: str) -> QFunction:
    """
    从文件加载Q函数。

    Args:
        filepath: 文件路径

    Returns:
        Q函数字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误

    Example:
        >>> q_function = load_q_function('models/q_function.pkl')
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 将字符串键转回元组
        q_function = {}
        for k, v in data.items():
            try:
                # 尝试将字符串解析为元组
                key = eval(k)  # 安全警告：仅用于受信任的文件
                q_function[key] = float(v)
            except Exception:
                q_function[k] = float(v)
        return q_function
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def save_value_function(
    value_function: ValueFunction,
    filepath: str
) -> None:
    """
    保存状态价值函数到文件。

    Args:
        value_function: 状态价值函数 {state: value}
        filepath: 文件路径
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if filepath.endswith('.json'):
        serializable = {
            str(k): float(v) for k, v in value_function.items()
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(value_function, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_value_function(filepath: str) -> ValueFunction:
    """
    从文件加载状态价值函数。

    Args:
        filepath: 文件路径

    Returns:
        状态价值函数字典
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        value_function = {}
        for k, v in data.items():
            try:
                key = int(k) if k.isdigit() else eval(k)
                value_function[key] = float(v)
            except Exception:
                value_function[k] = float(v)
        return value_function
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def save_experiment_results(
    results: Dict[str, Any],
    filepath: str
) -> None:
    """
    保存实验结果到JSON文件。

    核心思想 (Core Idea):
    --------------------
    将实验结果转换为可序列化格式并保存，
    便于后续分析和比较。

    Args:
        results: 实验结果字典
        filepath: 文件路径

    Example:
        >>> results = run_multi_seed_experiment(...)
        >>> save_experiment_results(results, 'results/sarsa_experiment.json')
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj: Any) -> Any:
        """递归转换为可序列化类型。"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    serializable = convert(results)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def load_experiment_results(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载实验结果。

    Args:
        filepath: 文件路径

    Returns:
        实验结果字典

    Example:
        >>> results = load_experiment_results('results/sarsa_experiment.json')
        >>> print(f"平均奖励: {np.mean(results['mean_reward'][-100:]):.2f}")
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将列表转换为numpy数组（如果适用）
    def convert_arrays(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # 检查是否是数值列表
            if obj and all(isinstance(x, (int, float)) for x in obj):
                return np.array(obj)
            return [convert_arrays(v) for v in obj]
        return obj

    return convert_arrays(data)


def save_learner(
    learner: Any,
    filepath: str,
    save_metrics: bool = True
) -> None:
    """
    保存学习器完整状态。

    核心思想 (Core Idea):
    --------------------
    保存学习器的所有状态，包括Q函数、配置和指标，
    便于后续恢复训练或部署。

    Args:
        learner: 学习器实例
        filepath: 文件路径
        save_metrics: 是否保存训练指标

    Example:
        >>> save_learner(sarsa_learner, 'models/sarsa_checkpoint.pkl')
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 构建状态字典
    state = {
        'class_name': learner.__class__.__name__,
        'q_table': dict(learner.q_table),
        'config': learner.config.__dict__ if hasattr(learner.config, '__dict__') else learner.config,
    }

    # 可选保存指标
    if save_metrics and hasattr(learner, 'metrics'):
        state['metrics'] = {
            'episode_rewards': learner.metrics.episode_rewards,
            'episode_lengths': learner.metrics.episode_lengths,
            'td_errors': learner.metrics.td_errors,
            'value_changes': learner.metrics.value_changes,
        }

    with open(filepath, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_learner(
    filepath: str,
    learner_class: Optional[type] = None
) -> Union[Dict[str, Any], Any]:
    """
    从文件加载学习器状态。

    Args:
        filepath: 文件路径
        learner_class: 学习器类（如果提供则创建实例）

    Returns:
        如果提供learner_class，返回学习器实例
        否则返回状态字典

    Example:
        >>> # 方式1：仅加载状态
        >>> state = load_learner('models/sarsa_checkpoint.pkl')
        >>> # 方式2：恢复学习器
        >>> learner = load_learner('models/sarsa_checkpoint.pkl', SARSA)
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    with open(filepath, 'rb') as f:
        state = pickle.load(f)

    if learner_class is None:
        return state

    # 尝试恢复学习器
    from ..core import TDConfig

    # 创建配置
    config_data = state.get('config', {})
    config = TDConfig(**{k: v for k, v in config_data.items()
                         if k in TDConfig.__dataclass_fields__})

    # 创建学习器
    learner = learner_class(config)

    # 恢复Q表
    learner.q_table.update(state.get('q_table', {}))

    # 恢复指标
    if 'metrics' in state and hasattr(learner, 'metrics'):
        metrics_data = state['metrics']
        learner.metrics.episode_rewards = metrics_data.get('episode_rewards', [])
        learner.metrics.episode_lengths = metrics_data.get('episode_lengths', [])
        learner.metrics.td_errors = metrics_data.get('td_errors', [])
        learner.metrics.value_changes = metrics_data.get('value_changes', [])

    return learner


def export_to_numpy(
    q_function: QFunction,
    n_states: int,
    n_actions: int,
    filepath: Optional[str] = None
) -> np.ndarray:
    """
    将Q函数导出为NumPy数组。

    核心思想 (Core Idea):
    --------------------
    将稀疏的字典表示转换为稠密的数组表示，
    便于与深度学习框架集成。

    Args:
        q_function: Q函数字典
        n_states: 状态数量
        n_actions: 动作数量
        filepath: 可选保存路径

    Returns:
        形状为(n_states, n_actions)的NumPy数组

    Example:
        >>> q_array = export_to_numpy(q_function, 48, 4, 'models/q_table.npy')
        >>> print(q_array.shape)  # (48, 4)
    """
    q_array = np.zeros((n_states, n_actions))

    for (state, action), value in q_function.items():
        if isinstance(state, int) and 0 <= state < n_states:
            if 0 <= action < n_actions:
                q_array[state, action] = value

    if filepath:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, q_array)

    return q_array


def import_from_numpy(
    filepath: str,
    default_value: float = 0.0
) -> QFunction:
    """
    从NumPy数组导入Q函数。

    Args:
        filepath: NumPy文件路径
        default_value: 默认Q值（用于过滤）

    Returns:
        Q函数字典

    Example:
        >>> q_function = import_from_numpy('models/q_table.npy')
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    q_array = np.load(filepath)
    n_states, n_actions = q_array.shape

    q_function = {}
    for state in range(n_states):
        for action in range(n_actions):
            value = q_array[state, action]
            if value != default_value:  # 只存储非默认值
                q_function[(state, action)] = float(value)

    return q_function
