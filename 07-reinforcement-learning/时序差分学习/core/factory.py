"""
算法工厂模块 (Factory Module)
============================

提供统一的算法创建接口。
"""

from typing import Optional

from .config import TDConfig
from .base import BaseTDLearner
from .td_prediction import TD0ValueLearner
from .td_control import SARSA, ExpectedSARSA, QLearning
from .advanced import (
    DoubleQLearning,
    NStepTD,
    TDLambda,
    SARSALambda,
    WatkinsQLambda,
)


def create_td_learner(
    algorithm: str,
    config: Optional[TDConfig] = None,
    **kwargs
) -> BaseTDLearner:
    """
    TD学习算法工厂函数。

    核心思想 (Core Idea):
    --------------------
    提供统一的接口创建各种TD算法实例，简化使用流程。
    支持通过算法名称字符串创建对应的学习器。

    Args:
        algorithm: 算法名称，支持以下选项:
            - 'td0': TD(0)状态价值学习
            - 'sarsa': SARSA (On-Policy TD控制)
            - 'expected_sarsa': Expected SARSA
            - 'q_learning': Q-Learning (Off-Policy TD控制)
            - 'double_q': Double Q-Learning
            - 'n_step': N-Step TD
            - 'td_lambda': TD(λ)
            - 'sarsa_lambda': SARSA(λ)
            - 'watkins_q_lambda': Watkins's Q(λ)

        config: TD学习配置对象，None则使用默认配置
        **kwargs: 传递给TDConfig的额外参数（当config为None时）

    Returns:
        对应的TD学习器实例

    Raises:
        ValueError: 当算法名称未知时

    Example:
        >>> # 使用默认配置
        >>> learner = create_td_learner('sarsa')

        >>> # 自定义参数
        >>> learner = create_td_learner('q_learning', alpha=0.1, gamma=0.99)

        >>> # 使用配置对象
        >>> config = TDConfig(alpha=0.5, lambda_=0.9)
        >>> learner = create_td_learner('td_lambda', config=config)
    """
    # 创建配置
    if config is None:
        config = TDConfig(**kwargs)

    # 算法名称到类的映射
    algorithm_map = {
        'td0': TD0ValueLearner,
        'sarsa': SARSA,
        'expected_sarsa': ExpectedSARSA,
        'q_learning': QLearning,
        'double_q': DoubleQLearning,
        'n_step': NStepTD,
        'td_lambda': TDLambda,
        'sarsa_lambda': SARSALambda,
        'watkins_q_lambda': WatkinsQLambda,
    }

    # 名称标准化（小写，支持多种写法）
    algorithm = algorithm.lower().replace('-', '_').replace(' ', '_')

    # 别名映射
    aliases = {
        'qlearning': 'q_learning',
        'double_qlearning': 'double_q',
        'doubleq': 'double_q',
        'nstep': 'n_step',
        'n_step_td': 'n_step',
        'tdlambda': 'td_lambda',
        'td_λ': 'td_lambda',
        'sarsalambda': 'sarsa_lambda',
        'sarsa_λ': 'sarsa_lambda',
        'watkins': 'watkins_q_lambda',
        'watkins_q': 'watkins_q_lambda',
    }

    algorithm = aliases.get(algorithm, algorithm)

    if algorithm not in algorithm_map:
        available = list(algorithm_map.keys())
        raise ValueError(
            f"未知算法: '{algorithm}'。\n"
            f"支持的算法: {available}"
        )

    return algorithm_map[algorithm](config)


# 算法信息查询
ALGORITHM_INFO = {
    'td0': {
        'name': 'TD(0)',
        'type': 'prediction',
        'policy_type': 'on-policy',
        'description': '最基础的单步时序差分预测算法',
    },
    'sarsa': {
        'name': 'SARSA',
        'type': 'control',
        'policy_type': 'on-policy',
        'description': 'On-Policy TD控制，学习行为策略的价值',
    },
    'expected_sarsa': {
        'name': 'Expected SARSA',
        'type': 'control',
        'policy_type': 'on-policy',
        'description': '使用期望消除动作采样方差的SARSA变体',
    },
    'q_learning': {
        'name': 'Q-Learning',
        'type': 'control',
        'policy_type': 'off-policy',
        'description': 'Off-Policy TD控制，直接学习最优价值函数',
    },
    'double_q': {
        'name': 'Double Q-Learning',
        'type': 'control',
        'policy_type': 'off-policy',
        'description': '解决最大化偏差的Q-Learning变体',
    },
    'n_step': {
        'name': 'N-Step TD',
        'type': 'control',
        'policy_type': 'on-policy',
        'description': '使用n步回报的TD方法',
    },
    'td_lambda': {
        'name': 'TD(λ)',
        'type': 'control',
        'policy_type': 'flexible',
        'description': '通过资格迹统一TD(0)和Monte Carlo',
    },
    'sarsa_lambda': {
        'name': 'SARSA(λ)',
        'type': 'control',
        'policy_type': 'on-policy',
        'description': 'On-Policy的TD(λ)控制算法',
    },
    'watkins_q_lambda': {
        'name': "Watkins's Q(λ)",
        'type': 'control',
        'policy_type': 'off-policy',
        'description': 'Off-Policy安全的TD(λ)算法',
    },
}


def list_algorithms() -> None:
    """打印所有可用算法及其简介。"""
    print("=" * 60)
    print("可用的TD学习算法")
    print("=" * 60)

    for algo_id, info in ALGORITHM_INFO.items():
        print(f"\n{info['name']} ('{algo_id}')")
        print(f"  类型: {info['type']}")
        print(f"  策略: {info['policy_type']}")
        print(f"  说明: {info['description']}")

    print("\n" + "=" * 60)
