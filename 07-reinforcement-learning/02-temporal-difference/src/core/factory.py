"""
TD算法工厂函数 (TD Algorithm Factory)
====================================
"""

from typing import Optional
from .config import TDConfig
from .base import BaseTDLearner
from .td_prediction import TD0ValueLearner
from .td_control import SARSA, ExpectedSARSA, QLearning
from .advanced import DoubleQLearning, NStepTD, TDLambda, SARSALambda, WatkinsQLambda


def create_td_learner(algorithm: str, config: Optional[TDConfig] = None, **kwargs) -> BaseTDLearner:
    """
    TD学习算法工厂函数。
    
    Args:
        algorithm: 算法名称
            - 'td0': TD(0)状态价值学习
            - 'sarsa': SARSA
            - 'expected_sarsa': Expected SARSA
            - 'q_learning': Q-Learning
            - 'double_q': Double Q-Learning
            - 'n_step': N-Step TD
            - 'td_lambda': TD(λ)
            - 'sarsa_lambda': SARSA(λ)
            - 'watkins_q_lambda': Watkins Q(λ)
        config: TD配置，None使用默认
        **kwargs: 传递给TDConfig的参数
    
    Returns:
        对应的TD学习器实例
    """
    if config is None:
        config = TDConfig(**kwargs)

    algorithms = {
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

    algorithm = algorithm.lower()
    if algorithm not in algorithms:
        raise ValueError(f"未知算法: {algorithm}. 支持: {list(algorithms.keys())}")
    return algorithms[algorithm](config)
