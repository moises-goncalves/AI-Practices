"""
时序差分学习核心模块 (Temporal Difference Learning Core Module)
================================================================

模块结构:
--------
- config.py: 配置类和超参数管理
- base.py: 基类定义、协议、类型
- td_prediction.py: TD预测算法 (TD(0))
- td_control.py: TD控制算法 (SARSA, Q-Learning, Expected SARSA)
- advanced.py: 高级算法 (Double Q-Learning, N-Step TD, TD(λ))
- factory.py: 算法工厂函数
"""

from .config import TDConfig, TrainingMetrics, EligibilityTraceType
from .base import BaseTDLearner, Environment, Policy
from .td_prediction import TD0ValueLearner
from .td_control import SARSA, ExpectedSARSA, QLearning
from .advanced import DoubleQLearning, NStepTD, TDLambda, SARSALambda, WatkinsQLambda
from .factory import create_td_learner

__all__ = [
    "TDConfig", "TrainingMetrics", "EligibilityTraceType",
    "BaseTDLearner", "Environment", "Policy",
    "TD0ValueLearner",
    "SARSA", "ExpectedSARSA", "QLearning",
    "DoubleQLearning", "NStepTD", "TDLambda", "SARSALambda", "WatkinsQLambda",
    "create_td_learner",
]
