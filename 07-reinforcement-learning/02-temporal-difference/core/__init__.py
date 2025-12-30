"""
时序差分学习核心模块 (Temporal Difference Learning Core Module)
================================================================

本模块提供TD学习的核心抽象和基础设施。

模块结构:
--------
- base.py: 基类定义、协议、类型
- config.py: 配置类和超参数管理
- td_prediction.py: TD预测算法 (TD(0), TD(λ))
- td_control.py: TD控制算法 (SARSA, Q-Learning, Expected SARSA)
- advanced.py: 高级算法 (Double Q-Learning, N-Step TD, Watkins Q(λ))
"""

from .config import (
    TDConfig,
    TrainingMetrics,
    EligibilityTraceType,
)

from .base import (
    BaseTDLearner,
    Environment,
    Policy,
)

from .td_prediction import (
    TD0ValueLearner,
)

from .td_control import (
    SARSA,
    ExpectedSARSA,
    QLearning,
)

from .advanced import (
    DoubleQLearning,
    NStepTD,
    TDLambda,
    SARSALambda,
    WatkinsQLambda,
)

from .factory import create_td_learner

__all__ = [
    # 配置
    "TDConfig",
    "TrainingMetrics",
    "EligibilityTraceType",
    # 基类
    "BaseTDLearner",
    "Environment",
    "Policy",
    # 预测算法
    "TD0ValueLearner",
    # 控制算法
    "SARSA",
    "ExpectedSARSA",
    "QLearning",
    # 高级算法
    "DoubleQLearning",
    "NStepTD",
    "TDLambda",
    "SARSALambda",
    "WatkinsQLambda",
    # 工厂函数
    "create_td_learner",
]
