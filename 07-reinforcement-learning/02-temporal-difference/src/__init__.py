"""
Temporal Difference Learning - 时序差分学习

核心组件:
    - TD预测: TD(0)值函数学习
    - TD控制: SARSA, Q-Learning, Expected SARSA
    - 高级算法: Double Q-Learning, N-Step TD, TD(λ)
    - 环境: GridWorld, CliffWalking, WindyGridWorld, RandomWalk
    - 工具: 可视化与分析

数学基础:
    TD更新规则: V(s) ← V(s) + α[r + γV(s') - V(s)]
    其中 δ = r + γV(s') - V(s) 为TD误差
"""

__version__ = "1.0.0"
__author__ = "AI-Practices"

from .core import (
    TDConfig, TrainingMetrics, EligibilityTraceType,
    BaseTDLearner, Environment, Policy,
    TD0ValueLearner,
    SARSA, ExpectedSARSA, QLearning,
    DoubleQLearning, NStepTD, TDLambda, SARSALambda, WatkinsQLambda,
    create_td_learner,
)

from .environments import (
    GridWorld, GridWorldConfig,
    CliffWalkingEnv,
    WindyGridWorld,
    RandomWalk,
)

from .utils import (
    plot_training_curve, plot_value_function, plot_policy,
    compute_rmse, extract_policy, compare_algorithms,
)

__all__ = [
    # Config
    "TDConfig", "TrainingMetrics", "EligibilityTraceType",
    # Base
    "BaseTDLearner", "Environment", "Policy",
    # Prediction
    "TD0ValueLearner",
    # Control
    "SARSA", "ExpectedSARSA", "QLearning",
    # Advanced
    "DoubleQLearning", "NStepTD", "TDLambda", "SARSALambda", "WatkinsQLambda",
    # Factory
    "create_td_learner",
    # Environments
    "GridWorld", "GridWorldConfig", "CliffWalkingEnv", "WindyGridWorld", "RandomWalk",
    # Utils
    "plot_training_curve", "plot_value_function", "plot_policy",
    "compute_rmse", "extract_policy", "compare_algorithms",
]
