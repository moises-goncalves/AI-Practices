"""
Actor-Critic Methods - Actor-Critic方法

核心组件:
    - Algorithms: REINFORCE, A2C, PPO
    - Networks: Discrete/Continuous Policy, Value/Q Networks, ActorCriticNetwork
    - Core: 配置、类型定义、枚举
    - Utils: 训练工具、可视化、学习率调度

数学基础:
    Actor-Critic结合了策略梯度和值函数方法:
    - Actor: 策略 π_θ(a|s) 负责选择动作
    - Critic: 价值函数 V_φ(s) 评估状态好坏
    - 优势函数: A(s,a) = Q(s,a) - V(s)
"""

__version__ = "1.0.0"
__author__ = "AI-Practices"

from .algorithms import (
    REINFORCE,
    A2C,
    PPO,
)

from .core import (
    PolicyGradientConfig,
    Trajectory,
    Transition,
    TrainingMetrics,
    PolicyType,
    AdvantageEstimator,
    NetworkArchitecture,
)

from .networks import (
    MLP,
    init_weights,
    get_activation,
    DiscretePolicy,
    ContinuousPolicy,
    SquashedGaussianPolicy,
    ValueNetwork,
    QNetwork,
    ActorCriticNetwork,
)

from .utils import (
    set_seed,
    RunningMeanStd,
    LearningRateScheduler,
    MetricsTracker,
    Checkpointer,
    compute_explained_variance,
    polyak_update,
    smooth_data,
    plot_training_curves,
    plot_comparison,
    plot_policy_distribution,
    plot_value_function_1d,
    plot_value_function_2d,
    create_training_dashboard,
    LivePlotter,
)

__all__ = [
    # Algorithms
    "REINFORCE", "A2C", "PPO",
    # Core
    "PolicyGradientConfig", "Trajectory", "Transition", "TrainingMetrics",
    "PolicyType", "AdvantageEstimator", "NetworkArchitecture",
    # Networks
    "MLP", "init_weights", "get_activation",
    "DiscretePolicy", "ContinuousPolicy", "SquashedGaussianPolicy",
    "ValueNetwork", "QNetwork", "ActorCriticNetwork",
    # Utils
    "set_seed", "RunningMeanStd", "LearningRateScheduler",
    "MetricsTracker", "Checkpointer", "compute_explained_variance", "polyak_update",
    "smooth_data", "plot_training_curves", "plot_comparison",
    "plot_policy_distribution", "plot_value_function_1d", "plot_value_function_2d",
    "create_training_dashboard", "LivePlotter",
]
