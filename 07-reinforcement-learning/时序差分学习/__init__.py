"""
时序差分学习模块 (Temporal Difference Learning Module)
=====================================================

本模块提供完整的时序差分学习算法实现，包括:

核心算法:
--------
- TD(0): 单步时序差分学习
- SARSA: On-policy TD控制
- Q-Learning: Off-policy TD控制
- Expected SARSA: 期望SARSA
- Double Q-Learning: 双Q学习（解决过估计）
- N-Step TD: N步时序差分
- TD(λ): 带资格迹的时序差分
- SARSA(λ): 带资格迹的SARSA
- Watkins's Q(λ): Watkins的Q(λ)算法

环境:
----
- GridWorld: 可配置的网格世界
- CliffWalkingEnv: 悬崖行走环境
- WindyGridWorld: 有风的网格世界
- RandomWalk: 随机游走环境
- Blackjack: 21点环境

工具:
----
- 可视化: 训练曲线、价值函数热力图、策略箭头图
- 实验管理: 多种子实验、超参数搜索
- 分析: 收敛检测、RMSE计算、策略提取

使用示例:
--------
>>> from td_algorithms import create_td_learner, TDConfig
>>> from environments import CliffWalkingEnv
>>>
>>> # 创建环境和配置
>>> env = CliffWalkingEnv()
>>> config = TDConfig(alpha=0.5, gamma=0.99, epsilon=0.1)
>>>
>>> # 创建并训练SARSA
>>> sarsa = create_td_learner('sarsa', config=config)
>>> metrics = sarsa.train(env, n_episodes=500)
>>>
>>> # 评估
>>> mean_reward, std_reward = sarsa.evaluate(env, n_episodes=100)
>>> print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")

参考文献:
--------
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
- Watkins, C. J. C. D. (1989). Learning from delayed rewards.
- Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems.
"""

from td_algorithms import (
    # 配置类
    TDConfig,
    TrainingMetrics,
    EligibilityTraceType,

    # 基类
    BaseTDLearner,

    # 核心算法
    TD0ValueLearner,
    SARSA,
    ExpectedSARSA,
    QLearning,
    DoubleQLearning,
    NStepTD,
    TDLambda,
    SARSALambda,
    WatkinsQLambda,

    # 工厂函数
    create_td_learner,
)

from environments import (
    # 环境
    GridWorld,
    GridWorldConfig,
    CliffWalkingEnv,
    WindyGridWorld,
    RandomWalk,
    Blackjack,

    # 动作枚举
    Action,

    # 空间
    DiscreteSpace,
)

from utils import (
    # 可视化
    plot_training_curves,
    plot_value_heatmap,
    plot_q_value_heatmap,
    plot_policy_arrows,
    plot_td_error_analysis,
    plot_lambda_comparison,
    visualize_cliff_walking_path,

    # 实验管理
    ExperimentConfig,
    ExperimentResult,
    run_multi_seed_experiment,
    plot_multi_seed_comparison,

    # 分析工具
    compute_rmse,
    extract_greedy_policy,
    compute_state_visitation,
    detect_convergence,

    # 序列化
    save_q_function,
    load_q_function,
    save_experiment_results,
)

__version__ = "1.0.0"
__author__ = "AI-Practices"
__all__ = [
    # 配置
    "TDConfig",
    "TrainingMetrics",
    "EligibilityTraceType",

    # 算法
    "BaseTDLearner",
    "TD0ValueLearner",
    "SARSA",
    "ExpectedSARSA",
    "QLearning",
    "DoubleQLearning",
    "NStepTD",
    "TDLambda",
    "SARSALambda",
    "WatkinsQLambda",
    "create_td_learner",

    # 环境
    "GridWorld",
    "GridWorldConfig",
    "CliffWalkingEnv",
    "WindyGridWorld",
    "RandomWalk",
    "Blackjack",
    "Action",
    "DiscreteSpace",

    # 可视化
    "plot_training_curves",
    "plot_value_heatmap",
    "plot_q_value_heatmap",
    "plot_policy_arrows",
    "plot_td_error_analysis",
    "plot_lambda_comparison",
    "visualize_cliff_walking_path",

    # 实验管理
    "ExperimentConfig",
    "ExperimentResult",
    "run_multi_seed_experiment",
    "plot_multi_seed_comparison",

    # 分析
    "compute_rmse",
    "extract_greedy_policy",
    "compute_state_visitation",
    "detect_convergence",

    # 序列化
    "save_q_function",
    "load_q_function",
    "save_experiment_results",
]
