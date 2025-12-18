"""
时序差分学习模块 (Temporal Difference Learning Module)
=====================================================

核心思想 (Core Idea):
--------------------
时序差分学习是强化学习的核心方法之一，结合了动态规划的自举思想
和蒙特卡洛方法的采样思想。TD方法可以从不完整的回合中学习，
并且不需要环境模型。

本模块提供完整的时序差分学习算法实现。

数学原理 (Mathematical Theory):
------------------------------
TD学习的核心是TD误差:
    δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

更新规则:
    V(S_t) ← V(S_t) + α × δ_t

TD(λ)引入资格迹，实现多步回溯:
    e_t(s) = γλe_{t-1}(s) + 𝟙[S_t = s]
    V(s) ← V(s) + α × δ_t × e_t(s)

核心算法 (Core Algorithms):
--------------------------
- TD(0): 单步TD预测，使用下一状态的价值估计作为目标
- SARSA: On-Policy TD控制，使用实际执行的动作更新
- Q-Learning: Off-Policy TD控制，使用max Q值更新
- Expected SARSA: 使用期望Q值更新，减少方差
- Double Q-Learning: 使用两个Q表消除最大化偏差
- N-Step TD: 使用n步回报作为目标
- TD(λ): 使用资格迹实现多步回溯
- SARSA(λ): 带资格迹的SARSA
- Watkins Q(λ): 带资格迹的Q-Learning（非贪婪时切断迹）

环境 (Environments):
-------------------
- RandomWalk: TD预测的标准测试床
- CliffWalking: On/Off-Policy对比的经典环境
- WindyGridWorld: 测试智能体应对环境动态的能力
- GridWorld: 可配置的通用网格世界
- Blackjack: MC和TD方法的测试环境

算法对比 (Comparison):
---------------------
┌─────────────────┬────────────┬────────────┬───────────────┐
│      算法       │   更新类型 │  偏差/方差 │   适用场景    │
├─────────────────┼────────────┼────────────┼───────────────┤
│   TD(0)         │  On-line   │  低方差    │   快速学习    │
│   SARSA         │  On-Policy │  稳定      │   安全导航    │
│   Q-Learning    │  Off-Policy│  最优性    │   最优策略    │
│   Expected SARSA│  On-Policy │  更低方差  │   中间选择    │
│   TD(λ)         │  多步回溯  │  可调      │   稀疏奖励    │
└─────────────────┴────────────┴────────────┴───────────────┘

使用示例 (Example):
-----------------
>>> from 时序差分学习 import SARSA, TDConfig, CliffWalkingEnv
>>>
>>> # 创建环境和学习器
>>> env = CliffWalkingEnv()
>>> config = TDConfig(alpha=0.5, gamma=1.0, epsilon=0.1)
>>> learner = SARSA(config)
>>>
>>> # 训练
>>> metrics = learner.train(env, n_episodes=500)
>>>
>>> # 评估
>>> mean_reward, std_reward = learner.evaluate(env, n_episodes=100)
>>> print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")

模块结构 (Module Structure):
---------------------------
```
时序差分学习/
├── core/           # 核心算法实现
│   ├── config.py       # 配置类
│   ├── base.py         # 基类
│   ├── td_prediction.py # TD预测
│   ├── td_control.py    # TD控制
│   ├── advanced.py      # 高级算法
│   └── factory.py       # 工厂函数
├── environments/   # 强化学习环境
│   ├── base.py         # 基础组件
│   ├── grid_world.py   # 网格世界
│   ├── cliff_walking.py # 悬崖行走
│   ├── windy_grid.py   # 有风网格
│   ├── random_walk.py  # 随机游走
│   └── blackjack.py    # 二十一点
├── utils/          # 工具函数
│   ├── visualization.py # 可视化
│   ├── experiment.py    # 实验管理
│   ├── analysis.py      # 分析工具
│   └── serialization.py # 序列化
├── networks/       # 神经网络组件
├── tests/          # 单元测试
└── notebooks/      # Jupyter教程
```

参考文献 (References):
--------------------
1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning:
   An Introduction (2nd ed.). MIT Press.
2. Watkins, C. J. C. H. (1989). Learning from Delayed Rewards.
   PhD thesis, Cambridge University.
3. Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning
   using connectionist systems. Technical Report CUED/F-INFENG/TR 166.
"""

from .core import (
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
