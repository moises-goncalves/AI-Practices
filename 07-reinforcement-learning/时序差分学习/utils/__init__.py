"""
工具模块 (Utilities Module)
==========================

本模块提供TD学习的辅助工具，包括可视化、实验管理、分析和序列化功能。

模块结构:
--------
- visualization.py: 训练曲线、热力图、策略可视化
- experiment.py: 多种子实验、超参数搜索
- analysis.py: 收敛检测、RMSE计算、策略提取
- serialization.py: 模型保存/加载
"""

from .visualization import (
    plot_training_curves,
    plot_value_heatmap,
    plot_q_value_heatmap,
    plot_policy_arrows,
    plot_td_error_analysis,
    plot_lambda_comparison,
    plot_multi_seed_comparison,
    visualize_cliff_walking_path,
    MATPLOTLIB_AVAILABLE,
)

from .experiment import (
    ExperimentConfig,
    ExperimentResult,
    run_multi_seed_experiment,
    run_hyperparameter_search,
)

from .analysis import (
    compute_rmse,
    extract_greedy_policy,
    compute_state_visitation,
    detect_convergence,
    evaluate_policy,
)

from .serialization import (
    save_q_function,
    load_q_function,
    save_experiment_results,
    load_experiment_results,
    save_learner,
    load_learner,
)

__all__ = [
    # 可视化
    "plot_training_curves",
    "plot_value_heatmap",
    "plot_q_value_heatmap",
    "plot_policy_arrows",
    "plot_td_error_analysis",
    "plot_lambda_comparison",
    "plot_multi_seed_comparison",
    "visualize_cliff_walking_path",
    "MATPLOTLIB_AVAILABLE",
    # 实验管理
    "ExperimentConfig",
    "ExperimentResult",
    "run_multi_seed_experiment",
    "run_hyperparameter_search",
    # 分析工具
    "compute_rmse",
    "extract_greedy_policy",
    "compute_state_visitation",
    "detect_convergence",
    "evaluate_policy",
    # 序列化
    "save_q_function",
    "load_q_function",
    "save_experiment_results",
    "load_experiment_results",
    "save_learner",
    "load_learner",
]
