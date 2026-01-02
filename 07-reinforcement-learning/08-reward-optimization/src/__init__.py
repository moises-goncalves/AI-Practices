"""
Reward Optimization - 奖励优化方法

核心组件:
    - Reward Shaping: 基于势函数的奖励塑形 (PBRS)
    - Inverse RL: 逆强化学习 (MaxMargin, MaxEntropy, GAIL)
    - Curiosity: 好奇心驱动探索 (ICM, RND)
    - Hindsight: 事后经验回放 (HER)
    - Evolution: 进化策略 (CMA-ES, CEM)
    - Buffers: 目标条件缓冲、演示缓冲
    - Networks: 特征编码器、动力学模型、判别器

数学基础:
    Reward Shaping: R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)
    MaxEnt IRL: max_θ Σ_τ log P(τ|θ)
    Curiosity: r_i = η · ||f(s') - f̂(s,a)||²
    HER: (s, a, r, s', g) → (s, a, r', s', g')
"""

__version__ = "1.0.0"
__author__ = "AI-Practices"

from .core import (
    # Reward Shaping
    ShapedRewardConfig,
    RewardShaper,
    DistanceBasedShaper,
    SubgoalBasedShaper,
    LearnedPotentialShaper,
    AdaptiveRewardShaper,
    DynamicShapingConfig,
    compute_optimal_potential_from_value,
    verify_policy_invariance,
    # Inverse RL
    IRLConfig,
    Demonstration,
    LinearFeatureExtractor,
    InverseRLBase,
    MaxMarginIRL,
    MaxEntropyIRL,
    DeepIRL,
    GAILConfig,
    compute_feature_matching_loss,
    reward_ambiguity_analysis,
    # Curiosity
    CuriosityConfig,
    IntrinsicCuriosityModule,
    RandomNetworkDistillation,
    CountBasedExploration,
    EpisodicNoveltyModule,
    compute_exploration_efficiency,
    # Hindsight
    GoalSelectionStrategy,
    Transition,
    Episode,
    HERConfig,
    HindsightExperienceReplay,
    PrioritizedHER,
    CurriculumHER,
    GoalGenerator,
    compute_success_rate,
    analyze_goal_coverage,
)

from .buffers import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    GoalTransition,
    GoalConditionedBuffer,
    HERBuffer,
    DemonstrationBuffer,
    MixedBuffer,
)

from .networks import (
    FeatureEncoder,
    CNNFeatureEncoder,
    MLPFeatureEncoder,
    RandomFeatureEncoder,
    ForwardDynamicsModel,
    InverseDynamicsModel,
    EnsembleDynamicsModel,
    GAILDiscriminator,
    AIRLDiscriminator,
    StateActionDiscriminator,
    ValueNetwork,
    DuelingValueNetwork,
    PotentialNetwork,
)

from .utils import (
    compute_episode_statistics,
    compute_reward_statistics,
    compute_exploration_metrics,
    plot_learning_curves,
    plot_reward_distribution,
    plot_value_heatmap,
    plot_trajectory,
)

__all__ = [
    # Core - Reward Shaping
    "ShapedRewardConfig", "RewardShaper", "DistanceBasedShaper",
    "SubgoalBasedShaper", "LearnedPotentialShaper", "AdaptiveRewardShaper",
    "DynamicShapingConfig", "compute_optimal_potential_from_value", "verify_policy_invariance",
    # Core - Inverse RL
    "IRLConfig", "Demonstration", "LinearFeatureExtractor", "InverseRLBase",
    "MaxMarginIRL", "MaxEntropyIRL", "DeepIRL", "GAILConfig",
    "compute_feature_matching_loss", "reward_ambiguity_analysis",
    # Core - Curiosity
    "CuriosityConfig", "IntrinsicCuriosityModule", "RandomNetworkDistillation",
    "CountBasedExploration", "EpisodicNoveltyModule", "compute_exploration_efficiency",
    # Core - Hindsight
    "GoalSelectionStrategy", "Transition", "Episode", "HERConfig",
    "HindsightExperienceReplay", "PrioritizedHER", "CurriculumHER",
    "GoalGenerator", "compute_success_rate", "analyze_goal_coverage",
    # Buffers
    "ReplayBuffer", "PrioritizedReplayBuffer", "GoalTransition",
    "GoalConditionedBuffer", "HERBuffer", "DemonstrationBuffer", "MixedBuffer",
    # Networks
    "FeatureEncoder", "CNNFeatureEncoder", "MLPFeatureEncoder", "RandomFeatureEncoder",
    "ForwardDynamicsModel", "InverseDynamicsModel", "EnsembleDynamicsModel",
    "GAILDiscriminator", "AIRLDiscriminator", "StateActionDiscriminator",
    "ValueNetwork", "DuelingValueNetwork", "PotentialNetwork",
    # Utils
    "compute_episode_statistics", "compute_reward_statistics", "compute_exploration_metrics",
    "plot_learning_curves", "plot_reward_distribution", "plot_value_heatmap", "plot_trajectory",
]
