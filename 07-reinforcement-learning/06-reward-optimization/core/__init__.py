"""
Core Reward Optimization Algorithms
====================================

This subpackage contains the fundamental reward optimization algorithms
organized by methodology:

Modules:
    reward_shaping: Potential-based reward shaping (PBRS)
    inverse_rl: Inverse Reinforcement Learning algorithms
    curiosity: Curiosity-driven exploration methods
    hindsight: Hindsight Experience Replay variants

Mathematical Foundation:
    All algorithms address the sparse reward problem in RL through different
    approaches:

    1. Reward Shaping: Augment rewards while preserving optimality
       - R'(s,a,s') = R(s,a,s') + γΦ(s') - Φ(s)

    2. Inverse RL: Learn rewards from expert demonstrations
       - max_θ Σ_τ log P(τ|θ)

    3. Curiosity: Generate intrinsic rewards from prediction errors
       - r_i = η · ||f(s') - f̂(s,a)||²

    4. Hindsight: Relabel goals to create successful experiences
       - (s, a, r, s', g) → (s, a, r', s', g')

References:
    [1] Ng et al. (1999). Policy invariance under reward transformations.
    [2] Ziebart et al. (2008). Maximum entropy inverse RL.
    [3] Pathak et al. (2017). Curiosity-driven exploration.
    [4] Andrychowicz et al. (2017). Hindsight experience replay.
"""

from .reward_shaping import (
    ShapedRewardConfig,
    RewardShaper,
    DistanceBasedShaper,
    SubgoalBasedShaper,
    LearnedPotentialShaper,
    AdaptiveRewardShaper,
    DynamicShapingConfig,
    compute_optimal_potential_from_value,
    verify_policy_invariance,
)

from .inverse_rl import (
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
)

from .curiosity import (
    CuriosityConfig,
    IntrinsicCuriosityModule,
    RandomNetworkDistillation,
    CountBasedExploration,
    EpisodicNoveltyModule,
    compute_exploration_efficiency,
)

from .hindsight import (
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

__all__ = [
    # Reward Shaping
    "ShapedRewardConfig",
    "RewardShaper",
    "DistanceBasedShaper",
    "SubgoalBasedShaper",
    "LearnedPotentialShaper",
    "AdaptiveRewardShaper",
    "DynamicShapingConfig",
    "compute_optimal_potential_from_value",
    "verify_policy_invariance",
    # Inverse RL
    "IRLConfig",
    "Demonstration",
    "LinearFeatureExtractor",
    "InverseRLBase",
    "MaxMarginIRL",
    "MaxEntropyIRL",
    "DeepIRL",
    "GAILConfig",
    "compute_feature_matching_loss",
    "reward_ambiguity_analysis",
    # Curiosity
    "CuriosityConfig",
    "IntrinsicCuriosityModule",
    "RandomNetworkDistillation",
    "CountBasedExploration",
    "EpisodicNoveltyModule",
    "compute_exploration_efficiency",
    # Hindsight
    "GoalSelectionStrategy",
    "Transition",
    "Episode",
    "HERConfig",
    "HindsightExperienceReplay",
    "PrioritizedHER",
    "CurriculumHER",
    "GoalGenerator",
    "compute_success_rate",
    "analyze_goal_coverage",
]
