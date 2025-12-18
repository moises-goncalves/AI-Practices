"""
Learning to Optimize Rewards: A Comprehensive Module Collection

================================================================================
OVERVIEW
================================================================================
This package provides production-grade implementations of reward optimization
techniques for reinforcement learning. All implementations follow research-grade
standards with comprehensive documentation and mathematical foundations.

================================================================================
MODULE STRUCTURE
================================================================================
学习优化奖励/
├── core/                    # Core algorithms
│   ├── reward_shaping.py    # Potential-based reward shaping (PBRS)
│   ├── inverse_rl.py        # Inverse reinforcement learning
│   ├── curiosity.py         # Curiosity-driven exploration
│   └── hindsight.py         # Hindsight experience replay
├── networks/                # Neural network architectures
│   ├── feature_encoders.py  # State representation networks
│   ├── dynamics_models.py   # Forward/inverse dynamics
│   ├── discriminators.py    # GAIL/AIRL discriminators
│   └── value_networks.py    # Value/potential networks
├── buffers/                 # Experience replay implementations
│   ├── replay_buffer.py     # Standard and prioritized replay
│   ├── goal_buffer.py       # Goal-conditioned buffers for HER
│   └── demonstration_buffer.py  # Expert demonstration storage
├── utils/                   # Utility functions
│   ├── metrics.py           # Performance metrics
│   └── visualization.py     # Plotting utilities
├── notebooks/               # Interactive tutorials
│   ├── 01-Reward-Shaping-Fundamentals.ipynb
│   ├── 02-Inverse-RL-Tutorial.ipynb
│   ├── 03-Curiosity-Driven-Exploration.ipynb
│   └── 04-HER-Tutorial.ipynb
└── 核心知识点.md             # Knowledge summary document

================================================================================
CORE ALGORITHMS
================================================================================
1. Potential-Based Reward Shaping (PBRS)
   - Augments sparse rewards while preserving optimal policy
   - F(s, s') = γΦ(s') - Φ(s)
   - Implementations: Distance-based, Subgoal-based, Learned, Adaptive

2. Inverse Reinforcement Learning (IRL)
   - Learns reward function from expert demonstrations
   - Algorithms: Max-Margin IRL, Max-Entropy IRL, Deep IRL, GAIL, AIRL

3. Curiosity-Driven Exploration
   - Generates intrinsic rewards from prediction errors
   - Methods: ICM, RND, Count-based, Ensemble disagreement

4. Hindsight Experience Replay (HER)
   - Relabels failed episodes as successes with different goals
   - Strategies: Final, Future, Episode, Random

================================================================================
MATHEMATICAL FOUNDATIONS
================================================================================
All implementations include detailed mathematical documentation:
- Core theorems and proofs
- Complexity analysis
- Algorithm comparisons
- References to original papers

================================================================================
QUICK START
================================================================================
```python
from 学习优化奖励 import (
    # Reward Shaping
    DistanceBasedShaper,
    ShapedRewardConfig,

    # Inverse RL
    MaxEntropyIRL,
    GAILDiscriminator,

    # Curiosity
    IntrinsicCuriosityModule,
    RandomNetworkDistillation,

    # HER
    HindsightExperienceReplay,
    GoalSelectionStrategy,
)

# Example: Distance-based reward shaping
goal = np.array([10.0, 10.0])
shaper = DistanceBasedShaper(
    goal_position=goal,
    config=ShapedRewardConfig(discount_factor=0.99)
)

# In training loop:
shaped_reward = shaper.shape_reward(state, action, next_state, reward, done)
```

================================================================================
REFERENCES
================================================================================
[1] Ng et al. (1999). Policy invariance under reward transformations. ICML.
[2] Ziebart et al. (2008). Maximum entropy inverse RL. AAAI.
[3] Pathak et al. (2017). Curiosity-driven exploration. ICML.
[4] Andrychowicz et al. (2017). Hindsight experience replay. NeurIPS.
[5] Ho & Ermon (2016). Generative adversarial imitation learning. NeurIPS.
[6] Fu et al. (2018). Learning robust rewards with AIRL. ICLR.
[7] Burda et al. (2019). Exploration by random network distillation. ICLR.
"""

__version__ = "2.0.0"
__author__ = "AI-Practices Contributors"

# Core algorithms
from .core.reward_shaping import (
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

# Import from existing modules for backwards compatibility
from .inverse_rl import (
    IRLConfig,
    Demonstration,
    LinearFeatureExtractor,
    InverseRLBase,
    MaxMarginIRL,
    MaxEntropyIRL,
    DeepIRL,
    GAILDiscriminator,
    GAILConfig,
    compute_feature_matching_loss,
    reward_ambiguity_analysis,
)

from .curiosity_driven import (
    CuriosityConfig,
    FeatureEncoder,
    ForwardDynamicsModel,
    InverseDynamicsModel,
    IntrinsicCuriosityModule,
    RandomNetworkDistillation,
    CountBasedExploration,
    EpisodicNoveltyModule,
    compute_exploration_efficiency,
)

from .hindsight_experience_replay import (
    GoalSelectionStrategy,
    Transition,
    Episode,
    HERConfig,
    GoalConditionedReplayBuffer,
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
    "GAILDiscriminator",
    "GAILConfig",
    "compute_feature_matching_loss",
    "reward_ambiguity_analysis",
    # Curiosity
    "CuriosityConfig",
    "FeatureEncoder",
    "ForwardDynamicsModel",
    "InverseDynamicsModel",
    "IntrinsicCuriosityModule",
    "RandomNetworkDistillation",
    "CountBasedExploration",
    "EpisodicNoveltyModule",
    "compute_exploration_efficiency",
    # HER
    "GoalSelectionStrategy",
    "Transition",
    "Episode",
    "HERConfig",
    "GoalConditionedReplayBuffer",
    "HindsightExperienceReplay",
    "PrioritizedHER",
    "CurriculumHER",
    "GoalGenerator",
    "compute_success_rate",
    "analyze_goal_coverage",
]
