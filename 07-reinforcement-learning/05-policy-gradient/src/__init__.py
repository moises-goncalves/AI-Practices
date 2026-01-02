"""
Policy Gradient Methods - 策略梯度方法

核心组件:
    - Algorithms: REINFORCE, Actor-Critic, A2C
    - Networks: Discrete/Continuous/Gaussian Policy, Value Networks
    - Core: Policy Gradient Agent, Trajectory Buffer
    - Utils: GAE计算、优势函数归一化、策略评估

数学基础:
    策略梯度定理: ∇J(θ) = E[∇log π(a|s;θ) · Q^π(s,a)]
    REINFORCE: ∇J(θ) = E[Σ_t ∇log π(a_t|s_t;θ) · G_t]
"""

__version__ = "1.0.0"
__author__ = "AI-Practices"

from .algorithms import (
    REINFORCE,
    ActorCritic,
    A2C,
)

from .core import (
    PolicyGradientAgent,
    BasePolicy,
    BaseValueFunction,
    Trajectory,
    TrajectoryBuffer,
)

from .networks import (
    DiscretePolicy,
    ContinuousPolicy,
    GaussianPolicy,
    ValueNetwork,
    DuelingValueNetwork,
)

from .utils import (
    compute_gae,
    normalize_advantages,
    compute_returns,
    evaluate_policy,
    collect_trajectories,
)

__all__ = [
    # Algorithms
    "REINFORCE", "ActorCritic", "A2C",
    # Core
    "PolicyGradientAgent", "BasePolicy", "BaseValueFunction",
    "Trajectory", "TrajectoryBuffer",
    # Networks
    "DiscretePolicy", "ContinuousPolicy", "GaussianPolicy",
    "ValueNetwork", "DuelingValueNetwork",
    # Utils
    "compute_gae", "normalize_advantages", "compute_returns",
    "evaluate_policy", "collect_trajectories",
]
