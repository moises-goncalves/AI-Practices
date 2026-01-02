"""
Deep Q-Learning - 深度Q学习

核心组件:
    - DQN Variants: Vanilla DQN, Double DQN, Dueling DQN, Noisy DQN, C51, Rainbow
    - Experience Replay: Uniform, Prioritized, N-Step
    - Networks: MLP, Dueling, Noisy, Categorical
    - Training Utils: 训练循环、可视化、评估

数学基础:
    DQN Loss: L = E[(r + γ max_a' Q(s', a'; θ⁻) - Q(s, a; θ))²]
    其中 θ⁻ 为目标网络参数
"""

__version__ = "1.0.0"
__author__ = "AI-Practices"

from .core import (
    DQNConfig,
    Transition,
    NStepTransition,
    NetworkType,
    LossType,
    ExplorationStrategy,
)

from .buffers import (
    ReplayBuffer,
    SumTree,
    PrioritizedReplayBuffer,
)

from .networks import (
    DQNNetwork,
    DuelingDQNNetwork,
    create_mlp,
    init_weights_orthogonal,
    init_weights_xavier,
)

from .agents import (
    DQNVariantAgent,
)

from .utils import (
    TrainingConfig,
    TrainingMetrics,
    plot_training_curves,
    plot_algorithm_comparison,
)

__all__ = [
    # Core
    "DQNConfig", "Transition", "NStepTransition",
    "NetworkType", "LossType", "ExplorationStrategy",
    # Buffers
    "ReplayBuffer", "SumTree", "PrioritizedReplayBuffer",
    # Networks
    "DQNNetwork", "DuelingDQNNetwork", "create_mlp",
    "init_weights_orthogonal", "init_weights_xavier",
    # Agents
    "DQNVariantAgent",
    # Utils
    "TrainingConfig", "TrainingMetrics",
    "plot_training_curves", "plot_algorithm_comparison",
]
