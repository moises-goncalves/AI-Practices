"""Q-Learning and SARSA Implementation Module.

Production-ready implementations of classic temporal-difference control algorithms
for discrete state-action spaces, designed for both research and production use.

Core Algorithms:
    - Q-Learning: Off-policy TD control
    - SARSA: On-policy TD control
    - Expected SARSA: Variance-reduced on-policy control
    - Double Q-Learning: Bias-corrected off-policy control

Example:
    >>> from src.agents import QLearningAgent
    >>> from src.environments import CliffWalkingEnv
    >>> from src.training import Trainer
    >>>
    >>> env = CliffWalkingEnv()
    >>> agent = QLearningAgent(n_actions=4)
    >>> trainer = Trainer(env, agent)
    >>> metrics = trainer.train(episodes=500)
"""

from .agents import (
    AgentConfig,
    BaseAgent,
    QLearningAgent,
    SARSAAgent,
    ExpectedSARSAAgent,
    DoubleQLearningAgent,
)
from .environments import CliffWalkingEnv
from .training import Trainer, TrainingMetrics
from .exploration import ExplorationStrategy, ExplorationMixin
from .utils import (
    extract_path,
    plot_learning_curves,
    visualize_q_table,
    visualize_policy,
)

__version__ = "1.0.0"
__author__ = "Research Implementation"

__all__ = [
    # Agents
    "AgentConfig",
    "BaseAgent",
    "QLearningAgent",
    "SARSAAgent",
    "ExpectedSARSAAgent",
    "DoubleQLearningAgent",
    # Environments
    "CliffWalkingEnv",
    # Training
    "Trainer",
    "TrainingMetrics",
    # Exploration
    "ExplorationStrategy",
    "ExplorationMixin",
    # Utilities
    "extract_path",
    "plot_learning_curves",
    "visualize_q_table",
    "visualize_policy",
]
