"""
Markov Decision Process (MDP) - Research-Grade Implementation

A comprehensive, production-ready implementation of Markov Decision Processes
with multiple solving algorithms and benchmark environments.

Core Components:
    - MDP Formulation: States, actions, transitions, rewards, policies, values
    - Solving Algorithms: Value Iteration, Policy Iteration, Linear Programming
    - Benchmark Environments: GridWorld, FrozenLake, CliffWalking
    - Visualization Tools: Policy plots, value heatmaps, convergence analysis

Mathematical Foundation:
    An MDP is a tuple M = (S, A, P, R, γ) where:
    - S: State space
    - A: Action space
    - P(s'|s,a): Transition probability model
    - R(s,a,s'): Reward function
    - γ ∈ [0,1]: Discount factor

    Bellman Optimality Equation:
    V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]

Key Features:
    1. Rigorous mathematical formulation with comprehensive documentation
    2. Multiple solving algorithms with convergence guarantees
    3. Benchmark environments for algorithm evaluation
    4. Production-grade code quality and error handling
    5. Extensive visualization and analysis tools

Usage Example:
    >>> from src.environments import GridWorld
    >>> from src.solvers import ValueIterationSolver
    >>>
    >>> # Create environment
    >>> env = GridWorld(height=5, width=5)
    >>> env.set_start(0, 0)
    >>> env.set_goal(4, 4)
    >>> env.build_transitions()
    >>>
    >>> # Solve with value iteration
    >>> solver = ValueIterationSolver(env)
    >>> value_fn, policy = solver.solve(verbose=True)
    >>>
    >>> # Visualize results
    >>> print(env.render(policy=policy))

References:
    - Bellman, R. E. (1957). Dynamic Programming
    - Puterman, M. L. (1994). Markov Decision Processes
    - Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning
"""

__version__ = "1.0.0"
__author__ = "AI-Practices"

from .core import (
    State,
    Action,
    TransitionModel,
    RewardFunction,
    MarkovDecisionProcess,
    Policy,
    DeterministicPolicy,
    StochasticPolicy,
    ValueFunction,
    StateValueFunction,
    ActionValueFunction,
)

from .solvers import (
    ValueIterationSolver,
    PolicyIterationSolver,
    LinearProgrammingSolver,
)

from .environments import (
    GridWorld,
    FrozenLake,
    CliffWalking,
)

from .utils import (
    MDPVisualizer,
)

__all__ = [
    # Core
    "State",
    "Action",
    "TransitionModel",
    "RewardFunction",
    "MarkovDecisionProcess",
    "Policy",
    "DeterministicPolicy",
    "StochasticPolicy",
    "ValueFunction",
    "StateValueFunction",
    "ActionValueFunction",
    # Solvers
    "ValueIterationSolver",
    "PolicyIterationSolver",
    "LinearProgrammingSolver",
    # Environments
    "GridWorld",
    "FrozenLake",
    "CliffWalking",
    # Utils
    "MDPVisualizer",
]
