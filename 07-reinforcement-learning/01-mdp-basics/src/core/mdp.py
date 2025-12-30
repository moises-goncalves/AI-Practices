"""
Markov Decision Process (MDP) Core Data Structures

This module provides the fundamental building blocks for Markov Decision Processes,
including state spaces, action spaces, transition models, and reward functions.

Core Idea:
    An MDP formalizes sequential decision-making under uncertainty. It decomposes
    the problem into: (1) states representing the environment, (2) actions the agent
    can take, (3) stochastic transitions between states, and (4) rewards guiding
    the agent's behavior.

Mathematical Theory:
    An MDP is formally defined as a tuple M = (S, A, P, R, γ):
    - S: State space (finite or infinite)
    - A: Action space (finite or infinite)
    - P(s'|s,a): Transition probability model
    - R(s,a,s'): Reward function
    - γ ∈ [0,1]: Discount factor

    The Bellman equation forms the foundation:
    V(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]

Problem Statement:
    Sequential decision-making requires balancing immediate rewards with long-term
    consequences. MDPs provide a principled framework to model this trade-off.

Complexity:
    - State space: O(|S|)
    - Action space: O(|A|)
    - Transition model storage: O(|S|² × |A|)
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class State:
    """
    Immutable representation of an environment state.

    A state encapsulates all relevant information needed to make decisions.
    States are hashable to enable use in dictionaries and sets.

    Attributes:
        state_id: Unique identifier for the state
        features: Optional feature vector for continuous state representations
    """
    state_id: int
    features: Optional[np.ndarray] = None

    def __hash__(self) -> int:
        """Enable use as dictionary key."""
        return hash(self.state_id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on state_id."""
        if not isinstance(other, State):
            return NotImplemented
        return self.state_id == other.state_id


@dataclass(frozen=True)
class Action:
    """
    Immutable representation of an action.

    Actions are the decisions available to the agent at each state.
    Actions are hashable to enable use in dictionaries and sets.

    Attributes:
        action_id: Unique identifier for the action
        name: Human-readable action name
    """
    action_id: int
    name: str = ""

    def __hash__(self) -> int:
        """Enable use as dictionary key."""
        return hash(self.action_id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on action_id."""
        if not isinstance(other, Action):
            return NotImplemented
        return self.action_id == other.action_id


class TransitionModel:
    """
    Stochastic transition probability model P(s'|s,a).

    Core Idea:
        Captures the environment dynamics. Given current state and action,
        defines probability distribution over next states.

    Mathematical Theory:
        P(s'|s,a) ≥ 0 for all s,a,s'
        Σ_{s'} P(s'|s,a) = 1 for all s,a (probability conservation)

    Storage Complexity: O(|S|² × |A|) for dense representation
    """

    def __init__(self):
        """Initialize empty transition model."""
        self._transitions: Dict[Tuple[State, Action], Dict[State, float]] = {}

    def set_transition(self, state: State, action: Action,
                      next_state: State, probability: float) -> None:
        """
        Set transition probability P(next_state|state,action).

        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            probability: Transition probability (must be in [0,1])

        Raises:
            ValueError: If probability not in [0,1]
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be in [0,1], got {probability}")

        key = (state, action)
        if key not in self._transitions:
            self._transitions[key] = {}
        self._transitions[key][next_state] = probability

    def get_transition_distribution(self, state: State, action: Action) -> Dict[State, float]:
        """
        Get probability distribution over next states.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Dictionary mapping next states to probabilities
        """
        key = (state, action)
        return self._transitions.get(key, {})

    def sample_next_state(self, state: State, action: Action,
                         rng: np.random.Generator) -> State:
        """
        Sample next state from transition distribution.

        Args:
            state: Current state
            action: Action taken
            rng: Random number generator

        Returns:
            Sampled next state
        """
        distribution = self.get_transition_distribution(state, action)
        if not distribution:
            raise ValueError(f"No transitions defined for state={state}, action={action}")

        next_states = list(distribution.keys())
        probabilities = list(distribution.values())
        return rng.choice(next_states, p=probabilities)

    def validate_probabilities(self) -> bool:
        """
        Validate that all transition probabilities sum to 1.

        Returns:
            True if valid, raises ValueError otherwise
        """
        for (state, action), distribution in self._transitions.items():
            prob_sum = sum(distribution.values())
            if not np.isclose(prob_sum, 1.0, atol=1e-6):
                raise ValueError(
                    f"Probabilities for state={state}, action={action} "
                    f"sum to {prob_sum}, not 1.0"
                )
        return True


class RewardFunction:
    """
    Reward function R(s,a,s') or R(s,a).

    Core Idea:
        Encodes the agent's objectives. Rewards guide learning by providing
        immediate feedback on action quality.

    Mathematical Theory:
        Expected reward: E[R(s,a)] = Σ_{s'} P(s'|s,a) × R(s,a,s')

    Problem Statement:
        Reward design is critical for agent behavior. Sparse rewards lead to
        exploration challenges; dense rewards may cause local optima.
    """

    def __init__(self, reward_type: str = "state_action_next_state"):
        """
        Initialize reward function.

        Args:
            reward_type: Type of reward function
                - "state_action_next_state": R(s,a,s')
                - "state_action": R(s,a)
                - "state": R(s)
        """
        self.reward_type = reward_type
        self._rewards: Dict[Any, float] = {}

    def set_reward(self, *args, reward: float) -> None:
        """
        Set reward value.

        Args:
            *args: State, action, next_state (depending on reward_type)
            reward: Reward value
        """
        key = tuple(args)
        self._rewards[key] = reward

    def get_reward(self, *args) -> float:
        """
        Get reward value.

        Args:
            *args: State, action, next_state (depending on reward_type)

        Returns:
            Reward value (0.0 if not defined)
        """
        key = tuple(args)
        return self._rewards.get(key, 0.0)

    def get_expected_reward(self, state: State, action: Action,
                           transition_model: TransitionModel) -> float:
        """
        Compute expected reward E[R(s,a)].

        Args:
            state: Current state
            action: Action taken
            transition_model: Transition probability model

        Returns:
            Expected reward
        """
        if self.reward_type == "state_action":
            return self.get_reward(state, action)

        elif self.reward_type == "state_action_next_state":
            distribution = transition_model.get_transition_distribution(state, action)
            expected = 0.0
            for next_state, prob in distribution.items():
                reward = self.get_reward(state, action, next_state)
                expected += prob * reward
            return expected

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")


class MarkovDecisionProcess:
    """
    Complete Markov Decision Process specification.

    Core Idea:
        Encapsulates all components of an MDP: state space, action space,
        transition dynamics, rewards, and discount factor.

    Mathematical Theory:
        M = (S, A, P, R, γ)
        - Markov property: P(s_{t+1}|s_t, a_t, s_{t-1}, ...) = P(s_{t+1}|s_t, a_t)
        - Stationary: Transition and reward functions don't change over time

    Problem Statement:
        Provides unified interface for MDP solvers and environments.

    Complexity:
        - Storage: O(|S|² × |A|) for transition model
        - Query: O(1) for state/action lookups
    """

    def __init__(self, discount_factor: float = 0.99):
        """
        Initialize MDP.

        Args:
            discount_factor: γ ∈ [0,1], controls importance of future rewards

        Raises:
            ValueError: If discount_factor not in [0,1]
        """
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError(f"Discount factor must be in [0,1], got {discount_factor}")

        self.gamma = discount_factor
        self.states: Set[State] = set()
        self.actions: Set[Action] = set()
        self.transition_model = TransitionModel()
        self.reward_function = RewardFunction()
        self._initial_state: Optional[State] = None
        self._terminal_states: Set[State] = set()

    def add_state(self, state: State) -> None:
        """Add state to state space."""
        self.states.add(state)

    def add_action(self, action: Action) -> None:
        """Add action to action space."""
        self.actions.add(action)

    def set_initial_state(self, state: State) -> None:
        """Set initial state for episodes."""
        if state not in self.states:
            raise ValueError(f"State {state} not in state space")
        self._initial_state = state

    def add_terminal_state(self, state: State) -> None:
        """Mark state as terminal (episode ends)."""
        if state not in self.states:
            raise ValueError(f"State {state} not in state space")
        self._terminal_states.add(state)

    def is_terminal(self, state: State) -> bool:
        """Check if state is terminal."""
        return state in self._terminal_states

    def get_initial_state(self) -> State:
        """Get initial state."""
        if self._initial_state is None:
            raise RuntimeError("Initial state not set")
        return self._initial_state

    def validate(self) -> bool:
        """
        Validate MDP consistency.

        Returns:
            True if valid, raises ValueError otherwise
        """
        if not self.states:
            raise ValueError("State space is empty")
        if not self.actions:
            raise ValueError("Action space is empty")
        if self._initial_state is None:
            raise ValueError("Initial state not set")

        self.transition_model.validate_probabilities()
        return True

    def get_state_space_size(self) -> int:
        """Get number of states."""
        return len(self.states)

    def get_action_space_size(self) -> int:
        """Get number of actions."""
        return len(self.actions)
