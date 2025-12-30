"""
Policy Representation and Management

This module defines policy abstractions for MDPs, including deterministic and
stochastic policies, with support for policy evaluation and improvement.

Core Idea:
    A policy π is a mapping from states to actions (deterministic) or probability
    distributions over actions (stochastic). Policies guide agent behavior and
    are the primary output of MDP solvers.

Mathematical Theory:
    Deterministic policy: π(s) → a
    Stochastic policy: π(a|s) = P(a|s)

    Policy value function:
    V^π(s) = E[Σ_{t=0}^∞ γ^t R_t | s_0=s, π]
           = E[R_0 + γV^π(s_1) | s_0=s, π]

    Action value function:
    Q^π(s,a) = E[R_0 + γV^π(s_1) | s_0=s, a_0=a, π]

Problem Statement:
    Policies must be efficiently represented, evaluated, and improved.
    Different policy types suit different problem structures.

Complexity:
    - Storage: O(|S| × |A|) for stochastic, O(|S|) for deterministic
    - Evaluation: O(|S|²) per iteration for policy evaluation
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
from .mdp import State, Action, MarkovDecisionProcess


class Policy(ABC):
    """
    Abstract base class for policies.

    A policy defines the agent's behavior: how to select actions given states.
    """

    @abstractmethod
    def get_action(self, state: State) -> Action:
        """
        Get action for given state.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        pass

    @abstractmethod
    def get_action_probability(self, state: State, action: Action) -> float:
        """
        Get probability of selecting action in state.

        Args:
            state: Current state
            action: Action to evaluate

        Returns:
            Probability π(a|s) ∈ [0,1]
        """
        pass

    @abstractmethod
    def get_all_action_probabilities(self, state: State) -> Dict[Action, float]:
        """
        Get probability distribution over all actions in state.

        Args:
            state: Current state

        Returns:
            Dictionary mapping actions to probabilities
        """
        pass


class DeterministicPolicy(Policy):
    """
    Deterministic policy: π(s) → a.

    Core Idea:
        Maps each state to exactly one action. Simplest policy representation,
        often optimal for finite MDPs.

    Mathematical Theory:
        π(a|s) = 1 if a = π(s), else 0
        V^π(s) = R(s,π(s)) + γ Σ_{s'} P(s'|s,π(s)) V^π(s')

    Problem Statement:
        Deterministic policies are easier to represent and often sufficient
        for optimal control in finite MDPs.

    Complexity:
        - Storage: O(|S|)
        - Action selection: O(1)
    """

    def __init__(self, mdp: MarkovDecisionProcess):
        """
        Initialize deterministic policy.

        Args:
            mdp: The MDP this policy operates on
        """
        self.mdp = mdp
        self._policy: Dict[State, Action] = {}

    def set_action(self, state: State, action: Action) -> None:
        """
        Set action for state.

        Args:
            state: State to set action for
            action: Action to take in this state

        Raises:
            ValueError: If state or action not in MDP
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")
        if action not in self.mdp.actions:
            raise ValueError(f"Action {action} not in MDP")

        self._policy[state] = action

    def get_action(self, state: State) -> Action:
        """
        Get deterministic action for state.

        Args:
            state: Current state

        Returns:
            The unique action for this state

        Raises:
            ValueError: If action not defined for state
        """
        if state not in self._policy:
            raise ValueError(f"No action defined for state {state}")
        return self._policy[state]

    def get_action_probability(self, state: State, action: Action) -> float:
        """
        Get probability of action in state.

        Returns:
            1.0 if action is the policy action, 0.0 otherwise
        """
        if state not in self._policy:
            return 0.0
        return 1.0 if self._policy[state] == action else 0.0

    def get_all_action_probabilities(self, state: State) -> Dict[Action, float]:
        """
        Get action probability distribution.

        Returns:
            Dictionary with single action having probability 1.0
        """
        if state not in self._policy:
            return {}

        action = self._policy[state]
        return {action: 1.0}

    def is_complete(self) -> bool:
        """Check if policy is defined for all states."""
        return len(self._policy) == len(self.mdp.states)

    def to_array(self) -> np.ndarray:
        """
        Convert policy to array representation.

        Returns:
            Array where index i contains action_id for state i
        """
        if not self.is_complete():
            raise ValueError("Policy not defined for all states")

        state_list = sorted(self.mdp.states, key=lambda s: s.state_id)
        action_ids = np.array([self._policy[s].action_id for s in state_list])
        return action_ids


class StochasticPolicy(Policy):
    """
    Stochastic policy: π(a|s) = P(a|s).

    Core Idea:
        Maps each state to a probability distribution over actions. Enables
        exploration and is useful for theoretical analysis.

    Mathematical Theory:
        π(a|s) ≥ 0 for all a,s
        Σ_a π(a|s) = 1 for all s

        V^π(s) = Σ_a π(a|s) Q^π(s,a)
        Q^π(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V^π(s')

    Problem Statement:
        Stochastic policies enable exploration and are necessary for
        convergence guarantees in some algorithms.

    Complexity:
        - Storage: O(|S| × |A|)
        - Action selection: O(|A|) for sampling
    """

    def __init__(self, mdp: MarkovDecisionProcess):
        """
        Initialize stochastic policy.

        Args:
            mdp: The MDP this policy operates on
        """
        self.mdp = mdp
        self._policy: Dict[State, Dict[Action, float]] = {}
        self._rng = np.random.default_rng()

    def set_action_probability(self, state: State, action: Action,
                              probability: float) -> None:
        """
        Set probability of action in state.

        Args:
            state: State to set probability for
            action: Action to set probability for
            probability: Probability value in [0,1]

        Raises:
            ValueError: If probability not in [0,1] or state/action invalid
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be in [0,1], got {probability}")
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")
        if action not in self.mdp.actions:
            raise ValueError(f"Action {action} not in MDP")

        if state not in self._policy:
            self._policy[state] = {}
        self._policy[state][action] = probability

    def set_uniform_policy(self) -> None:
        """Set uniform random policy: π(a|s) = 1/|A| for all s,a."""
        uniform_prob = 1.0 / len(self.mdp.actions)
        for state in self.mdp.states:
            for action in self.mdp.actions:
                self.set_action_probability(state, action, uniform_prob)

    def get_action(self, state: State) -> Action:
        """
        Sample action from policy distribution.

        Args:
            state: Current state

        Returns:
            Sampled action

        Raises:
            ValueError: If policy not defined for state
        """
        if state not in self._policy:
            raise ValueError(f"No policy defined for state {state}")

        distribution = self._policy[state]
        actions = list(distribution.keys())
        probabilities = list(distribution.values())

        return self._rng.choice(actions, p=probabilities)

    def get_action_probability(self, state: State, action: Action) -> float:
        """
        Get probability of action in state.

        Args:
            state: Current state
            action: Action to evaluate

        Returns:
            Probability π(a|s)
        """
        if state not in self._policy:
            return 0.0
        return self._policy[state].get(action, 0.0)

    def get_all_action_probabilities(self, state: State) -> Dict[Action, float]:
        """
        Get action probability distribution.

        Args:
            state: Current state

        Returns:
            Dictionary mapping actions to probabilities
        """
        if state not in self._policy:
            return {}
        return dict(self._policy[state])

    def validate_probabilities(self) -> bool:
        """
        Validate that probabilities sum to 1 for all states.

        Returns:
            True if valid, raises ValueError otherwise
        """
        for state, distribution in self._policy.items():
            prob_sum = sum(distribution.values())
            if not np.isclose(prob_sum, 1.0, atol=1e-6):
                raise ValueError(
                    f"Probabilities for state {state} sum to {prob_sum}, not 1.0"
                )
        return True

    def is_complete(self) -> bool:
        """Check if policy is defined for all states."""
        return len(self._policy) == len(self.mdp.states)

    def to_array(self) -> np.ndarray:
        """
        Convert policy to array representation.

        Returns:
            Array of shape (|S|, |A|) with policy probabilities
        """
        if not self.is_complete():
            raise ValueError("Policy not defined for all states")

        state_list = sorted(self.mdp.states, key=lambda s: s.state_id)
        action_list = sorted(self.mdp.actions, key=lambda a: a.action_id)

        policy_array = np.zeros((len(state_list), len(action_list)))
        for i, state in enumerate(state_list):
            for j, action in enumerate(action_list):
                policy_array[i, j] = self.get_action_probability(state, action)

        return policy_array
