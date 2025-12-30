"""
Value Function Representation and Management

This module defines value function abstractions for MDPs, including state value
functions and action value functions (Q-functions), with support for efficient
storage and computation.

Core Idea:
    Value functions quantify the long-term expected reward from states or
    state-action pairs. They are central to dynamic programming and reinforcement
    learning algorithms.

Mathematical Theory:
    State value function:
    V(s) = E[Σ_{t=0}^∞ γ^t R_t | s_0=s]

    Action value function (Q-function):
    Q(s,a) = E[Σ_{t=0}^∞ γ^t R_t | s_0=s, a_0=a]

    Bellman equations:
    V(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
    Q(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q(s',a')]

Problem Statement:
    Efficient value function representation is critical for scalability.
    Different representations suit different problem structures.

Complexity:
    - State value storage: O(|S|)
    - Action value storage: O(|S| × |A|)
    - Lookup: O(1)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np
from .mdp import State, Action, MarkovDecisionProcess


class ValueFunction(ABC):
    """
    Abstract base class for value functions.

    A value function estimates the long-term expected reward from states.
    """

    @abstractmethod
    def get_value(self, state: State) -> float:
        """
        Get value of state.

        Args:
            state: State to evaluate

        Returns:
            Value V(s)
        """
        pass

    @abstractmethod
    def set_value(self, state: State, value: float) -> None:
        """
        Set value of state.

        Args:
            state: State to set value for
            value: Value to assign
        """
        pass


class StateValueFunction(ValueFunction):
    """
    State value function V(s).

    Core Idea:
        Maps each state to its expected long-term reward. Fundamental for
        policy evaluation and improvement.

    Mathematical Theory:
        V^π(s) = E[Σ_{t=0}^∞ γ^t R_t | s_0=s, π]

        Bellman expectation equation:
        V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

        Bellman optimality equation:
        V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]

    Problem Statement:
        Computing V(s) requires solving a system of linear equations or
        iterative approximation.

    Complexity:
        - Storage: O(|S|)
        - Update: O(1)
    """

    def __init__(self, mdp: MarkovDecisionProcess, initial_value: float = 0.0):
        """
        Initialize state value function.

        Args:
            mdp: The MDP this value function operates on
            initial_value: Initial value for all states
        """
        self.mdp = mdp
        self._values: Dict[State, float] = {}

        # Initialize all states with initial value
        for state in mdp.states:
            self._values[state] = initial_value

    def get_value(self, state: State) -> float:
        """
        Get value of state.

        Args:
            state: State to evaluate

        Returns:
            Value V(s)

        Raises:
            ValueError: If state not in MDP
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")
        return self._values.get(state, 0.0)

    def set_value(self, state: State, value: float) -> None:
        """
        Set value of state.

        Args:
            state: State to set value for
            value: Value to assign

        Raises:
            ValueError: If state not in MDP
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")
        self._values[state] = value

    def update_value(self, state: State, delta: float) -> None:
        """
        Update value by adding delta.

        Args:
            state: State to update
            delta: Change in value
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")
        self._values[state] += delta

    def get_max_value(self) -> float:
        """Get maximum value across all states."""
        if not self._values:
            return 0.0
        return max(self._values.values())

    def get_min_value(self) -> float:
        """Get minimum value across all states."""
        if not self._values:
            return 0.0
        return min(self._values.values())

    def get_mean_value(self) -> float:
        """Get mean value across all states."""
        if not self._values:
            return 0.0
        return np.mean(list(self._values.values()))

    def to_array(self) -> np.ndarray:
        """
        Convert value function to array representation.

        Returns:
            Array where index i contains value of state i
        """
        state_list = sorted(self.mdp.states, key=lambda s: s.state_id)
        values = np.array([self._values[s] for s in state_list])
        return values

    def copy(self) -> 'StateValueFunction':
        """Create a deep copy of this value function."""
        new_vf = StateValueFunction(self.mdp, initial_value=0.0)
        new_vf._values = dict(self._values)
        return new_vf


class ActionValueFunction(ValueFunction):
    """
    Action value function Q(s,a).

    Core Idea:
        Maps each state-action pair to its expected long-term reward. Enables
        direct policy extraction without explicit transition model.

    Mathematical Theory:
        Q^π(s,a) = E[Σ_{t=0}^∞ γ^t R_t | s_0=s, a_0=a, π]

        Bellman expectation equation:
        Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ Σ_{a'} π(a'|s') Q^π(s',a')]

        Bellman optimality equation:
        Q*(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γ max_{a'} Q*(s',a')]

    Problem Statement:
        Q-functions enable model-free learning but require more storage than
        state value functions.

    Complexity:
        - Storage: O(|S| × |A|)
        - Update: O(1)
    """

    def __init__(self, mdp: MarkovDecisionProcess, initial_value: float = 0.0):
        """
        Initialize action value function.

        Args:
            mdp: The MDP this value function operates on
            initial_value: Initial value for all state-action pairs
        """
        self.mdp = mdp
        self._values: Dict[Tuple[State, Action], float] = {}

        # Initialize all state-action pairs with initial value
        for state in mdp.states:
            for action in mdp.actions:
                self._values[(state, action)] = initial_value

    def get_value(self, state: State, action: Optional[Action] = None) -> float:
        """
        Get value of state or state-action pair.

        Args:
            state: State to evaluate
            action: Action to evaluate (if None, returns max over actions)

        Returns:
            Value Q(s,a) or max_a Q(s,a)

        Raises:
            ValueError: If state or action not in MDP
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")

        if action is None:
            # Return max Q-value for this state
            return self.get_max_action_value(state)

        if action not in self.mdp.actions:
            raise ValueError(f"Action {action} not in MDP")

        return self._values.get((state, action), 0.0)

    def set_value(self, state: State, action: Action, value: float) -> None:
        """
        Set value of state-action pair.

        Args:
            state: State to set value for
            action: Action to set value for
            value: Value to assign

        Raises:
            ValueError: If state or action not in MDP
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")
        if action not in self.mdp.actions:
            raise ValueError(f"Action {action} not in MDP")

        self._values[(state, action)] = value

    def update_value(self, state: State, action: Action, delta: float) -> None:
        """
        Update value by adding delta.

        Args:
            state: State to update
            action: Action to update
            delta: Change in value
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")
        if action not in self.mdp.actions:
            raise ValueError(f"Action {action} not in MDP")

        self._values[(state, action)] += delta

    def get_max_action_value(self, state: State) -> float:
        """
        Get maximum Q-value for state.

        Args:
            state: State to evaluate

        Returns:
            max_a Q(s,a)
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")

        max_value = float('-inf')
        for action in self.mdp.actions:
            value = self._values.get((state, action), 0.0)
            max_value = max(max_value, value)

        return max_value if max_value != float('-inf') else 0.0

    def get_best_action(self, state: State) -> Action:
        """
        Get action with highest Q-value in state.

        Args:
            state: State to evaluate

        Returns:
            Action with max Q(s,a)

        Raises:
            ValueError: If state not in MDP
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")

        best_action = None
        best_value = float('-inf')

        for action in self.mdp.actions:
            value = self._values.get((state, action), 0.0)
            if value > best_value:
                best_value = value
                best_action = action

        if best_action is None:
            raise RuntimeError(f"No actions available for state {state}")

        return best_action

    def get_action_values(self, state: State) -> Dict[Action, float]:
        """
        Get Q-values for all actions in state.

        Args:
            state: State to evaluate

        Returns:
            Dictionary mapping actions to Q-values
        """
        if state not in self.mdp.states:
            raise ValueError(f"State {state} not in MDP")

        return {action: self._values.get((state, action), 0.0)
                for action in self.mdp.actions}

    def get_max_value(self) -> float:
        """Get maximum Q-value across all state-action pairs."""
        if not self._values:
            return 0.0
        return max(self._values.values())

    def get_min_value(self) -> float:
        """Get minimum Q-value across all state-action pairs."""
        if not self._values:
            return 0.0
        return min(self._values.values())

    def get_mean_value(self) -> float:
        """Get mean Q-value across all state-action pairs."""
        if not self._values:
            return 0.0
        return np.mean(list(self._values.values()))

    def to_array(self) -> np.ndarray:
        """
        Convert Q-function to array representation.

        Returns:
            Array of shape (|S|, |A|) with Q-values
        """
        state_list = sorted(self.mdp.states, key=lambda s: s.state_id)
        action_list = sorted(self.mdp.actions, key=lambda a: a.action_id)

        q_array = np.zeros((len(state_list), len(action_list)))
        for i, state in enumerate(state_list):
            for j, action in enumerate(action_list):
                q_array[i, j] = self._values.get((state, action), 0.0)

        return q_array

    def copy(self) -> 'ActionValueFunction':
        """Create a deep copy of this Q-function."""
        new_qf = ActionValueFunction(self.mdp, initial_value=0.0)
        new_qf._values = dict(self._values)
        return new_qf
