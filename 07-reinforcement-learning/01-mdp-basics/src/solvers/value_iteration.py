"""
Value Iteration Solver for Markov Decision Processes

This module implements the value iteration algorithm, a fundamental dynamic
programming approach for computing optimal value functions and policies.

Core Idea:
    Value iteration repeatedly applies the Bellman optimality operator until
    convergence. It directly computes the optimal value function without
    explicitly maintaining a policy.

Mathematical Theory:
    Bellman optimality operator:
    (TV)(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]

    Value iteration update:
    V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV_k(s')]

    Convergence: ||V_{k+1} - V_k||_∞ → 0 as k → ∞
    Optimal policy: π*(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]

Problem Statement:
    Value iteration is guaranteed to converge to optimal values in finite MDPs.
    It's simpler than policy iteration but may require more iterations.

Algorithm Comparison:
    vs Policy Iteration: VI doesn't maintain explicit policy, PI does
    vs Linear Programming: VI is iterative, LP is direct but computationally expensive
    Complexity: O(|S|²|A|) per iteration

Complexity:
    - Time: O(k|S|²|A|) where k is number of iterations
    - Space: O(|S|)
    - Convergence rate: Geometric with rate γ
"""

from typing import Tuple, Optional, Callable
import numpy as np
from ..core.mdp import MarkovDecisionProcess, State, Action
from ..core.value_function import StateValueFunction, ActionValueFunction
from ..core.policy import DeterministicPolicy


class ValueIterationSolver:
    """
    Value Iteration solver for MDPs.

    Core Idea:
        Iteratively improves value estimates by applying the Bellman optimality
        operator. Converges to optimal value function.

    Mathematical Theory:
        V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV_k(s')]

    Problem Statement:
        Provides guaranteed convergence to optimal values with simple implementation.

    Complexity:
        - Per iteration: O(|S|²|A|)
        - Total iterations: O(log(1/ε) / (1-γ))
    """

    def __init__(self, mdp: MarkovDecisionProcess, theta: float = 1e-6):
        """
        Initialize Value Iteration solver.

        Args:
            mdp: The MDP to solve
            theta: Convergence threshold for value changes

        Raises:
            ValueError: If MDP is invalid or theta <= 0
        """
        if theta <= 0:
            raise ValueError(f"Theta must be positive, got {theta}")

        self.mdp = mdp
        self.theta = theta
        self.mdp.validate()

        self.value_function: Optional[StateValueFunction] = None
        self.policy: Optional[DeterministicPolicy] = None
        self.iteration_count = 0
        self.convergence_history = []

    def solve(self, max_iterations: int = 1000,
             verbose: bool = False) -> Tuple[StateValueFunction, DeterministicPolicy]:
        """
        Solve MDP using value iteration.

        Args:
            max_iterations: Maximum number of iterations
            verbose: Print convergence information

        Returns:
            Tuple of (optimal value function, optimal policy)

        Raises:
            RuntimeError: If fails to converge within max_iterations
        """
        # Initialize value function
        self.value_function = StateValueFunction(self.mdp, initial_value=0.0)
        self.convergence_history = []
        self.iteration_count = 0

        for iteration in range(max_iterations):
            # Store old values for convergence check
            old_values = self.value_function.to_array().copy()

            # Apply Bellman optimality operator to all states
            for state in self.mdp.states:
                self._update_state_value(state)

            # Check convergence
            new_values = self.value_function.to_array()
            max_change = np.max(np.abs(new_values - old_values))
            self.convergence_history.append(max_change)
            self.iteration_count = iteration + 1

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: max_change = {max_change:.6e}")

            if max_change < self.theta:
                if verbose:
                    print(f"Converged in {iteration + 1} iterations")
                break
        else:
            raise RuntimeError(
                f"Value iteration failed to converge within {max_iterations} iterations. "
                f"Final max_change: {max_change:.6e}"
            )

        # Extract optimal policy from value function
        self.policy = self._extract_policy()

        return self.value_function, self.policy

    def _update_state_value(self, state: State) -> None:
        """
        Update value of single state using Bellman optimality operator.

        V(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]

        Args:
            state: State to update
        """
        if self.mdp.is_terminal(state):
            self.value_function.set_value(state, 0.0)
            return

        max_value = float('-inf')

        for action in self.mdp.actions:
            # Compute Q(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
            q_value = self._compute_q_value(state, action)
            max_value = max(max_value, q_value)

        self.value_function.set_value(state, max_value)

    def _compute_q_value(self, state: State, action: Action) -> float:
        """
        Compute Q-value for state-action pair.

        Q(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]

        Args:
            state: Current state
            action: Action to evaluate

        Returns:
            Q-value
        """
        q_value = 0.0
        transition_dist = self.mdp.transition_model.get_transition_distribution(
            state, action
        )

        for next_state, probability in transition_dist.items():
            reward = self.mdp.reward_function.get_reward(state, action, next_state)
            next_value = self.value_function.get_value(next_state)
            q_value += probability * (reward + self.mdp.gamma * next_value)

        return q_value

    def _extract_policy(self) -> DeterministicPolicy:
        """
        Extract greedy policy from value function.

        π(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]

        Returns:
            Optimal deterministic policy
        """
        policy = DeterministicPolicy(self.mdp)

        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                # For terminal states, pick any action (won't be used)
                policy.set_action(state, list(self.mdp.actions)[0])
                continue

            best_action = None
            best_q_value = float('-inf')

            for action in self.mdp.actions:
                q_value = self._compute_q_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            policy.set_action(state, best_action)

        return policy

    def get_convergence_history(self) -> list:
        """Get history of max value changes per iteration."""
        return self.convergence_history

    def get_iteration_count(self) -> int:
        """Get number of iterations performed."""
        return self.iteration_count
