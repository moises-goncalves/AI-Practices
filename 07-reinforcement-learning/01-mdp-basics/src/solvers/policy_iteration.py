"""
Policy Iteration Solver for Markov Decision Processes

This module implements the policy iteration algorithm, which alternates between
policy evaluation and policy improvement until convergence to optimal policy.

Core Idea:
    Policy iteration maintains an explicit policy and iteratively improves it.
    Each iteration consists of: (1) evaluate current policy, (2) improve policy
    greedily with respect to value function.

Mathematical Theory:
    Policy Evaluation (compute V^π):
    V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

    Policy Improvement:
    π'(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

    Policy Iteration Theorem:
    If π' = π (policy stable), then π is optimal.

Problem Statement:
    Policy iteration often converges in fewer iterations than value iteration,
    but each iteration is more expensive due to policy evaluation.

Algorithm Comparison:
    vs Value Iteration: PI maintains explicit policy, VI doesn't
    vs Linear Programming: PI is iterative, LP is direct
    Complexity: O(k|S|²|A|) where k is number of policy iterations

Complexity:
    - Time: O(k(|S|³ + |S|²|A|)) where k is policy iterations
    - Space: O(|S| × |A|)
    - Convergence: Typically faster than value iteration
"""

from typing import Tuple, Optional
import numpy as np
from ..core.mdp import MarkovDecisionProcess, State, Action
from ..core.value_function import StateValueFunction
from ..core.policy import DeterministicPolicy


class PolicyIterationSolver:
    """
    Policy Iteration solver for MDPs.

    Core Idea:
        Alternates between policy evaluation (compute V^π) and policy
        improvement (update π greedily) until convergence.

    Mathematical Theory:
        Repeat:
        1. Policy Evaluation: Solve V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
        2. Policy Improvement: π'(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
        Until π' = π

    Problem Statement:
        Provides guaranteed convergence with typically fewer iterations than VI.

    Complexity:
        - Per iteration: O(|S|³ + |S|²|A|)
        - Total iterations: Usually O(log(1/ε))
    """

    def __init__(self, mdp: MarkovDecisionProcess, theta: float = 1e-6):
        """
        Initialize Policy Iteration solver.

        Args:
            mdp: The MDP to solve
            theta: Convergence threshold for policy evaluation

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
        self.policy_stable_history = []

    def solve(self, max_iterations: int = 100,
             verbose: bool = False) -> Tuple[StateValueFunction, DeterministicPolicy]:
        """
        Solve MDP using policy iteration.

        Args:
            max_iterations: Maximum number of policy iterations
            verbose: Print convergence information

        Returns:
            Tuple of (optimal value function, optimal policy)

        Raises:
            RuntimeError: If fails to converge within max_iterations
        """
        # Initialize with random policy
        self.policy = self._initialize_random_policy()
        self.value_function = StateValueFunction(self.mdp, initial_value=0.0)
        self.policy_stable_history = []
        self.iteration_count = 0

        for iteration in range(max_iterations):
            # Step 1: Policy Evaluation
            self._evaluate_policy()

            # Step 2: Policy Improvement
            policy_stable = self._improve_policy()
            self.policy_stable_history.append(policy_stable)
            self.iteration_count = iteration + 1

            if verbose:
                print(f"Iteration {iteration + 1}: policy_stable = {policy_stable}")

            if policy_stable:
                if verbose:
                    print(f"Converged in {iteration + 1} iterations")
                break
        else:
            raise RuntimeError(
                f"Policy iteration failed to converge within {max_iterations} iterations"
            )

        return self.value_function, self.policy

    def _initialize_random_policy(self) -> DeterministicPolicy:
        """
        Initialize policy with random actions.

        Returns:
            Random deterministic policy
        """
        policy = DeterministicPolicy(self.mdp)
        actions_list = list(self.mdp.actions)

        for state in self.mdp.states:
            # Pick first action (could be random)
            policy.set_action(state, actions_list[0])

        return policy

    def _evaluate_policy(self) -> None:
        """
        Evaluate current policy by solving system of linear equations.

        Solves: V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

        Uses iterative approach (similar to value iteration but with fixed policy).
        """
        max_iterations = 1000

        for iteration in range(max_iterations):
            old_values = self.value_function.to_array().copy()

            for state in self.mdp.states:
                if self.mdp.is_terminal(state):
                    self.value_function.set_value(state, 0.0)
                    continue

                # Get action from current policy
                action = self.policy.get_action(state)

                # Compute V^π(s) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
                v_value = self._compute_policy_value(state, action)
                self.value_function.set_value(state, v_value)

            # Check convergence
            new_values = self.value_function.to_array()
            max_change = np.max(np.abs(new_values - old_values))

            if max_change < self.theta:
                break

    def _compute_policy_value(self, state: State, action: Action) -> float:
        """
        Compute value of state under current policy.

        V^π(s) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

        Args:
            state: Current state
            action: Action from policy

        Returns:
            Policy value
        """
        v_value = 0.0
        transition_dist = self.mdp.transition_model.get_transition_distribution(
            state, action
        )

        for next_state, probability in transition_dist.items():
            reward = self.mdp.reward_function.get_reward(state, action, next_state)
            next_value = self.value_function.get_value(next_state)
            v_value += probability * (reward + self.mdp.gamma * next_value)

        return v_value

    def _improve_policy(self) -> bool:
        """
        Improve policy greedily with respect to current value function.

        π'(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]

        Returns:
            True if policy is stable (unchanged), False otherwise
        """
        policy_stable = True

        for state in self.mdp.states:
            if self.mdp.is_terminal(state):
                continue

            # Get current action from policy
            old_action = self.policy.get_action(state)

            # Find best action
            best_action = None
            best_q_value = float('-inf')

            for action in self.mdp.actions:
                q_value = self._compute_q_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            # Update policy if action changed
            if best_action != old_action:
                policy_stable = False
                self.policy.set_action(state, best_action)

        return policy_stable

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

    def get_policy_stable_history(self) -> list:
        """Get history of policy stability per iteration."""
        return self.policy_stable_history

    def get_iteration_count(self) -> int:
        """Get number of iterations performed."""
        return self.iteration_count
