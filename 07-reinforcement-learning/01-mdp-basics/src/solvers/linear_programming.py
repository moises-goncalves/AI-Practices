"""
Linear Programming Solver for Markov Decision Processes

This module implements the linear programming approach to solving MDPs,
which formulates the optimal value function computation as a linear program.

Core Idea:
    The optimal value function can be computed by solving a linear program.
    This approach is theoretically elegant and guarantees optimality but may
    be computationally expensive for large state spaces.

Mathematical Theory:
    Primal LP formulation:
    minimize Σ_s V(s)
    subject to: V(s) ≥ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s') for all s,a

    Dual LP formulation:
    maximize Σ_{s,a} π(s,a) R(s,a)
    subject to: Σ_a π(s,a) = ρ(s) for all s
                Σ_a π(s,a) - γ Σ_{s',a'} π(s',a') P(s|s',a') = ρ(s) for all s
                π(s,a) ≥ 0 for all s,a

    where ρ(s) is the stationary distribution.

Problem Statement:
    LP approach provides theoretical guarantees but requires solving a potentially
    large linear program. Useful for small to medium-sized MDPs.

Algorithm Comparison:
    vs Value Iteration: LP is direct but expensive, VI is iterative and cheaper
    vs Policy Iteration: LP guarantees optimality in one solve, PI iterates
    Complexity: O(|S|³) using interior point methods

Complexity:
    - Time: O(|S|³) to O(|S|^3.5) depending on LP solver
    - Space: O(|S|² × |A|) for constraint matrix
    - Convergence: Single solve (no iterations)
"""

from typing import Tuple, Optional
import numpy as np
from scipy.optimize import linprog

from ..core.mdp import MarkovDecisionProcess, State, Action
from ..core.value_function import StateValueFunction
from ..core.policy import DeterministicPolicy


class LinearProgrammingSolver:
    """
    Linear Programming solver for MDPs.

    Core Idea:
        Formulates optimal value function computation as a linear program
        and solves it directly using standard LP solvers.

    Mathematical Theory:
        Primal formulation:
        min Σ_s V(s)
        s.t. V(s) ≥ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ∀s,a

    Problem Statement:
        Provides guaranteed optimal solution but with higher computational cost.

    Complexity:
        - Time: O(|S|³) to O(|S|^3.5)
        - Space: O(|S|² × |A|)
    """

    def __init__(self, mdp: MarkovDecisionProcess):
        """
        Initialize Linear Programming solver.

        Args:
            mdp: The MDP to solve

        Raises:
            ValueError: If MDP is invalid
        """
        self.mdp = mdp
        self.mdp.validate()

        self.value_function: Optional[StateValueFunction] = None
        self.policy: Optional[DeterministicPolicy] = None

    def solve(self, verbose: bool = False) -> Tuple[StateValueFunction, DeterministicPolicy]:
        """
        Solve MDP using linear programming.

        Args:
            verbose: Print solver information

        Returns:
            Tuple of (optimal value function, optimal policy)

        Raises:
            RuntimeError: If LP solver fails
        """
        # Build LP problem
        c, A_ub, b_ub = self._build_lp_problem()

        # Solve LP
        if verbose:
            print("Solving linear program...")

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs', bounds=(None, None))

        if not result.success:
            raise RuntimeError(f"LP solver failed: {result.message}")

        if verbose:
            print(f"LP solved successfully. Objective value: {result.fun:.6f}")

        # Extract value function from solution
        self.value_function = self._extract_value_function(result.x)

        # Extract optimal policy
        self.policy = self._extract_policy()

        return self.value_function, self.policy

    def _build_lp_problem(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build linear program for MDP.

        Primal formulation:
        min Σ_s V(s)
        s.t. V(s) ≥ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ∀s,a

        Converted to standard form:
        min c^T x
        s.t. A_ub x ≤ b_ub

        Returns:
            Tuple of (c, A_ub, b_ub)
        """
        state_list = sorted(self.mdp.states, key=lambda s: s.state_id)
        num_states = len(state_list)
        state_to_idx = {s: i for i, s in enumerate(state_list)}

        # Objective: minimize sum of values
        c = np.ones(num_states)

        # Build constraints: -V(s) + γ Σ_{s'} P(s'|s,a) V(s') ≤ -R(s,a)
        # This is equivalent to: V(s) ≥ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')
        constraints = []
        bounds = []

        for state in state_list:
            if self.mdp.is_terminal(state):
                # Terminal states have value 0
                # Add constraint: V(s) ≤ 0
                constraint = np.zeros(num_states)
                constraint[state_to_idx[state]] = 1.0
                constraints.append(constraint)
                bounds.append(0.0)
                continue

            for action in self.mdp.actions:
                # Build constraint for (s,a) pair
                constraint = np.zeros(num_states)
                constraint[state_to_idx[state]] = -1.0  # -V(s)

                # Add γ Σ_{s'} P(s'|s,a) V(s')
                transition_dist = self.mdp.transition_model.get_transition_distribution(
                    state, action
                )
                for next_state, prob in transition_dist.items():
                    constraint[state_to_idx[next_state]] += self.mdp.gamma * prob

                # Right-hand side: -R(s,a)
                reward = self.mdp.reward_function.get_expected_reward(
                    state, action, self.mdp.transition_model
                )
                bound = -reward

                constraints.append(constraint)
                bounds.append(bound)

        A_ub = np.array(constraints) if constraints else np.zeros((0, num_states))
        b_ub = np.array(bounds) if bounds else np.zeros(0)

        return c, A_ub, b_ub

    def _extract_value_function(self, x: np.ndarray) -> StateValueFunction:
        """
        Extract value function from LP solution.

        Args:
            x: Solution vector from LP solver

        Returns:
            State value function
        """
        state_list = sorted(self.mdp.states, key=lambda s: s.state_id)
        value_function = StateValueFunction(self.mdp, initial_value=0.0)

        for i, state in enumerate(state_list):
            value_function.set_value(state, x[i])

        return value_function

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
                # For terminal states, pick any action
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
