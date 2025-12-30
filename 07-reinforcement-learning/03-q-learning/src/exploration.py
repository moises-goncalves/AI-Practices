"""Exploration Strategies for Reinforcement Learning.

This module implements various exploration strategies that balance the fundamental
exploration-exploitation tradeoff in reinforcement learning.

Core Idea:
    Exploration strategies determine how an agent selects actions during learning.
    Too much exploitation leads to suboptimal convergence (local minima); too much
    exploration wastes samples and slows learning.

Mathematical Foundation:
    The exploration-exploitation dilemma is formalized in multi-armed bandit theory.
    The regret bound for an optimal strategy is O(√(K·T·ln(T))) for K actions over
    T timesteps, achieved by algorithms like UCB.

Implemented Strategies:
    1. ε-Greedy: Simple probabilistic exploration
    2. Softmax (Boltzmann): Temperature-controlled probabilistic selection
    3. UCB (Upper Confidence Bound): Optimism in the face of uncertainty

References:
    [1] Auer, P. et al. (2002). Finite-time Analysis of the Multiarmed Bandit Problem.
    [2] Sutton & Barto (2018). Reinforcement Learning: An Introduction, Chapter 2.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np


class ExplorationStrategy(Enum):
    """Enumeration of available exploration strategies.

    Attributes:
        EPSILON_GREEDY: ε-greedy with uniform random exploration
        SOFTMAX: Boltzmann exploration with temperature parameter
        UCB: Upper Confidence Bound balancing exploitation and uncertainty
    """

    EPSILON_GREEDY = "epsilon_greedy"
    SOFTMAX = "softmax"
    UCB = "ucb"


class ExplorationMixin:
    """Mixin class providing exploration strategy implementations.

    Core Idea:
        Encapsulates exploration logic separately from value learning, enabling
        flexible combination with different TD algorithms (Q-Learning, SARSA, etc.).

    Mathematical Principles:
        Each strategy provides a different answer to: "Given Q-value estimates
        Q(s,a) for all actions, how should we select the next action?"

    Usage:
        Inherit from this mixin alongside your agent class to gain access to
        exploration methods. The agent's get_action() method should delegate
        to the appropriate exploration strategy.

    Example:
        >>> class MyAgent(BaseAgent, ExplorationMixin):
        ...     def get_action(self, state, training=True):
        ...         if not training:
        ...             return np.argmax(self.q_table[state])
        ...         return self._epsilon_greedy(self.q_table[state], self.epsilon)
    """

    def _epsilon_greedy(self, q_values: np.ndarray, epsilon: float) -> int:
        """ε-Greedy action selection with random tie-breaking.

        Core Idea:
            With probability ε, select a random action (explore); otherwise,
            select the action with highest Q-value (exploit).

        Mathematical Formulation:
            π(a|s) = ε/|A| + (1-ε)·𝟙[a = argmax_a' Q(s,a')]

            where 𝟙[·] is the indicator function.

        Problem Context:
            Introduced as the simplest exploration strategy. While not optimal
            in terms of regret bounds, it is widely used due to simplicity and
            effectiveness in practice.

        Algorithm Comparison:
            vs Softmax: ε-greedy explores uniformly; softmax explores
                       proportionally to Q-values
            vs UCB: ε-greedy has no notion of uncertainty; UCB explicitly
                   models confidence bounds

        Complexity:
            Time: O(|A|) for argmax operation
            Space: O(1) additional space

        Args:
            q_values: Action-value estimates Q(s,·), shape (n_actions,)
            epsilon: Exploration probability, ε ∈ [0, 1]

        Returns:
            Selected action index

        Raises:
            ValueError: If epsilon is outside [0, 1]

        Implementation Notes:
            - Uses random tie-breaking when multiple actions share max Q-value
            - Numerical tolerance (rtol=1e-9) for floating-point comparison
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")

        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))

        # Greedy selection with tie-breaking
        max_q = np.max(q_values)
        max_actions = np.where(np.isclose(q_values, max_q, rtol=1e-9))[0]
        return int(np.random.choice(max_actions))

    def _softmax(self, q_values: np.ndarray, temperature: float) -> int:
        """Softmax (Boltzmann) exploration via Gibbs distribution.

        Core Idea:
            Select actions probabilistically based on their relative Q-values.
            Higher Q-values lead to higher selection probability, but all actions
            retain non-zero probability.

        Mathematical Formulation:
            π(a|s) = exp(Q(s,a)/τ) / Σ_{a'} exp(Q(s,a')/τ)

            This is the Gibbs (Boltzmann) distribution from statistical mechanics.

        Temperature Parameter (τ):
            - τ → 0: Approaches greedy selection (max exploitation)
            - τ → ∞: Approaches uniform random (max exploration)
            - τ = 1: Standard softmax, Q-values interpreted as log-probabilities

        Problem Context:
            Softmax addresses a limitation of ε-greedy: it considers the
            *magnitude* of Q-value differences. Actions with slightly lower
            Q-values are selected more often than actions with much lower values.

        Algorithm Comparison:
            vs ε-greedy: More nuanced exploration based on value differences
            vs UCB: No explicit uncertainty modeling; purely value-based

        Complexity:
            Time: O(|A|) for computing softmax distribution
            Space: O(|A|) for probability vector

        Numerical Stability:
            We subtract max(Q) before exponentiation to prevent overflow:
            exp(Q/τ) / Σexp(Q/τ) = exp((Q-max)/τ) / Σexp((Q-max)/τ)

        Args:
            q_values: Action-value estimates Q(s,·), shape (n_actions,)
            temperature: Temperature parameter τ > 0

        Returns:
            Sampled action from softmax distribution

        Raises:
            ValueError: If temperature is not positive

        Summary:
            Softmax provides smooth interpolation between random and greedy
            selection, making it suitable when Q-value magnitudes carry
            meaningful information about action quality.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        # Numerical stability: subtract max before exponentiating
        q_scaled = (q_values - np.max(q_values)) / max(temperature, 1e-8)

        # Clip to prevent overflow in exp()
        exp_q = np.exp(np.clip(q_scaled, -700, 700))

        # Normalize to probability distribution
        probs = exp_q / (np.sum(exp_q) + 1e-10)

        return int(np.random.choice(len(q_values), p=probs))

    def _ucb(
        self,
        q_values: np.ndarray,
        action_counts: np.ndarray,
        total_count: int,
        c: float,
    ) -> int:
        """Upper Confidence Bound (UCB) exploration strategy.

        Core Idea:
            "Optimism in the face of uncertainty" - select actions based on
            both estimated value AND uncertainty. Less-visited actions have
            larger confidence bounds, encouraging exploration.

        Mathematical Formulation:
            A_t = argmax_a [Q(s,a) + c·√(ln(t) / N(s,a))]

            Components:
            - Q(s,a): Exploitation term (estimated value)
            - c·√(ln(t)/N(s,a)): Exploration bonus (confidence bound)
            - N(s,a): Visit count for state-action pair
            - t: Total timesteps
            - c: Exploration coefficient (typically √2 for optimal regret)

        Theoretical Foundation:
            Based on Hoeffding inequality. With probability 1-δ:
            |Q̂(s,a) - Q*(s,a)| ≤ √(ln(1/δ) / 2N(s,a))

            Setting δ = 1/t gives the ln(t) term, ensuring all actions are
            tried infinitely often as t → ∞.

        Regret Bounds:
            UCB achieves O(√(K·T·ln(T))) regret for K-armed bandits,
            which is optimal up to logarithmic factors.

        Problem Context:
            UCB addresses the exploration problem more principally than
            ε-greedy or softmax by explicitly modeling uncertainty.

        Algorithm Comparison:
            vs ε-greedy: UCB's exploration is directed (toward uncertain
                        states); ε-greedy is random
            vs Softmax: UCB considers uncertainty; softmax only considers values

        Complexity:
            Time: O(|A|) for computing UCB values
            Space: O(|A|) for storing action counts

        Args:
            q_values: Current Q-value estimates, shape (n_actions,)
            action_counts: Visit counts N(s,·), shape (n_actions,)
            total_count: Total visits to this state
            c: Exploration coefficient, c ≥ 0

        Returns:
            Action with maximum UCB value

        Raises:
            ValueError: If c is negative

        Implementation Notes:
            - Unvisited actions have infinite UCB → explored first
            - Small constant (1e-10) added to denominator for stability

        Summary:
            UCB provides theoretically grounded exploration by maintaining
            confidence bounds on Q-value estimates. Particularly effective
            in stationary environments where uncertainty decreases with samples.
        """
        if c < 0:
            raise ValueError(f"UCB coefficient must be non-negative, got {c}")

        # Prioritize unvisited actions (infinite confidence bound)
        if np.any(action_counts == 0):
            unvisited = np.where(action_counts == 0)[0]
            return int(np.random.choice(unvisited))

        # Compute UCB values
        exploration_bonus = c * np.sqrt(
            np.log(total_count + 1) / (action_counts + 1e-10)
        )
        ucb_values = q_values + exploration_bonus

        return int(np.argmax(ucb_values))


def create_exploration_schedule(
    strategy: ExplorationStrategy,
    initial_value: float,
    final_value: float,
    decay_steps: int,
    decay_type: str = "exponential",
) -> callable:
    """Create a callable exploration parameter schedule.

    Core Idea:
        Exploration parameters (ε for ε-greedy, τ for softmax) typically
        decrease over training to shift from exploration to exploitation.

    Mathematical Formulation:
        Linear decay:
            value(t) = initial - (initial - final) × (t / decay_steps)

        Exponential decay:
            value(t) = final + (initial - final) × exp(-t / (decay_steps/5))

    Args:
        strategy: Which exploration strategy this schedule is for
        initial_value: Starting parameter value
        final_value: Minimum parameter value
        decay_steps: Number of steps over which to decay
        decay_type: "linear" or "exponential"

    Returns:
        Callable that takes current step and returns parameter value

    Example:
        >>> schedule = create_exploration_schedule(
        ...     ExplorationStrategy.EPSILON_GREEDY,
        ...     initial_value=1.0, final_value=0.01, decay_steps=10000
        ... )
        >>> schedule(0)    # Returns 1.0
        >>> schedule(5000) # Returns ~0.5 (linear) or ~0.37 (exponential)
    """
    if decay_type == "linear":

        def schedule(step: int) -> float:
            progress = min(step / max(decay_steps, 1), 1.0)
            return initial_value - (initial_value - final_value) * progress

    elif decay_type == "exponential":
        decay_rate = decay_steps / 5  # Decay constant

        def schedule(step: int) -> float:
            return final_value + (initial_value - final_value) * np.exp(
                -step / max(decay_rate, 1)
            )

    else:
        raise ValueError(f"Unknown decay type: {decay_type}")

    return schedule


if __name__ == "__main__":
    print("Exploration Strategy Unit Tests")
    print("=" * 60)

    # Test 1: ε-greedy basic functionality
    mixin = ExplorationMixin()
    q_vals = np.array([1.0, 3.0, 2.0, 0.5])

    # With ε=0, should always pick argmax
    actions = [mixin._epsilon_greedy(q_vals, 0.0) for _ in range(100)]
    assert all(a == 1 for a in actions), "ε=0 should be greedy"
    print("✓ Test 1: ε-greedy with ε=0 is greedy")

    # Test 2: ε-greedy explores with ε>0
    np.random.seed(42)
    actions = [mixin._epsilon_greedy(q_vals, 0.5) for _ in range(1000)]
    unique_actions = set(actions)
    assert len(unique_actions) > 1, "ε=0.5 should explore"
    print("✓ Test 2: ε-greedy explores with ε>0")

    # Test 3: Softmax temperature effect
    np.random.seed(42)
    cold_actions = [mixin._softmax(q_vals, 0.1) for _ in range(1000)]
    hot_actions = [mixin._softmax(q_vals, 10.0) for _ in range(1000)]

    # Low temperature should concentrate on best action
    cold_best_freq = cold_actions.count(1) / 1000
    hot_best_freq = hot_actions.count(1) / 1000
    assert cold_best_freq > hot_best_freq, "Low τ should be more greedy"
    print("✓ Test 3: Softmax temperature controls exploration")

    # Test 4: UCB prioritizes unvisited
    counts = np.array([10, 0, 5, 3])
    action = mixin._ucb(q_vals, counts, 18, c=2.0)
    assert action == 1, "UCB should pick unvisited action"
    print("✓ Test 4: UCB prioritizes unvisited actions")

    # Test 5: UCB exploration bonus
    counts = np.array([100, 100, 100, 1])  # Action 3 rarely visited
    np.random.seed(42)
    actions = [mixin._ucb(q_vals, counts, 301, c=2.0) for _ in range(100)]
    # Action 3 has lowest Q but highest uncertainty
    assert 3 in actions, "UCB should explore uncertain actions"
    print("✓ Test 5: UCB adds exploration bonus for uncertain actions")

    # Test 6: Schedule creation
    schedule = create_exploration_schedule(
        ExplorationStrategy.EPSILON_GREEDY,
        initial_value=1.0,
        final_value=0.01,
        decay_steps=1000,
        decay_type="linear",
    )
    assert np.isclose(schedule(0), 1.0), "Schedule start incorrect"
    assert np.isclose(schedule(1000), 0.01), "Schedule end incorrect"
    assert np.isclose(schedule(500), 0.505), "Schedule midpoint incorrect"
    print("✓ Test 6: Exploration schedule works correctly")

    print("=" * 60)
    print("All tests passed!")
