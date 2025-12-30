"""Tabular Reinforcement Learning Agents.

This module provides production-ready implementations of classic temporal-difference
control algorithms: Q-Learning, SARSA, Expected SARSA, and Double Q-Learning.

Core Mathematical Framework:
    All algorithms are based on the Bellman optimality equation:
    Q*(s,a) = E[R + γ max_{a'} Q*(s',a') | s, a]

    They differ in how they approximate the target value:
    - Q-Learning: Uses max (off-policy, biased)
    - SARSA: Uses sampled action (on-policy, higher variance)
    - Expected SARSA: Uses expectation under policy (on-policy, lower variance)
    - Double Q-Learning: Uses two Q-tables to reduce bias

References:
    [1] Watkins (1989). Learning from Delayed Rewards. PhD Thesis.
    [2] Rummery & Niranjan (1994). On-Line Q-Learning Using Connectionist Systems.
    [3] Van Hasselt (2010). Double Q-learning. NeurIPS.
    [4] Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.
"""

from __future__ import annotations

import json
import pickle
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from .exploration import ExplorationMixin, ExplorationStrategy
except ImportError:
    from exploration import ExplorationMixin, ExplorationStrategy


@dataclass
class AgentConfig:
    """Configuration parameters for tabular RL agents.

    Attributes:
        n_actions: Size of discrete action space |A|
        learning_rate: Step size α ∈ (0,1] for Q-value updates
        discount_factor: Discount γ ∈ [0,1] for future rewards
        epsilon: Initial exploration rate for ε-greedy
        epsilon_decay: Multiplicative decay per episode
        epsilon_min: Lower bound on exploration rate
        exploration: Strategy for exploration-exploitation balance
        temperature: Softmax temperature τ > 0
        ucb_c: UCB exploration coefficient c ≥ 0
    """

    n_actions: int
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    exploration: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    temperature: float = 1.0
    ucb_c: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {self.n_actions}")
        if not 0 < self.learning_rate <= 1:
            raise ValueError(
                f"learning_rate must be in (0,1], got {self.learning_rate}"
            )
        if not 0 <= self.discount_factor <= 1:
            raise ValueError(
                f"discount_factor must be in [0,1], got {self.discount_factor}"
            )
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"epsilon must be in [0,1], got {self.epsilon}")
        if self.epsilon_decay <= 0:
            raise ValueError(f"epsilon_decay must be positive, got {self.epsilon_decay}")
        # Auto-adjust epsilon_min if epsilon is set lower than default epsilon_min
        if self.epsilon_min > self.epsilon:
            object.__setattr__(self, "epsilon_min", self.epsilon)
        if not 0 <= self.epsilon_min <= self.epsilon:
            raise ValueError(
                f"epsilon_min must be in [0, epsilon], got {self.epsilon_min}"
            )


class BaseAgent(ABC, ExplorationMixin):
    """Abstract base class for tabular TD learning agents.

    Core Idea:
        Provides common infrastructure for Q-table management, action selection,
        and model persistence. Subclasses implement specific TD update rules.

    Architecture:
        - Hash-based Q-table using defaultdict for lazy initialization
        - Pluggable exploration strategies via ExplorationMixin
        - JSON/pickle serialization for model checkpointing

    Complexity:
        Space: O(|S| × |A|) for Q-table storage
        Action Selection: O(|A|) for exploration strategies
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize agent with configuration.

        Args:
            config: Validated AgentConfig object
        """
        self.config = config
        self.n_actions = config.n_actions
        self.lr = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.exploration = config.exploration
        self.temperature = config.temperature
        self.ucb_c = config.ucb_c

        # Q-table: state → action-value vector
        self.q_table: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )

        # UCB bookkeeping
        self.action_counts: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.int32)
        )
        self.total_steps = 0

    def get_action(self, state: Any, training: bool = True) -> int:
        """Select action according to exploration strategy.

        Args:
            state: Current environment state (must be hashable)
            training: If True, uses exploration; if False, pure greedy

        Returns:
            Selected action index ∈ {0, ..., |A|-1}
        """
        try:
            q_values = self.q_table[state]
        except TypeError:
            raise TypeError(f"State must be hashable, got {type(state)}")

        # Evaluation mode: greedy with tie-breaking
        if not training:
            max_q = np.max(q_values)
            max_actions = np.where(np.isclose(q_values, max_q, rtol=1e-9))[0]
            return int(np.random.choice(max_actions))

        # Training mode: apply exploration strategy
        if self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(q_values, self.epsilon)
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            action = self._softmax(q_values, self.temperature)
        elif self.exploration == ExplorationStrategy.UCB:
            action = self._ucb(
                q_values, self.action_counts[state], self.total_steps, self.ucb_c
            )
        else:
            warnings.warn(
                f"Unknown exploration {self.exploration}, defaulting to ε-greedy"
            )
            action = self._epsilon_greedy(q_values, self.epsilon)

        # Update statistics
        self.action_counts[state][action] += 1
        self.total_steps += 1

        return int(action)

    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """Update Q-values based on experience.

        Must be implemented by subclasses.

        Returns:
            TD error δ = target - Q(s,a)
        """
        pass

    def decay_epsilon(self) -> None:
        """Exponentially decay exploration rate.

        Update Rule:
            ε ← max(ε_min, ε × decay_rate)
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_greedy_policy(self) -> Dict[Any, int]:
        """Extract greedy policy π*(s) = argmax_a Q(s,a).

        Returns:
            Dictionary mapping states to optimal actions
        """
        return {state: int(np.argmax(q_vals)) for state, q_vals in self.q_table.items()}

    def get_value_function(self) -> Dict[Any, float]:
        """Compute state-value function V(s) = max_a Q(s,a).

        Returns:
            Dictionary mapping states to values
        """
        return {
            state: float(np.max(q_vals)) for state, q_vals in self.q_table.items()
        }

    def reset(self) -> None:
        """Reset agent to initial state."""
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )
        self.action_counts = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.int32)
        )
        self.total_steps = 0
        self.epsilon = self.config.epsilon

    def save(self, filepath: Union[str, Path]) -> None:
        """Save agent to disk.

        Args:
            filepath: Output path (.json for human-readable, .pkl for binary)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": {
                "n_actions": self.n_actions,
                "learning_rate": self.lr,
                "discount_factor": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
            },
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "total_steps": self.total_steps,
        }

        if filepath.suffix == ".json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath: Union[str, Path]) -> None:
        """Load agent from checkpoint.

        Args:
            filepath: Path to saved model
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        if filepath.suffix == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

        # Restore Q-table
        self.q_table = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )
        for k, v in data["q_table"].items():
            try:
                key = eval(k)
            except:
                key = int(k) if k.isdigit() else k
            self.q_table[key] = np.array(v, dtype=np.float32)

        self.total_steps = data.get("total_steps", 0)


class QLearningAgent(BaseAgent):
    """Q-Learning: Off-policy temporal-difference control.

    Core Idea:
        Learn the optimal action-value function Q* by using the max operator
        in the TD target, independent of the behavior policy. This allows
        learning from any exploration strategy while converging to optimality.

    Mathematical Principle:
        Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

        TD Target: y = r + γ max_{a'} Q(s',a')
        TD Error: δ = y - Q(s,a)

        The max operator makes this off-policy: we're learning about the
        greedy policy while possibly following an exploratory policy.

    Problem Context:
        Introduced by Watkins (1989) as a breakthrough enabling optimal
        policy learning without environment models. Key insight: separate
        behavior (exploration) from learning target (optimal policy).

    Comparison with Alternatives:
        vs SARSA:
            + Converges to Q* (optimal) regardless of exploration
            + Enables experience replay (DQN)
            - Higher variance, potentially unsafe during training
            - Maximization bias (overestimation)

        vs Double Q-Learning:
            + Simpler with single Q-table
            - Systematic overestimation due to max operator

    Complexity Analysis:
        Time per update: O(|A|) for max operation
        Space: O(|S| × |A|) for Q-table

    Theoretical Properties:
        Convergence: To Q* w.p. 1 under conditions:
            1. All (s,a) pairs visited infinitely often
            2. Learning rate satisfies Σαₜ=∞, Σαₜ²<∞

    Summary:
        Q-Learning revolutionized RL by proving optimal policies can be learned
        from experience without environment models. Its off-policy nature enables
        flexible data collection and reuse, forming the foundation for DQN.

    Example:
        >>> config = AgentConfig(n_actions=4, learning_rate=0.1)
        >>> agent = QLearningAgent(config)
        >>> state = env.reset()
        >>> action = agent.get_action(state)
        >>> next_state, reward, done = env.step(action)
        >>> td_error = agent.update(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        """Initialize Q-Learning agent.

        Args:
            config: Configuration object (overrides other params if provided)
            n_actions: Action space size
            learning_rate: TD step size α
            discount_factor: Discount γ
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate per episode
            epsilon_min: Minimum exploration rate
        """
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
            )
        super().__init__(config)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """Apply Q-Learning update rule.

        Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

        Args:
            state: Current state s
            action: Action taken a
            reward: Reward received r
            next_state: Next state s'
            done: Whether s' is terminal

        Returns:
            TD error δ
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} out of range [0, {self.n_actions})")

        current_q = self.q_table[state][action]

        # TD target with max (off-policy)
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        return float(td_error)


class SARSAAgent(BaseAgent):
    """SARSA: On-policy temporal-difference control.

    Core Idea:
        Learn the value of the policy being executed by using the actual
        next action in the TD target. This accounts for exploration risk,
        leading to more conservative but safer behavior during training.

    Mathematical Principle:
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

        Key Difference from Q-Learning:
            Q-Learning: y = r + γ max_{a'} Q(s',a')  [off-policy]
            SARSA:      y = r + γ Q(s',a')           [on-policy]

    Problem Context:
        SARSA addresses safety concerns in Q-Learning. In risky environments
        (e.g., cliff walking), Q-Learning learns the optimal but dangerous
        path along the cliff, while SARSA learns a safer detour.

    Comparison with Q-Learning:
        + More stable training (accounts for exploration)
        + Safer behavior (considers actual policy risk)
        - Converges to Q^π (policy value), not Q* (optimal)
        - Cannot use experience replay (needs on-policy data)

    Cliff Walking Example:
        Q-Learning: Learns path along cliff edge (optimal but falls often)
        SARSA: Learns path far from cliff (suboptimal but safe)

    Complexity Analysis:
        Time per update: O(1) - no max operation needed
        Space: O(|S| × |A|) for Q-table

    Theoretical Properties:
        Convergence: To Q^π under Robbins-Monro conditions
        GLIE: If ε → 0 in limit, converges to Q*

    Summary:
        SARSA represents the conservative alternative to Q-Learning, trading
        optimal asymptotic performance for safer training behavior. Essential
        for real-world applications where exploration mistakes have consequences.

    Example:
        >>> agent = SARSAAgent(config)
        >>> state = env.reset()
        >>> action = agent.get_action(state)
        >>> while not done:
        ...     next_state, reward, done = env.step(action)
        ...     next_action = agent.get_action(next_state)
        ...     agent.update(state, action, reward, next_state, next_action, done)
        ...     state, action = next_state, next_action
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        """Initialize SARSA agent."""
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
            )
        super().__init__(config)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_action: int,
        done: bool,
    ) -> float:
        """Apply SARSA update rule.

        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

        Args:
            state: Current state s
            action: Action taken a
            reward: Reward received r
            next_state: Next state s'
            next_action: Next action a' (from policy)
            done: Whether s' is terminal

        Returns:
            TD error δ
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} out of range")
        if not 0 <= next_action < self.n_actions:
            raise ValueError(f"Next action {next_action} out of range")

        current_q = self.q_table[state][action]

        # TD target with actual next action (on-policy)
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state][next_action]

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        return float(td_error)


class ExpectedSARSAAgent(BaseAgent):
    """Expected SARSA: Variance-reduced on-policy TD control.

    Core Idea:
        Instead of sampling a single next action (SARSA) or taking max
        (Q-Learning), use the expected Q-value under the current policy.
        This reduces variance while maintaining on-policy properties.

    Mathematical Principle:
        Q(s,a) ← Q(s,a) + α[r + γ E_π[Q(s',·)] - Q(s,a)]

        For ε-greedy policy:
        E_π[Q(s',·)] = (ε/|A|) Σ_a Q(s',a) + (1-ε) max_a Q(s',a)

    Problem Context:
        Expected SARSA bridges the gap between SARSA and Q-Learning:
        - Like SARSA: On-policy, accounts for exploration
        - Like Q-Learning: Low variance (uses expectation, not sample)

    Special Case:
        When ε = 0, Expected SARSA reduces to Q-Learning.

    Comparison:
        vs SARSA:
            + Lower variance (expectation vs sample)
            + Faster convergence in practice
            - O(|A|) instead of O(1) per update

        vs Q-Learning:
            + On-policy (can account for exploration risk)
            + Same computational cost
            - Converges to Q^π, not Q*

    Complexity Analysis:
        Time per update: O(|A|) for computing expectation
        Space: O(|S| × |A|) for Q-table

    Summary:
        Expected SARSA offers the best of both worlds: on-policy learning
        with Q-Learning's variance characteristics. Recommended default
        for TD control when computational overhead is acceptable.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        """Initialize Expected SARSA agent."""
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
            )
        super().__init__(config)

    def _get_expected_q(self, state: Any) -> float:
        """Compute expected Q-value under ε-greedy policy.

        E_π[Q(s,·)] = (ε/|A|) Σ_a Q(s,a) + (1-ε) max_a Q(s,a)

        Args:
            state: State to compute expected value for

        Returns:
            Expected Q-value as scalar
        """
        q_values = self.q_table[state]
        n_actions = len(q_values)

        # ε-greedy action probabilities
        probs = np.ones(n_actions, dtype=np.float32) * (self.epsilon / n_actions)
        best_action = int(np.argmax(q_values))
        probs[best_action] += 1.0 - self.epsilon

        return float(np.dot(probs, q_values))

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """Apply Expected SARSA update rule.

        Q(s,a) ← Q(s,a) + α[r + γ E_π[Q(s',·)] - Q(s,a)]

        Args:
            state: Current state s
            action: Action taken a
            reward: Reward received r
            next_state: Next state s'
            done: Whether s' is terminal

        Returns:
            TD error δ
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} out of range")

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            expected_q = self._get_expected_q(next_state)
            target = reward + self.gamma * expected_q

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        return float(td_error)


class DoubleQLearningAgent(BaseAgent):
    """Double Q-Learning: Bias-corrected off-policy TD control.

    Core Idea:
        Standard Q-Learning overestimates Q-values because E[max X] ≥ max E[X].
        Double Q-Learning decouples action selection from value estimation
        using two independent Q-tables, eliminating this positive bias.

    Mathematical Principle:
        Maintain two Q-tables Q₁ and Q₂. With probability 0.5:

        Update Q₁:
            Q₁(s,a) ← Q₁(s,a) + α[r + γ Q₂(s', argmax_{a'} Q₁(s',a')) - Q₁(s,a)]

        Update Q₂:
            Q₂(s,a) ← Q₂(s,a) + α[r + γ Q₁(s', argmax_{a'} Q₂(s',a')) - Q₂(s,a)]

        Key insight: Action selection uses one table; value estimation uses the other.

    Problem Context:
        The maximization bias in Q-Learning causes:
        - Overconfident value estimates
        - Suboptimal policy choices
        - Slower convergence in stochastic environments

        This was later addressed in DQN via target networks, which is a
        related but distinct approach.

    Proof Sketch (Bias Reduction):
        For i.i.d. random variables X₁, ..., Xₙ with E[Xᵢ] = μ:
        - E[max Xᵢ] ≥ μ (Jensen's inequality)
        - E[X_j], where j = argmax Xᵢ for independent X, is unbiased

    Comparison:
        vs Q-Learning:
            + Eliminates overestimation bias
            + More accurate value estimates
            - Double memory for two Q-tables
            - Slightly slower convergence in deterministic domains

    Complexity Analysis:
        Time per update: O(|A|) for max operation
        Space: O(2 × |S| × |A|) for two Q-tables

    Summary:
        Double Q-Learning provides a principled solution to the maximization
        bias inherent in Q-Learning. Essential foundation for understanding
        modern deep RL algorithms like DDQN.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        """Initialize Double Q-Learning agent."""
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
            )
        super().__init__(config)

        # Second Q-table
        self.q_table2: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )

    def get_action(self, state: Any, training: bool = True) -> int:
        """Select action using combined Q-values.

        For action selection, we use Q₁ + Q₂ to leverage both estimates.
        """
        q_combined = self.q_table[state] + self.q_table2[state]

        if not training:
            max_q = np.max(q_combined)
            max_actions = np.where(np.isclose(q_combined, max_q, rtol=1e-9))[0]
            return int(np.random.choice(max_actions))

        if self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(q_combined, self.epsilon)
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            action = self._softmax(q_combined, self.temperature)
        else:
            action = self._ucb(
                q_combined, self.action_counts[state], self.total_steps, self.ucb_c
            )

        self.action_counts[state][action] += 1
        self.total_steps += 1
        return int(action)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> float:
        """Apply Double Q-Learning update.

        Randomly updates Q₁ or Q₂, using the other for value estimation.

        Args:
            state: Current state s
            action: Action taken a
            reward: Reward received r
            next_state: Next state s'
            done: Whether s' is terminal

        Returns:
            TD error δ
        """
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action {action} out of range")

        if np.random.random() < 0.5:
            # Update Q₁: select with Q₁, evaluate with Q₂
            current_q = self.q_table[state][action]
            if done:
                target = reward
            else:
                best_action = int(np.argmax(self.q_table[next_state]))
                target = reward + self.gamma * self.q_table2[next_state][best_action]
            td_error = target - current_q
            self.q_table[state][action] += self.lr * td_error
        else:
            # Update Q₂: select with Q₂, evaluate with Q₁
            current_q = self.q_table2[state][action]
            if done:
                target = reward
            else:
                best_action = int(np.argmax(self.q_table2[next_state]))
                target = reward + self.gamma * self.q_table[next_state][best_action]
            td_error = target - current_q
            self.q_table2[state][action] += self.lr * td_error

        return float(td_error)

    def reset(self) -> None:
        """Reset both Q-tables."""
        super().reset()
        self.q_table2 = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )

    def get_q_values(self, state: Any) -> np.ndarray:
        """Get average Q-values from both tables."""
        return (self.q_table[state] + self.q_table2[state]) / 2


if __name__ == "__main__":
    print("Agent Unit Tests")
    print("=" * 60)

    # Test 1: Q-Learning update correctness
    agent = QLearningAgent(n_actions=4, learning_rate=0.5, discount_factor=0.9)
    state, next_state = (0, 0), (0, 1)
    agent.q_table[next_state] = np.array([1.0, 2.0, 0.5, 0.3], dtype=np.float32)

    td_error = agent.update(state, 0, -1.0, next_state, False)
    expected_q = 0.5 * (-1.0 + 0.9 * 2.0)  # α[r + γ max Q']

    assert np.isclose(agent.q_table[state][0], expected_q, atol=1e-6)
    print(f"✓ Test 1: Q-Learning update correct (Q = {agent.q_table[state][0]:.4f})")

    # Test 2: SARSA update correctness
    agent = SARSAAgent(n_actions=4, learning_rate=0.5, discount_factor=0.9)
    agent.q_table[next_state] = np.array([1.0, 2.0, 0.5, 0.3], dtype=np.float32)

    td_error = agent.update(state, 0, -1.0, next_state, 1, False)  # next_action=1
    expected_q = 0.5 * (-1.0 + 0.9 * 2.0)

    assert np.isclose(agent.q_table[state][0], expected_q, atol=1e-6)
    print(f"✓ Test 2: SARSA update correct (Q = {agent.q_table[state][0]:.4f})")

    # Test 3: Expected SARSA expected value calculation
    agent = ExpectedSARSAAgent(n_actions=4, epsilon=0.2)
    agent.q_table[state] = np.array([1.0, 2.0, 0.5, 0.5], dtype=np.float32)

    expected_q = agent._get_expected_q(state)
    # ε=0.2: best action (1) has prob 0.8 + 0.05 = 0.85, others 0.05 each
    manual_expected = 0.05 * 1.0 + 0.85 * 2.0 + 0.05 * 0.5 + 0.05 * 0.5
    assert np.isclose(expected_q, manual_expected, atol=1e-6)
    print(f"✓ Test 3: Expected SARSA expectation correct (E[Q] = {expected_q:.4f})")

    # Test 4: Double Q-Learning updates both tables
    agent = DoubleQLearningAgent(n_actions=4, learning_rate=0.5)
    np.random.seed(42)
    for _ in range(20):
        agent.update(state, 0, -1.0, next_state, False)

    q1_updated = agent.q_table[state][0] != 0
    q2_updated = agent.q_table2[state][0] != 0
    assert q1_updated or q2_updated
    print(f"✓ Test 4: Double Q-Learning updates tables (Q1={q1_updated}, Q2={q2_updated})")

    # Test 5: Greedy action selection
    agent = QLearningAgent(n_actions=4, epsilon=0.0)
    agent.q_table[state] = np.array([1.0, 3.0, 2.0, 0.5], dtype=np.float32)

    actions = [agent.get_action(state, training=True) for _ in range(100)]
    assert all(a == 1 for a in actions)
    print("✓ Test 5: Greedy selection correct (ε=0)")

    # Test 6: Save/load functionality
    import tempfile
    agent = QLearningAgent(n_actions=4)
    agent.q_table[(0, 0)] = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        agent.save(f.name)
        new_agent = QLearningAgent(n_actions=4)
        new_agent.load(f.name)
        assert np.allclose(new_agent.q_table[(0, 0)], agent.q_table[(0, 0)])
    print("✓ Test 6: Save/load works correctly")

    print("=" * 60)
    print("All tests passed!")
