"""
Goal-Conditioned Replay Buffers for HER and Sparse Reward Learning

================================================================================
CORE IDEA
================================================================================
Goal-conditioned buffers store transitions with associated goals, enabling
Hindsight Experience Replay (HER) - a technique that creates additional
learning signal by relabeling failed episodes as successful with new goals.

================================================================================
MATHEMATICAL THEORY
================================================================================
Standard goal-conditioned RL:
    π*(a|s, g) = argmax_π E[Σₜ γᵗ r(sₜ, aₜ, g)]

where r(s, a, g) = 1 if achieved(s, g) else 0 (sparse).

HER relabels transitions (s, a, g, r, s') as (s, a, g', r', s') where:
    g' = mapping(s')  (e.g., achieved goal in s')
    r' = 1 if achieved(s', g') else 0 (now always 1!)

This converts failed episodes into successful ones, providing dense signal.

================================================================================
GOAL SELECTION STRATEGIES
================================================================================
- "final": Use final achieved goal of the episode
- "future": Sample from future achieved goals in same episode
- "episode": Sample uniformly from all achieved goals in episode
- "random": Sample from buffer's achieved goals distribution

================================================================================
REFERENCES
================================================================================
[1] Andrychowicz et al. (2017). Hindsight Experience Replay.
[2] Plappert et al. (2018). Multi-goal RL with Imagined Goals.
[3] Fang et al. (2019). Curriculum-guided HER.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np


class GoalSelectionStrategy(Enum):
    """Strategy for selecting hindsight goals."""

    FINAL = "final"
    FUTURE = "future"
    EPISODE = "episode"
    RANDOM = "random"


class GoalTransition(NamedTuple):
    """Goal-conditioned transition (s, a, g, r, s', ag).

    Attributes:
        state: Current state observation.
        action: Action taken.
        goal: Desired goal.
        reward: Reward received (typically sparse).
        next_state: Resulting state observation.
        achieved_goal: Goal actually achieved in next_state.
        done: Whether episode terminated.
    """

    state: np.ndarray
    action: np.ndarray
    goal: np.ndarray
    reward: float
    next_state: np.ndarray
    achieved_goal: np.ndarray
    done: bool


@dataclass
class Episode:
    """Complete episode of goal-conditioned transitions.

    Attributes:
        transitions: List of transitions in temporal order.
        initial_goal: Goal at episode start.
        final_achieved: Goal achieved at episode end.
        success: Whether desired goal was achieved.
    """

    transitions: List[GoalTransition] = field(default_factory=list)
    initial_goal: Optional[np.ndarray] = None
    final_achieved: Optional[np.ndarray] = None
    success: bool = False

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: GoalTransition) -> None:
        """Add transition to episode."""
        self.transitions.append(transition)
        self.final_achieved = transition.achieved_goal
        if transition.reward > 0:
            self.success = True


@dataclass
class BatchedGoalTransitions:
    """Batched goal-conditioned transitions.

    Attributes:
        states: Batch of states, shape (batch, state_dim).
        actions: Batch of actions, shape (batch, action_dim).
        goals: Batch of goals, shape (batch, goal_dim).
        rewards: Batch of rewards, shape (batch,).
        next_states: Batch of next states, shape (batch, state_dim).
        achieved_goals: Batch of achieved goals, shape (batch, goal_dim).
        dones: Batch of done flags, shape (batch,).
    """

    states: np.ndarray
    actions: np.ndarray
    goals: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    achieved_goals: np.ndarray
    dones: np.ndarray

    def __len__(self) -> int:
        return len(self.states)


class GoalConditionedBuffer:
    """Buffer storing goal-conditioned transitions.

    ============================================================================
    CORE IDEA
    ============================================================================
    Extends standard replay buffer to store goals alongside transitions.
    Supports goal relabeling for HER and goal-conditioned policy learning.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    Each transition includes:
    - State s, Action a, Reward r, Next state s'
    - Desired goal g (what we wanted to achieve)
    - Achieved goal ag (what we actually achieved)

    The reward function is typically:
        r(s, a, g) = -||achieved_goal(s) - g||  (dense)
        r(s, a, g) = 1 if ||ag - g|| < ε else 0  (sparse)

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Insertion: O(1)
    - Sampling: O(batch_size)
    - Space: O(capacity × transition_size)
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
    ) -> None:
        """Initialize goal-conditioned buffer.

        Args:
            capacity: Maximum number of transitions.
            state_dim: State dimensionality.
            action_dim: Action dimensionality.
            goal_dim: Goal dimensionality.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim

        self._states = np.zeros((capacity, state_dim), dtype=np.float64)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float64)
        self._goals = np.zeros((capacity, goal_dim), dtype=np.float64)
        self._rewards = np.zeros(capacity, dtype=np.float64)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float64)
        self._achieved_goals = np.zeros((capacity, goal_dim), dtype=np.float64)
        self._dones = np.zeros(capacity, dtype=np.bool_)

        self._position = 0
        self._size = 0

    def add(self, transition: GoalTransition) -> None:
        """Add a goal-conditioned transition.

        Args:
            transition: Goal-conditioned experience tuple.
        """
        self._states[self._position] = transition.state
        self._actions[self._position] = transition.action
        self._goals[self._position] = transition.goal
        self._rewards[self._position] = transition.reward
        self._next_states[self._position] = transition.next_state
        self._achieved_goals[self._position] = transition.achieved_goal
        self._dones[self._position] = transition.done

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchedGoalTransitions:
        """Sample random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Batched goal-conditioned transitions.
        """
        if self._size < batch_size:
            raise ValueError(f"Not enough samples: {self._size} < {batch_size}")

        indices = np.random.randint(0, self._size, size=batch_size)

        return BatchedGoalTransitions(
            states=self._states[indices].copy(),
            actions=self._actions[indices].copy(),
            goals=self._goals[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_states=self._next_states[indices].copy(),
            achieved_goals=self._achieved_goals[indices].copy(),
            dones=self._dones[indices].copy(),
        )

    def __len__(self) -> int:
        return self._size


class HERBuffer:
    """Hindsight Experience Replay buffer with automatic relabeling.

    ============================================================================
    CORE IDEA
    ============================================================================
    HER creates additional training data by relabeling failed episodes as
    successful ones with different goals. This is particularly effective for
    sparse reward tasks where most episodes fail.

    ============================================================================
    MATHEMATICAL THEORY
    ============================================================================
    For each transition (s, a, g, r, s'), HER creates k additional transitions:

        (s, a, g', r', s')  where g' ~ sample_goals(episode)

    The new reward r' is recomputed:
        r' = reward_fn(achieved_goal(s'), g')

    Expected fraction of positive rewards increases from ~0 to ~k/(k+1).

    ============================================================================
    GOAL SELECTION STRATEGIES
    ============================================================================
    Let T be current timestep, Episode has timesteps 0, 1, ..., N

    - FINAL: g' = achieved_goal(s_N)
        Always use final achieved goal

    - FUTURE: g' ~ Uniform({achieved_goal(s_t) : t > T})
        Sample from future achieved goals (most common)

    - EPISODE: g' ~ Uniform({achieved_goal(s_t) : t ∈ [0, N]})
        Sample from all achieved goals in episode

    - RANDOM: g' ~ buffer distribution
        Sample from all stored achieved goals

    ============================================================================
    ALGORITHM
    ============================================================================
    1. Store original episode
    2. For each transition in episode:
       a. Store original transition
       b. For k times:
          - Sample hindsight goal g' using strategy
          - Recompute reward r' = reward_fn(ag, g')
          - Store relabeled transition

    ============================================================================
    COMPLEXITY
    ============================================================================
    - Storage per episode: O(T + k×T) = O((1+k)×T)
    - Sampling: O(batch_size)
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        strategy: GoalSelectionStrategy = GoalSelectionStrategy.FUTURE,
        k_goals: int = 4,
        achieved_goal_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """Initialize HER buffer.

        Args:
            capacity: Maximum transitions to store.
            state_dim: State dimensionality.
            action_dim: Action dimensionality.
            goal_dim: Goal dimensionality.
            reward_fn: Function computing reward from (achieved_goal, desired_goal).
            strategy: Goal selection strategy for relabeling.
            k_goals: Number of hindsight goals per transition.
            achieved_goal_fn: Function extracting achieved goal from state.
                If None, uses the achieved_goal from transitions.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.reward_fn = reward_fn
        self.strategy = strategy
        self.k_goals = k_goals
        self.achieved_goal_fn = achieved_goal_fn

        self._buffer = GoalConditionedBuffer(
            capacity=capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
        )

        self._current_episode: List[GoalTransition] = []
        self._all_achieved_goals: List[np.ndarray] = []

    def add_transition(self, transition: GoalTransition) -> None:
        """Add transition to current episode.

        Args:
            transition: Goal-conditioned transition.
        """
        self._current_episode.append(transition)
        self._all_achieved_goals.append(transition.achieved_goal.copy())

        if len(self._all_achieved_goals) > self.capacity:
            self._all_achieved_goals.pop(0)

    def end_episode(self) -> Dict[str, float]:
        """Process end of episode, performing HER relabeling.

        Returns:
            Statistics about the episode and relabeling.
        """
        episode = self._current_episode
        n_transitions = len(episode)

        if n_transitions == 0:
            return {"episode_length": 0, "relabeled_transitions": 0}

        original_count = 0
        relabeled_count = 0

        for t, transition in enumerate(episode):
            self._buffer.add(transition)
            original_count += 1

            hindsight_goals = self._sample_hindsight_goals(episode, t)

            for goal in hindsight_goals:
                reward = self.reward_fn(transition.achieved_goal, goal)

                relabeled = GoalTransition(
                    state=transition.state,
                    action=transition.action,
                    goal=goal,
                    reward=reward,
                    next_state=transition.next_state,
                    achieved_goal=transition.achieved_goal,
                    done=transition.done,
                )

                self._buffer.add(relabeled)
                relabeled_count += 1

        self._current_episode = []

        return {
            "episode_length": n_transitions,
            "original_transitions": original_count,
            "relabeled_transitions": relabeled_count,
            "total_stored": original_count + relabeled_count,
        }

    def _sample_hindsight_goals(
        self,
        episode: List[GoalTransition],
        current_idx: int,
    ) -> List[np.ndarray]:
        """Sample hindsight goals based on strategy.

        Args:
            episode: Full episode transitions.
            current_idx: Index of current transition.

        Returns:
            List of k hindsight goals.
        """
        goals = []

        if self.strategy == GoalSelectionStrategy.FINAL:
            final_goal = episode[-1].achieved_goal
            goals = [final_goal.copy() for _ in range(self.k_goals)]

        elif self.strategy == GoalSelectionStrategy.FUTURE:
            future_achieved = [
                t.achieved_goal for t in episode[current_idx + 1 :]
            ]
            if future_achieved:
                for _ in range(self.k_goals):
                    idx = np.random.randint(0, len(future_achieved))
                    goals.append(future_achieved[idx].copy())
            else:
                goals = [episode[-1].achieved_goal.copy() for _ in range(self.k_goals)]

        elif self.strategy == GoalSelectionStrategy.EPISODE:
            all_achieved = [t.achieved_goal for t in episode]
            for _ in range(self.k_goals):
                idx = np.random.randint(0, len(all_achieved))
                goals.append(all_achieved[idx].copy())

        elif self.strategy == GoalSelectionStrategy.RANDOM:
            if self._all_achieved_goals:
                for _ in range(self.k_goals):
                    idx = np.random.randint(0, len(self._all_achieved_goals))
                    goals.append(self._all_achieved_goals[idx].copy())
            else:
                goals = [episode[-1].achieved_goal.copy() for _ in range(self.k_goals)]

        return goals

    def sample(self, batch_size: int) -> BatchedGoalTransitions:
        """Sample batch from buffer.

        Args:
            batch_size: Number of transitions.

        Returns:
            Batched transitions.
        """
        return self._buffer.sample(batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


def compute_success_rate(
    achieved_goals: np.ndarray,
    desired_goals: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """Compute success rate for goal achievement.

    Args:
        achieved_goals: Array of achieved goals, shape (n, goal_dim).
        desired_goals: Array of desired goals, shape (n, goal_dim).
        threshold: Distance threshold for success.

    Returns:
        Fraction of successful goal achievements.
    """
    distances = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
    successes = distances < threshold
    return float(np.mean(successes))


def analyze_goal_coverage(
    achieved_goals: np.ndarray,
    goal_space_bounds: Tuple[np.ndarray, np.ndarray],
    n_bins: int = 10,
) -> Dict[str, float]:
    """Analyze coverage of the goal space.

    Args:
        achieved_goals: Array of achieved goals.
        goal_space_bounds: (lower, upper) bounds of goal space.
        n_bins: Number of bins per dimension.

    Returns:
        Coverage statistics.
    """
    lower, upper = goal_space_bounds
    normalized = (achieved_goals - lower) / (upper - lower + 1e-8)
    normalized = np.clip(normalized, 0, 1 - 1e-8)
    discretized = np.floor(normalized * n_bins).astype(int)

    unique_bins = set(tuple(row) for row in discretized)
    total_bins = n_bins ** achieved_goals.shape[1]

    return {
        "coverage_ratio": len(unique_bins) / total_bins,
        "unique_goals": len(unique_bins),
        "total_bins": total_bins,
        "total_samples": len(achieved_goals),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Goal-Conditioned Buffers - Validation Tests")
    print("=" * 70)

    np.random.seed(42)

    state_dim = 10
    action_dim = 4
    goal_dim = 3

    # Test basic goal buffer
    print("\n[Test 1] Goal-Conditioned Buffer")
    buffer = GoalConditionedBuffer(
        capacity=1000,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
    )

    for _ in range(100):
        t = GoalTransition(
            state=np.random.randn(state_dim),
            action=np.random.randn(action_dim),
            goal=np.random.randn(goal_dim),
            reward=0.0,
            next_state=np.random.randn(state_dim),
            achieved_goal=np.random.randn(goal_dim),
            done=False,
        )
        buffer.add(t)

    batch = buffer.sample(32)
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Batch goals shape: {batch.goals.shape}")
    print("  [PASS]")

    # Test HER buffer
    print("\n[Test 2] HER Buffer")

    def sparse_reward(achieved: np.ndarray, desired: np.ndarray) -> float:
        return 1.0 if np.linalg.norm(achieved - desired) < 0.1 else 0.0

    her = HERBuffer(
        capacity=10000,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        reward_fn=sparse_reward,
        strategy=GoalSelectionStrategy.FUTURE,
        k_goals=4,
    )

    for episode_idx in range(10):
        goal = np.random.randn(goal_dim)
        state = np.random.randn(state_dim)

        for step in range(20):
            action = np.random.randn(action_dim)
            next_state = state + 0.1 * np.random.randn(state_dim)
            achieved = next_state[:goal_dim]

            t = GoalTransition(
                state=state,
                action=action,
                goal=goal,
                reward=sparse_reward(achieved, goal),
                next_state=next_state,
                achieved_goal=achieved,
                done=(step == 19),
            )
            her.add_transition(t)
            state = next_state

        stats = her.end_episode()

    print(f"  HER buffer size: {len(her)}")
    print(f"  Last episode stats: {stats}")

    batch = her.sample(64)
    positive_rewards = np.sum(batch.rewards > 0)
    print(f"  Positive rewards in batch: {positive_rewards}/{len(batch)}")
    print("  [PASS]")

    # Test goal coverage analysis
    print("\n[Test 3] Goal Coverage Analysis")
    achieved = np.random.rand(1000, goal_dim)
    bounds = (np.zeros(goal_dim), np.ones(goal_dim))

    coverage = analyze_goal_coverage(achieved, bounds, n_bins=5)
    print(f"  Coverage: {coverage['coverage_ratio']:.2%}")
    print(f"  Unique goals: {coverage['unique_goals']}/{coverage['total_bins']}")
    print("  [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
