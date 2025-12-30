"""
GridWorld Environment for Markov Decision Processes

This module implements the classic GridWorld environment, a fundamental benchmark
for MDP algorithms. The agent navigates a grid, avoiding obstacles and reaching goals.

Core Idea:
    GridWorld is a canonical MDP environment where the agent moves on a 2D grid.
    States are grid positions, actions are movements (up, down, left, right),
    and rewards are given for reaching goal states.

Mathematical Theory:
    State space: S = {(i,j) | 0 ≤ i < height, 0 ≤ j < width}
    Action space: A = {up, down, left, right}
    Transition: Deterministic or stochastic movement
    Reward: R(s,a,s') = +1 if s' is goal, -1 if s' is obstacle, 0 otherwise

Problem Statement:
    GridWorld provides a simple yet rich environment for testing MDP solvers.
    It's intuitive to visualize and understand, making it ideal for education.

Complexity:
    - State space: O(height × width)
    - Action space: O(4)
    - Transition model: O(height × width × 4)
"""

from typing import Set, Tuple, Optional
import numpy as np
from ..core.mdp import (
    MarkovDecisionProcess, State, Action, TransitionModel, RewardFunction
)


class GridWorld(MarkovDecisionProcess):
    """
    GridWorld environment for MDPs.

    Core Idea:
        2D grid where agent navigates from start to goal, avoiding obstacles.
        Provides intuitive visualization and analysis of MDP algorithms.

    Mathematical Theory:
        State: (row, col) position on grid
        Actions: {UP, DOWN, LEFT, RIGHT}
        Transitions: Deterministic or stochastic
        Rewards: Goal +1, Obstacle -1, Empty 0

    Problem Statement:
        Canonical benchmark for MDP algorithms with clear optimal policy.

    Complexity:
        - States: O(height × width)
        - Transitions: O(height × width × 4)
    """

    # Action definitions
    UP = Action(0, "UP")
    DOWN = Action(1, "DOWN")
    LEFT = Action(2, "LEFT")
    RIGHT = Action(3, "RIGHT")

    def __init__(self, height: int = 5, width: int = 5, discount_factor: float = 0.99):
        """
        Initialize GridWorld.

        Args:
            height: Grid height
            width: Grid width
            discount_factor: Discount factor γ

        Raises:
            ValueError: If dimensions invalid
        """
        if height <= 0 or width <= 0:
            raise ValueError(f"Grid dimensions must be positive, got {height}x{width}")

        super().__init__(discount_factor=discount_factor)

        self.height = height
        self.width = width
        self._goal_states: Set[Tuple[int, int]] = set()
        self._obstacle_states: Set[Tuple[int, int]] = set()
        self._stochasticity = 0.0  # Probability of random action

        # Add all grid positions as states
        for i in range(height):
            for j in range(width):
                state = State(i * width + j, features=np.array([i, j]))
                self.add_state(state)

        # Add actions
        self.add_action(self.UP)
        self.add_action(self.DOWN)
        self.add_action(self.LEFT)
        self.add_action(self.RIGHT)

        # Initialize transition model and reward function
        self.transition_model = TransitionModel()
        self.reward_function = RewardFunction(reward_type="state_action_next_state")

    def set_goal(self, row: int, col: int, reward: float = 1.0) -> None:
        """
        Set goal state.

        Args:
            row: Goal row
            col: Goal column
            reward: Reward for reaching goal

        Raises:
            ValueError: If position invalid
        """
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"Invalid position ({row}, {col})")

        self._goal_states.add((row, col))
        state = self._get_state(row, col)
        self.add_terminal_state(state)

    def set_obstacle(self, row: int, col: int) -> None:
        """
        Set obstacle state (impassable).

        Args:
            row: Obstacle row
            col: Obstacle column

        Raises:
            ValueError: If position invalid
        """
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"Invalid position ({row}, {col})")

        self._obstacle_states.add((row, col))

    def set_start(self, row: int, col: int) -> None:
        """
        Set start state.

        Args:
            row: Start row
            col: Start column

        Raises:
            ValueError: If position invalid
        """
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"Invalid position ({row}, {col})")

        state = self._get_state(row, col)
        self.set_initial_state(state)

    def set_stochasticity(self, stochasticity: float) -> None:
        """
        Set action stochasticity.

        With probability stochasticity, agent takes random action instead of intended.

        Args:
            stochasticity: Probability of random action in [0,1]

        Raises:
            ValueError: If stochasticity not in [0,1]
        """
        if not 0.0 <= stochasticity <= 1.0:
            raise ValueError(f"Stochasticity must be in [0,1], got {stochasticity}")

        self._stochasticity = stochasticity

    def build_transitions(self) -> None:
        """
        Build transition model for all state-action pairs.

        Handles:
        - Boundary conditions (agent stays in place)
        - Obstacles (agent stays in place)
        - Stochastic actions (random movement)
        """
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) in self._obstacle_states:
                    continue

                state = self._get_state(row, col)

                for action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
                    self._build_transition_for_action(state, row, col, action)

    def _build_transition_for_action(self, state: State, row: int, col: int,
                                     action: Action) -> None:
        """
        Build transition probabilities for single state-action pair.

        Args:
            state: Current state
            row: Current row
            col: Current column
            action: Action to take
        """
        # Determine intended next position
        next_row, next_col = self._get_next_position(row, col, action)

        # Handle stochasticity
        if self._stochasticity > 0:
            # With probability (1-stochasticity), take intended action
            self._add_transition(state, action, next_row, next_col, 1.0 - self._stochasticity)

            # With probability stochasticity, take random action
            random_prob = self._stochasticity / 4.0
            for random_action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
                rand_row, rand_col = self._get_next_position(row, col, random_action)
                self._add_transition(state, action, rand_row, rand_col, random_prob)
        else:
            # Deterministic transition
            self._add_transition(state, action, next_row, next_col, 1.0)

    def _add_transition(self, state: State, action: Action, next_row: int,
                       next_col: int, probability: float) -> None:
        """
        Add transition probability.

        Args:
            state: Current state
            action: Action taken
            next_row: Next row
            next_col: Next column
            probability: Transition probability
        """
        next_state = self._get_state(next_row, next_col)

        # Add transition
        self.transition_model.set_transition(state, action, next_state, probability)

        # Add reward
        if (next_row, next_col) in self._goal_states:
            reward = 1.0
        elif (next_row, next_col) in self._obstacle_states:
            reward = -1.0
        else:
            reward = -0.01  # Small penalty for each step

        self.reward_function.set_reward(state, action, next_state, reward=reward)

    def _get_next_position(self, row: int, col: int, action: Action) -> Tuple[int, int]:
        """
        Get next position after taking action.

        Handles boundary conditions and obstacles.

        Args:
            row: Current row
            col: Current column
            action: Action to take

        Returns:
            Tuple of (next_row, next_col)
        """
        if action == self.UP:
            next_row, next_col = row - 1, col
        elif action == self.DOWN:
            next_row, next_col = row + 1, col
        elif action == self.LEFT:
            next_row, next_col = row, col - 1
        elif action == self.RIGHT:
            next_row, next_col = row, col + 1
        else:
            raise ValueError(f"Unknown action: {action}")

        # Handle boundaries
        next_row = max(0, min(next_row, self.height - 1))
        next_col = max(0, min(next_col, self.width - 1))

        # Handle obstacles
        if (next_row, next_col) in self._obstacle_states:
            next_row, next_col = row, col

        return next_row, next_col

    def _get_state(self, row: int, col: int) -> State:
        """
        Get state object for grid position.

        Args:
            row: Row index
            col: Column index

        Returns:
            State object

        Raises:
            ValueError: If position invalid
        """
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"Invalid position ({row}, {col})")

        state_id = row * self.width + col
        for state in self.states:
            if state.state_id == state_id:
                return state

        raise RuntimeError(f"State not found for position ({row}, {col})")

    def get_position(self, state: State) -> Tuple[int, int]:
        """
        Get grid position from state.

        Args:
            state: State object

        Returns:
            Tuple of (row, col)
        """
        state_id = state.state_id
        row = state_id // self.width
        col = state_id % self.width
        return row, col

    def render(self, policy=None, value_function=None) -> str:
        """
        Render grid visualization.

        Args:
            policy: Optional policy to display
            value_function: Optional value function to display

        Returns:
            String representation of grid
        """
        output = []

        for row in range(self.height):
            line = ""
            for col in range(self.width):
                if (row, col) in self._goal_states:
                    line += "G "
                elif (row, col) in self._obstacle_states:
                    line += "# "
                elif policy:
                    state = self._get_state(row, col)
                    try:
                        action = policy.get_action(state)
                        if action == self.UP:
                            line += "↑ "
                        elif action == self.DOWN:
                            line += "↓ "
                        elif action == self.LEFT:
                            line += "← "
                        elif action == self.RIGHT:
                            line += "→ "
                    except:
                        line += ". "
                else:
                    line += ". "

            output.append(line)

        return "\n".join(output)
