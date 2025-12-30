"""
CliffWalking Environment for Markov Decision Processes

This module implements the CliffWalking environment, a classic benchmark that
demonstrates the trade-off between safety and optimality in MDPs.

Core Idea:
    The agent navigates a grid with a cliff. The optimal path is close to the
    cliff edge (risky but short), while the safe path is longer. This environment
    illustrates the exploration-exploitation trade-off and the importance of
    robust policies.

Mathematical Theory:
    State space: S = {(i,j) | 0 ≤ i < 4, 0 ≤ j < 12} (4x12 grid)
    Action space: A = {UP, DOWN, LEFT, RIGHT}
    Transition: Deterministic movement
    Reward: R(s,a,s') = -1 for each step, -100 if falling off cliff, +0 at goal

Problem Statement:
    CliffWalking demonstrates that optimal policies can be risky. It's used to
    compare on-policy (SARSA) vs off-policy (Q-learning) algorithms.

Complexity:
    - State space: 48 states (4x12 grid)
    - Action space: 4 actions
    - Transition model: Deterministic
"""

from typing import Set, Tuple, Optional
import numpy as np
from ..core.mdp import (
    MarkovDecisionProcess, State, Action, TransitionModel, RewardFunction
)


class CliffWalking(MarkovDecisionProcess):
    """
    CliffWalking environment for MDPs.

    Core Idea:
        4x12 grid where agent starts at bottom-left, goal is bottom-right.
        Bottom row (except start and goal) is a cliff. Falling off cliff gives
        large negative reward and resets to start.

    Mathematical Theory:
        State: (row, col) on 4x12 grid
        Actions: {UP, DOWN, LEFT, RIGHT}
        Transitions: Deterministic
        Rewards: -1 per step, -100 for cliff, 0 at goal

    Problem Statement:
        Demonstrates risk-reward trade-off. Optimal policy hugs the cliff
        (risky but efficient), while safe policy stays away from cliff.

    Complexity:
        - States: 48
        - Actions: 4
        - Transitions: Deterministic
    """

    # Action definitions
    UP = Action(0, "UP")
    DOWN = Action(1, "DOWN")
    LEFT = Action(2, "LEFT")
    RIGHT = Action(3, "RIGHT")

    # Standard CliffWalking dimensions
    HEIGHT = 4
    WIDTH = 12

    def __init__(self, height: int = HEIGHT, width: int = WIDTH,
                 discount_factor: float = 0.99):
        """
        Initialize CliffWalking.

        Args:
            height: Grid height (default 4)
            width: Grid width (default 12)
            discount_factor: Discount factor γ

        Raises:
            ValueError: If dimensions invalid
        """
        if height <= 0 or width <= 0:
            raise ValueError(f"Grid dimensions must be positive, got {height}x{width}")
        if height < 2 or width < 2:
            raise ValueError(f"Grid must be at least 2x2, got {height}x{width}")

        super().__init__(discount_factor=discount_factor)

        self.height = height
        self.width = width

        # Add all grid positions as states
        for i in range(height):
            for j in range(width):
                state_id = i * width + j
                state = State(state_id, features=np.array([i, j]))
                self.add_state(state)

        # Add actions
        self.add_action(self.UP)
        self.add_action(self.DOWN)
        self.add_action(self.LEFT)
        self.add_action(self.RIGHT)

        # Initialize transition model and reward function
        self.transition_model = TransitionModel()
        self.reward_function = RewardFunction(reward_type="state_action_next_state")

        # Set start and goal
        self._start_pos = (height - 1, 0)
        self._goal_pos = (height - 1, width - 1)
        self.set_initial_state(self._get_state(*self._start_pos))
        self.add_terminal_state(self._get_state(*self._goal_pos))

    def build_transitions(self) -> None:
        """
        Build deterministic transition model.

        Handles:
        - Boundary conditions (agent stays in place)
        - Cliff (agent falls and resets to start)
        - Goal (terminal state)
        """
        for row in range(self.height):
            for col in range(self.width):
                state = self._get_state(row, col)

                # Terminal states have no transitions
                if self.is_terminal(state):
                    continue

                for action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
                    self._build_transition_for_action(state, row, col, action)

    def _build_transition_for_action(self, state: State, row: int, col: int,
                                     action: Action) -> None:
        """
        Build transition for single state-action pair.

        Args:
            state: Current state
            row: Current row
            col: Current column
            action: Action to take
        """
        # Determine next position
        next_row, next_col = self._get_next_position(row, col, action)

        # Check if falling off cliff
        if self._is_cliff(next_row, next_col):
            # Fall off cliff - reset to start
            next_state = self._get_state(*self._start_pos)
            reward = -100.0
        else:
            next_state = self._get_state(next_row, next_col)
            if (next_row, next_col) == self._goal_pos:
                reward = 0.0
            else:
                reward = -1.0

        # Add deterministic transition
        self.transition_model.set_transition(state, action, next_state, 1.0)
        self.reward_function.set_reward(state, action, next_state, reward=reward)

    def _get_next_position(self, row: int, col: int, action: Action) -> Tuple[int, int]:
        """
        Get next position after taking action.

        Handles boundary conditions.

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

        # Handle boundaries - stay in place
        next_row = max(0, min(next_row, self.height - 1))
        next_col = max(0, min(next_col, self.width - 1))

        return next_row, next_col

    def _is_cliff(self, row: int, col: int) -> bool:
        """
        Check if position is on cliff.

        Cliff is the bottom row except start and goal positions.

        Args:
            row: Row index
            col: Column index

        Returns:
            True if on cliff, False otherwise
        """
        if row != self.height - 1:
            return False

        # Bottom row is cliff except start (0) and goal (width-1)
        return 0 < col < self.width - 1

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

    def _get_position(self, state: State) -> Tuple[int, int]:
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
                if (row, col) == self._goal_pos:
                    line += "G "
                elif self._is_cliff(row, col):
                    line += "C "
                elif (row, col) == self._start_pos:
                    if policy:
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
                            line += "S "
                    else:
                        line += "S "
                else:
                    if policy:
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

    def get_grid_info(self) -> str:
        """Get information about grid layout."""
        return (
            f"CliffWalking Grid: {self.height}x{self.width}\n"
            f"Start: {self._start_pos}\n"
            f"Goal: {self._goal_pos}\n"
            f"Cliff: Bottom row (row {self.height-1}), columns 1 to {self.width-2}"
        )
