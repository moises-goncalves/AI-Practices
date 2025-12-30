"""
FrozenLake Environment for Markov Decision Processes

This module implements the FrozenLake environment, a classic benchmark from
OpenAI Gym. The agent must navigate a frozen lake to reach the goal while
avoiding falling into holes.

Core Idea:
    FrozenLake is a stochastic environment where the agent's actions don't
    always succeed. The agent slides on ice with some probability, making
    the environment inherently uncertain and challenging.

Mathematical Theory:
    State space: S = {(i,j) | 0 ≤ i < 4, 0 ≤ j < 4} (4x4 grid)
    Action space: A = {LEFT, DOWN, RIGHT, UP}
    Transition: Stochastic - agent moves in intended direction with probability 1/3
                and in perpendicular directions with probability 1/3 each
    Reward: R(s,a,s') = +1 if s' is goal, 0 otherwise

Problem Statement:
    FrozenLake demonstrates the importance of handling stochasticity in MDPs.
    The optimal policy must account for the uncertainty in action outcomes.

Complexity:
    - State space: 16 states (4x4 grid)
    - Action space: 4 actions
    - Transition model: Stochastic with 3 possible outcomes per action
"""

from typing import Set, Tuple, Optional
import numpy as np
from ..core.mdp import (
    MarkovDecisionProcess, State, Action, TransitionModel, RewardFunction
)


class FrozenLake(MarkovDecisionProcess):
    """
    FrozenLake environment for MDPs.

    Core Idea:
        Agent navigates a 4x4 frozen lake grid. Some tiles are safe (frozen),
        some are holes (fall through). Agent must reach goal while avoiding holes.
        Actions are stochastic due to slippery ice.

    Mathematical Theory:
        State: (row, col) on 4x4 grid
        Actions: {LEFT, DOWN, RIGHT, UP}
        Transitions: Stochastic - 1/3 probability for each of 3 directions
        Rewards: Goal +1, Hole 0 (episode ends), Safe 0

    Problem Statement:
        Demonstrates stochastic MDPs where optimal policy must handle uncertainty.

    Complexity:
        - States: 16
        - Actions: 4
        - Transitions: Stochastic with 3 outcomes per action
    """

    # Action definitions
    LEFT = Action(0, "LEFT")
    DOWN = Action(1, "DOWN")
    RIGHT = Action(2, "RIGHT")
    UP = Action(3, "UP")

    # Standard 4x4 FrozenLake layout
    # S: Start, F: Frozen, H: Hole, G: Goal
    STANDARD_LAYOUT = [
        "SFFF",
        "FHFH",
        "FFFF",
        "HFFG"
    ]

    def __init__(self, layout: Optional[list] = None, discount_factor: float = 0.99):
        """
        Initialize FrozenLake.

        Args:
            layout: List of strings defining grid layout
                   S: Start, F: Frozen, H: Hole, G: Goal
            discount_factor: Discount factor γ

        Raises:
            ValueError: If layout invalid
        """
        super().__init__(discount_factor=discount_factor)

        if layout is None:
            layout = self.STANDARD_LAYOUT

        self.layout = layout
        self.height = len(layout)
        self.width = len(layout[0]) if layout else 0

        if self.height == 0 or self.width == 0:
            raise ValueError("Layout must be non-empty")

        # Validate layout
        for row in layout:
            if len(row) != self.width:
                raise ValueError("All rows must have same width")

        self._start_pos: Optional[Tuple[int, int]] = None
        self._goal_pos: Optional[Tuple[int, int]] = None
        self._hole_positions: Set[Tuple[int, int]] = set()

        # Parse layout and create states
        self._parse_layout()

        # Add actions
        self.add_action(self.LEFT)
        self.add_action(self.DOWN)
        self.add_action(self.RIGHT)
        self.add_action(self.UP)

        # Initialize transition model and reward function
        self.transition_model = TransitionModel()
        self.reward_function = RewardFunction(reward_type="state_action_next_state")

    def _parse_layout(self) -> None:
        """
        Parse layout string and create states.

        Raises:
            ValueError: If layout invalid
        """
        for row in range(self.height):
            for col in range(self.width):
                cell = self.layout[row][col]

                if cell not in ['S', 'F', 'H', 'G']:
                    raise ValueError(f"Invalid cell '{cell}' at ({row}, {col})")

                state_id = row * self.width + col
                state = State(state_id, features=np.array([row, col]))
                self.add_state(state)

                if cell == 'S':
                    if self._start_pos is not None:
                        raise ValueError("Multiple start positions")
                    self._start_pos = (row, col)
                    self.set_initial_state(state)

                elif cell == 'G':
                    if self._goal_pos is not None:
                        raise ValueError("Multiple goal positions")
                    self._goal_pos = (row, col)
                    self.add_terminal_state(state)

                elif cell == 'H':
                    self._hole_positions.add((row, col))
                    self.add_terminal_state(state)

        if self._start_pos is None:
            raise ValueError("No start position in layout")
        if self._goal_pos is None:
            raise ValueError("No goal position in layout")

    def build_transitions(self) -> None:
        """
        Build stochastic transition model.

        With probability 1/3, agent moves in intended direction.
        With probability 1/3 each, agent moves in perpendicular directions.
        """
        for row in range(self.height):
            for col in range(self.width):
                if self.layout[row][col] in ['H', 'G']:
                    # Terminal states have no transitions
                    continue

                state = self._get_state(row, col)

                for action in [self.LEFT, self.DOWN, self.RIGHT, self.UP]:
                    self._build_stochastic_transition(state, row, col, action)

    def _build_stochastic_transition(self, state: State, row: int, col: int,
                                     action: Action) -> None:
        """
        Build stochastic transitions for state-action pair.

        Agent moves in intended direction with 1/3 probability,
        and in perpendicular directions with 1/3 probability each.

        Args:
            state: Current state
            row: Current row
            col: Current column
            action: Action to take
        """
        # Get intended direction and perpendicular directions
        if action == self.LEFT:
            intended = (row, col - 1)
            perpendicular = [(row - 1, col), (row + 1, col)]
        elif action == self.DOWN:
            intended = (row + 1, col)
            perpendicular = [(row, col - 1), (row, col + 1)]
        elif action == self.RIGHT:
            intended = (row, col + 1)
            perpendicular = [(row - 1, col), (row + 1, col)]
        elif action == self.UP:
            intended = (row - 1, col)
            perpendicular = [(row, col - 1), (row, col + 1)]
        else:
            raise ValueError(f"Unknown action: {action}")

        # Add transitions with 1/3 probability each
        self._add_stochastic_transition(state, action, intended, 1.0 / 3.0)
        for perp_pos in perpendicular:
            self._add_stochastic_transition(state, action, perp_pos, 1.0 / 3.0)

    def _add_stochastic_transition(self, state: State, action: Action,
                                   next_pos: Tuple[int, int],
                                   probability: float) -> None:
        """
        Add stochastic transition.

        Args:
            state: Current state
            action: Action taken
            next_pos: Next position (may be out of bounds)
            probability: Transition probability
        """
        # Handle boundaries - stay in place if out of bounds
        next_row, next_col = next_pos
        if not (0 <= next_row < self.height and 0 <= next_col < self.width):
            next_row, next_col = self._get_position(state)

        next_state = self._get_state(next_row, next_col)

        # Get existing probability and add to it
        existing_prob = self.transition_model.get_transition_distribution(state, action).get(next_state, 0.0)
        new_prob = existing_prob + probability

        # Add transition
        self.transition_model.set_transition(state, action, next_state, new_prob)

        # Add reward
        if (next_row, next_col) == self._goal_pos:
            reward = 1.0
        else:
            reward = 0.0

        self.reward_function.set_reward(state, action, next_state, reward=reward)

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

    def render(self, policy=None) -> str:
        """
        Render grid visualization.

        Args:
            policy: Optional policy to display

        Returns:
            String representation of grid
        """
        output = []

        for row in range(self.height):
            line = ""
            for col in range(self.width):
                cell = self.layout[row][col]

                if cell == 'G':
                    line += "G "
                elif cell == 'H':
                    line += "H "
                elif cell == 'S':
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
                else:  # 'F'
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
                            line += "F "
                    else:
                        line += "F "

            output.append(line)

        return "\n".join(output)

    def get_layout_description(self) -> str:
        """Get description of layout."""
        return "\n".join(self.layout)
