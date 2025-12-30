"""Reinforcement Learning Environments.

This module provides gridworld environments for testing and comparing
temporal-difference control algorithms.

Core Environment:
    CliffWalkingEnv: Classic environment demonstrating on-policy vs off-policy
    behavioral differences, particularly between Q-Learning and SARSA.

Design Principles:
    - Gymnasium-compatible interface (reset, step, render)
    - Deterministic transitions for reproducible experiments
    - Configurable grid dimensions
    - ASCII visualization support

References:
    [1] Sutton & Barto (2018). Reinforcement Learning: An Introduction, Example 6.6
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class CliffWalkingEnv:
    """Cliff Walking Gridworld Environment.

    Core Idea:
        Classic tabular RL benchmark demonstrating stark behavioral differences
        between on-policy (SARSA) and off-policy (Q-Learning) methods. The agent
        must navigate from start to goal while avoiding a cliff that causes
        severe penalty and reset.

    Environment Layout (Default 4×12):
        ┌─────────────────────────────────────────────┐
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 0
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 1
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 2
        │ S  C  C  C  C  C  C  C  C  C  C  G  │  row 3 (cliff row)
        └─────────────────────────────────────────────┘
          0  1  2  3  4  5  6  7  8  9 10 11  columns

        Legend:
            S: Start position (bottom-left)
            G: Goal position (bottom-right)
            C: Cliff cells (deadly)
            .: Safe traversable cells

    Mathematical Properties:
        State Space: |S| = height × width (default 48)
        Action Space: |A| = 4 (Up, Right, Down, Left)
        Transitions: Deterministic, boundary-clipped

    Reward Structure:
        - Step cost: -1 (encourages efficiency)
        - Cliff penalty: -100 (severe punishment) + reset to start
        - Goal reached: 0 (episode terminates)

    Optimal vs Safe Paths:
        Optimal path: Along cliff edge, 13 steps, return = -13
        Safe path: Detour via top, ~30 steps, return ≈ -30

    Behavioral Analysis:
        Q-Learning (Off-Policy):
            - Learns optimal policy (along cliff)
            - Updates use max, ignoring exploration risk
            - High asymptotic performance
            - Dangerous during training (frequent cliff falls)

        SARSA (On-Policy):
            - Learns safe policy (far from cliff)
            - Updates account for ε-greedy exploration
            - Lower asymptotic performance
            - Safer training (rare cliff falls)

        This demonstrates the fundamental difference: Q-Learning learns
        "what's optimal" while SARSA learns "what will I actually achieve."

    Problem Context:
        Introduced by Sutton & Barto to illustrate on-policy vs off-policy
        learning. The environment is simple enough for tabular methods yet
        exhibits interesting behavioral phenomena.

    Complexity:
        State representation: O(1) space per state
        Transition: O(1) time per step
        Render: O(height × width) time

    Example:
        >>> env = CliffWalkingEnv()
        >>> state = env.reset()  # Returns (3, 0)
        >>> next_state, reward, done = env.step(1)  # Move right
        >>> # If next_state in cliff: reward=-100, next_state=start
        >>> env.render()

    Attributes:
        height: Grid height (rows)
        width: Grid width (columns)
        start: Start cell coordinates (row, col)
        goal: Goal cell coordinates
        cliff: List of cliff cell coordinates
        state: Current agent position
        n_states: Total number of states
        n_actions: Number of available actions (4)
        ACTIONS: Mapping from action index to (row_delta, col_delta)
        ACTION_NAMES: Human-readable action names
    """

    # Action space definition
    ACTIONS = {
        0: (-1, 0),  # Up: decrease row
        1: (0, 1),   # Right: increase column
        2: (1, 0),   # Down: increase row
        3: (0, -1),  # Left: decrease column
    }
    ACTION_NAMES = ["Up", "Right", "Down", "Left"]
    ACTION_NAMES_CN = ["上", "右", "下", "左"]

    def __init__(self, height: int = 4, width: int = 12) -> None:
        """Initialize cliff walking gridworld.

        Args:
            height: Number of rows (≥2 required for cliff)
            width: Number of columns (≥3 for start/cliff/goal)

        Raises:
            ValueError: If grid dimensions are too small
        """
        if height < 2:
            raise ValueError(f"Height must be ≥2, got {height}")
        if width < 3:
            raise ValueError(f"Width must be ≥3, got {width}")

        self.height = height
        self.width = width

        # Special positions
        self.start = (height - 1, 0)           # Bottom-left
        self.goal = (height - 1, width - 1)    # Bottom-right
        self.cliff = [
            (height - 1, j) for j in range(1, width - 1)
        ]  # Bottom row excluding start/goal

        # Current state
        self.state = self.start

        # Environment properties
        self.n_states = height * width
        self.n_actions = 4

    def reset(self) -> Tuple[int, int]:
        """Reset environment to initial state.

        Returns:
            Initial state coordinates (row, col)
        """
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Execute action and observe transition.

        Transition Dynamics:
            1. Compute target position from action
            2. Clip to grid boundaries (no wrap-around)
            3. Check for cliff (reset + heavy penalty)
            4. Check for goal (episode termination)
            5. Normal step (standard cost)

        Args:
            action: Action index ∈ {0, 1, 2, 3}

        Returns:
            Tuple of (next_state, reward, done):
                - next_state: (row, col) coordinates
                - reward: Immediate reward signal
                - done: True if episode terminates (goal reached)

        Raises:
            ValueError: If action is invalid
        """
        if action not in self.ACTIONS:
            raise ValueError(
                f"Invalid action {action}, must be in {list(self.ACTIONS.keys())}"
            )

        # Compute next position with boundary clipping
        di, dj = self.ACTIONS[action]
        new_i = int(np.clip(self.state[0] + di, 0, self.height - 1))
        new_j = int(np.clip(self.state[1] + dj, 0, self.width - 1))
        next_state = (new_i, new_j)

        # Check cliff collision
        if next_state in self.cliff:
            self.state = self.start
            return self.state, -100.0, False

        self.state = next_state

        # Check goal reached
        if self.state == self.goal:
            return self.state, 0.0, True

        # Standard step
        return self.state, -1.0, False

    def render(
        self,
        path: Optional[List[Tuple[int, int]]] = None,
        show_agent: bool = True,
    ) -> str:
        """Render current state as ASCII art.

        Args:
            path: Optional trajectory to visualize with '*' markers
            show_agent: Whether to show current agent position with '@'

        Returns:
            String representation of gridworld
        """
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        # Mark cliff
        for pos in self.cliff:
            grid[pos[0]][pos[1]] = "C"

        # Mark start and goal
        grid[self.start[0]][self.start[1]] = "S"
        grid[self.goal[0]][self.goal[1]] = "G"

        # Mark trajectory
        if path:
            for pos in path[1:-1]:
                if (
                    pos not in self.cliff
                    and pos != self.start
                    and pos != self.goal
                ):
                    grid[pos[0]][pos[1]] = "*"

        # Mark current agent position
        if show_agent and self.state != self.start and self.state != self.goal:
            if self.state not in self.cliff:
                grid[self.state[0]][self.state[1]] = "@"

        # Build bordered output
        border_h = "┌" + "─" * (self.width * 2 + 1) + "┐"
        border_b = "└" + "─" * (self.width * 2 + 1) + "┘"

        lines = [border_h]
        for row in grid:
            lines.append("│ " + " ".join(row) + " │")
        lines.append(border_b)

        output = "\n".join(lines)
        print(output)
        return output

    def get_optimal_path(self) -> List[Tuple[int, int]]:
        """Return optimal (risky) path along cliff edge.

        Returns:
            Sequence of states from start to goal (shortest path)
        """
        return [(self.height - 1, j) for j in range(self.width)]

    def get_safe_path(self) -> List[Tuple[int, int]]:
        """Return conservative path avoiding cliff.

        Returns:
            Sequence of states taking detour via top of grid
        """
        path = [self.start]

        # Move up to top row
        for i in range(self.height - 2, -1, -1):
            path.append((i, 0))

        # Move right along top row
        for j in range(1, self.width):
            path.append((0, j))

        # Move down to goal
        for i in range(1, self.height):
            path.append((i, self.width - 1))

        return path

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) state to flat index.

        Args:
            state: (row, col) coordinates

        Returns:
            Flat index in range [0, n_states)
        """
        return state[0] * self.width + state[1]

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat index to (row, col) state.

        Args:
            index: Flat index in range [0, n_states)

        Returns:
            (row, col) coordinates
        """
        return (index // self.width, index % self.width)


class FrozenLakeEnv:
    """Frozen Lake Environment (Simplified Implementation).

    Core Idea:
        Navigate a frozen lake from start to goal, avoiding holes.
        Unlike CliffWalking, transitions can be stochastic (slippery ice).

    Layout (4×4):
        S F F F     S: Start
        F H F H     F: Frozen (safe)
        F F F H     H: Hole (fall through)
        H F F G     G: Goal

    This is a simplified version; for full features use Gymnasium's FrozenLake-v1.
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = ["Up", "Right", "Down", "Left"]

    def __init__(
        self,
        map_name: str = "4x4",
        is_slippery: bool = False,
    ) -> None:
        """Initialize Frozen Lake environment.

        Args:
            map_name: Map size ("4x4" or "8x8")
            is_slippery: If True, transitions are stochastic
        """
        if map_name == "4x4":
            self.desc = [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG",
            ]
        else:
            self.desc = [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG",
            ]

        self.height = len(self.desc)
        self.width = len(self.desc[0])
        self.is_slippery = is_slippery

        # Find special positions
        for i, row in enumerate(self.desc):
            for j, cell in enumerate(row):
                if cell == "S":
                    self.start = (i, j)
                elif cell == "G":
                    self.goal = (i, j)

        self.holes = [
            (i, j)
            for i, row in enumerate(self.desc)
            for j, cell in enumerate(row)
            if cell == "H"
        ]

        self.state = self.start
        self.n_states = self.height * self.width
        self.n_actions = 4

    def reset(self) -> Tuple[int, int]:
        """Reset to start state."""
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Execute action.

        If slippery, there's 1/3 chance of intended direction,
        1/3 chance of perpendicular directions each.
        """
        if self.is_slippery:
            # Stochastic transitions
            perpendicular = [(action - 1) % 4, (action + 1) % 4]
            choices = [action] + perpendicular
            action = np.random.choice(choices)

        di, dj = self.ACTIONS[action]
        new_i = int(np.clip(self.state[0] + di, 0, self.height - 1))
        new_j = int(np.clip(self.state[1] + dj, 0, self.width - 1))
        next_state = (new_i, new_j)

        self.state = next_state

        # Check termination
        if self.state in self.holes:
            return self.state, 0.0, True  # Fell in hole

        if self.state == self.goal:
            return self.state, 1.0, True  # Reached goal

        return self.state, 0.0, False  # Continue

    def render(self) -> str:
        """Render current state."""
        output_lines = []
        for i, row in enumerate(self.desc):
            line = ""
            for j, cell in enumerate(row):
                if (i, j) == self.state:
                    line += "@ "
                else:
                    line += cell + " "
            output_lines.append(line)
        output = "\n".join(output_lines)
        print(output)
        return output


if __name__ == "__main__":
    print("Environment Unit Tests")
    print("=" * 60)

    # Test 1: CliffWalking initialization
    env = CliffWalkingEnv()
    assert env.start == (3, 0), "Start position incorrect"
    assert env.goal == (3, 11), "Goal position incorrect"
    assert len(env.cliff) == 10, "Cliff length incorrect"
    print("✓ Test 1: CliffWalking initialization correct")

    # Test 2: Reset returns start state
    state = env.reset()
    assert state == env.start, "Reset should return start state"
    print("✓ Test 2: Reset returns start state")

    # Test 3: Step movement
    env.reset()
    next_state, reward, done = env.step(0)  # Up
    assert next_state == (2, 0), f"Move up incorrect: {next_state}"
    assert reward == -1.0, "Step reward should be -1"
    assert not done, "Episode should not end"
    print("✓ Test 3: Step movement correct")

    # Test 4: Boundary clipping
    env.reset()
    next_state, _, _ = env.step(3)  # Left (at left edge)
    assert next_state == env.start, "Should stay at boundary"
    print("✓ Test 4: Boundary clipping works")

    # Test 5: Cliff penalty
    env.reset()
    next_state, reward, done = env.step(1)  # Right into cliff
    assert reward == -100.0, "Cliff should give -100 reward"
    assert next_state == env.start, "Should reset to start"
    assert not done, "Episode continues after cliff"
    print("✓ Test 5: Cliff penalty correct")

    # Test 6: Goal termination
    env.reset()
    env.state = (3, 10)  # Position before goal
    next_state, reward, done = env.step(1)  # Right to goal
    assert next_state == env.goal, "Should reach goal"
    assert reward == 0.0, "Goal reward should be 0"
    assert done, "Episode should terminate"
    print("✓ Test 6: Goal termination correct")

    # Test 7: Optimal path
    path = env.get_optimal_path()
    assert path[0] == env.start, "Path should start at start"
    assert path[-1] == env.goal, "Path should end at goal"
    assert len(path) == env.width, "Optimal path should be direct"
    print("✓ Test 7: Optimal path generation correct")

    # Test 8: Safe path
    safe_path = env.get_safe_path()
    assert safe_path[0] == env.start, "Safe path should start at start"
    assert safe_path[-1] == env.goal, "Safe path should end at goal"
    assert len(safe_path) > len(path), "Safe path should be longer"
    print("✓ Test 8: Safe path generation correct")

    # Test 9: Render output
    env.reset()
    output = env.render(show_agent=False)
    assert "S" in output, "Render should show start"
    assert "G" in output, "Render should show goal"
    assert "C" in output, "Render should show cliff"
    print("✓ Test 9: Render produces valid output")

    # Test 10: FrozenLake basic functionality
    lake = FrozenLakeEnv(is_slippery=False)
    state = lake.reset()
    assert state == lake.start, "FrozenLake reset incorrect"
    print("✓ Test 10: FrozenLake initialization correct")

    print("=" * 60)
    print("All tests passed!")
