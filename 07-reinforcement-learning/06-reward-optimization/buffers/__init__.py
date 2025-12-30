"""
Experience Replay Buffers for Reward Optimization

================================================================================
OVERVIEW
================================================================================
This module provides specialized replay buffer implementations for reward
optimization algorithms, including:

- GoalConditionedBuffer: For HER and goal-conditioned RL
- DemonstrationBuffer: For inverse RL with expert data
- PrioritizedBuffer: For prioritized experience replay
- CuriosityBuffer: For intrinsic motivation with auxiliary data

================================================================================
DESIGN PRINCIPLES
================================================================================
1. Efficiency: O(1) sampling, O(1) insertion
2. Flexibility: Support various transition types
3. Integration: Easy to combine with shaping/HER/curiosity

================================================================================
REFERENCES
================================================================================
[1] Andrychowicz et al. (2017). Hindsight Experience Replay
[2] Schaul et al. (2015). Prioritized Experience Replay
[3] Ng & Russell (2000). Algorithms for Inverse RL (demonstration buffers)
"""

from .replay_buffer import (
    Transition,
    ReplayBuffer,
    PrioritizedReplayBuffer,
)

from .goal_buffer import (
    GoalTransition,
    Episode,
    GoalConditionedBuffer,
    HERBuffer,
)

from .demonstration_buffer import (
    Demonstration,
    DemonstrationBuffer,
    MixedBuffer,
)

__all__ = [
    # Basic replay
    "Transition",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    # Goal-conditioned
    "GoalTransition",
    "Episode",
    "GoalConditionedBuffer",
    "HERBuffer",
    # Demonstrations
    "Demonstration",
    "DemonstrationBuffer",
    "MixedBuffer",
]
