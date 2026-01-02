"""
Trajectory buffer for storing and processing experience data.

This module provides efficient data structures for storing trajectories and computing
returns and advantages for policy gradient algorithms.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class Trajectory:
    """
    Data class representing a single trajectory (episode).

    Attributes:
        states: List of states visited
        actions: List of actions taken
        rewards: List of rewards received
        dones: List of done flags
        log_probs: List of log probabilities of actions
    """
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    log_probs: List[float]

    def __len__(self) -> int:
        """Return the length of the trajectory."""
        return len(self.states)


class TrajectoryBuffer:
    """
    Buffer for storing and processing trajectories.

    Core Idea:
        Efficiently store trajectories and compute returns/advantages for batch updates.
        This is essential for policy gradient algorithms which require computing
        cumulative returns and advantage estimates.

    Mathematical Theory:
        Return: G_t = ∑_{k=0}^{T-t-1} γ^k r_{t+k}
        Advantage: A_t = G_t - V(s_t)
        Generalized Advantage Estimation (GAE):
        A_t^GAE = ∑_{l=0}^{T-t-1} (γλ)^l δ_t^V
        where δ_t^V = r_t + γV(s_{t+1}) - V(s_t)

    Problem Statement:
        Computing returns and advantages efficiently is critical for training stability
        and sample efficiency in policy gradient methods.

    Complexity:
        Time: O(T) for computing returns/advantages where T is trajectory length
        Space: O(T) for storing trajectory data
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize the trajectory buffer.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter for advantage estimation
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.trajectories: List[Trajectory] = []

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Add a trajectory to the buffer.

        Args:
            trajectory: Trajectory object to add
        """
        self.trajectories.append(trajectory)

    def clear(self) -> None:
        """Clear all trajectories from the buffer."""
        self.trajectories = []

    def compute_returns(self, trajectory: Trajectory) -> np.ndarray:
        """
        Compute discounted returns for a trajectory.

        Args:
            trajectory: Trajectory object

        Returns:
            Array of returns for each timestep
        """
        returns = np.zeros(len(trajectory))
        cumulative_return = 0.0

        for t in reversed(range(len(trajectory))):
            cumulative_return = trajectory.rewards[t] + self.gamma * cumulative_return * (1 - trajectory.dones[t])
            returns[t] = cumulative_return

        return returns

    def compute_advantages(
        self,
        trajectory: Trajectory,
        values: np.ndarray
    ) -> np.ndarray:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            trajectory: Trajectory object
            values: Value function estimates for each state

        Returns:
            Array of advantages for each timestep
        """
        advantages = np.zeros(len(trajectory))
        gae = 0.0

        for t in reversed(range(len(trajectory))):
            if t == len(trajectory) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = trajectory.rewards[t] + self.gamma * next_value * (1 - trajectory.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - trajectory.dones[t]) * gae
            advantages[t] = gae

        return advantages

    def get_batch(
        self,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all trajectories as batched tensors.

        Args:
            device: Device to place tensors on

        Returns:
            Tuple of (states, actions, returns, log_probs) as tensors
        """
        states_list = []
        actions_list = []
        returns_list = []
        log_probs_list = []

        for trajectory in self.trajectories:
            returns = self.compute_returns(trajectory)

            states_list.extend(trajectory.states)
            actions_list.extend(trajectory.actions)
            returns_list.extend(returns)
            log_probs_list.extend(trajectory.log_probs)

        states = torch.FloatTensor(np.array(states_list)).to(device)
        actions = torch.FloatTensor(np.array(actions_list)).to(device)
        returns = torch.FloatTensor(returns_list).unsqueeze(1).to(device)
        log_probs = torch.FloatTensor(log_probs_list).unsqueeze(1).to(device)

        return states, actions, returns, log_probs

    def get_batch_with_advantages(
        self,
        values: List[np.ndarray],
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all trajectories with computed advantages.

        Args:
            values: List of value estimates for each trajectory
            device: Device to place tensors on

        Returns:
            Tuple of (states, actions, advantages, log_probs) as tensors
        """
        states_list = []
        actions_list = []
        advantages_list = []
        log_probs_list = []

        for trajectory, traj_values in zip(self.trajectories, values):
            advantages = self.compute_advantages(trajectory, traj_values)

            states_list.extend(trajectory.states)
            actions_list.extend(trajectory.actions)
            advantages_list.extend(advantages)
            log_probs_list.extend(trajectory.log_probs)

        states = torch.FloatTensor(np.array(states_list)).to(device)
        actions = torch.FloatTensor(np.array(actions_list)).to(device)
        advantages = torch.FloatTensor(advantages_list).unsqueeze(1).to(device)
        log_probs = torch.FloatTensor(log_probs_list).unsqueeze(1).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, advantages, log_probs

    def __len__(self) -> int:
        """Return the number of trajectories in the buffer."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        """Get a trajectory by index."""
        return self.trajectories[idx]
