"""Training utility functions for policy gradient algorithms."""

from typing import Tuple, List
import numpy as np
import torch


def compute_returns(
    rewards: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute discounted cumulative returns for a trajectory.

    Mathematical Theory:
        G_t = ∑_{k=0}^{T-t-1} γ^k r_{t+k}

        This is computed efficiently in reverse:
        G_t = r_t + γ * G_{t+1} * (1 - done_t)

    Args:
        rewards: List of rewards from trajectory
        dones: List of done flags
        gamma: Discount factor
        normalize: Whether to normalize returns

    Returns:
        Array of returns for each timestep
    """
    returns = np.zeros(len(rewards))
    cumulative_return = 0.0

    for t in reversed(range(len(rewards))):
        cumulative_return = rewards[t] + gamma * cumulative_return * (1 - dones[t])
        returns[t] = cumulative_return

    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def compute_gae(
    rewards: List[float],
    values: np.ndarray,
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Mathematical Theory:
        GAE provides a trade-off between bias and variance in advantage estimation:

        A_t^GAE(γ,λ) = ∑_{l=0}^{∞} (γλ)^l δ_t^V

        where δ_t^V = r_t + γV(s_{t+1}) - V(s_t) is the TD error.

        This can be computed efficiently as:
        A_t = δ_t + (γλ) * A_{t+1} * (1 - done_t)

        Returns are then: G_t = A_t + V(s_t)

    Problem Statement:
        GAE balances the bias-variance trade-off:
        - λ = 0: Uses only one-step TD error (low variance, high bias)
        - λ = 1: Uses full Monte Carlo returns (high variance, low bias)
        - 0 < λ < 1: Interpolates between the two

    Args:
        rewards: List of rewards
        values: Value function estimates for each state
        dones: List of done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns)
    """
    advantages = np.zeros(len(rewards))
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        # Compute TD error
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # Compute GAE
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    # Compute returns from advantages and values
    returns = advantages + values

    return advantages, returns


def normalize_advantages(
    advantages: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalize advantages to have zero mean and unit variance.

    Mathematical Theory:
        Normalized advantages: A_norm = (A - mean(A)) / (std(A) + ε)

        This normalization:
        1. Improves training stability
        2. Makes learning rate less sensitive to reward scale
        3. Helps with gradient flow

    Args:
        advantages: Array of advantages
        epsilon: Small constant for numerical stability

    Returns:
        Normalized advantages
    """
    return (advantages - advantages.mean()) / (advantages.std() + epsilon)


def compute_policy_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    entropy_coeff: float = 0.01
) -> torch.Tensor:
    """
    Compute policy loss with entropy regularization.

    Mathematical Theory:
        L_policy = -E[log π(a|s) * A(s,a)] - β * E[H(π(·|s))]

        where:
        - A(s,a) is the advantage
        - H(π(·|s)) is the entropy of the policy
        - β is the entropy coefficient

    Args:
        log_probs: Log probabilities of actions
        advantages: Advantage estimates
        entropy: Policy entropy
        entropy_coeff: Coefficient for entropy regularization

    Returns:
        Policy loss
    """
    policy_loss = -(log_probs * advantages.detach()).mean()
    entropy_loss = -entropy_coeff * entropy.mean()

    return policy_loss + entropy_loss


def compute_value_loss(
    predicted_values: torch.Tensor,
    target_values: torch.Tensor
) -> torch.Tensor:
    """
    Compute value function loss (MSE).

    Mathematical Theory:
        L_value = E[(V_φ(s) - G_t)²]

    Args:
        predicted_values: Predicted values from network
        target_values: Target values (returns or TD targets)

    Returns:
        Value loss
    """
    return torch.nn.functional.mse_loss(predicted_values, target_values)
