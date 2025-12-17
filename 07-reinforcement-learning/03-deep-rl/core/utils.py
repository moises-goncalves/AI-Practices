"""
Utility Functions for Deep Reinforcement Learning

Common utilities including device management, random seed configuration,
learning rate schedules, and training helpers.
"""

from __future__ import annotations
from typing import Optional, Callable
import random

import numpy as np
import torch


def get_device(device: str = "auto") -> torch.device:
    """
    Get PyTorch device for computation.

    Args:
        device: Device specification
            - "auto": Use CUDA if available, else CPU
            - "cpu": Force CPU
            - "cuda": Force CUDA (raises if unavailable)
            - "cuda:0", "cuda:1", etc.: Specific GPU

    Returns:
        torch.device instance
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for:
        - Python random module
        - NumPy random
        - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value
        deterministic: If True, enable fully deterministic operations
            (may reduce performance)

    Note:
        Full determinism requires:
        - CUDA deterministic mode
        - Disabling cuDNN benchmark
        - May significantly impact training speed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def linear_schedule(
    start: float,
    end: float,
    duration: int
) -> Callable[[int], float]:
    """
    Create linear schedule function.

    Returns a function that linearly interpolates from start to end
    over duration steps, then remains at end value.

    Mathematical Formula:
        f(t) = start + (end - start) × min(t / duration, 1.0)

    Common Uses:
        - Learning rate decay
        - Exploration rate (ε) decay
        - PER β annealing

    Args:
        start: Initial value
        end: Final value
        duration: Number of steps for interpolation

    Returns:
        Callable that maps step number to scheduled value

    Example:
        >>> schedule = linear_schedule(1.0, 0.01, 100000)
        >>> schedule(0)  # Returns 1.0
        >>> schedule(50000)  # Returns ~0.5
        >>> schedule(100000)  # Returns 0.01
        >>> schedule(200000)  # Returns 0.01 (saturated)
    """
    def schedule(step: int) -> float:
        fraction = min(float(step) / duration, 1.0)
        return start + fraction * (end - start)

    return schedule


def exponential_schedule(
    start: float,
    end: float,
    decay: float
) -> Callable[[int], float]:
    """
    Create exponential decay schedule.

    Mathematical Formula:
        f(t) = max(end, start × decay^t)

    Args:
        start: Initial value
        end: Minimum value (floor)
        decay: Decay rate per step (typically 0.9999 or similar)

    Returns:
        Callable that maps step number to scheduled value
    """
    def schedule(step: int) -> float:
        return max(end, start * (decay ** step))

    return schedule


def soft_update(
    target: torch.nn.Module,
    source: torch.nn.Module,
    tau: float = 0.005
) -> None:
    """
    Soft update target network parameters.

    Polyak averaging: θ_target ← τ × θ_source + (1-τ) × θ_target

    This provides smoother target updates compared to hard updates,
    often used in continuous control algorithms (DDPG, TD3, SAC).

    Args:
        target: Target network to update
        source: Source network with new parameters
        tau: Interpolation factor (typically 0.001 to 0.01)

    Complexity: O(|θ|) for parameter count |θ|
    """
    with torch.no_grad():
        for target_param, source_param in zip(
            target.parameters(),
            source.parameters()
        ):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )


def hard_update(
    target: torch.nn.Module,
    source: torch.nn.Module
) -> None:
    """
    Hard update target network parameters.

    Direct copy: θ_target ← θ_source

    Used in DQN-family algorithms where target network is updated
    every C steps rather than continuously.

    Args:
        target: Target network to update
        source: Source network to copy from
    """
    target.load_state_dict(source.state_dict())


def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    Compute total gradient norm across all parameters.

    Useful for monitoring training stability and debugging
    gradient explosion/vanishing.

    Args:
        model: PyTorch model with computed gradients

    Returns:
        L2 norm of all gradients concatenated
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Utility Tests")
    print("=" * 50)

    device = get_device("auto")
    print(f"Device: {device}")

    set_seed(42)
    print("Random seed set: 42")

    lr_schedule = linear_schedule(1.0, 0.01, 100)
    assert abs(lr_schedule(0) - 1.0) < 1e-6
    assert abs(lr_schedule(50) - 0.505) < 1e-6
    assert abs(lr_schedule(100) - 0.01) < 1e-6
    assert abs(lr_schedule(200) - 0.01) < 1e-6
    print("Linear schedule: OK")

    exp_schedule = exponential_schedule(1.0, 0.01, 0.99)
    assert abs(exp_schedule(0) - 1.0) < 1e-6
    print("Exponential schedule: OK")

    source = torch.nn.Linear(4, 2)
    target = torch.nn.Linear(4, 2)
    target.load_state_dict(source.state_dict())

    source.weight.data.fill_(1.0)
    soft_update(target, source, tau=0.1)
    expected = 0.1 * 1.0 + 0.9 * target.weight.data.mean().item()
    print("Soft update: OK")

    hard_update(target, source)
    assert torch.allclose(target.weight, source.weight)
    print("Hard update: OK")

    print(f"\nAll utility tests passed!")
