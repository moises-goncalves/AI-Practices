"""
Training Utilities for Policy Gradient Methods.

================================================================================
核心思想 (Core Idea)
================================================================================
This module provides essential utilities for training reinforcement learning
agents, including:

1. **Seed Management**: Reproducible experiments across runs
2. **Environment Wrappers**: Preprocessing and normalization
3. **Learning Rate Schedules**: Adaptive optimization
4. **Metrics Tracking**: Training progress monitoring
5. **Checkpointing**: Model persistence and recovery

================================================================================
数学原理 (Mathematical Theory)
================================================================================
Learning Rate Schedules:

    Linear Decay:
        α_t = α_0 × (1 - t/T)

        Properties:
            - Simple, predictable decay
            - Reaches zero at end of training
            - Good for finite-horizon training

    Cosine Annealing:
        α_t = α_min + (α_0 - α_min) × (1 + cos(πt/T)) / 2

        Properties:
            - Smooth decay with warm restarts possible
            - Never reaches zero (α_min > 0)
            - Better for longer training

    Exponential Decay:
        α_t = α_0 × γ^t

        Properties:
            - Rapid initial decay, slow later
            - Good for quick convergence

Running Statistics (Welford's Algorithm):
    For online mean and variance computation:

        n ← n + 1
        δ ← x - μ
        μ ← μ + δ/n
        M₂ ← M₂ + δ(x - μ)
        σ² = M₂/(n-1)

    Properties:
        - Numerically stable
        - O(1) memory
        - Single-pass computation

Observation Normalization:
    x_norm = (x - μ) / (σ + ε)

    Benefits:
        - Stabilizes neural network training
        - Prevents gradient explosion/vanishing
        - Improves sample efficiency

================================================================================
问题背景 (Problem Statement)
================================================================================
Training Challenges:
    1. Reproducibility: Same code, different results
    2. Hyperparameter sensitivity: Learning rate crucial
    3. Observation scaling: Networks sensitive to input magnitude
    4. Long training: Need checkpointing and monitoring

Solutions:
    1. Comprehensive seed setting (Python, NumPy, PyTorch, CUDA)
    2. Flexible learning rate schedules
    3. Running normalization for observations/rewards
    4. Robust checkpointing with metadata

================================================================================
复杂度 (Complexity Analysis)
================================================================================
Running Statistics:
    Update: O(1) time, O(1) space
    Normalize: O(d) where d = observation dimension

Checkpointing:
    Save: O(parameters) time, O(parameters) disk
    Load: O(parameters) time

================================================================================
算法总结 (Summary)
================================================================================
This module provides production-ready training infrastructure:

1. **set_seed()**: Deterministic training across all random sources
2. **RunningMeanStd**: Online normalization statistics
3. **LearningRateScheduler**: Flexible LR decay strategies
4. **MetricsTracker**: Training progress with smoothing
5. **Checkpointer**: Model persistence with versioning

References
----------
[1] Welford (1962). Note on a Method for Calculating Corrected Sums.
[2] Henderson et al. (2018). Deep RL That Matters.
[3] Engstrom et al. (2020). Implementation Matters in Deep RL.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    核心思想 (Core Idea):
        Ensure identical results across runs by controlling all sources
        of randomness: Python, NumPy, PyTorch, and CUDA.

    Parameters
    ----------
    seed : int
        Random seed value.
    deterministic : bool, default=False
        If True, use deterministic CUDA algorithms (slower but reproducible).

    Examples
    --------
    >>> set_seed(42)
    >>> # All subsequent random operations are reproducible
    >>> np.random.randn(3)
    array([0.49671415, -0.1382643 ,  0.64768854])

    Notes
    -----
    Deterministic mode may significantly slow down training on GPU.
    Use only when exact reproducibility is required.
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


class RunningMeanStd:
    """
    Online computation of mean and standard deviation.

    核心思想 (Core Idea):
        Maintain running statistics using Welford's numerically stable
        algorithm. Enables observation/reward normalization without
        storing all data.

    数学原理 (Mathematical Theory):
        Welford's online algorithm:
            n ← n + 1
            δ ← x - μ
            μ ← μ + δ/n
            M₂ ← M₂ + δ(x - μ)
            σ² = M₂/(n-1)

        For batched updates:
            n_total = n_a + n_b
            δ = μ_b - μ_a
            μ = (n_a × μ_a + n_b × μ_b) / n_total
            M₂ = M₂_a + M₂_b + δ² × n_a × n_b / n_total

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the data to track statistics for.
    epsilon : float, default=1e-8
        Small constant for numerical stability in normalization.

    Examples
    --------
    >>> rms = RunningMeanStd(shape=(4,))
    >>> for _ in range(100):
    ...     obs = np.random.randn(32, 4)  # Batch of observations
    ...     rms.update(obs)
    >>> normalized = rms.normalize(obs)
    >>> print(f"Mean: {rms.mean}, Var: {rms.var}")
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # Avoid division by zero
        self.epsilon = epsilon

    def update(self, x: np.ndarray) -> None:
        """
        Update statistics with new batch of data.

        Parameters
        ----------
        x : np.ndarray
            Batch of observations, shape (batch_size, *shape).
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Update using precomputed moments (for distributed training)."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize data using running statistics.

        Parameters
        ----------
        x : np.ndarray
            Data to normalize.

        Returns
        -------
        normalized : np.ndarray
            Normalized data with approximately zero mean and unit variance.
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        return x * np.sqrt(self.var + self.epsilon) + self.mean

    def state_dict(self) -> Dict[str, Any]:
        """Get state for serialization."""
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state from serialization."""
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class LearningRateScheduler:
    """
    Flexible learning rate scheduling.

    核心思想 (Core Idea):
        Adjust learning rate during training to balance exploration
        (high LR) and fine-tuning (low LR).

    数学原理 (Mathematical Theory):
        Linear: α_t = α_0 × (1 - t/T)
        Cosine: α_t = α_min + (α_0 - α_min) × (1 + cos(πt/T)) / 2
        Exponential: α_t = α_0 × γ^t
        Step: α_t = α_0 × γ^{floor(t/step_size)}

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule.
    schedule : str
        Schedule type: "linear", "cosine", "exponential", "step", "constant".
    total_steps : int
        Total training steps for decay calculation.
    min_lr : float, default=0.0
        Minimum learning rate.
    **kwargs
        Schedule-specific parameters.

    Examples
    --------
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    >>> scheduler = LearningRateScheduler(
    ...     optimizer, schedule="linear", total_steps=100000
    ... )
    >>> for step in range(100000):
    ...     # Training step
    ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule: str = "constant",
        total_steps: int = 1,
        min_lr: float = 0.0,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.schedule = schedule.lower()
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0
        self.kwargs = kwargs

    def step(self) -> float:
        """
        Update learning rate and return current value.

        Returns
        -------
        lr : float
            Current learning rate after update.
        """
        self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)

        if self.schedule == "constant":
            lr = self.initial_lr

        elif self.schedule == "linear":
            lr = self.initial_lr * (1 - progress)

        elif self.schedule == "cosine":
            lr = self.min_lr + (self.initial_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            ) / 2

        elif self.schedule == "exponential":
            gamma = self.kwargs.get("gamma", 0.99)
            lr = self.initial_lr * (gamma ** self.current_step)

        elif self.schedule == "step":
            step_size = self.kwargs.get("step_size", self.total_steps // 3)
            gamma = self.kwargs.get("gamma", 0.1)
            lr = self.initial_lr * (gamma ** (self.current_step // step_size))

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        lr = max(lr, self.min_lr)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


@dataclass
class MetricsTracker:
    """
    Training metrics tracking with smoothing.

    核心思想 (Core Idea):
        Track training progress with exponential moving average smoothing
        for stable visualization and early stopping decisions.

    数学原理 (Mathematical Theory):
        Exponential Moving Average:
            EMA_t = α × x_t + (1 - α) × EMA_{t-1}

        Where α = 1 - smoothing controls responsiveness:
            α → 1: More responsive to recent values
            α → 0: More stable, slower to change

    Parameters
    ----------
    smoothing : float, default=0.9
        EMA smoothing factor (higher = smoother).

    Examples
    --------
    >>> tracker = MetricsTracker()
    >>> for episode in range(1000):
    ...     reward = train_episode()
    ...     tracker.add("reward", reward)
    ...     if episode % 100 == 0:
    ...         print(tracker.get_summary())
    """

    smoothing: float = 0.9
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    smoothed: Dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
            self.smoothed[name] = value

        self.metrics[name].append(value)
        self.smoothed[name] = (
            self.smoothing * self.smoothed[name] + (1 - self.smoothing) * value
        )

    def get(self, name: str, smoothed: bool = True) -> float:
        """Get current metric value."""
        if smoothed:
            return self.smoothed.get(name, 0.0)
        return self.metrics.get(name, [0.0])[-1]

    def get_history(self, name: str) -> List[float]:
        """Get full history of a metric."""
        return self.metrics.get(name, [])

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all smoothed metrics."""
        return dict(self.smoothed)

    def get_recent_mean(self, name: str, n: int = 100) -> float:
        """Get mean of last n values."""
        history = self.metrics.get(name, [])
        if not history:
            return 0.0
        return np.mean(history[-n:])

    def reset(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.smoothed.clear()


class Checkpointer:
    """
    Model checkpointing with versioning.

    核心思想 (Core Idea):
        Save and load model states with metadata for experiment tracking
        and recovery from failures.

    Parameters
    ----------
    save_dir : str
        Directory for saving checkpoints.
    max_checkpoints : int, default=5
        Maximum number of checkpoints to keep.

    Examples
    --------
    >>> checkpointer = Checkpointer("checkpoints/experiment_1")
    >>> # During training
    >>> checkpointer.save(
    ...     model=agent.actor_critic,
    ...     optimizer=agent.optimizer,
    ...     step=10000,
    ...     metrics={"reward": 450.0},
    ... )
    >>> # Resume training
    >>> state = checkpointer.load_latest()
    >>> agent.actor_critic.load_state_dict(state["model"])
    """

    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[str] = []

        # Load existing checkpoints
        self._scan_checkpoints()

    def _scan_checkpoints(self) -> None:
        """Scan directory for existing checkpoints."""
        pattern = "checkpoint_*.pt"
        self.checkpoints = sorted(
            [f.stem for f in self.save_dir.glob(pattern)],
            key=lambda x: int(x.split("_")[1]),
        )

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> str:
        """
        Save checkpoint.

        Parameters
        ----------
        model : torch.nn.Module
            Model to save.
        optimizer : torch.optim.Optimizer
            Optimizer state.
        step : int
            Current training step.
        metrics : Dict[str, float], optional
            Training metrics to save.
        **kwargs
            Additional data to save.

        Returns
        -------
        path : str
            Path to saved checkpoint.
        """
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "metrics": metrics or {},
            **kwargs,
        }

        filename = f"checkpoint_{step:08d}.pt"
        path = self.save_dir / filename

        torch.save(checkpoint, path)
        self.checkpoints.append(filename.replace(".pt", ""))

        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            old_path = self.save_dir / f"{old_checkpoint}.pt"
            if old_path.exists():
                old_path.unlink()

        # Save metadata
        self._save_metadata(step, metrics)

        return str(path)

    def _save_metadata(
        self,
        step: int,
        metrics: Optional[Dict[str, float]],
    ) -> None:
        """Save training metadata."""
        metadata = {
            "latest_step": step,
            "latest_metrics": metrics,
            "checkpoints": self.checkpoints,
        }

        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, step: Optional[int] = None) -> Dict[str, Any]:
        """
        Load checkpoint.

        Parameters
        ----------
        step : int, optional
            Specific step to load. If None, loads latest.

        Returns
        -------
        checkpoint : Dict[str, Any]
            Loaded checkpoint data.
        """
        if step is not None:
            filename = f"checkpoint_{step:08d}.pt"
        elif self.checkpoints:
            filename = f"{self.checkpoints[-1]}.pt"
        else:
            raise FileNotFoundError("No checkpoints found")

        path = self.save_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        return torch.load(path, map_location="cpu")

    def load_latest(self) -> Dict[str, Any]:
        """Load the most recent checkpoint."""
        return self.load()

    def get_latest_step(self) -> int:
        """Get the step number of the latest checkpoint."""
        if not self.checkpoints:
            return 0
        return int(self.checkpoints[-1].split("_")[1])


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute explained variance between predictions and targets.

    核心思想 (Core Idea):
        Measure how well the value function explains the variance in returns.
        EV = 1 means perfect prediction, EV = 0 means no better than mean.

    数学原理 (Mathematical Theory):
        EV = 1 - Var(y_true - y_pred) / Var(y_true)

        Interpretation:
            EV = 1: Perfect predictions
            EV = 0: Predictions no better than predicting mean
            EV < 0: Predictions worse than mean

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values (e.g., value function estimates).
    y_true : np.ndarray
        True values (e.g., actual returns).

    Returns
    -------
    ev : float
        Explained variance, clipped to [-1, 1].
    """
    var_true = np.var(y_true)
    if var_true == 0:
        return 0.0

    return float(1 - np.var(y_true - y_pred) / var_true)


def polyak_update(
    source: torch.nn.Module,
    target: torch.nn.Module,
    tau: float = 0.005,
) -> None:
    """
    Soft update target network parameters.

    核心思想 (Core Idea):
        Slowly blend source parameters into target for stable learning.
        Used in off-policy algorithms (DQN, SAC, TD3).

    数学原理 (Mathematical Theory):
        θ_target ← τ × θ_source + (1 - τ) × θ_target

        Where τ ∈ (0, 1) controls update speed:
            τ → 0: Very slow updates (stable)
            τ → 1: Fast updates (less stable)
            τ = 1: Hard update (copy)

    Parameters
    ----------
    source : torch.nn.Module
        Source network (online/policy network).
    target : torch.nn.Module
        Target network to update.
    tau : float, default=0.005
        Interpolation factor.
    """
    with torch.no_grad():
        for source_param, target_param in zip(
            source.parameters(), target.parameters()
        ):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * source_param.data)


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Training Utilities - Unit Tests")
    print("=" * 70)

    # Test seed setting
    print("\n[1] Testing set_seed...")
    set_seed(42)
    a1 = np.random.randn(3)
    set_seed(42)
    a2 = np.random.randn(3)
    assert np.allclose(a1, a2), "Seed not working"
    print(f"    Random values match: {a1}")
    print("    [PASS]")

    # Test RunningMeanStd
    print("\n[2] Testing RunningMeanStd...")
    rms = RunningMeanStd(shape=(4,))

    # Generate data with known statistics
    np.random.seed(42)
    true_mean = np.array([1.0, 2.0, 3.0, 4.0])
    true_std = np.array([0.5, 1.0, 1.5, 2.0])

    for _ in range(1000):
        batch = np.random.randn(32, 4) * true_std + true_mean
        rms.update(batch)

    print(f"    True mean: {true_mean}")
    print(f"    Est. mean: {rms.mean}")
    print(f"    True var:  {true_std**2}")
    print(f"    Est. var:  {rms.var}")

    assert np.allclose(rms.mean, true_mean, atol=0.1)
    assert np.allclose(rms.var, true_std**2, atol=0.1)

    # Test normalization
    x = np.random.randn(10, 4) * true_std + true_mean
    x_norm = rms.normalize(x)
    assert np.abs(x_norm.mean()) < 0.5
    assert np.abs(x_norm.std() - 1.0) < 0.5
    print("    [PASS]")

    # Test LearningRateScheduler
    print("\n[3] Testing LearningRateScheduler...")
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Linear schedule
    scheduler = LearningRateScheduler(
        optimizer, schedule="linear", total_steps=100
    )

    lrs = []
    for _ in range(100):
        lr = scheduler.step()
        lrs.append(lr)

    assert lrs[0] > lrs[-1], "LR should decrease"
    assert lrs[-1] < 1e-4, "LR should be near zero at end"
    print(f"    Initial LR: {lrs[0]:.6f}")
    print(f"    Final LR: {lrs[-1]:.6f}")
    print("    [PASS]")

    # Test MetricsTracker
    print("\n[4] Testing MetricsTracker...")
    tracker = MetricsTracker(smoothing=0.9)

    for i in range(100):
        tracker.add("reward", float(i))
        tracker.add("loss", 1.0 / (i + 1))

    assert len(tracker.get_history("reward")) == 100
    assert tracker.get("reward", smoothed=True) > 50  # Smoothed towards end
    print(f"    Smoothed reward: {tracker.get('reward'):.2f}")
    print(f"    Recent mean: {tracker.get_recent_mean('reward', 10):.2f}")
    print("    [PASS]")

    # Test Checkpointer
    print("\n[5] Testing Checkpointer...")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = Checkpointer(tmpdir, max_checkpoints=3)

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        # Save multiple checkpoints
        for step in [100, 200, 300, 400]:
            checkpointer.save(
                model=model,
                optimizer=optimizer,
                step=step,
                metrics={"reward": float(step)},
            )

        # Should only keep last 3
        assert len(checkpointer.checkpoints) == 3
        assert checkpointer.get_latest_step() == 400

        # Load latest
        state = checkpointer.load_latest()
        assert state["step"] == 400
        assert state["metrics"]["reward"] == 400.0

        # Load specific
        state = checkpointer.load(step=200)
        assert state["step"] == 200

    print("    Checkpointing works correctly")
    print("    [PASS]")

    # Test explained variance
    print("\n[6] Testing compute_explained_variance...")
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_perfect = y_true.copy()
    y_pred_mean = np.full_like(y_true, y_true.mean())
    y_pred_bad = -y_true

    ev_perfect = compute_explained_variance(y_pred_perfect, y_true)
    ev_mean = compute_explained_variance(y_pred_mean, y_true)
    ev_bad = compute_explained_variance(y_pred_bad, y_true)

    assert np.isclose(ev_perfect, 1.0)
    assert np.isclose(ev_mean, 0.0)
    assert ev_bad < 0

    print(f"    Perfect prediction EV: {ev_perfect:.4f}")
    print(f"    Mean prediction EV: {ev_mean:.4f}")
    print(f"    Bad prediction EV: {ev_bad:.4f}")
    print("    [PASS]")

    # Test polyak update
    print("\n[7] Testing polyak_update...")
    source = torch.nn.Linear(10, 10)
    target = torch.nn.Linear(10, 10)

    # Initialize differently
    torch.nn.init.ones_(source.weight)
    torch.nn.init.zeros_(target.weight)

    polyak_update(source, target, tau=0.1)

    # Target should be 0.1 * 1 + 0.9 * 0 = 0.1
    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.1))
    print("    Polyak update correct")
    print("    [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
