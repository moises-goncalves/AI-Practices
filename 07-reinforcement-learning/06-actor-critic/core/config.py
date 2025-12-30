"""
Training Configuration for Policy Gradient Methods.

================================================================================
核心思想 (Core Idea)
================================================================================
Centralized configuration management using Python dataclasses. This approach:
1. Provides type safety and IDE autocompletion
2. Enables easy serialization for experiment tracking
3. Groups related hyperparameters logically
4. Supports default values with override capability

================================================================================
数学原理 (Mathematical Theory)
================================================================================
Key Hyperparameters and Their Effects:

Learning Rate (α):
    θ_{t+1} = θ_t + α ∇_θ J(θ)

    Too high: Oscillation, divergence
    Too low: Slow convergence
    Typical range: 1e-4 to 3e-4 for policy gradient

Discount Factor (γ):
    G_t = Σ_{k=0}^∞ γ^k r_{t+k}

    γ → 0: Myopic, only immediate rewards
    γ → 1: Far-sighted, values future equally
    Typical: 0.99 for most tasks

GAE Lambda (λ):
    A^{GAE}_t = Σ_{k=0}^∞ (γλ)^k δ_{t+k}

    λ = 0: TD(0), low variance, high bias
    λ = 1: Monte Carlo, high variance, no bias
    Typical: 0.95 for PPO

Entropy Coefficient (c_ent):
    L = L_policy + c_v L_value - c_ent H(π)

    Higher: More exploration, slower convergence
    Lower: Less exploration, risk of premature convergence
    Typical: 0.01 for discrete, 0.001 for continuous

PPO Clip Range (ε):
    L^{CLIP} = min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)

    Larger ε: Bigger policy updates, less stable
    Smaller ε: Conservative updates, slower learning
    Typical: 0.2

================================================================================
问题背景 (Problem Statement)
================================================================================
Challenge: Managing dozens of hyperparameters across experiments.

Problems with ad-hoc configuration:
    1. Scattered parameters across codebase
    2. No type checking or validation
    3. Difficult to reproduce experiments
    4. Hard to track what was changed

Solution: Structured dataclass with sensible defaults and validation.

================================================================================
算法对比 (Comparison)
================================================================================
| Config Method    | Type Safety | Serialization | IDE Support | Validation |
|------------------|-------------|---------------|-------------|------------|
| Dict             | No          | Easy          | No          | Manual     |
| Namespace        | No          | Medium        | Partial     | Manual     |
| Dataclass (this) | Yes         | Easy          | Yes         | Built-in   |
| Pydantic         | Yes         | Easy          | Yes         | Advanced   |
| Hydra/OmegaConf  | Yes         | Advanced      | Yes         | Advanced   |

================================================================================
复杂度 (Complexity Analysis)
================================================================================
Memory: O(1) - fixed number of scalar parameters
Serialization: O(n) where n = number of parameters
Validation: O(n) one-time at construction

================================================================================
算法总结 (Summary)
================================================================================
This module provides a clean, type-safe configuration system:

1. **TrainingConfig**: All hyperparameters in one place
2. **Default values**: Tuned for common benchmarks (CartPole, Pendulum)
3. **Validation**: Catches invalid configurations early
4. **Serialization**: Easy save/load for reproducibility

References
----------
[1] Schulman et al. (2017). PPO - recommended hyperparameters
[2] Andrychowicz et al. (2020). What Matters in On-Policy RL
[3] Henderson et al. (2018). Deep RL That Matters
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration for policy gradient algorithms.

    核心思想 (Core Idea):
        Single source of truth for all training hyperparameters.
        Organized into logical groups: environment, network, optimization,
        algorithm-specific, and logging.

    数学原理 (Mathematical Theory):
        Hyperparameter sensitivity analysis shows:

        High sensitivity (tune carefully):
            - learning_rate: 10x change → training failure
            - clip_range (PPO): affects stability significantly
            - entropy_coef: exploration-exploitation balance

        Medium sensitivity:
            - gamma: task-dependent horizon
            - gae_lambda: bias-variance trade-off
            - hidden_dims: capacity vs. overfitting

        Low sensitivity (defaults usually fine):
            - max_grad_norm: rarely needs tuning
            - value_coef: 0.5 works broadly

    Parameters
    ----------
    env_name : str
        Gymnasium environment identifier.
    seed : int
        Random seed for reproducibility.
    total_timesteps : int
        Total environment interactions for training.

    hidden_dims : List[int]
        Hidden layer dimensions for policy/value networks.
    activation : str
        Activation function name ("relu", "tanh").

    learning_rate : float
        Optimizer learning rate.
    gamma : float
        Discount factor for returns.
    gae_lambda : float
        GAE lambda for advantage estimation.

    n_steps : int
        Steps per rollout before update (A2C, PPO).
    n_epochs : int
        SGD epochs per rollout (PPO).
    batch_size : int
        Mini-batch size for SGD.

    clip_range : float
        PPO clipping parameter epsilon.
    entropy_coef : float
        Entropy bonus coefficient.
    value_coef : float
        Value loss coefficient.
    max_grad_norm : float
        Gradient clipping threshold.

    normalize_advantage : bool
        Whether to normalize advantages.
    use_gae : bool
        Whether to use GAE (vs. simple returns).

    log_interval : int
        Episodes between logging.
    save_interval : int
        Episodes between checkpoints.
    eval_interval : int
        Episodes between evaluation runs.
    n_eval_episodes : int
        Number of episodes for evaluation.

    Examples
    --------
    >>> # Default configuration
    >>> config = TrainingConfig()
    >>>
    >>> # Custom configuration
    >>> config = TrainingConfig(
    ...     env_name="LunarLander-v2",
    ...     learning_rate=2.5e-4,
    ...     total_timesteps=1_000_000,
    ... )
    >>>
    >>> # Save and load
    >>> config.save("experiment_config.json")
    >>> loaded = TrainingConfig.load("experiment_config.json")
    >>>
    >>> # Convert to dict for logging
    >>> wandb.config.update(config.to_dict())
    """

    # Environment
    env_name: str = "CartPole-v1"
    seed: int = 42
    total_timesteps: int = 100_000
    max_episode_steps: Optional[int] = None

    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    shared_backbone: bool = True

    # Optimization
    learning_rate: float = 3e-4
    lr_schedule: str = "constant"  # "constant", "linear", "cosine"
    optimizer: str = "adam"
    weight_decay: float = 0.0

    # RL fundamentals
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Rollout collection
    n_steps: int = 2048
    n_envs: int = 1

    # PPO specific
    n_epochs: int = 10
    batch_size: int = 64
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    target_kl: Optional[float] = None

    # Loss coefficients
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Normalization
    normalize_advantage: bool = True
    normalize_observations: bool = False
    normalize_rewards: bool = False

    # Algorithm selection
    algorithm: str = "ppo"  # "reinforce", "a2c", "ppo"
    use_gae: bool = True

    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    n_eval_episodes: int = 10
    log_dir: str = "logs"
    save_dir: str = "checkpoints"

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """
        Validate hyperparameter ranges and consistency.

        Raises
        ------
        ValueError
            If any hyperparameter is invalid.
        """
        # Learning rate
        if not 1e-6 <= self.learning_rate <= 1.0:
            raise ValueError(
                f"learning_rate={self.learning_rate} outside valid range [1e-6, 1.0]"
            )

        # Discount factor
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma={self.gamma} must be in [0, 1]")

        # GAE lambda
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda={self.gae_lambda} must be in [0, 1]")

        # PPO clip range
        if not 0.0 < self.clip_range <= 1.0:
            raise ValueError(f"clip_range={self.clip_range} must be in (0, 1]")

        # Coefficients
        if self.entropy_coef < 0:
            raise ValueError(f"entropy_coef={self.entropy_coef} must be non-negative")
        if self.value_coef < 0:
            raise ValueError(f"value_coef={self.value_coef} must be non-negative")

        # Batch size
        if self.batch_size > self.n_steps * self.n_envs:
            raise ValueError(
                f"batch_size={self.batch_size} > buffer_size={self.n_steps * self.n_envs}"
            )

        # Algorithm
        valid_algorithms = {"reinforce", "a2c", "ppo"}
        if self.algorithm.lower() not in valid_algorithms:
            raise ValueError(
                f"algorithm={self.algorithm} not in {valid_algorithms}"
            )

        # Activation
        valid_activations = {"relu", "tanh", "elu", "leaky_relu"}
        if self.activation.lower() not in valid_activations:
            raise ValueError(
                f"activation={self.activation} not in {valid_activations}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> TrainingConfig:
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> TrainingConfig:
        """
        Load configuration from JSON file.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        config : TrainingConfig
            Loaded configuration.
        """
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def get_device(self) -> str:
        """
        Resolve device string to actual device.

        Returns
        -------
        device : str
            Resolved device ("cpu", "cuda", or "mps").
        """
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device

    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = [f"{self.__class__.__name__}("]
        for key, value in self.to_dict().items():
            lines.append(f"    {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)


# =============================================================================
# Preset Configurations
# =============================================================================

def get_cartpole_config() -> TrainingConfig:
    """
    Optimized configuration for CartPole-v1.

    CartPole is a simple balancing task:
        - State: 4D (position, velocity, angle, angular velocity)
        - Action: 2 discrete (left, right)
        - Reward: +1 per step
        - Solved: Average reward >= 475 over 100 episodes
    """
    return TrainingConfig(
        env_name="CartPole-v1",
        total_timesteps=50_000,
        hidden_dims=[64, 64],
        learning_rate=3e-4,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
    )


def get_lunarlander_config() -> TrainingConfig:
    """
    Optimized configuration for LunarLander-v2.

    LunarLander is a rocket landing task:
        - State: 8D (position, velocity, angle, leg contact)
        - Action: 4 discrete (noop, left, main, right engine)
        - Reward: Landing pad bonus, crash penalty, fuel cost
        - Solved: Average reward >= 200 over 100 episodes
    """
    return TrainingConfig(
        env_name="LunarLander-v2",
        total_timesteps=500_000,
        hidden_dims=[64, 64],
        learning_rate=2.5e-4,
        n_steps=2048,
        n_epochs=4,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
    )


def get_pendulum_config() -> TrainingConfig:
    """
    Optimized configuration for Pendulum-v1 (continuous control).

    Pendulum is a swing-up task:
        - State: 3D (cos(θ), sin(θ), angular velocity)
        - Action: 1D continuous torque in [-2, 2]
        - Reward: -(θ² + 0.1*ω² + 0.001*u²)
        - Goal: Keep pendulum upright (θ ≈ 0)
    """
    return TrainingConfig(
        env_name="Pendulum-v1",
        total_timesteps=100_000,
        hidden_dims=[64, 64],
        learning_rate=1e-3,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.0,  # Continuous actions don't need entropy bonus
    )


# =============================================================================
# Module Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Training Configuration Module - Unit Tests")
    print("=" * 70)

    # Test default configuration
    print("\n[1] Testing default configuration...")
    config = TrainingConfig()
    print(f"    Environment: {config.env_name}")
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Device: {config.get_device()}")
    print("    [PASS]")

    # Test custom configuration
    print("\n[2] Testing custom configuration...")
    custom_config = TrainingConfig(
        env_name="LunarLander-v2",
        learning_rate=1e-4,
        total_timesteps=1_000_000,
        hidden_dims=[128, 128],
    )
    assert custom_config.env_name == "LunarLander-v2"
    assert custom_config.learning_rate == 1e-4
    print(f"    Custom env: {custom_config.env_name}")
    print("    [PASS]")

    # Test validation
    print("\n[3] Testing validation...")
    try:
        invalid_config = TrainingConfig(learning_rate=-0.1)
        print("    [FAIL] Should have raised ValueError")
    except ValueError as e:
        print(f"    Caught expected error: {e}")
        print("    [PASS]")

    try:
        invalid_config = TrainingConfig(gamma=1.5)
        print("    [FAIL] Should have raised ValueError")
    except ValueError as e:
        print(f"    Caught expected error: {e}")
        print("    [PASS]")

    # Test serialization
    print("\n[4] Testing serialization...")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "config.json")
        config.save(save_path)

        loaded_config = TrainingConfig.load(save_path)
        assert loaded_config.env_name == config.env_name
        assert loaded_config.learning_rate == config.learning_rate
        print(f"    Saved and loaded from: {save_path}")
    print("    [PASS]")

    # Test dict conversion
    print("\n[5] Testing dict conversion...")
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "learning_rate" in config_dict

    reconstructed = TrainingConfig.from_dict(config_dict)
    assert reconstructed.learning_rate == config.learning_rate
    print(f"    Dict keys: {len(config_dict)}")
    print("    [PASS]")

    # Test preset configurations
    print("\n[6] Testing preset configurations...")
    cartpole = get_cartpole_config()
    assert cartpole.env_name == "CartPole-v1"
    print(f"    CartPole config: {cartpole.total_timesteps} timesteps")

    lunarlander = get_lunarlander_config()
    assert lunarlander.env_name == "LunarLander-v2"
    print(f"    LunarLander config: {lunarlander.total_timesteps} timesteps")

    pendulum = get_pendulum_config()
    assert pendulum.env_name == "Pendulum-v1"
    print(f"    Pendulum config: {pendulum.total_timesteps} timesteps")
    print("    [PASS]")

    # Test repr
    print("\n[7] Testing repr...")
    repr_str = repr(config)
    assert "TrainingConfig" in repr_str
    assert "learning_rate" in repr_str
    print("    Repr output valid")
    print("    [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
