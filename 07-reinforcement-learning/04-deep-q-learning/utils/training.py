"""
训练工具模块

提供DQN训练的完整基础设施：
- 训练循环抽象
- 实时监控和指标跟踪
- 可视化工具
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import Env
    from gymnasium.spaces import Box, Discrete
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    warnings.warn("gymnasium未安装。环境功能禁用。安装: pip install gymnasium")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib未安装。绘图功能禁用。安装: pip install matplotlib")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import DQNAgent
from core.config import DQNConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    训练超参数配置
    
    将训练循环参数与算法超参数分离，允许在不同训练场景中
    复用相同的agent配置（如快速调试 vs 完整训练）。
    """
    num_episodes: int = 500
    max_steps_per_episode: int = 500
    eval_frequency: int = 50
    eval_episodes: int = 10
    log_frequency: int = 10
    save_frequency: int = 100
    checkpoint_dir: str = "./checkpoints"
    render: bool = False
    early_stopping_reward: Optional[float] = None
    early_stopping_episodes: int = 10
    warmup_episodes: int = 0
    
    def __post_init__(self) -> None:
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes必须为正数，得到{self.num_episodes}")
        if self.max_steps_per_episode <= 0:
            raise ValueError(f"max_steps_per_episode必须为正数")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_episodes": self.num_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "eval_frequency": self.eval_frequency,
            "eval_episodes": self.eval_episodes,
            "log_frequency": self.log_frequency,
            "save_frequency": self.save_frequency,
            "checkpoint_dir": self.checkpoint_dir,
            "render": self.render,
            "early_stopping_reward": self.early_stopping_reward,
            "early_stopping_episodes": self.early_stopping_episodes,
        }


@dataclass
class TrainingMetrics:
    """
    训练指标记录和分析
    
    记录训练过程中的关键指标：
    - 回合回报: G = Σ_{t=0}^T γ^t r_t
    - 移动平均: MA_n(G) = (1/n) Σ_{i=max(1,k-n+1)}^k G_i
    """
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    eval_rewards: List[Tuple[int, float, float]] = field(default_factory=list)
    q_values: List[float] = field(default_factory=list)
    training_time: float = 0.0
    total_steps: int = 0
    
    def add_episode(
        self,
        reward: float,
        length: int,
        epsilon: float,
        loss: Optional[float] = None,
        q_value: Optional[float] = None,
    ) -> None:
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilon_history.append(epsilon)
        self.total_steps += length
        if loss is not None:
            self.losses.append(loss)
        if q_value is not None:
            self.q_values.append(q_value)
    
    def add_evaluation(self, episode: int, mean_reward: float, std_reward: float) -> None:
        self.eval_rewards.append((episode, mean_reward, std_reward))
    
    def get_moving_average(self, window: int = 100) -> NDArray[np.floating[Any]]:
        if len(self.episode_rewards) < window:
            return np.array(self.episode_rewards, dtype=np.float64)
        kernel = np.ones(window) / window
        return np.convolve(self.episode_rewards, kernel, mode="valid")
    
    def get_statistics(self, last_n: int = 100) -> Dict[str, float]:
        rewards = self.episode_rewards[-last_n:] if self.episode_rewards else []
        lengths = self.episode_lengths[-last_n:] if self.episode_lengths else []
        losses = self.losses[-last_n:] if self.losses else []
        
        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "total_episodes": len(self.episode_rewards),
            "total_steps": self.total_steps,
            "training_time": self.training_time,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
            "epsilon_history": self.epsilon_history,
            "eval_rewards": self.eval_rewards,
            "q_values": self.q_values,
            "training_time": self.training_time,
            "total_steps": self.total_steps,
            "statistics": self.get_statistics(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingMetrics":
        with open(path, "r") as f:
            data = json.load(f)
        metrics = cls()
        metrics.episode_rewards = data.get("episode_rewards", [])
        metrics.episode_lengths = data.get("episode_lengths", [])
        metrics.losses = data.get("losses", [])
        metrics.epsilon_history = data.get("epsilon_history", [])
        metrics.eval_rewards = [tuple(e) for e in data.get("eval_rewards", [])]
        metrics.q_values = data.get("q_values", [])
        metrics.training_time = data.get("training_time", 0.0)
        metrics.total_steps = data.get("total_steps", 0)
        return metrics
