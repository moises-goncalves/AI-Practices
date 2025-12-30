"""
类型定义模块 (Type Definitions Module)

本模块定义策略梯度算法中使用的数据结构和类型别名。

核心思想 (Core Idea):
    使用结构化数据类型替代松散的字典和元组，提供:
    1. 类型安全和IDE支持
    2. 清晰的数据契约
    3. 不可变性保证（NamedTuple）
    4. 序列化支持

设计原则:
    - Transition: 单步交互数据
    - Trajectory: 完整轨迹数据
    - TrainingMetrics: 训练指标
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch


class Transition(NamedTuple):
    """
    单步状态转移数据结构。

    核心思想 (Core Idea):
        封装强化学习中的基本交互单元 (s, a, r, s', done)，
        以及策略梯度所需的额外信息 (log_prob, value, entropy)。

    数学背景 (Mathematical Background):
        MDP转移: (s_t, a_t) → (r_t, s_{t+1})

        策略梯度需要:
            - log π_θ(a_t|s_t): 用于梯度计算
            - V(s_t): 用于优势估计
            - H(π(·|s_t)): 用于熵正则化

    Attributes
    ----------
    state : np.ndarray
        当前状态观测 s_t，形状为 (state_dim,)

    action : Union[int, np.ndarray]
        执行的动作 a_t
        - 离散动作: int
        - 连续动作: np.ndarray of shape (action_dim,)

    reward : float
        即时奖励 r_t = R(s_t, a_t, s_{t+1})

    next_state : np.ndarray
        下一状态观测 s_{t+1}，形状为 (state_dim,)

    done : bool
        回合终止标志
        - True: 终止状态（不再bootstrap）
        - False: 非终止状态

    log_prob : Optional[float]
        动作对数概率 log π_θ(a_t|s_t)
        用于策略梯度计算

    value : Optional[float]
        状态价值估计 V_φ(s_t)
        用于优势计算和Critic训练

    entropy : Optional[float]
        策略熵 H(π_θ(·|s_t)) = -Σ_a π(a|s) log π(a|s)
        用于熵正则化

    info : Optional[Dict[str, Any]]
        额外信息（环境返回的info字典）

    Examples
    --------
    >>> # 创建离散动作转移
    >>> transition = Transition(
    ...     state=np.array([1.0, 2.0, 3.0, 4.0]),
    ...     action=1,
    ...     reward=1.0,
    ...     next_state=np.array([1.1, 2.1, 3.1, 4.1]),
    ...     done=False,
    ...     log_prob=-0.693,
    ...     value=5.0,
    ...     entropy=0.693
    ... )

    >>> # 创建连续动作转移
    >>> transition = Transition(
    ...     state=np.array([0.5, -0.3, 0.1]),
    ...     action=np.array([0.7, -0.2]),
    ...     reward=0.5,
    ...     next_state=np.array([0.6, -0.2, 0.15]),
    ...     done=False,
    ...     log_prob=-1.5,
    ...     value=2.3
    ... )

    Notes
    -----
    内存布局:
        NamedTuple是不可变的，适合作为经验回放的基本单元。
        对于大规模存储，考虑使用numpy数组的结构化存储。

    复杂度:
        - 创建: O(1)
        - 访问: O(1)
        - 内存: O(state_dim + action_dim)
    """

    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: Optional[float] = None
    value: Optional[float] = None
    entropy: Optional[float] = None
    info: Optional[Dict[str, Any]] = None


@dataclass
class Trajectory:
    """
    完整轨迹数据结构。

    核心思想 (Core Idea):
        轨迹是一系列连续的状态转移，构成策略梯度更新的基本单位。
        τ = (s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T)

    数学背景 (Mathematical Background):
        策略梯度目标:
            J(θ) = E_{τ~π_θ}[R(τ)] = E_{τ~π_θ}[Σ_t γ^t r_t]

        梯度估计:
            ∇_θ J(θ) ≈ (1/N) Σ_i Σ_t ∇_θ log π_θ(a_t^i|s_t^i) · Ψ_t^i

        其中 Ψ_t 可以是:
            - G_t: 蒙特卡洛回报
            - A_t: 优势函数
            - δ_t: TD误差

    Attributes
    ----------
    states : List[np.ndarray]
        状态序列 [s_0, s_1, ..., s_{T-1}]

    actions : List[Union[int, np.ndarray]]
        动作序列 [a_0, a_1, ..., a_{T-1}]

    rewards : List[float]
        奖励序列 [r_1, r_2, ..., r_T]
        注意: r_t 是执行 a_{t-1} 后获得的奖励

    log_probs : List[float]
        对数概率序列 [log π(a_0|s_0), ..., log π(a_{T-1}|s_{T-1})]

    values : List[float]
        价值估计序列 [V(s_0), V(s_1), ..., V(s_{T-1})]

    dones : List[bool]
        终止标志序列

    entropies : List[float]
        熵序列

    next_states : List[np.ndarray]
        下一状态序列（可选，用于某些算法）

    Properties
    ----------
    length : int
        轨迹长度 T

    total_reward : float
        未折扣总奖励 Σ_t r_t

    discounted_return : float
        折扣总回报 Σ_t γ^t r_t

    Examples
    --------
    >>> trajectory = Trajectory()
    >>> for t in range(100):
    ...     action, log_prob, entropy = policy.sample(state)
    ...     value = critic(state)
    ...     next_state, reward, done, info = env.step(action)
    ...     trajectory.add(
    ...         state=state,
    ...         action=action,
    ...         reward=reward,
    ...         log_prob=log_prob,
    ...         value=value,
    ...         done=done,
    ...         entropy=entropy
    ...     )
    ...     if done:
    ...         break
    ...     state = next_state

    Notes
    -----
    复杂度分析:
        - add(): O(1) 摊销
        - length: O(1)
        - total_reward: O(T)
        - to_tensor(): O(T)
    """

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[Union[int, np.ndarray]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    next_states: List[np.ndarray] = field(default_factory=list)

    def add(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        log_prob: float,
        value: Optional[float] = None,
        done: bool = False,
        entropy: Optional[float] = None,
        next_state: Optional[np.ndarray] = None,
    ) -> None:
        """添加单步转移到轨迹。"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

        if value is not None:
            self.values.append(value)
        if entropy is not None:
            self.entropies.append(entropy)
        if next_state is not None:
            self.next_states.append(next_state)

    def add_transition(self, transition: Transition) -> None:
        """从Transition对象添加数据。"""
        self.add(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            log_prob=transition.log_prob,
            value=transition.value,
            done=transition.done,
            entropy=transition.entropy,
            next_state=transition.next_state,
        )

    @property
    def length(self) -> int:
        """轨迹长度。"""
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        """未折扣总奖励。"""
        return sum(self.rewards)

    def discounted_return(self, gamma: float = 0.99) -> float:
        """
        计算折扣总回报。

        Parameters
        ----------
        gamma : float
            折扣因子

        Returns
        -------
        float
            G_0 = Σ_t γ^t r_t
        """
        G = 0.0
        for t, r in enumerate(self.rewards):
            G += (gamma ** t) * r
        return G

    def clear(self) -> None:
        """清空轨迹数据。"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.entropies.clear()
        self.next_states.clear()

    def to_tensors(
        self,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        转换为PyTorch张量字典。

        Parameters
        ----------
        device : str
            目标设备

        Returns
        -------
        Dict[str, torch.Tensor]
            包含所有数据的张量字典
        """
        tensors = {
            "states": torch.tensor(
                np.array(self.states), dtype=torch.float32, device=device
            ),
            "rewards": torch.tensor(
                self.rewards, dtype=torch.float32, device=device
            ),
            "log_probs": torch.tensor(
                self.log_probs, dtype=torch.float32, device=device
            ),
            "dones": torch.tensor(
                self.dones, dtype=torch.float32, device=device
            ),
        }

        # 处理动作（离散或连续）
        if isinstance(self.actions[0], (int, np.integer)):
            tensors["actions"] = torch.tensor(
                self.actions, dtype=torch.long, device=device
            )
        else:
            tensors["actions"] = torch.tensor(
                np.array(self.actions), dtype=torch.float32, device=device
            )

        if self.values:
            tensors["values"] = torch.tensor(
                self.values, dtype=torch.float32, device=device
            )

        if self.entropies:
            tensors["entropies"] = torch.tensor(
                self.entropies, dtype=torch.float32, device=device
            )

        return tensors


@dataclass
class TrainingMetrics:
    """
    训练指标数据结构。

    核心思想 (Core Idea):
        集中管理训练过程中的各种指标，便于监控、日志记录和可视化。

    Attributes
    ----------
    episode : int
        当前回合数

    total_steps : int
        总训练步数

    episode_reward : float
        回合总奖励

    episode_length : int
        回合长度

    policy_loss : float
        策略损失 L_policy = -E[log π(a|s) · A]

    value_loss : float
        价值损失 L_value = E[(V(s) - V_target)²]

    entropy : float
        平均策略熵

    kl_divergence : Optional[float]
        KL散度 KL(π_old || π_new)，用于TRPO/PPO

    clip_fraction : Optional[float]
        PPO中被裁剪的比例

    explained_variance : Optional[float]
        价值函数解释方差
        EV = 1 - Var(V_target - V_pred) / Var(V_target)

    learning_rate : Optional[float]
        当前学习率

    advantage_mean : Optional[float]
        优势函数均值

    advantage_std : Optional[float]
        优势函数标准差

    Examples
    --------
    >>> metrics = TrainingMetrics(
    ...     episode=100,
    ...     total_steps=50000,
    ...     episode_reward=195.0,
    ...     episode_length=200,
    ...     policy_loss=0.05,
    ...     value_loss=0.1,
    ...     entropy=0.5
    ... )
    >>> print(metrics.to_dict())
    """

    episode: int
    total_steps: int
    episode_reward: float
    episode_length: int
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: Optional[float] = None
    clip_fraction: Optional[float] = None
    explained_variance: Optional[float] = None
    learning_rate: Optional[float] = None
    advantage_mean: Optional[float] = None
    advantage_std: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式。"""
        result = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
        }

        if self.kl_divergence is not None:
            result["kl_divergence"] = self.kl_divergence
        if self.clip_fraction is not None:
            result["clip_fraction"] = self.clip_fraction
        if self.explained_variance is not None:
            result["explained_variance"] = self.explained_variance
        if self.learning_rate is not None:
            result["learning_rate"] = self.learning_rate
        if self.advantage_mean is not None:
            result["advantage_mean"] = self.advantage_mean
        if self.advantage_std is not None:
            result["advantage_std"] = self.advantage_std

        result.update(self.extra)
        return result

    def __str__(self) -> str:
        """格式化输出。"""
        return (
            f"Episode {self.episode} | "
            f"Steps: {self.total_steps} | "
            f"Reward: {self.episode_reward:.2f} | "
            f"Length: {self.episode_length} | "
            f"Policy Loss: {self.policy_loss:.4f} | "
            f"Value Loss: {self.value_loss:.4f} | "
            f"Entropy: {self.entropy:.4f}"
        )


# 类型别名
StateType = Union[np.ndarray, torch.Tensor]
ActionType = Union[int, np.ndarray, torch.Tensor]
RewardType = Union[float, np.floating]
DoneType = Union[bool, np.bool_]

# 批量类型
BatchStates = Union[np.ndarray, torch.Tensor]  # (batch, state_dim)
BatchActions = Union[np.ndarray, torch.Tensor]  # (batch,) or (batch, action_dim)
BatchRewards = Union[np.ndarray, torch.Tensor]  # (batch,)
BatchDones = Union[np.ndarray, torch.Tensor]  # (batch,)
BatchLogProbs = Union[np.ndarray, torch.Tensor]  # (batch,)
BatchValues = Union[np.ndarray, torch.Tensor]  # (batch,)
BatchAdvantages = Union[np.ndarray, torch.Tensor]  # (batch,)
BatchReturns = Union[np.ndarray, torch.Tensor]  # (batch,)

# 网络输出类型
PolicyOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # action, log_prob, entropy
ValueOutput = torch.Tensor  # value
ActorCriticOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]  # action, log_prob, entropy, value
