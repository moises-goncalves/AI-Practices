"""
轨迹缓冲区模块 (Trajectory Buffer Module)

本模块实现策略梯度算法所需的经验存储数据结构。

核心思想 (Core Idea):
    策略梯度方法需要收集轨迹数据 τ = (s_0, a_0, r_1, s_1, ...) 来估计梯度。
    不同算法对数据存储有不同需求:
    - REINFORCE: 完整episode，计算MC回报
    - A2C: 固定长度rollout，计算n-step回报或GAE
    - PPO: 固定长度rollout + 多epoch重用

数学原理 (Mathematical Theory):
    策略梯度估计需要的数据:
        ∇_θ J(θ) ≈ (1/N) Σ_i Σ_t ∇_θ log π_θ(a_t|s_t) · Ψ_t

    存储需求:
        - states: s_t ∈ R^d
        - actions: a_t (离散或连续)
        - rewards: r_t ∈ R
        - log_probs: log π_θ(a_t|s_t) ∈ R
        - values: V(s_t) ∈ R (用于优势计算)
        - dones: 终止标志 (用于正确处理episode边界)

    GAE计算需要:
        δ_t = r_t + γ(1-done_t)V(s_{t+1}) - V(s_t)
        A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

问题背景 (Problem Statement):
    | 缓冲区类型    | 容量      | 数据访问    | 适用算法      |
    |---------------|-----------|-------------|---------------|
    | EpisodeBuffer | 动态      | 顺序        | REINFORCE     |
    | RolloutBuffer | 固定      | 批量        | A2C, PPO      |
    | GAEBuffer     | 固定      | 批量+GAE    | PPO, TRPO     |

复杂度 (Complexity):
    - 存储: O(T × d) 其中 T 是轨迹长度，d 是状态维度
    - 添加: O(1) 摊销
    - GAE计算: O(T) 单次反向遍历

References:
    [1] Schulman et al. (2016). GAE - High-dimensional continuous control.
    [2] Schulman et al. (2017). PPO - Proximal Policy Optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class EpisodeBuffer:
    """
    Episode轨迹缓冲区（动态大小）。

    核心思想 (Core Idea):
        存储完整episode的轨迹数据，支持REINFORCE等需要完整回报的算法。
        使用Python列表实现动态增长。

    数学原理 (Mathematical Theory):
        存储轨迹 τ = (s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T)

        用于计算:
            - MC回报: G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}
            - 策略梯度: ∇_θ J ≈ Σ_t ∇_θ log π(a_t|s_t) · G_t

    Attributes
    ----------
    states : List[np.ndarray]
        状态序列
    actions : List[Union[int, np.ndarray]]
        动作序列
    rewards : List[float]
        奖励序列
    log_probs : List[float]
        对数概率序列
    values : List[float]
        价值估计序列
    dones : List[bool]
        终止标志序列
    entropies : List[float]
        熵序列

    Examples
    --------
    >>> buffer = EpisodeBuffer()
    >>> for step in range(100):
    ...     action, log_prob, entropy = policy.sample(state)
    ...     value = critic(state)
    ...     next_state, reward, done, _ = env.step(action)
    ...     buffer.store(
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
    >>> print(f"Episode length: {len(buffer)}")
    >>> print(f"Total reward: {buffer.total_reward}")
    """

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[Union[int, np.ndarray]] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)

    def store(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        log_prob: float,
        value: Optional[float] = None,
        done: bool = False,
        entropy: Optional[float] = None,
    ) -> None:
        """
        存储单步转移。

        Parameters
        ----------
        state : np.ndarray
            当前状态
        action : Union[int, np.ndarray]
            执行的动作
        reward : float
            获得的奖励
        log_prob : float
            动作对数概率
        value : float, optional
            状态价值估计
        done : bool
            是否终止
        entropy : float, optional
            策略熵
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

        if value is not None:
            self.values.append(value)
        if entropy is not None:
            self.entropies.append(entropy)

    def clear(self) -> None:
        """清空缓冲区。"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.entropies.clear()

    def __len__(self) -> int:
        """返回存储的转移数量。"""
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        """计算总奖励。"""
        return sum(self.rewards)

    def to_tensors(
        self,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        转换为PyTorch张量。

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

        # 处理动作类型
        if self.actions and isinstance(self.actions[0], (int, np.integer)):
            tensors["actions"] = torch.tensor(
                self.actions, dtype=torch.long, device=device
            )
        elif self.actions:
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
class RolloutBuffer:
    """
    固定大小Rollout缓冲区。

    核心思想 (Core Idea):
        预分配固定大小的numpy数组，支持高效的批量操作。
        适用于A2C、PPO等需要固定长度rollout的算法。

    数学原理 (Mathematical Theory):
        存储固定长度 T 的rollout:
            {(s_t, a_t, r_t, log π(a_t|s_t), V(s_t), done_t)}_{t=0}^{T-1}

        用于计算:
            - n-step回报: G_t^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})
            - GAE: A_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}

    Parameters
    ----------
    buffer_size : int
        缓冲区大小（rollout长度）
    state_dim : int
        状态维度
    action_dim : int
        动作维度（连续动作）
    discrete : bool
        是否离散动作空间
    device : str
        计算设备

    Attributes
    ----------
    states : np.ndarray
        状态数组 (buffer_size, state_dim)
    actions : np.ndarray
        动作数组
    rewards : np.ndarray
        奖励数组 (buffer_size,)
    log_probs : np.ndarray
        对数概率数组 (buffer_size,)
    values : np.ndarray
        价值数组 (buffer_size,)
    dones : np.ndarray
        终止标志数组 (buffer_size,)
    advantages : np.ndarray
        优势数组 (buffer_size,)
    returns : np.ndarray
        回报数组 (buffer_size,)

    Examples
    --------
    >>> buffer = RolloutBuffer(
    ...     buffer_size=2048,
    ...     state_dim=4,
    ...     action_dim=1,
    ...     discrete=True
    ... )
    >>> for step in range(2048):
    ...     buffer.store(step, state, action, reward, log_prob, value, done)
    >>> buffer.compute_gae(next_value=0.0, gamma=0.99, gae_lambda=0.95)
    >>> for batch in buffer.get_batches(batch_size=64):
    ...     # 训练
    ...     pass
    """

    buffer_size: int
    state_dim: int
    action_dim: int = 1
    discrete: bool = True
    device: str = "cpu"

    def __post_init__(self):
        """初始化预分配数组。"""
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)

        if self.discrete:
            self.actions = np.zeros(self.buffer_size, dtype=np.int64)
        else:
            self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)

        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.entropies = np.zeros(self.buffer_size, dtype=np.float32)

        # GAE计算结果
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def store(
        self,
        idx: int,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        entropy: float = 0.0,
    ) -> None:
        """
        在指定位置存储转移。

        Parameters
        ----------
        idx : int
            存储位置索引
        state : np.ndarray
            状态
        action : Union[int, np.ndarray]
            动作
        reward : float
            奖励
        log_prob : float
            对数概率
        value : float
            价值估计
        done : bool
            终止标志
        entropy : float
            策略熵
        """
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.dones[idx] = float(done)
        self.entropies[idx] = entropy

    def add(
        self,
        state: np.ndarray,
        action: Union[int, np.ndarray],
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        entropy: float = 0.0,
    ) -> None:
        """
        添加转移到当前位置（自动递增指针）。
        """
        self.store(self.ptr, state, action, reward, log_prob, value, done, entropy)
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_gae(
        self,
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        计算广义优势估计 (GAE)。

        核心思想 (Core Idea):
            GAE通过指数加权平均多步TD误差，在偏差和方差之间取得平衡。

        数学原理 (Mathematical Theory):
            TD误差:
                δ_t = r_t + γ(1-done_t)V(s_{t+1}) - V(s_t)

            GAE (反向递归):
                A_T = δ_T
                A_t = δ_t + γλ(1-done_t)A_{t+1}

            回报目标:
                G_t = A_t + V(s_t)

        Parameters
        ----------
        next_value : float
            最后状态的价值估计 V(s_T)
        gamma : float
            折扣因子
        gae_lambda : float
            GAE λ参数

        Notes
        -----
        复杂度: O(T) 单次反向遍历
        """
        gae = 0.0
        size = self.buffer_size if self.full else self.ptr

        for t in reversed(range(size)):
            if t == size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = self.values[t + 1]

            # TD误差
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]

            # GAE递归
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        # 回报 = 优势 + 价值
        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def compute_returns(
        self,
        next_value: float,
        gamma: float = 0.99,
    ) -> None:
        """
        计算折扣回报（不使用GAE）。

        数学原理:
            G_T = r_T + γ(1-done_T)V(s_{T+1})
            G_t = r_t + γ(1-done_t)G_{t+1}
        """
        size = self.buffer_size if self.full else self.ptr
        G = next_value

        for t in reversed(range(size)):
            G = self.rewards[t] + gamma * (1.0 - self.dones[t]) * G
            self.returns[t] = G

        # 优势 = 回报 - 价值
        self.advantages[:size] = self.returns[:size] - self.values[:size]

    def normalize_advantages(self, eps: float = 1e-8) -> None:
        """标准化优势函数。"""
        size = self.buffer_size if self.full else self.ptr
        adv = self.advantages[:size]
        self.advantages[:size] = (adv - adv.mean()) / (adv.std() + eps)

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        生成训练批次。

        Parameters
        ----------
        batch_size : int
            批次大小
        shuffle : bool
            是否打乱顺序

        Yields
        ------
        Dict[str, torch.Tensor]
            包含批次数据的字典
        """
        size = self.buffer_size if self.full else self.ptr
        indices = np.arange(size)

        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_indices = indices[start:end]

            yield {
                "states": torch.tensor(
                    self.states[batch_indices], dtype=torch.float32, device=self.device
                ),
                "actions": torch.tensor(
                    self.actions[batch_indices],
                    dtype=torch.long if self.discrete else torch.float32,
                    device=self.device,
                ),
                "log_probs": torch.tensor(
                    self.log_probs[batch_indices], dtype=torch.float32, device=self.device
                ),
                "values": torch.tensor(
                    self.values[batch_indices], dtype=torch.float32, device=self.device
                ),
                "advantages": torch.tensor(
                    self.advantages[batch_indices], dtype=torch.float32, device=self.device
                ),
                "returns": torch.tensor(
                    self.returns[batch_indices], dtype=torch.float32, device=self.device
                ),
            }

    def get_all(self) -> Dict[str, torch.Tensor]:
        """获取所有数据。"""
        size = self.buffer_size if self.full else self.ptr
        return {
            "states": torch.tensor(
                self.states[:size], dtype=torch.float32, device=self.device
            ),
            "actions": torch.tensor(
                self.actions[:size],
                dtype=torch.long if self.discrete else torch.float32,
                device=self.device,
            ),
            "log_probs": torch.tensor(
                self.log_probs[:size], dtype=torch.float32, device=self.device
            ),
            "values": torch.tensor(
                self.values[:size], dtype=torch.float32, device=self.device
            ),
            "advantages": torch.tensor(
                self.advantages[:size], dtype=torch.float32, device=self.device
            ),
            "returns": torch.tensor(
                self.returns[:size], dtype=torch.float32, device=self.device
            ),
        }

    def reset(self) -> None:
        """重置缓冲区指针。"""
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        """返回当前存储的数据量。"""
        return self.buffer_size if self.full else self.ptr


class GAEBuffer(RolloutBuffer):
    """
    带GAE支持的Rollout缓冲区（RolloutBuffer的别名）。

    提供与RolloutBuffer相同的功能，命名更明确表示其用途。
    """
    pass


# ==================== 模块测试 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("Trajectory Buffers - Unit Tests")
    print("=" * 70)

    # 测试EpisodeBuffer
    print("\n[1] Testing EpisodeBuffer...")
    buffer = EpisodeBuffer()

    # 模拟episode
    for t in range(100):
        state = np.random.randn(4).astype(np.float32)
        action = np.random.randint(0, 2)
        reward = 1.0
        log_prob = -0.5
        value = 5.0
        done = (t == 99)
        entropy = 0.5

        buffer.store(state, action, reward, log_prob, value, done, entropy)

    assert len(buffer) == 100
    assert buffer.total_reward == 100.0
    print(f"    Episode length: {len(buffer)}")
    print(f"    Total reward: {buffer.total_reward}")

    # 测试转换为张量
    tensors = buffer.to_tensors()
    assert tensors["states"].shape == (100, 4)
    assert tensors["actions"].shape == (100,)
    print(f"    States tensor shape: {tensors['states'].shape}")
    print("    [PASS]")

    # 测试RolloutBuffer
    print("\n[2] Testing RolloutBuffer...")
    rollout = RolloutBuffer(
        buffer_size=128,
        state_dim=4,
        action_dim=1,
        discrete=True,
    )

    # 填充数据
    for t in range(128):
        state = np.random.randn(4).astype(np.float32)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        log_prob = -0.5
        value = np.random.randn()
        done = (t == 127)

        rollout.add(state, action, reward, log_prob, value, done)

    assert len(rollout) == 128
    print(f"    Buffer size: {len(rollout)}")

    # 测试GAE计算
    rollout.compute_gae(next_value=0.0, gamma=0.99, gae_lambda=0.95)
    print(f"    Advantages mean: {rollout.advantages.mean():.4f}")
    print(f"    Advantages std: {rollout.advantages.std():.4f}")
    print(f"    Returns mean: {rollout.returns.mean():.4f}")

    # 测试标准化
    rollout.normalize_advantages()
    assert abs(rollout.advantages.mean()) < 1e-6
    assert abs(rollout.advantages.std() - 1.0) < 1e-6
    print(f"    Normalized advantages mean: {rollout.advantages.mean():.6f}")
    print(f"    Normalized advantages std: {rollout.advantages.std():.6f}")
    print("    [PASS]")

    # 测试批次生成
    print("\n[3] Testing batch generation...")
    batch_count = 0
    total_samples = 0

    for batch in rollout.get_batches(batch_size=32, shuffle=True):
        batch_count += 1
        total_samples += batch["states"].shape[0]
        assert batch["states"].shape[1] == 4
        assert batch["advantages"].shape[0] == batch["states"].shape[0]

    assert total_samples == 128
    print(f"    Number of batches: {batch_count}")
    print(f"    Total samples: {total_samples}")
    print("    [PASS]")

    # 测试连续动作
    print("\n[4] Testing continuous action buffer...")
    cont_buffer = RolloutBuffer(
        buffer_size=64,
        state_dim=3,
        action_dim=2,
        discrete=False,
    )

    for t in range(64):
        state = np.random.randn(3).astype(np.float32)
        action = np.random.randn(2).astype(np.float32)
        reward = np.random.randn()
        log_prob = -1.0
        value = np.random.randn()
        done = False

        cont_buffer.add(state, action, reward, log_prob, value, done)

    cont_buffer.compute_gae(next_value=0.0)
    data = cont_buffer.get_all()

    assert data["states"].shape == (64, 3)
    assert data["actions"].shape == (64, 2)
    print(f"    States shape: {data['states'].shape}")
    print(f"    Actions shape: {data['actions'].shape}")
    print("    [PASS]")

    # 测试重置
    print("\n[5] Testing buffer reset...")
    rollout.reset()
    assert len(rollout) == 0
    assert rollout.ptr == 0
    assert not rollout.full
    print("    Buffer reset successful")
    print("    [PASS]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
