"""
DQN Agent实现

============================================================
核心思想 (Core Idea)
============================================================
Deep Q-Network通过神经网络逼近Q函数，实现高维状态空间的端到端学习。
结合经验回放和目标网络实现训练稳定性。

============================================================
数学基础 (Mathematical Foundation)
============================================================
训练目标：最小化TD误差

    L(θ) = E[(y - Q(s, a; θ))²]

TD目标：
    y = r + γ max_{a'} Q(s', a'; θ⁻)  (标准DQN)
    y = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)  (Double DQN)

策略：ε-贪婪
    π(a|s) = { argmax_a Q(s, a; θ), 概率 1 - ε
             { 均匀随机,            概率 ε

============================================================
算法流程 (Algorithm Flow)
============================================================
1. 初始化: 在线网络θ, 目标网络θ⁻ ← θ, 回放缓冲区D
2. 每一步:
    a. 选择动作: a = ε-greedy(Q(s, ·; θ))
    b. 执行动作: (r, s') = env.step(a)
    c. 存储转移: D.push((s, a, r, s', done))
    d. 采样批次: B ~ D
    e. 计算目标: y = r + γ (1-done) max_{a'} Q(s', a'; θ⁻)
    f. 更新网络: θ ← θ - α ∇_θ (y - Q(s, a; θ))²
    g. 同步目标: if step % C == 0: θ⁻ ← θ

============================================================
参考文献 (References)
============================================================
[1] Mnih, V., et al. (2015). Human-level control through deep RL. Nature.
[2] van Hasselt, H., et al. (2016). Deep RL with Double Q-learning. AAAI.
[3] Wang, Z., et al. (2016). Dueling Network Architectures. ICML.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray

import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.config import DQNConfig
from core.enums import NetworkType, LossType
from core.types import FloatArray
from buffers.base import ReplayBuffer
from buffers.prioritized import PrioritizedReplayBuffer
from networks.base import DQNNetwork
from networks.dueling import DuelingDQNNetwork


class DQNAgent:
    """
    Deep Q-Network Agent
    
    ============================================================
    算法对比 (Algorithm Comparison)
    ============================================================
    标准 vs Double DQN:
    - 标准: 使用max同时选择和评估，导致过估计
    - Double: 解耦选择(θ)和评估(θ⁻)，减少偏差
    
    vs 策略梯度:
    + 更高的样本效率（离策略回放）
    + 实现更简单
    - 只支持离散动作
    - 确定性策略（需要外部探索）
    
    ============================================================
    复杂度分析 (Complexity Analysis)
    ============================================================
    每步: O(B × |θ|) 前向 + 反向
    空间: O(N × d + |θ|) 缓冲区 + 网络
    """
    
    def __init__(self, config: DQNConfig) -> None:
        """
        初始化DQN Agent
        
        Parameters
        ----------
        config : DQNConfig
            超参数配置对象
        """
        self.config = config
        self.device = config.get_device()
        
        if config.seed is not None:
            self._set_seed(config.seed)
        
        self._init_networks()
        self._init_replay_buffer()
        
        self._epsilon = config.epsilon_start
        self._training_step = 0
        self._update_count = 0
        
        # 训练统计
        self._losses: List[float] = []
        self._q_values: List[float] = []
        self._td_errors: List[float] = []
    
    def _set_seed(self, seed: int) -> None:
        """设置随机种子以保证可复现性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _init_networks(self) -> None:
        """初始化在线网络和目标网络"""
        NetworkClass: Type[nn.Module]
        if self.config.network_type == NetworkType.DUELING:
            NetworkClass = DuelingDQNNetwork
        else:
            NetworkClass = DQNNetwork
        
        self.q_network: nn.Module = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)
        
        self.target_network: nn.Module = NetworkClass(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
        ).to(self.device)
        
        # 同步目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
        )
    
    def _init_replay_buffer(self) -> None:
        """初始化经验回放缓冲区"""
        self.replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer] = ReplayBuffer(
            self.config.buffer_size
        )
    
    @property
    def epsilon(self) -> float:
        """当前探索率"""
        return self._epsilon
    
    @property
    def training_step(self) -> int:
        """当前训练步数"""
        return self._training_step
    
    @property
    def update_count(self) -> int:
        """当前网络更新次数"""
        return self._update_count
    
    @property
    def losses(self) -> List[float]:
        """训练损失历史"""
        return self._losses.copy()
    
    @property
    def q_values(self) -> List[float]:
        """平均Q值历史"""
        return self._q_values.copy()
    
    def select_action(
        self,
        state: FloatArray,
        training: bool = True,
    ) -> int:
        """
        使用ε-贪婪策略选择动作
        
        Parameters
        ----------
        state : FloatArray
            当前状态
        training : bool
            是否处于训练模式（启用探索）
        
        Returns
        -------
        int
            动作索引
        """
        if training and random.random() < self._epsilon:
            return random.randint(0, self.config.action_dim - 1)
        
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return int(q_values.argmax(dim=1).item())
    
    def store_transition(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> None:
        """存储转移到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        执行一次梯度更新步骤
        
        Returns
        -------
        Optional[float]
            损失值，如果缓冲区样本不足则返回None
        """
        min_size = self.config.min_buffer_size or self.config.batch_size
        if not self.replay_buffer.is_ready(min_size):
            return None
        
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            return self._update_prioritized()
        return self._update_uniform()
    
    def _update_uniform(self) -> float:
        """使用均匀回放采样更新"""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        # 转换为张量
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states_t)
        current_q = current_q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: 用在线网络选择动作，用目标网络评估
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # 标准DQN
                next_q = self.target_network(next_states_t).max(dim=1)[0]
            
            target_q = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)
        
        # 计算损失
        if self.config.loss_type == LossType.HUBER:
            loss = F.smooth_l1_loss(current_q, target_q)
        else:
            loss = F.mse_loss(current_q, target_q)
        
        # 优化
        self._optimize(loss)
        
        loss_value = loss.item()
        self._losses.append(loss_value)
        self._q_values.append(current_q.mean().item())
        
        return loss_value
    
    def _update_prioritized(self) -> float:
        """使用优先回放采样更新"""
        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
        
        (
            states, actions, rewards, next_states, dones, indices, weights,
        ) = self.replay_buffer.sample(self.config.batch_size)
        
        # 转换为张量
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        
        # 计算当前Q值
        current_q_values = self.q_network(states_t)
        current_q = current_q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.q_network(next_states_t).argmax(dim=1)
                next_q = self.target_network(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_network(next_states_t).max(dim=1)[0]
            
            target_q = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)
        
        # 计算TD误差
        td_errors = target_q - current_q
        
        # 计算加权损失
        if self.config.loss_type == LossType.HUBER:
            element_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
        else:
            element_loss = F.mse_loss(current_q, target_q, reduction="none")
        
        loss = (weights_t * element_loss).mean()
        
        # 优化
        self._optimize(loss)
        
        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        loss_value = loss.item()
        self._losses.append(loss_value)
        self._q_values.append(current_q.mean().item())
        self._td_errors.append(td_errors.abs().mean().item())
        
        return loss_value
    
    def _optimize(self, loss: torch.Tensor) -> None:
        """执行优化步骤"""
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.q_network.parameters(),
                self.config.grad_clip,
            )
        
        self.optimizer.step()
        
        self._update_count += 1
        self._sync_target_network()
    
    def _sync_target_network(self) -> None:
        """同步目标网络参数"""
        if self.config.soft_update_tau is not None:
            # 软更新: θ⁻ ← τθ + (1-τ)θ⁻
            tau = self.config.soft_update_tau
            for target_param, online_param in zip(
                self.target_network.parameters(),
                self.q_network.parameters(),
            ):
                target_param.data.copy_(
                    tau * online_param.data + (1.0 - tau) * target_param.data
                )
        elif self._update_count % self.config.target_update_freq == 0:
            # 硬更新: θ⁻ ← θ
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self) -> None:
        """线性衰减探索率"""
        self._training_step += 1
        decay_progress = min(1.0, self._training_step / self.config.epsilon_decay)
        self._epsilon = (
            self.config.epsilon_start
            + (self.config.epsilon_end - self.config.epsilon_start) * decay_progress
        )
    
    def train_step(
        self,
        state: FloatArray,
        action: int,
        reward: float,
        next_state: FloatArray,
        done: bool,
    ) -> Optional[float]:
        """
        完整训练步骤：存储 + 更新 + 衰减epsilon
        
        Parameters
        ----------
        state : FloatArray
            当前状态
        action : int
            执行的动作
        reward : float
            即时奖励
        next_state : FloatArray
            下一状态
        done : bool
            终止标志
        
        Returns
        -------
        Optional[float]
            损失值
        """
        self.store_transition(state, action, reward, next_state, done)
        loss = self.update()
        self.decay_epsilon()
        return loss
    
    def save(self, path: Union[str, Path]) -> None:
        """
        保存模型检查点
        
        Parameters
        ----------
        path : Union[str, Path]
            保存路径
        """
        checkpoint = {
            "config": self.config.to_dict(),
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self._epsilon,
            "training_step": self._training_step,
            "update_count": self._update_count,
            "losses": self._losses[-1000:],
            "q_values": self._q_values[-1000:],
        }
        torch.save(checkpoint, path)
    
    def load(self, path: Union[str, Path], load_optimizer: bool = True) -> None:
        """
        加载模型检查点
        
        Parameters
        ----------
        path : Union[str, Path]
            检查点路径
        load_optimizer : bool
            是否加载优化器状态
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._epsilon = checkpoint.get("epsilon", self.config.epsilon_end)
        self._training_step = checkpoint.get("training_step", 0)
        self._update_count = checkpoint.get("update_count", 0)
        self._losses = checkpoint.get("losses", [])
        self._q_values = checkpoint.get("q_values", [])
    
    def get_q_values(self, state: FloatArray) -> NDArray[np.floating[Any]]:
        """获取状态的Q值（用于调试和可视化）"""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]
    
    def set_eval_mode(self) -> None:
        """设置网络为评估模式"""
        self.q_network.eval()
    
    def set_train_mode(self) -> None:
        """设置网络为训练模式"""
        self.q_network.train()
    
    def __repr__(self) -> str:
        return (
            f"DQNAgent(state_dim={self.config.state_dim}, "
            f"action_dim={self.config.action_dim}, "
            f"double_dqn={self.config.double_dqn}, "
            f"dueling={self.config.dueling}, "
            f"device={self.device})"
        )


def create_dqn_agent(
    state_dim: int,
    action_dim: int,
    double_dqn: bool = True,
    dueling: bool = False,
    prioritized_replay: bool = False,
    **kwargs: Any,
) -> DQNAgent:
    """
    工厂函数创建DQN Agent
    
    Parameters
    ----------
    state_dim : int
        状态空间维度
    action_dim : int
        离散动作数量
    double_dqn : bool
        使用Double DQN
    dueling : bool
        使用Dueling架构
    prioritized_replay : bool
        使用优先经验回放
    **kwargs : Any
        其他DQNConfig参数
    
    Returns
    -------
    DQNAgent
        配置好的DQN Agent
    """
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        double_dqn=double_dqn,
        dueling=dueling,
        **kwargs,
    )
    agent = DQNAgent(config)
    
    if prioritized_replay:
        agent.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            alpha=kwargs.get("per_alpha", 0.6),
            beta_start=kwargs.get("per_beta_start", 0.4),
            beta_frames=kwargs.get("per_beta_frames", 100000),
        )
    
    return agent
