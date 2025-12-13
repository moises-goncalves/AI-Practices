"""
Q-Learning 与 SARSA 算法实现

本模块提供表格型强化学习算法的完整实现，包括：
- Q-Learning (Off-Policy TD Control)
- SARSA (On-Policy TD Control)
- Double Q-Learning
- 多种探索策略 (epsilon-greedy, softmax, UCB)
- 悬崖行走环境

理论基础:
    Q-Learning 更新公式:
        Q(S,A) <- Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]

    SARSA 更新公式:
        Q(S,A) <- Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]

参考文献:
    [1] Watkins, C.J.C.H. (1989). Learning from Delayed Rewards. PhD Thesis.
    [2] Rummery & Niranjan (1994). On-Line Q-Learning Using Connectionist Systems.
    [3] Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.

依赖:
    pip install numpy matplotlib gymnasium

作者: Ziming Ding
日期: 2024
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np

# ============================================================
# 类型定义
# ============================================================

State = TypeVar('State')
Action = TypeVar('Action', bound=int)


class ExplorationStrategy(Enum):
    """探索策略枚举"""
    EPSILON_GREEDY = "epsilon_greedy"
    SOFTMAX = "softmax"
    UCB = "ucb"


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class AgentConfig:
    """智能体配置参数

    Attributes:
        n_actions: 动作空间大小
        learning_rate: 学习率 α，控制Q值更新步长
        discount_factor: 折扣因子 γ，权衡即时与未来奖励
        epsilon: 初始探索率，epsilon-greedy策略中随机探索的概率
        epsilon_decay: 探索率衰减系数
        epsilon_min: 最小探索率下限
        exploration: 探索策略类型
        temperature: Softmax策略的温度参数
        ucb_c: UCB策略的探索系数
    """
    n_actions: int
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    exploration: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    temperature: float = 1.0
    ucb_c: float = 2.0


@dataclass
class TrainingMetrics:
    """训练指标记录

    Attributes:
        episode_rewards: 每回合累计奖励
        episode_lengths: 每回合步数
        epsilon_history: 探索率变化历史
        td_errors: TD误差历史 (可选)
    """
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    epsilon_history: List[float] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)

    def get_moving_average(self, window: int = 100) -> np.ndarray:
        """计算奖励的移动平均"""
        if len(self.episode_rewards) < window:
            return np.array(self.episode_rewards)
        return np.convolve(
            self.episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )


# ============================================================
# 探索策略实现
# ============================================================

class ExplorationMixin:
    """探索策略混入类，提供多种动作选择方法"""

    def _epsilon_greedy(
        self,
        q_values: np.ndarray,
        epsilon: float
    ) -> int:
        """Epsilon-Greedy策略

        以概率epsilon随机选择动作，否则选择Q值最大的动作。
        当存在多个最大Q值时，随机选择其中之一。

        Args:
            q_values: 当前状态的Q值数组
            epsilon: 探索概率

        Returns:
            选择的动作索引
        """
        if np.random.random() < epsilon:
            return np.random.randint(len(q_values))

        # 处理多个最大值情况，随机打破平局
        max_q = np.max(q_values)
        max_actions = np.where(np.isclose(q_values, max_q))[0]
        return np.random.choice(max_actions)

    def _softmax(
        self,
        q_values: np.ndarray,
        temperature: float
    ) -> int:
        """Softmax (Boltzmann) 策略

        根据Q值的softmax分布选择动作:
            P(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)

        温度τ控制探索程度:
            - τ→0: 趋向贪心选择
            - τ→∞: 趋向均匀随机

        Args:
            q_values: 当前状态的Q值数组
            temperature: 温度参数τ

        Returns:
            选择的动作索引
        """
        # 数值稳定性：减去最大值防止溢出
        q_scaled = (q_values - np.max(q_values)) / max(temperature, 1e-8)
        exp_q = np.exp(q_scaled)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probs)

    def _ucb(
        self,
        q_values: np.ndarray,
        action_counts: np.ndarray,
        total_count: int,
        c: float
    ) -> int:
        """Upper Confidence Bound (UCB) 策略

        选择置信上界最大的动作:
            A = argmax[Q(s,a) + c * sqrt(ln(t) / N(s,a))]

        平衡利用(Q值)和探索(不确定性)。

        Args:
            q_values: 当前状态的Q值数组
            action_counts: 每个动作的选择次数
            total_count: 总选择次数
            c: 探索系数，控制探索倾向

        Returns:
            选择的动作索引
        """
        # 未访问的动作优先选择
        if np.any(action_counts == 0):
            return np.random.choice(np.where(action_counts == 0)[0])

        ucb_values = q_values + c * np.sqrt(
            np.log(total_count + 1) / (action_counts + 1e-8)
        )
        return np.argmax(ucb_values)


# ============================================================
# 基础智能体抽象类
# ============================================================

class BaseAgent(ABC, ExplorationMixin):
    """表格型强化学习智能体基类

    提供Q表管理、探索策略、模型保存/加载等通用功能。
    子类需实现update方法定义具体的学习规则。
    """

    def __init__(self, config: AgentConfig):
        """初始化智能体

        Args:
            config: 智能体配置参数
        """
        self.config = config
        self.n_actions = config.n_actions
        self.lr = config.learning_rate
        self.gamma = config.discount_factor
        self.epsilon = config.epsilon
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.exploration = config.exploration
        self.temperature = config.temperature
        self.ucb_c = config.ucb_c

        # Q表：状态 -> 动作Q值数组
        self.q_table: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )

        # UCB策略需要的计数器
        self.action_counts: Dict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions)
        )
        self.total_steps = 0

        # 训练指标
        self.metrics = TrainingMetrics()

    def get_action(self, state: Any, training: bool = True) -> int:
        """根据当前策略选择动作

        Args:
            state: 当前环境状态
            training: 是否为训练模式。测试时使用贪心策略

        Returns:
            选择的动作索引
        """
        q_values = self.q_table[state]

        if not training:
            # 测试模式：纯贪心
            max_q = np.max(q_values)
            max_actions = np.where(np.isclose(q_values, max_q))[0]
            return np.random.choice(max_actions)

        # 训练模式：根据配置选择探索策略
        if self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(q_values, self.epsilon)
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            action = self._softmax(q_values, self.temperature)
        elif self.exploration == ExplorationStrategy.UCB:
            action = self._ucb(
                q_values,
                self.action_counts[state],
                self.total_steps,
                self.ucb_c
            )
        else:
            action = self._epsilon_greedy(q_values, self.epsilon)

        # 更新计数器
        self.action_counts[state][action] += 1
        self.total_steps += 1

        return action

    @abstractmethod
    def update(self, *args, **kwargs) -> float:
        """更新Q值，由子类实现

        Returns:
            TD误差值
        """
        pass

    def decay_epsilon(self) -> None:
        """衰减探索率"""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    def get_greedy_policy(self) -> Dict[Any, int]:
        """提取当前贪心策略

        Returns:
            状态到最优动作的映射字典
        """
        policy = {}
        for state in self.q_table:
            policy[state] = int(np.argmax(self.q_table[state]))
        return policy

    def get_value_function(self) -> Dict[Any, float]:
        """提取状态价值函数 V(s) = max_a Q(s,a)

        Returns:
            状态到价值的映射字典
        """
        return {
            state: float(np.max(q_values))
            for state, q_values in self.q_table.items()
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """保存模型到文件

        Args:
            filepath: 保存路径，支持.pkl和.json格式
        """
        filepath = Path(filepath)

        data = {
            'config': {
                'n_actions': self.n_actions,
                'learning_rate': self.lr,
                'discount_factor': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
            },
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'total_steps': self.total_steps,
        }

        if filepath.suffix == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

    def load(self, filepath: Union[str, Path]) -> None:
        """从文件加载模型

        Args:
            filepath: 模型文件路径
        """
        filepath = Path(filepath)

        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        # 恢复Q表
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for k, v in data['q_table'].items():
            # 尝试还原元组键
            try:
                key = eval(k)
            except:
                key = k
            self.q_table[key] = np.array(v)

        self.total_steps = data.get('total_steps', 0)

    def reset(self) -> None:
        """重置智能体状态"""
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.action_counts = defaultdict(lambda: np.zeros(self.n_actions))
        self.total_steps = 0
        self.epsilon = self.config.epsilon
        self.metrics = TrainingMetrics()


# ============================================================
# Q-Learning 智能体
# ============================================================

class QLearningAgent(BaseAgent):
    """Q-Learning智能体 (Off-Policy TD Control)

    Q-Learning是一种离策略(off-policy)时序差分控制算法。
    它直接学习最优动作价值函数Q*，更新时使用下一状态的最大Q值，
    与实际采取的动作无关。

    更新公式:
        Q(S,A) <- Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]

    特点:
        - 离策略：行为策略和目标策略可以不同
        - 学习最优策略，不受探索影响
        - 可能过估计Q值（可用Double Q-Learning缓解）

    Example:
        >>> config = AgentConfig(n_actions=4)
        >>> agent = QLearningAgent(config)
        >>> action = agent.get_action(state)
        >>> td_error = agent.update(state, action, reward, next_state, done)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        double_q: bool = False
    ):
        """初始化Q-Learning智能体

        Args:
            config: 配置对象，若提供则忽略其他参数
            n_actions: 动作数量
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 初始探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            double_q: 是否使用Double Q-Learning
        """
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )
        super().__init__(config)

        self.double_q = double_q
        if double_q:
            self.q_table2: Dict[Any, np.ndarray] = defaultdict(
                lambda: np.zeros(self.n_actions)
            )

    def get_action(self, state: Any, training: bool = True) -> int:
        """选择动作（Double Q-Learning使用两个Q表的和）"""
        if self.double_q:
            q_values = self.q_table[state] + self.q_table2[state]
        else:
            q_values = self.q_table[state]

        if not training:
            max_q = np.max(q_values)
            max_actions = np.where(np.isclose(q_values, max_q))[0]
            return np.random.choice(max_actions)

        if self.exploration == ExplorationStrategy.EPSILON_GREEDY:
            action = self._epsilon_greedy(q_values, self.epsilon)
        elif self.exploration == ExplorationStrategy.SOFTMAX:
            action = self._softmax(q_values, self.temperature)
        else:
            action = self._ucb(
                q_values,
                self.action_counts[state],
                self.total_steps,
                self.ucb_c
            )

        self.action_counts[state][action] += 1
        self.total_steps += 1
        return action

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> float:
        """Q-Learning更新规则

        使用TD目标 R + γ max_a Q(S',a) 更新Q值。

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 转移到的下一状态
            done: 是否为终止状态

        Returns:
            TD误差 δ = target - Q(S,A)
        """
        if self.double_q:
            return self._double_q_update(state, action, reward, next_state, done)

        current_q = self.q_table[state][action]

        # TD目标
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD误差
        td_error = target - current_q

        # 更新Q值
        self.q_table[state][action] += self.lr * td_error

        return td_error

    def _double_q_update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> float:
        """Double Q-Learning更新

        解耦动作选择和价值评估，减少过估计偏差:
            - 以0.5概率选择更新Q1或Q2
            - 更新Q1时：用Q1选择动作，Q2评估价值
            - 更新Q2时：用Q2选择动作，Q1评估价值
        """
        if np.random.random() < 0.5:
            # 更新Q1
            current_q = self.q_table[state][action]
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q_table[next_state])
                target = reward + self.gamma * self.q_table2[next_state][best_action]
            td_error = target - current_q
            self.q_table[state][action] += self.lr * td_error
        else:
            # 更新Q2
            current_q = self.q_table2[state][action]
            if done:
                target = reward
            else:
                best_action = np.argmax(self.q_table2[next_state])
                target = reward + self.gamma * self.q_table[next_state][best_action]
            td_error = target - current_q
            self.q_table2[state][action] += self.lr * td_error

        return td_error


# ============================================================
# SARSA 智能体
# ============================================================

class SARSAAgent(BaseAgent):
    """SARSA智能体 (On-Policy TD Control)

    SARSA (State-Action-Reward-State-Action) 是一种在策略(on-policy)
    时序差分控制算法。它学习当前行为策略的价值函数。

    更新公式:
        Q(S,A) <- Q(S,A) + α[R + γ Q(S',A') - Q(S,A)]

    与Q-Learning的关键区别:
        - 使用实际采取的下一动作A'，而非max
        - 学习的是当前策略的价值，而非最优策略
        - 更保守，会考虑探索带来的风险

    Example:
        >>> agent = SARSAAgent(config)
        >>> action = agent.get_action(state)
        >>> next_action = agent.get_action(next_state)
        >>> td_error = agent.update(state, action, reward, next_state, next_action, done)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """初始化SARSA智能体"""
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )
        super().__init__(config)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        next_action: int,
        done: bool
    ) -> float:
        """SARSA更新规则

        使用TD目标 R + γ Q(S',A') 更新Q值。
        注意：需要提供下一步实际采取的动作。

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            next_action: 下一状态实际采取的动作
            done: 是否终止

        Returns:
            TD误差
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            # SARSA: 使用实际的next_action
            target = reward + self.gamma * self.q_table[next_state][next_action]

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        return td_error


# ============================================================
# Expected SARSA 智能体
# ============================================================

class ExpectedSARSAAgent(BaseAgent):
    """Expected SARSA智能体

    Expected SARSA使用下一状态Q值的期望，而非采样值。
    它结合了Q-Learning的低方差和SARSA的在策略特性。

    更新公式:
        Q(S,A) <- Q(S,A) + α[R + γ E[Q(S',A')] - Q(S,A)]

    其中期望在当前策略下计算:
        E[Q(S',A')] = Σ_a π(a|S') Q(S',a)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        if config is None:
            config = AgentConfig(
                n_actions=n_actions,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            )
        super().__init__(config)

    def _get_expected_q(self, state: Any) -> float:
        """计算ε-greedy策略下的期望Q值"""
        q_values = self.q_table[state]

        # ε-greedy策略下的动作概率
        n_actions = len(q_values)
        probs = np.ones(n_actions) * self.epsilon / n_actions
        best_action = np.argmax(q_values)
        probs[best_action] += 1 - self.epsilon

        return np.dot(probs, q_values)

    def update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> float:
        """Expected SARSA更新"""
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self._get_expected_q(next_state)

        td_error = target - current_q
        self.q_table[state][action] += self.lr * td_error

        return td_error


# ============================================================
# 悬崖行走环境
# ============================================================

class CliffWalkingEnv:
    """悬崖行走环境 (Cliff Walking)

    经典的强化学习测试环境，用于对比Q-Learning和SARSA的行为差异。

    环境布局 (4x12 网格):
        ┌─────────────────────────────────────────────┐
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 0
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 1
        │ .  .  .  .  .  .  .  .  .  .  .  .  │  row 2
        │ S  C  C  C  C  C  C  C  C  C  C  G  │  row 3
        └─────────────────────────────────────────────┘
          0  1  2  3  4  5  6  7  8  9 10 11    columns

    符号说明:
        S: 起点 (3, 0)
        G: 目标 (3, 11)
        C: 悬崖 (3, 1) ~ (3, 10)
        .: 普通格子

    动作空间: {0: 上, 1: 右, 2: 下, 3: 左}

    奖励设计:
        - 每步: -1 (鼓励快速到达)
        - 掉入悬崖: -100，并重置到起点
        - 到达目标: 0，回合结束

    算法行为差异:
        - Q-Learning: 学习最短路径（沿悬崖边），因为更新时不考虑探索风险
        - SARSA: 学习安全路径（远离悬崖），因为会考虑探索时掉下悬崖的可能
    """

    # 动作映射
    ACTIONS = {
        0: (-1, 0),   # 上
        1: (0, 1),    # 右
        2: (1, 0),    # 下
        3: (0, -1)    # 左
    }
    ACTION_NAMES = ['上', '右', '下', '左']

    def __init__(self, height: int = 4, width: int = 12):
        """初始化环境

        Args:
            height: 网格高度
            width: 网格宽度
        """
        self.height = height
        self.width = width

        # 特殊位置
        self.start = (height - 1, 0)
        self.goal = (height - 1, width - 1)
        self.cliff = [(height - 1, j) for j in range(1, width - 1)]

        # 当前状态
        self.state = self.start

        # 环境属性
        self.n_states = height * width
        self.n_actions = 4

    def reset(self) -> Tuple[int, int]:
        """重置环境到初始状态

        Returns:
            起始状态坐标 (row, col)
        """
        self.state = self.start
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """执行动作，返回转移结果

        Args:
            action: 动作索引 (0-3)

        Returns:
            (next_state, reward, done) 三元组
        """
        # 计算下一位置
        di, dj = self.ACTIONS[action]
        new_i = np.clip(self.state[0] + di, 0, self.height - 1)
        new_j = np.clip(self.state[1] + dj, 0, self.width - 1)
        next_state = (int(new_i), int(new_j))

        # 检查是否掉入悬崖
        if next_state in self.cliff:
            self.state = self.start
            return self.state, -100.0, False

        self.state = next_state

        # 检查是否到达目标
        if self.state == self.goal:
            return self.state, 0.0, True

        return self.state, -1.0, False

    def render(self, path: Optional[List[Tuple[int, int]]] = None) -> str:
        """渲染环境状态

        Args:
            path: 可选的路径点列表，用于可视化策略

        Returns:
            环境的字符串表示
        """
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]

        # 标记悬崖
        for pos in self.cliff:
            grid[pos[0]][pos[1]] = 'C'

        # 标记起点和终点
        grid[self.start[0]][self.start[1]] = 'S'
        grid[self.goal[0]][self.goal[1]] = 'G'

        # 标记路径
        if path:
            for pos in path[1:-1]:
                if pos not in self.cliff and pos != self.start and pos != self.goal:
                    grid[pos[0]][pos[1]] = '*'

        # 标记当前位置
        if self.state != self.start and self.state != self.goal:
            if self.state not in self.cliff:
                grid[self.state[0]][self.state[1]] = '@'

        # 构建输出字符串
        border = "┌" + "─" * (self.width * 2 + 1) + "┐"
        lines = [border]
        for row in grid:
            lines.append("│ " + " ".join(row) + " │")
        lines.append("└" + "─" * (self.width * 2 + 1) + "┘")

        output = "\n".join(lines)
        print(output)
        return output

    def get_optimal_path(self) -> List[Tuple[int, int]]:
        """获取最短路径（不考虑悬崖风险）

        Returns:
            从起点到终点的最短路径
        """
        # 沿着悬崖边走的最短路径
        path = [self.start]
        # 先向右走到终点正上方
        for j in range(1, self.width):
            path.append((self.height - 1, j))
        return path

    def get_safe_path(self) -> List[Tuple[int, int]]:
        """获取安全路径（远离悬崖）

        Returns:
            避开悬崖的安全路径
        """
        path = [self.start]
        # 先向上
        for i in range(self.height - 2, -1, -1):
            path.append((i, 0))
        # 向右
        for j in range(1, self.width):
            path.append((0, j))
        # 向下到达目标
        for i in range(1, self.height):
            path.append((i, self.width - 1))
        return path


# ============================================================
# 训练工具函数
# ============================================================

def train_q_learning(
    env,
    agent: QLearningAgent,
    episodes: int = 500,
    max_steps: int = 200,
    verbose: bool = True,
    log_interval: int = 100
) -> TrainingMetrics:
    """训练Q-Learning智能体

    Args:
        env: 环境实例（需实现reset()和step()方法）
        agent: Q-Learning智能体
        episodes: 训练回合数
        max_steps: 每回合最大步数
        verbose: 是否打印训练进度
        log_interval: 日志打印间隔

    Returns:
        训练指标记录
    """
    metrics = TrainingMetrics()

    for episode in range(episodes):
        # 环境重置
        if hasattr(env, 'reset'):
            result = env.reset()
            state = result[0] if isinstance(result, tuple) else result
        else:
            state = env.reset()

        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.get_action(state, training=True)

            # 执行动作
            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # 更新Q值
            td_error = agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # 衰减探索率
        agent.decay_epsilon()

        # 记录指标
        metrics.episode_rewards.append(total_reward)
        metrics.episode_lengths.append(steps)
        metrics.epsilon_history.append(agent.epsilon)

        # 打印进度
        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(metrics.episode_rewards[-log_interval:])
            avg_steps = np.mean(metrics.episode_lengths[-log_interval:])
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Avg Steps: {avg_steps:6.1f} | "
                  f"ε: {agent.epsilon:.4f}")

    agent.metrics = metrics
    return metrics


def train_sarsa(
    env,
    agent: SARSAAgent,
    episodes: int = 500,
    max_steps: int = 200,
    verbose: bool = True,
    log_interval: int = 100
) -> TrainingMetrics:
    """训练SARSA智能体

    Args:
        env: 环境实例
        agent: SARSA智能体
        episodes: 训练回合数
        max_steps: 每回合最大步数
        verbose: 是否打印训练进度
        log_interval: 日志打印间隔

    Returns:
        训练指标记录
    """
    metrics = TrainingMetrics()

    for episode in range(episodes):
        # 环境重置
        if hasattr(env, 'reset'):
            result = env.reset()
            state = result[0] if isinstance(result, tuple) else result
        else:
            state = env.reset()

        # SARSA需要先选择初始动作
        action = agent.get_action(state, training=True)

        total_reward = 0.0
        steps = 0

        for step in range(max_steps):
            # 执行动作
            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
            else:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated

            # 选择下一个动作
            next_action = agent.get_action(next_state, training=True)

            # SARSA更新（需要next_action）
            td_error = agent.update(
                state, action, reward, next_state, next_action, done
            )

            state = next_state
            action = next_action
            total_reward += reward
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        metrics.episode_rewards.append(total_reward)
        metrics.episode_lengths.append(steps)
        metrics.epsilon_history.append(agent.epsilon)

        if verbose and (episode + 1) % log_interval == 0:
            avg_reward = np.mean(metrics.episode_rewards[-log_interval:])
            avg_steps = np.mean(metrics.episode_lengths[-log_interval:])
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Avg Steps: {avg_steps:6.1f} | "
                  f"ε: {agent.epsilon:.4f}")

    agent.metrics = metrics
    return metrics


def extract_path(
    agent: BaseAgent,
    env: CliffWalkingEnv,
    max_steps: int = 50
) -> List[Tuple[int, int]]:
    """从训练好的智能体提取贪心策略路径

    Args:
        agent: 训练好的智能体
        env: 环境实例
        max_steps: 最大步数（防止无限循环）

    Returns:
        策略产生的状态序列
    """
    state = env.reset()
    path = [state]

    for _ in range(max_steps):
        action = agent.get_action(state, training=False)
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
        if done:
            break

    return path


# ============================================================
# 可视化工具
# ============================================================

def plot_learning_curves(
    metrics_dict: Dict[str, TrainingMetrics],
    window: int = 10,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> None:
    """绘制学习曲线对比图

    Args:
        metrics_dict: {算法名称: 训练指标} 字典
        window: 平滑窗口大小
        figsize: 图形尺寸
        save_path: 保存路径（可选）
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 奖励曲线
    ax1 = axes[0]
    for name, metrics in metrics_dict.items():
        smoothed = np.convolve(
            metrics.episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        ax1.plot(smoothed, label=name, alpha=0.8)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Learning Curve: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 步数曲线
    ax2 = axes[1]
    for name, metrics in metrics_dict.items():
        smoothed = np.convolve(
            metrics.episode_lengths,
            np.ones(window) / window,
            mode='valid'
        )
        ax2.plot(smoothed, label=name, alpha=0.8)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Learning Curve: Episode Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def visualize_q_table(
    agent: BaseAgent,
    env: CliffWalkingEnv,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None
) -> None:
    """可视化Q表和策略

    Args:
        agent: 训练好的智能体
        env: 环境实例
        figsize: 图形尺寸
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 准备数据
    v_table = np.zeros((env.height, env.width))
    policy_arrows = np.zeros((env.height, env.width), dtype=int)

    for i in range(env.height):
        for j in range(env.width):
            state = (i, j)
            if state in agent.q_table:
                v_table[i, j] = np.max(agent.q_table[state])
                policy_arrows[i, j] = np.argmax(agent.q_table[state])

    # 价值函数热力图
    ax1 = axes[0]
    im = ax1.imshow(v_table, cmap='RdYlGn')
    ax1.set_title('Value Function V(s)')
    plt.colorbar(im, ax=ax1)

    # 标记特殊位置
    for pos in env.cliff:
        ax1.add_patch(plt.Rectangle(
            (pos[1]-0.5, pos[0]-0.5), 1, 1,
            fill=True, color='black', alpha=0.5
        ))

    # 策略箭头图
    ax2 = axes[1]
    arrow_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    for i in range(env.height):
        for j in range(env.width):
            if (i, j) in env.cliff:
                ax2.text(j, i, 'X', ha='center', va='center', fontsize=12)
            elif (i, j) == env.goal:
                ax2.text(j, i, 'G', ha='center', va='center', fontsize=12, color='green')
            elif (i, j) == env.start:
                ax2.text(j, i, 'S', ha='center', va='center', fontsize=12, color='blue')
            else:
                ax2.text(j, i, arrow_map[policy_arrows[i, j]],
                        ha='center', va='center', fontsize=14)

    ax2.set_xlim(-0.5, env.width - 0.5)
    ax2.set_ylim(env.height - 0.5, -0.5)
    ax2.set_title('Greedy Policy')
    ax2.grid(True)

    # Q值分布
    ax3 = axes[2]
    q_max_values = [np.max(q) for q in agent.q_table.values() if np.any(q != 0)]
    if q_max_values:
        ax3.hist(q_max_values, bins=30, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Q Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Q Value Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


# ============================================================
# 主程序入口
# ============================================================

def compare_cliff_walking(
    episodes: int = 500,
    learning_rate: float = 0.5,
    epsilon: float = 0.1,
    show_plots: bool = True
) -> Tuple[QLearningAgent, SARSAAgent]:
    """在悬崖行走环境上对比Q-Learning和SARSA

    Args:
        episodes: 训练回合数
        learning_rate: 学习率
        epsilon: 固定探索率
        show_plots: 是否显示图表

    Returns:
        (q_agent, sarsa_agent) 元组
    """
    print("=" * 60)
    print("悬崖行走环境: Q-Learning vs SARSA 对比实验")
    print("=" * 60)

    env = CliffWalkingEnv()

    # 创建智能体（固定epsilon，不衰减）
    q_agent = QLearningAgent(
        n_actions=4,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=1.0,
        epsilon_min=epsilon
    )

    sarsa_agent = SARSAAgent(
        n_actions=4,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=1.0,
        epsilon_min=epsilon
    )

    # 训练
    print("\n训练 Q-Learning...")
    q_metrics = train_q_learning(
        env, q_agent, episodes=episodes, verbose=True, log_interval=100
    )

    print("\n训练 SARSA...")
    sarsa_metrics = train_sarsa(
        env, sarsa_agent, episodes=episodes, verbose=True, log_interval=100
    )

    # 显示学到的路径
    print("\n" + "=" * 60)
    print("学习到的策略路径")
    print("=" * 60)

    print("\nQ-Learning (倾向最短路径):")
    q_path = extract_path(q_agent, env)
    env.render(q_path)

    env.reset()
    print("\nSARSA (倾向安全路径):")
    sarsa_path = extract_path(sarsa_agent, env)
    env.render(sarsa_path)

    # 统计
    print("\n" + "=" * 60)
    print("训练统计")
    print("=" * 60)
    print(f"Q-Learning 最后100回合平均奖励: {np.mean(q_metrics.episode_rewards[-100:]):.2f}")
    print(f"SARSA 最后100回合平均奖励: {np.mean(sarsa_metrics.episode_rewards[-100:]):.2f}")

    if show_plots:
        plot_learning_curves(
            {'Q-Learning': q_metrics, 'SARSA': sarsa_metrics},
            window=10,
            save_path='cliff_walking_comparison.png'
        )

    return q_agent, sarsa_agent


def train_taxi(
    episodes: int = 2000,
    show_plots: bool = True
) -> QLearningAgent:
    """在Taxi-v3环境上训练Q-Learning

    Args:
        episodes: 训练回合数
        show_plots: 是否显示图表

    Returns:
        训练好的智能体
    """
    try:
        import gymnasium as gym
    except ImportError:
        print("请安装gymnasium: pip install gymnasium")
        return None

    print("\n" + "=" * 60)
    print("Taxi-v3 Q-Learning 训练")
    print("=" * 60)

    env = gym.make('Taxi-v3')

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    metrics = train_q_learning(
        env, agent, episodes=episodes, verbose=True, log_interval=200
    )

    env.close()

    if show_plots:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        window = 50
        smoothed_rewards = np.convolve(
            metrics.episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        axes[0].plot(smoothed_rewards)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Taxi-v3: Reward per Episode')
        axes[0].grid(True, alpha=0.3)

        smoothed_steps = np.convolve(
            metrics.episode_lengths,
            np.ones(window) / window,
            mode='valid'
        )
        axes[1].plot(smoothed_steps)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].set_title('Taxi-v3: Steps per Episode')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('taxi_training.png', dpi=150)
        plt.show()

    return agent


def main():
    """主程序入口"""
    import argparse

    parser = argparse.ArgumentParser(description='Q-Learning 与 SARSA 实验')
    parser.add_argument('--exp', type=str, default='cliff',
                       choices=['cliff', 'taxi', 'all'],
                       help='实验类型: cliff/taxi/all')
    parser.add_argument('--episodes', type=int, default=500,
                       help='训练回合数')
    parser.add_argument('--no-plot', action='store_true',
                       help='不显示图表')

    args = parser.parse_args()

    if args.exp in ['cliff', 'all']:
        compare_cliff_walking(
            episodes=args.episodes,
            show_plots=not args.no_plot
        )

    if args.exp in ['taxi', 'all']:
        train_taxi(
            episodes=args.episodes * 4 if args.exp == 'all' else args.episodes,
            show_plots=not args.no_plot
        )


if __name__ == "__main__":
    main()
