"""
时序差分学习算法 (Temporal Difference Learning Algorithms)
============================================================

核心思想 (Core Idea):
--------------------
时序差分学习是强化学习的核心范式，它结合了蒙特卡洛方法的采样思想和动态规划的
自举(Bootstrapping)思想。TD方法无需等待回合结束，仅依赖下一步的估计值就能
更新当前状态的价值估计——这是"用猜测更新猜测"的精髓。

数学原理 (Mathematical Theory):
------------------------------
TD(0)更新规则:
    V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]

其中:
    - V(S_t): 状态S_t的价值估计
    - α: 学习率 (learning rate)
    - R_{t+1}: 从S_t转移到S_{t+1}获得的即时奖励
    - γ: 折扣因子 (discount factor), γ ∈ [0, 1]
    - R_{t+1} + γV(S_{t+1}): TD目标 (TD target)
    - δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t): TD误差 (TD error)

TD(λ)资格迹更新:
    E_t(s) = γλE_{t-1}(s) + 𝟙(S_t = s)  (累积迹)
    或
    E_t(s) = (1-α)γλE_{t-1}(s) + 𝟙(S_t = s)  (荷兰迹)

    V(s) ← V(s) + αδ_t E_t(s), ∀s

问题背景 (Problem Statement):
----------------------------
Monte Carlo方法需要等待整个回合结束才能更新价值估计，这在以下场景存在问题:
1. 回合很长或无限长
2. 需要在线学习(online learning)
3. 需要快速适应环境变化

TD方法通过自举解决了这些问题，同时保持了无模型(model-free)的优势。

算法对比 (Comparison):
---------------------
┌─────────────┬──────────────┬─────────────┬─────────────┐
│   算法      │   偏差       │    方差     │  数据效率   │
├─────────────┼──────────────┼─────────────┼─────────────┤
│ Monte Carlo │   无偏       │    高       │    低       │
│ TD(0)       │   有偏       │    低       │    高       │
│ TD(λ)       │   可调       │    可调     │    可调     │
│ n-step TD   │   可调       │    可调     │    可调     │
└─────────────┴──────────────┴─────────────┴─────────────┘

复杂度 (Complexity):
-------------------
- TD(0): 时间O(1)/步, 空间O(|S|)
- TD(λ): 时间O(|S|)/步, 空间O(|S|)
- SARSA: 时间O(1)/步, 空间O(|S|×|A|)

算法总结 (Summary):
-----------------
TD学习是一种在线、增量式的价值函数学习方法。它在每一步都能更新估计值，
无需等待回合结束。这使得TD方法特别适合连续任务和需要快速响应的场景。
TD(λ)通过资格迹统一了TD(0)和Monte Carlo，提供了偏差-方差权衡的灵活性。
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Optional, Tuple, Callable,
    Protocol, TypeVar, Generic, Any, Union
)
import warnings
from collections import defaultdict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 类型定义与协议
# =============================================================================

State = TypeVar('State')
Action = TypeVar('Action')


class Environment(Protocol[State, Action]):
    """
    环境协议，定义强化学习环境的最小接口。
    兼容OpenAI Gym/Gymnasium风格的环境。
    """

    def reset(self) -> Tuple[State, Dict[str, Any]]:
        """重置环境，返回初始状态和信息字典。"""
        ...

    def step(self, action: Action) -> Tuple[State, float, bool, bool, Dict[str, Any]]:
        """
        执行动作，返回(新状态, 奖励, 终止标志, 截断标志, 信息字典)。
        """
        ...

    @property
    def action_space(self) -> Any:
        """返回动作空间。"""
        ...

    @property
    def observation_space(self) -> Any:
        """返回观测空间。"""
        ...


class Policy(Protocol[State, Action]):
    """策略协议，根据状态选择动作。"""

    def __call__(self, state: State) -> Action:
        """根据状态返回动作。"""
        ...

    def action_probabilities(self, state: State) -> Dict[Action, float]:
        """返回状态下各动作的概率分布。"""
        ...


# =============================================================================
# 配置类
# =============================================================================

class EligibilityTraceType(Enum):
    """
    资格迹类型枚举。

    资格迹是TD(λ)的核心机制，用于追踪哪些状态对当前TD误差"负有责任"。
    """
    ACCUMULATING = auto()  # 累积迹: E(s) ← γλE(s) + 1
    REPLACING = auto()      # 替换迹: E(s) ← 1 (访问时重置为1)
    DUTCH = auto()          # 荷兰迹: E(s) ← (1-α)γλE(s) + 1 (解决累积迹的发散问题)


@dataclass
class TDConfig:
    """
    时序差分学习配置类。

    封装所有TD算法的超参数，便于实验管理和复现。

    Attributes:
        alpha: 学习率，控制新信息对估计值的影响程度。
               太大导致不稳定，太小收敛慢。典型值: 0.01-0.5
        gamma: 折扣因子，决定未来奖励的重要性。
               γ=0表示只关心即时奖励，γ=1表示长远奖励同等重要。
        lambda_: TD(λ)的λ参数，控制自举程度。
                 λ=0退化为TD(0)，λ=1退化为Monte Carlo。
        epsilon: ε-greedy策略的探索率。
        n_step: n-step TD的步数。
        trace_type: 资格迹类型。
        initial_value: 价值函数初始化值，乐观初始化可促进探索。
    """
    alpha: float = 0.1
    gamma: float = 0.99
    lambda_: float = 0.9
    epsilon: float = 0.1
    n_step: int = 1
    trace_type: EligibilityTraceType = EligibilityTraceType.ACCUMULATING
    initial_value: float = 0.0

    def __post_init__(self) -> None:
        """参数验证。"""
        if not 0 < self.alpha <= 1:
            raise ValueError(f"学习率alpha必须在(0, 1]范围内，当前值: {self.alpha}")
        if not 0 <= self.gamma <= 1:
            raise ValueError(f"折扣因子gamma必须在[0, 1]范围内，当前值: {self.gamma}")
        if not 0 <= self.lambda_ <= 1:
            raise ValueError(f"λ参数必须在[0, 1]范围内，当前值: {self.lambda_}")
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"探索率epsilon必须在[0, 1]范围内，当前值: {self.epsilon}")
        if self.n_step < 1:
            raise ValueError(f"n_step必须至少为1，当前值: {self.n_step}")


@dataclass
class TrainingMetrics:
    """
    训练指标记录类。

    用于追踪和分析训练过程中的各项指标。
    """
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)
    value_changes: List[float] = field(default_factory=list)

    def add_episode(
        self,
        reward: float,
        length: int,
        avg_td_error: float = 0.0,
        avg_value_change: float = 0.0
    ) -> None:
        """记录一个回合的指标。"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.td_errors.append(avg_td_error)
        self.value_changes.append(avg_value_change)

    def get_moving_average(self, window: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """计算奖励和回合长度的移动平均。"""
        if len(self.episode_rewards) < window:
            return np.array(self.episode_rewards), np.array(self.episode_lengths)

        rewards = np.convolve(
            self.episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )
        lengths = np.convolve(
            self.episode_lengths,
            np.ones(window) / window,
            mode='valid'
        )
        return rewards, lengths


# =============================================================================
# 基类
# =============================================================================

class BaseTDLearner(ABC, Generic[State, Action]):
    """
    时序差分学习算法基类。

    核心思想 (Core Idea):
    --------------------
    提供TD学习算法的通用框架，包括价值函数管理、策略实现和训练循环。
    子类只需实现特定的更新规则即可。

    设计模式:
    --------
    采用模板方法模式(Template Method Pattern)，将算法骨架定义在基类中，
    具体的更新步骤延迟到子类实现。
    """

    def __init__(self, config: TDConfig) -> None:
        """
        初始化TD学习器。

        Args:
            config: TD学习配置对象
        """
        self.config = config
        self._value_function: Dict[State, float] = defaultdict(
            lambda: config.initial_value
        )
        self._q_function: Dict[Tuple[State, Action], float] = defaultdict(
            lambda: config.initial_value
        )
        self.metrics = TrainingMetrics()
        self._action_space: Optional[List[Action]] = None

    @property
    def value_function(self) -> Dict[State, float]:
        """获取状态价值函数V(s)。"""
        return dict(self._value_function)

    @property
    def q_function(self) -> Dict[Tuple[State, Action], float]:
        """获取动作价值函数Q(s, a)。"""
        return dict(self._q_function)

    def get_value(self, state: State) -> float:
        """获取状态价值V(s)。"""
        return self._value_function[state]

    def get_q_value(self, state: State, action: Action) -> float:
        """获取动作价值Q(s, a)。"""
        return self._q_function[(state, action)]

    def set_action_space(self, actions: List[Action]) -> None:
        """设置动作空间。"""
        self._action_space = actions

    def epsilon_greedy_action(self, state: State) -> Action:
        """
        ε-greedy策略选择动作。

        数学原理:
            π(a|s) = ε/|A| + (1-ε)·𝟙(a = argmax Q(s,a'))

        以概率ε随机选择动作(探索)，以概率1-ε选择当前最优动作(利用)。

        Args:
            state: 当前状态

        Returns:
            选择的动作
        """
        if self._action_space is None:
            raise ValueError("未设置动作空间，请先调用set_action_space()")

        if np.random.random() < self.config.epsilon:
            return np.random.choice(self._action_space)

        q_values = [self.get_q_value(state, a) for a in self._action_space]
        max_q = max(q_values)
        best_actions = [
            a for a, q in zip(self._action_space, q_values)
            if np.isclose(q, max_q)
        ]
        return np.random.choice(best_actions)

    def greedy_action(self, state: State) -> Action:
        """
        贪婪策略选择动作（用于评估）。

        Args:
            state: 当前状态

        Returns:
            最优动作
        """
        if self._action_space is None:
            raise ValueError("未设置动作空间，请先调用set_action_space()")

        q_values = [self.get_q_value(state, a) for a in self._action_space]
        max_q = max(q_values)
        best_actions = [
            a for a, q in zip(self._action_space, q_values)
            if np.isclose(q, max_q)
        ]
        return np.random.choice(best_actions)

    @abstractmethod
    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行TD更新步骤。

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            next_action: 下一动作（SARSA需要）
            done: 是否终止

        Returns:
            TD误差δ
        """
        pass

    def train_episode(
        self,
        env: Environment[State, Action],
        max_steps: int = 10000
    ) -> Tuple[float, int]:
        """
        训练一个回合。

        Args:
            env: 环境实例
            max_steps: 最大步数限制

        Returns:
            (回合总奖励, 回合步数)
        """
        state, _ = env.reset()
        action = self.epsilon_greedy_action(state)

        total_reward = 0.0
        td_errors = []

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = None if done else self.epsilon_greedy_action(next_state)

            td_error = self.update(state, action, reward, next_state, next_action, done)
            td_errors.append(abs(td_error))

            total_reward += reward

            if done:
                break

            state = next_state
            action = next_action

        steps = step + 1
        avg_td_error = np.mean(td_errors) if td_errors else 0.0
        self.metrics.add_episode(total_reward, steps, avg_td_error)

        return total_reward, steps

    def train(
        self,
        env: Environment[State, Action],
        n_episodes: int = 1000,
        max_steps_per_episode: int = 10000,
        log_interval: int = 100,
        early_stop_reward: Optional[float] = None
    ) -> TrainingMetrics:
        """
        执行完整训练过程。

        Args:
            env: 环境实例
            n_episodes: 训练回合数
            max_steps_per_episode: 每回合最大步数
            log_interval: 日志输出间隔
            early_stop_reward: 早停奖励阈值

        Returns:
            训练指标
        """
        # 自动设置动作空间
        if self._action_space is None:
            if hasattr(env.action_space, 'n'):
                self.set_action_space(list(range(env.action_space.n)))
            else:
                raise ValueError("无法自动推断动作空间，请手动设置")

        for episode in range(n_episodes):
            reward, steps = self.train_episode(env, max_steps_per_episode)

            if (episode + 1) % log_interval == 0:
                recent_rewards = self.metrics.episode_rewards[-log_interval:]
                avg_reward = np.mean(recent_rewards)
                logger.info(
                    f"Episode {episode + 1}/{n_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Last Reward: {reward:.2f} | "
                    f"Steps: {steps}"
                )

            # 早停检查
            if early_stop_reward is not None:
                if len(self.metrics.episode_rewards) >= 100:
                    recent_avg = np.mean(self.metrics.episode_rewards[-100:])
                    if recent_avg >= early_stop_reward:
                        logger.info(
                            f"达到早停条件: 平均奖励 {recent_avg:.2f} >= {early_stop_reward}"
                        )
                        break

        return self.metrics

    def evaluate(
        self,
        env: Environment[State, Action],
        n_episodes: int = 100,
        max_steps: int = 10000
    ) -> Tuple[float, float]:
        """
        评估当前策略性能。

        Args:
            env: 环境实例
            n_episodes: 评估回合数
            max_steps: 每回合最大步数

        Returns:
            (平均奖励, 奖励标准差)
        """
        rewards = []

        for _ in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0.0

            for _ in range(max_steps):
                action = self.greedy_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

                if terminated or truncated:
                    break

            rewards.append(total_reward)

        return np.mean(rewards), np.std(rewards)


# =============================================================================
# TD(0) 状态价值学习
# =============================================================================

class TD0ValueLearner(BaseTDLearner[State, Action]):
    """
    TD(0)状态价值学习算法。

    核心思想 (Core Idea):
    --------------------
    TD(0)是最简单的TD方法，使用单步自举来更新价值估计。
    它只看下一步的奖励和下一状态的价值估计，不等待完整回合。

    数学原理 (Mathematical Theory):
    ------------------------------
    更新规则:
        V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]

    TD误差:
        δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

    收敛性:
        在满足Robbins-Monro条件(Σα=∞, Σα²<∞)且策略固定时，
        TD(0)以概率1收敛到真实价值函数。

    问题背景 (Problem Statement):
    ----------------------------
    给定一个固定策略π，估计该策略下的状态价值函数V^π(s)。
    这是策略评估(Policy Evaluation)问题，是策略迭代算法的基础。

    算法对比 (Comparison):
    ---------------------
    与Monte Carlo相比:
    - 优势: 无需等待回合结束，方差低，数据效率高
    - 劣势: 引入偏差(因为V(S_{t+1})本身是估计值)

    复杂度 (Complexity):
    -------------------
    - 时间: O(1) per step
    - 空间: O(|S|) for value function
    """

    def __init__(self, config: TDConfig, policy: Optional[Policy[State, Action]] = None):
        """
        初始化TD(0)价值学习器。

        Args:
            config: TD学习配置
            policy: 待评估的策略，None则使用ε-greedy
        """
        super().__init__(config)
        self._policy = policy

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行TD(0)更新。

        δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
        V(S_t) ← V(S_t) + αδ_t

        Args:
            state: 当前状态
            action: 执行的动作（本算法不使用）
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（本算法不使用）
            done: 是否终止

        Returns:
            TD误差δ_t
        """
        # 计算TD目标
        if done:
            td_target = reward
        else:
            td_target = reward + self.config.gamma * self._value_function[next_state]

        # 计算TD误差
        td_error = td_target - self._value_function[state]

        # 更新价值估计
        self._value_function[state] += self.config.alpha * td_error

        return td_error


# =============================================================================
# SARSA (State-Action-Reward-State-Action)
# =============================================================================

class SARSA(BaseTDLearner[State, Action]):
    """
    SARSA算法实现。

    核心思想 (Core Idea):
    --------------------
    SARSA是一种on-policy TD控制算法。其名称来源于更新所需的五元组:
    (State, Action, Reward, State', Action')。关键特点是使用实际执行的
    下一动作A'来计算TD目标，因此学习的是行为策略本身的价值。

    数学原理 (Mathematical Theory):
    ------------------------------
    更新规则:
        Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]

    TD目标:
        G_t^{(1)} = R_{t+1} + γQ(S_{t+1}, A_{t+1})

    收敛性:
        在满足GLIE(Greedy in the Limit with Infinite Exploration)条件时，
        SARSA收敛到最优策略。

    问题背景 (Problem Statement):
    ----------------------------
    SARSA解决的是控制问题(Control Problem)：找到最优策略π*。
    与Q-Learning的关键区别在于SARSA是on-policy的——它评估和改进的是
    实际执行的策略，包括探索行为。

    算法对比 (Comparison):
    ---------------------
    SARSA vs Q-Learning:
    ┌────────────────┬─────────────────┬─────────────────┐
    │    特性        │     SARSA       │    Q-Learning   │
    ├────────────────┼─────────────────┼─────────────────┤
    │    类型        │    on-policy    │    off-policy   │
    │  下一动作      │  实际采样A'     │    max_a Q      │
    │    安全性      │      高         │       低        │
    │  收敛速度      │      慢         │       快        │
    │  最终策略      │    保守         │      激进       │
    └────────────────┴─────────────────┴─────────────────┘

    在cliff walking等危险环境中，SARSA会学到更安全的路径，
    因为它考虑了探索时可能掉落的风险。

    复杂度 (Complexity):
    -------------------
    - 时间: O(1) per step
    - 空间: O(|S| × |A|) for Q-table

    算法总结 (Summary):
    -----------------
    SARSA通过五元组(S,A,R,S',A')进行学习。它忠实地评估当前策略
    （包括探索行为）的价值，因此在需要考虑探索风险的环境中
    往往能学到更安全、更保守的策略。
    """

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行SARSA更新。

        Q(S_t, A_t) ← Q(S_t, A_t) + α[R + γQ(S', A') - Q(S_t, A_t)]

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（SARSA必需）
            done: 是否终止

        Returns:
            TD误差
        """
        current_q = self._q_function[(state, action)]

        if done:
            td_target = reward
        else:
            if next_action is None:
                raise ValueError("SARSA需要next_action参数")
            td_target = reward + self.config.gamma * self._q_function[(next_state, next_action)]

        td_error = td_target - current_q
        self._q_function[(state, action)] += self.config.alpha * td_error

        return td_error


# =============================================================================
# Expected SARSA
# =============================================================================

class ExpectedSARSA(BaseTDLearner[State, Action]):
    """
    Expected SARSA算法实现。

    核心思想 (Core Idea):
    --------------------
    Expected SARSA是SARSA的变体，使用下一状态所有动作Q值的期望
    （按策略概率加权）作为TD目标，而不是单一采样动作的Q值。
    这消除了动作采样带来的方差，使学习更加稳定。

    数学原理 (Mathematical Theory):
    ------------------------------
    更新规则:
        Q(S_t, A_t) ← Q(S_t, A_t) + α[R + γ𝔼_π[Q(S', A')] - Q(S_t, A_t)]

    期望计算:
        𝔼_π[Q(S', A')] = Σ_a π(a|S') × Q(S', a)

    对于ε-greedy策略:
        𝔼_π[Q(S', A')] = ε/|A| × Σ_a Q(S', a) + (1-ε) × max_a Q(S', a)

    问题背景 (Problem Statement):
    ----------------------------
    SARSA的更新依赖于采样的下一动作，引入了额外方差。
    Expected SARSA通过计算期望消除这一方差源，获得更稳定的学习。

    算法对比 (Comparison):
    ---------------------
    Expected SARSA位于SARSA和Q-Learning之间:
    - 当ε=0时，退化为Q-Learning（确定性greedy策略）
    - 当仅考虑单一动作时，退化为SARSA
    - 结合了SARSA的on-policy特性和更低的方差

    ┌─────────────────┬────────────┬────────────┬────────────────┐
    │     算法        │    方差    │   偏差     │    计算成本    │
    ├─────────────────┼────────────┼────────────┼────────────────┤
    │    SARSA        │    高      │    低      │      O(1)      │
    │  Expected SARSA │    低      │    低      │     O(|A|)     │
    │   Q-Learning    │    中      │    有      │      O(|A|)    │
    └─────────────────┴────────────┴────────────┴────────────────┘

    复杂度 (Complexity):
    -------------------
    - 时间: O(|A|) per step (需要遍历所有动作计算期望)
    - 空间: O(|S| × |A|) for Q-table

    算法总结 (Summary):
    -----------------
    Expected SARSA通过计算策略在下一状态的期望价值，消除了SARSA中
    动作采样的方差。它在保持on-policy特性的同时获得更稳定的更新，
    是SARSA和Q-Learning之间的优雅折中。
    """

    def _compute_expected_q(self, state: State) -> float:
        """
        计算状态下Q值的期望。

        对于ε-greedy策略:
        𝔼[Q(s,·)] = ε/|A| × Σ_a Q(s,a) + (1-ε) × max_a Q(s,a)

        Args:
            state: 状态

        Returns:
            期望Q值
        """
        if self._action_space is None:
            raise ValueError("未设置动作空间")

        q_values = [self._q_function[(state, a)] for a in self._action_space]
        n_actions = len(self._action_space)

        # ε-greedy策略的期望计算
        # 探索部分: 每个动作概率 ε/|A|
        exploration_value = (self.config.epsilon / n_actions) * sum(q_values)

        # 利用部分: 最优动作概率 (1-ε)
        exploitation_value = (1 - self.config.epsilon) * max(q_values)

        return exploration_value + exploitation_value

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行Expected SARSA更新。

        Q(S, A) ← Q(S, A) + α[R + γ𝔼[Q(S', ·)] - Q(S, A)]

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（本算法不使用）
            done: 是否终止

        Returns:
            TD误差
        """
        current_q = self._q_function[(state, action)]

        if done:
            td_target = reward
        else:
            expected_q = self._compute_expected_q(next_state)
            td_target = reward + self.config.gamma * expected_q

        td_error = td_target - current_q
        self._q_function[(state, action)] += self.config.alpha * td_error

        return td_error


# =============================================================================
# Q-Learning
# =============================================================================

class QLearning(BaseTDLearner[State, Action]):
    """
    Q-Learning算法实现。

    核心思想 (Core Idea):
    --------------------
    Q-Learning是最著名的off-policy TD控制算法。无论行为策略如何，
    它总是学习最优策略的Q值——使用max操作选择下一状态的最优动作，
    而不是实际执行的动作。这种"乐观主义"使其能够直接学习最优策略。

    数学原理 (Mathematical Theory):
    ------------------------------
    更新规则:
        Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]

    TD目标 (最优Bellman方程的采样版本):
        G_t = R_{t+1} + γ max_a Q(S_{t+1}, a)

    这直接对应最优Bellman方程:
        Q*(s, a) = 𝔼[R + γ max_{a'} Q*(s', a') | s, a]

    收敛性定理 (Watkins, 1989):
        在以下条件下Q-Learning以概率1收敛到Q*:
        1. 所有状态-动作对被无限次访问
        2. 学习率满足: Σα_t = ∞ 且 Σα_t² < ∞

    问题背景 (Problem Statement):
    ----------------------------
    Q-Learning解决最优控制问题，直接学习最优动作价值函数Q*。
    其off-policy特性允许使用任意探索策略收集数据，同时学习最优策略。

    算法对比 (Comparison):
    ---------------------
    Q-Learning vs SARSA:
    - Q-Learning: off-policy, 更激进, 可能不安全的探索
    - SARSA: on-policy, 更保守, 考虑探索风险

    最大化偏差 (Maximization Bias):
        Q-Learning的max操作会导致系统性的过估计。
        在噪声环境中，max会选中估计值偏高的动作，导致过度乐观。
        Double Q-Learning通过解耦选择和评估来解决这一问题。

    复杂度 (Complexity):
    -------------------
    - 时间: O(|A|) per step (需要找max)
    - 空间: O(|S| × |A|) for Q-table

    算法总结 (Summary):
    -----------------
    Q-Learning通过"假装"行为策略是贪婪的来直接学习最优Q函数。
    这种off-policy特性使其可以从任何数据源学习，但也可能导致
    在危险环境中学到不安全的策略，以及在噪声环境中过估计Q值。
    """

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行Q-Learning更新。

        Q(S, A) ← Q(S, A) + α[R + γ max_a Q(S', a) - Q(S, A)]

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（Q-Learning不使用）
            done: 是否终止

        Returns:
            TD误差
        """
        current_q = self._q_function[(state, action)]

        if done:
            td_target = reward
        else:
            # 关键区别: 使用max而不是实际下一动作
            max_next_q = max(
                self._q_function[(next_state, a)]
                for a in self._action_space
            )
            td_target = reward + self.config.gamma * max_next_q

        td_error = td_target - current_q
        self._q_function[(state, action)] += self.config.alpha * td_error

        return td_error


# =============================================================================
# Double Q-Learning
# =============================================================================

class DoubleQLearning(BaseTDLearner[State, Action]):
    """
    Double Q-Learning算法实现。

    核心思想 (Core Idea):
    --------------------
    Double Q-Learning通过维护两个独立的Q表来解决Q-Learning的最大化偏差。
    一个Q表用于选择最优动作，另一个用于评估该动作的价值。
    这种"解耦"策略有效消除了过估计问题。

    数学原理 (Mathematical Theory):
    ------------------------------
    标准Q-Learning的问题:
        max_a Q(s', a) 使用同一个Q来选择和评估，当Q有噪声时导致过估计。

    Double Q-Learning解决方案:
        以0.5概率更新Q_A或Q_B:

        更新Q_A:
            a* = argmax_a Q_A(S', a)           (用Q_A选择)
            Q_A(S, A) ← Q_A(S, A) + α[R + γQ_B(S', a*) - Q_A(S, A)]  (用Q_B评估)

        更新Q_B:
            a* = argmax_a Q_B(S', a)           (用Q_B选择)
            Q_B(S, A) ← Q_B(S, A) + α[R + γQ_A(S', a*) - Q_B(S, A)]  (用Q_A评估)

    为什么有效:
        关键洞察: 𝔼[max(X, Y)] ≥ max(𝔼[X], 𝔼[Y])
        当估计有噪声时，max总是偏向高估。
        通过用独立的估计器评估选中的动作，期望值变得无偏。

    问题背景 (Problem Statement):
    ----------------------------
    在随机环境中，Q-Learning会系统性地过估计Q值，导致次优策略。
    经典例子: 在有随机奖励的MDP中，Q-Learning可能学到错误策略。

    算法对比 (Comparison):
    ---------------------
    ┌──────────────────┬────────────┬────────────┬────────────┐
    │      算法        │   偏差     │   方差     │   内存     │
    ├──────────────────┼────────────┼────────────┼────────────┤
    │   Q-Learning     │   过估计   │    中      │    1×      │
    │ Double Q-Learning│   无偏     │    中      │    2×      │
    └──────────────────┴────────────┴────────────┴────────────┘

    复杂度 (Complexity):
    -------------------
    - 时间: O(|A|) per step
    - 空间: O(2 × |S| × |A|) for two Q-tables

    算法总结 (Summary):
    -----------------
    Double Q-Learning是Q-Learning的去偏差版本。通过维护两个Q表
    并随机选择哪个用于选择、哪个用于评估，它消除了max操作引入的
    系统性过估计。代价是双倍的内存消耗。
    """

    def __init__(self, config: TDConfig) -> None:
        """初始化Double Q-Learning。"""
        super().__init__(config)
        # 维护两个独立的Q表
        self._q_function_a: Dict[Tuple[State, Action], float] = defaultdict(
            lambda: config.initial_value
        )
        self._q_function_b: Dict[Tuple[State, Action], float] = defaultdict(
            lambda: config.initial_value
        )

    def get_q_value(self, state: State, action: Action) -> float:
        """获取合并后的Q值（两个Q表的平均）。"""
        q_a = self._q_function_a[(state, action)]
        q_b = self._q_function_b[(state, action)]
        return (q_a + q_b) / 2

    @property
    def q_function(self) -> Dict[Tuple[State, Action], float]:
        """获取合并后的Q函数。"""
        all_keys = set(self._q_function_a.keys()) | set(self._q_function_b.keys())
        return {
            key: (self._q_function_a[key] + self._q_function_b[key]) / 2
            for key in all_keys
        }

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行Double Q-Learning更新。

        以0.5概率选择更新Q_A或Q_B，交叉使用另一个Q表进行评估。

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（不使用）
            done: 是否终止

        Returns:
            TD误差
        """
        # 随机选择更新哪个Q表
        if np.random.random() < 0.5:
            # 更新Q_A，用Q_B评估
            q_select = self._q_function_a
            q_eval = self._q_function_b
            q_update = self._q_function_a
        else:
            # 更新Q_B，用Q_A评估
            q_select = self._q_function_b
            q_eval = self._q_function_a
            q_update = self._q_function_b

        current_q = q_update[(state, action)]

        if done:
            td_target = reward
        else:
            # 用一个Q表选择最优动作
            best_action = max(
                self._action_space,
                key=lambda a: q_select[(next_state, a)]
            )
            # 用另一个Q表评估
            td_target = reward + self.config.gamma * q_eval[(next_state, best_action)]

        td_error = td_target - current_q
        q_update[(state, action)] += self.config.alpha * td_error

        return td_error


# =============================================================================
# N-Step TD
# =============================================================================

class NStepTD(BaseTDLearner[State, Action]):
    """
    N-Step TD算法实现。

    核心思想 (Core Idea):
    --------------------
    N-Step TD是TD(0)和Monte Carlo的中间方案。它使用n步的实际奖励
    加上第n+1步的价值估计作为TD目标。n越大，越接近Monte Carlo；
    n=1时就是TD(0)。

    数学原理 (Mathematical Theory):
    ------------------------------
    n-step回报:
        G_t^{(n)} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
                  = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})

    更新规则:
        V(S_t) ← V(S_t) + α[G_t^{(n)} - V(S_t)]

    对于Q函数:
        G_t^{(n)} = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n Q(S_{t+n}, A_{t+n})

    关键洞察:
        当n→∞，G_t^{(n)}变成Monte Carlo回报
        当n=1，G_t^{(n)}就是TD(0)目标

    问题背景 (Problem Statement):
    ----------------------------
    TD(0)偏差高、方差低；Monte Carlo偏差低、方差高。
    N-Step TD提供了一种在两者之间平滑过渡的方式，
    允许根据问题特性选择合适的n值。

    算法对比 (Comparison):
    ---------------------
    ┌───────────┬────────────┬────────────┬────────────┐
    │     n     │    偏差    │    方差    │   延迟     │
    ├───────────┼────────────┼────────────┼────────────┤
    │    1      │     高     │     低     │   1 step   │
    │    5      │     中     │     中     │   5 steps  │
    │   100     │     低     │     高     │  100 steps │
    │    ∞      │     无     │     高     │  episode   │
    └───────────┴────────────┴────────────┴────────────┘

    实践中最优n通常在4-10之间。

    复杂度 (Complexity):
    -------------------
    - 时间: O(1) per step (摊销)
    - 空间: O(n) for storing n-step buffer

    算法总结 (Summary):
    -----------------
    N-Step TD通过调整n值在偏差和方差之间权衡。较小的n更新更频繁但偏差大，
    较大的n能利用更多真实奖励信息但需要等待更长时间。
    它是理解TD(λ)的基础。
    """

    def __init__(self, config: TDConfig) -> None:
        """初始化N-Step TD。"""
        super().__init__(config)
        # 存储n步经验的缓冲区
        self._buffer: List[Tuple[State, Action, float]] = []
        self._states_buffer: List[State] = []

    def _compute_n_step_return(
        self,
        rewards: List[float],
        final_state: State,
        done: bool
    ) -> float:
        """
        计算n-step回报。

        G_t^{(n)} = Σγ^k R_{t+k+1} + γ^n V(S_{t+n})

        Args:
            rewards: n步奖励列表
            final_state: 最终状态
            done: 是否终止

        Returns:
            n-step回报
        """
        n_step_return = 0.0
        discount = 1.0

        for reward in rewards:
            n_step_return += discount * reward
            discount *= self.config.gamma

        if not done:
            n_step_return += discount * self._value_function[final_state]

        return n_step_return

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行N-Step TD更新。

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作
            done: 是否终止

        Returns:
            TD误差（如果触发更新）
        """
        # 添加到缓冲区
        self._buffer.append((state, action, reward))
        self._states_buffer.append(next_state)

        td_error = 0.0

        # 当缓冲区满或回合结束时，进行更新
        if len(self._buffer) >= self.config.n_step or done:
            # 提取要更新的状态和奖励
            update_state = self._buffer[0][0]
            update_action = self._buffer[0][1]
            rewards = [exp[2] for exp in self._buffer]

            # 计算n-step回报
            n_step_return = self._compute_n_step_return(rewards, next_state, done)

            # 更新Q值
            current_q = self._q_function[(update_state, update_action)]
            td_error = n_step_return - current_q
            self._q_function[(update_state, update_action)] += self.config.alpha * td_error

            # 移除最旧的经验
            self._buffer.pop(0)
            self._states_buffer.pop(0)

        # 回合结束时清空缓冲区并更新剩余状态
        if done:
            while self._buffer:
                update_state = self._buffer[0][0]
                update_action = self._buffer[0][1]
                rewards = [exp[2] for exp in self._buffer]

                n_step_return = self._compute_n_step_return(rewards, next_state, True)

                current_q = self._q_function[(update_state, update_action)]
                td_error = n_step_return - current_q
                self._q_function[(update_state, update_action)] += self.config.alpha * td_error

                self._buffer.pop(0)
                if self._states_buffer:
                    self._states_buffer.pop(0)

            # 清空缓冲区
            self._buffer = []
            self._states_buffer = []

        return td_error


# =============================================================================
# TD(λ) with Eligibility Traces
# =============================================================================

class TDLambda(BaseTDLearner[State, Action]):
    """
    TD(λ)算法实现 (带资格迹)。

    核心思想 (Core Idea):
    --------------------
    TD(λ)通过资格迹(Eligibility Traces)统一了TD(0)和Monte Carlo。
    资格迹追踪哪些状态"有资格"接收当前TD误差的更新——最近访问的状态
    资格最高，随时间指数衰减。这等价于在所有n-step回报上做几何加权平均。

    数学原理 (Mathematical Theory):
    ------------------------------
    λ-回报 (Forward View):
        G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t^{(n)}

    这是所有n-step回报的几何加权平均，权重(1-λ)λ^{n-1}。

    资格迹 (Backward View):
        累积迹: E_t(s) = γλE_{t-1}(s) + 𝟙(S_t = s)
        替换迹: E_t(s) = γλE_{t-1}(s); E_t(S_t) = 1
        荷兰迹: E_t(s) = γλE_{t-1}(s) + (1-αγλE_{t-1}(s))𝟙(S_t = s)

    更新规则:
        δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
        V(s) ← V(s) + αδ_t E_t(s), ∀s

    前向视图与后向视图等价性:
        在离线更新下，TD(λ)的后向视图产生的总更新量
        等于使用λ-回报的前向视图。

    问题背景 (Problem Statement):
    ----------------------------
    N-Step TD需要选择特定的n值，不同的n在不同环境中表现差异大。
    TD(λ)通过资格迹实现对所有n的加权组合，由单一参数λ控制。
    λ=0等价于TD(0)，λ=1等价于Monte Carlo。

    算法对比 (Comparison):
    ---------------------
    不同λ值的特性:
    ┌───────────┬────────────┬────────────┬────────────────┐
    │     λ     │   等价于   │    偏差    │      方差      │
    ├───────────┼────────────┼────────────┼────────────────┤
    │    0      │   TD(0)    │     高     │       低       │
    │   0.5     │   混合     │     中     │       中       │
    │   0.9     │ 接近MC     │     低     │       较高     │
    │    1      │   MC       │     无     │       高       │
    └───────────┴────────────┴────────────┴────────────────┘

    实践中λ=0.9是常用起点。

    资格迹类型对比:
    - 累积迹: 经典方法，但可能发散
    - 替换迹: 在部分状态重访问多的任务中更稳定
    - 荷兰迹: 结合两者优点，理论保证更好

    复杂度 (Complexity):
    -------------------
    - 时间: O(|S|) per step (需要更新所有有资格的状态)
    - 空间: O(|S|) for eligibility traces

    算法总结 (Summary):
    -----------------
    TD(λ)是TD学习的统一框架。通过资格迹，它在每一步将TD误差
    分配给所有最近访问的状态，分配量随时间和λ指数衰减。
    这巧妙地组合了所有n-step方法的优点，用单一参数λ控制权衡。
    """

    def __init__(self, config: TDConfig) -> None:
        """初始化TD(λ)。"""
        super().__init__(config)
        # 资格迹
        self._eligibility_traces: Dict[Tuple[State, Action], float] = defaultdict(float)

    def _update_traces(
        self,
        state: State,
        action: Action
    ) -> None:
        """
        更新资格迹。

        Args:
            state: 当前状态
            action: 当前动作
        """
        gamma_lambda = self.config.gamma * self.config.lambda_

        # 衰减所有现有的资格迹
        for key in list(self._eligibility_traces.keys()):
            self._eligibility_traces[key] *= gamma_lambda
            # 清除过小的迹以节省内存
            if self._eligibility_traces[key] < 1e-8:
                del self._eligibility_traces[key]

        # 更新当前状态-动作的资格迹
        if self.config.trace_type == EligibilityTraceType.ACCUMULATING:
            self._eligibility_traces[(state, action)] += 1.0
        elif self.config.trace_type == EligibilityTraceType.REPLACING:
            self._eligibility_traces[(state, action)] = 1.0
        elif self.config.trace_type == EligibilityTraceType.DUTCH:
            current_trace = self._eligibility_traces[(state, action)]
            self._eligibility_traces[(state, action)] = (
                (1 - self.config.alpha) * gamma_lambda * current_trace + 1.0
            )

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行TD(λ)更新。

        1. 计算TD误差δ
        2. 更新资格迹
        3. 用δ和资格迹更新所有状态-动作对的Q值

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作
            done: 是否终止

        Returns:
            TD误差
        """
        # 计算TD误差
        if done:
            td_target = reward
        else:
            if next_action is None:
                # Q-Learning风格
                max_next_q = max(
                    self._q_function[(next_state, a)]
                    for a in self._action_space
                )
                td_target = reward + self.config.gamma * max_next_q
            else:
                # SARSA风格
                td_target = reward + self.config.gamma * self._q_function[(next_state, next_action)]

        current_q = self._q_function[(state, action)]
        td_error = td_target - current_q

        # 更新资格迹
        self._update_traces(state, action)

        # 使用资格迹更新所有相关的Q值
        for (s, a), trace in self._eligibility_traces.items():
            self._q_function[(s, a)] += self.config.alpha * td_error * trace

        # 回合结束时清空资格迹
        if done:
            self._eligibility_traces.clear()

        return td_error


# =============================================================================
# SARSA(λ)
# =============================================================================

class SARSALambda(TDLambda):
    """
    SARSA(λ)算法实现。

    核心思想 (Core Idea):
    --------------------
    SARSA(λ)是SARSA与资格迹的结合。它是on-policy的TD(λ)控制算法，
    使用实际下一动作计算TD目标，同时通过资格迹实现多步信用分配。

    数学原理 (Mathematical Theory):
    ------------------------------
    TD误差:
        δ_t = R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)

    资格迹更新:
        E_t(s, a) = γλE_{t-1}(s, a) + 𝟙(S_t=s, A_t=a)

    Q值更新:
        Q(s, a) ← Q(s, a) + αδ_t E_t(s, a), ∀s, a

    与SARSA的关系:
        当λ=0时，退化为SARSA
        当λ=1时，变成完整回合的on-policy更新

    问题背景 (Problem Statement):
    ----------------------------
    SARSA的更新仅依赖单步信息，信用分配范围有限。
    SARSA(λ)通过资格迹将TD误差传播到所有最近访问的状态-动作对，
    实现更高效的学习。

    复杂度 (Complexity):
    -------------------
    - 时间: O(|S|×|A|) per step (最坏情况，实际取决于活跃迹数量)
    - 空间: O(|S|×|A|) for eligibility traces

    算法总结 (Summary):
    -----------------
    SARSA(λ)结合了SARSA的on-policy特性和资格迹的高效信用分配。
    它保持了SARSA的安全性（考虑探索风险），同时通过多步传播加速学习。
    """

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行SARSA(λ)更新。

        使用实际下一动作计算TD目标，配合资格迹更新。

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（SARSA必需）
            done: 是否终止

        Returns:
            TD误差
        """
        # 计算SARSA风格的TD误差
        if done:
            td_target = reward
        else:
            if next_action is None:
                raise ValueError("SARSA(λ)需要next_action参数")
            td_target = reward + self.config.gamma * self._q_function[(next_state, next_action)]

        current_q = self._q_function[(state, action)]
        td_error = td_target - current_q

        # 更新资格迹
        self._update_traces(state, action)

        # 使用资格迹更新所有Q值
        for (s, a), trace in self._eligibility_traces.items():
            self._q_function[(s, a)] += self.config.alpha * td_error * trace

        # 回合结束时清空资格迹
        if done:
            self._eligibility_traces.clear()

        return td_error


# =============================================================================
# Watkins's Q(λ)
# =============================================================================

class WatkinsQLambda(TDLambda):
    """
    Watkins's Q(λ)算法实现。

    核心思想 (Core Idea):
    --------------------
    Watkins's Q(λ)是Q-Learning与资格迹的结合，但有一个关键特点：
    当采取非贪婪动作（探索）时，资格迹被清零。这确保了算法在
    off-policy设置下的收敛性。

    数学原理 (Mathematical Theory):
    ------------------------------
    TD误差 (Q-Learning风格):
        δ_t = R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)

    资格迹更新 (关键区别):
        如果 A_{t+1} = argmax_a Q(S_{t+1}, a):
            E_t(s, a) = γλE_{t-1}(s, a) + 𝟙(S_t=s, A_t=a)
        否则 (探索动作):
            E_t(s, a) = 𝟙(S_t=s, A_t=a)  // 清除历史迹

    为什么清零资格迹:
        Q-Learning假设后续动作都是贪婪的。当实际采取探索动作时，
        这个假设被打破，继续传播TD误差到更早的状态-动作对会引入偏差。
        清零资格迹切断这种错误的信用分配链。

    问题背景 (Problem Statement):
    ----------------------------
    简单地将资格迹加入Q-Learning会导致在off-policy设置下不收敛。
    Watkins's Q(λ)通过在探索时切断资格迹来解决这一问题。
    缺点是在高探索率下，资格迹经常被清零，退化为近似Q-Learning。

    算法对比 (Comparison):
    ---------------------
    ┌────────────────┬─────────────────┬─────────────────┐
    │     算法       │   探索时的迹    │    收敛性       │
    ├────────────────┼─────────────────┼─────────────────┤
    │   Q(λ) naive   │     保留        │    不保证       │
    │  Watkins Q(λ)  │     清零        │    保证         │
    │    Peng's Q(λ) │   部分保留      │    弱保证       │
    └────────────────┴─────────────────┴─────────────────┘

    复杂度 (Complexity):
    -------------------
    - 时间: O(|S|×|A|) per step (最坏)
    - 空间: O(|S|×|A|) for eligibility traces

    算法总结 (Summary):
    -----------------
    Watkins's Q(λ)在off-policy学习中安全地使用资格迹。
    代价是当探索动作发生时，无法利用之前的经验进行信用分配。
    在低ε设置下效果较好，高ε时退化为Q-Learning。
    """

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_action: Optional[Action],
        done: bool
    ) -> float:
        """
        执行Watkins's Q(λ)更新。

        使用Q-Learning目标，但在探索动作时清零资格迹。

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（用于检测是否探索）
            done: 是否终止

        Returns:
            TD误差
        """
        # 计算Q-Learning风格的TD误差
        if done:
            td_target = reward
        else:
            max_next_q = max(
                self._q_function[(next_state, a)]
                for a in self._action_space
            )
            td_target = reward + self.config.gamma * max_next_q

        current_q = self._q_function[(state, action)]
        td_error = td_target - current_q

        # 更新资格迹
        self._update_traces(state, action)

        # 使用资格迹更新所有Q值
        for (s, a), trace in self._eligibility_traces.items():
            self._q_function[(s, a)] += self.config.alpha * td_error * trace

        # 检查是否采取了探索动作，如果是则清零资格迹
        if not done and next_action is not None:
            # 找到贪婪动作
            max_next_q = max(
                self._q_function[(next_state, a)]
                for a in self._action_space
            )
            greedy_actions = [
                a for a in self._action_space
                if np.isclose(self._q_function[(next_state, a)], max_next_q)
            ]

            # 如果下一动作不是贪婪动作，清零资格迹
            if next_action not in greedy_actions:
                self._eligibility_traces.clear()

        # 回合结束时清空资格迹
        if done:
            self._eligibility_traces.clear()

        return td_error


# =============================================================================
# 工厂函数
# =============================================================================

def create_td_learner(
    algorithm: str,
    config: Optional[TDConfig] = None,
    **kwargs
) -> BaseTDLearner:
    """
    TD学习算法工厂函数。

    Args:
        algorithm: 算法名称，可选:
            - 'td0': TD(0)状态价值学习
            - 'sarsa': SARSA
            - 'expected_sarsa': Expected SARSA
            - 'q_learning': Q-Learning
            - 'double_q': Double Q-Learning
            - 'n_step': N-Step TD
            - 'td_lambda': TD(λ)
            - 'sarsa_lambda': SARSA(λ)
            - 'watkins_q_lambda': Watkins's Q(λ)
        config: TD学习配置，None则使用默认配置
        **kwargs: 传递给TDConfig的额外参数

    Returns:
        对应的TD学习器实例

    Example:
        >>> learner = create_td_learner('sarsa', alpha=0.1, gamma=0.99)
        >>> learner = create_td_learner('td_lambda', config=TDConfig(lambda_=0.9))
    """
    if config is None:
        config = TDConfig(**kwargs)

    algorithm_map = {
        'td0': TD0ValueLearner,
        'sarsa': SARSA,
        'expected_sarsa': ExpectedSARSA,
        'q_learning': QLearning,
        'double_q': DoubleQLearning,
        'n_step': NStepTD,
        'td_lambda': TDLambda,
        'sarsa_lambda': SARSALambda,
        'watkins_q_lambda': WatkinsQLambda,
    }

    algorithm = algorithm.lower()
    if algorithm not in algorithm_map:
        raise ValueError(
            f"未知算法: {algorithm}. 支持的算法: {list(algorithm_map.keys())}"
        )

    return algorithm_map[algorithm](config)


# =============================================================================
# 单元测试
# =============================================================================

if __name__ == '__main__':
    import gymnasium as gym

    print("=" * 70)
    print("时序差分学习算法测试")
    print("=" * 70)

    # 测试配置
    config = TDConfig(
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1,
        lambda_=0.9,
        n_step=3
    )

    # 创建环境
    env = gym.make('CliffWalking-v0')

    # 测试各算法（使用极小参数快速验证）
    algorithms = ['sarsa', 'expected_sarsa', 'q_learning', 'double_q', 'n_step', 'sarsa_lambda']

    for algo_name in algorithms:
        print(f"\n测试 {algo_name}...")

        # 创建学习器
        test_config = TDConfig(
            alpha=0.5,
            gamma=0.99,
            epsilon=0.2,
            lambda_=0.8,
            n_step=3
        )
        learner = create_td_learner(algo_name, config=test_config)

        # 快速测试：仅运行少量回合验证代码正确性
        try:
            metrics = learner.train(
                env,
                n_episodes=5,  # 极小值用于测试
                max_steps_per_episode=100,
                log_interval=5
            )
            print(f"  ✓ {algo_name} 测试通过")
            print(f"    最后5回合平均奖励: {np.mean(metrics.episode_rewards[-5:]):.2f}")
        except Exception as e:
            print(f"  ✗ {algo_name} 测试失败: {e}")

    print("\n" + "=" * 70)
    print("完整训练测试 (SARSA on CliffWalking)")
    print("=" * 70)

    # 生产环境参数的完整训练
    production_config = TDConfig(
        alpha=0.5,
        gamma=0.99,
        epsilon=0.1
    )

    sarsa_learner = create_td_learner('sarsa', config=production_config)

    metrics = sarsa_learner.train(
        env,
        n_episodes=500,
        max_steps_per_episode=200,
        log_interval=100,
        early_stop_reward=-20.0
    )

    # 评估
    mean_reward, std_reward = sarsa_learner.evaluate(env, n_episodes=100)
    print(f"\n评估结果: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    print("\n所有测试完成!")
