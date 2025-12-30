"""
时序差分学习基类模块 (Base Classes Module)
=========================================

核心思想 (Core Idea):
--------------------
定义TD学习算法的通用接口、协议和基类。采用模板方法模式，
将算法骨架固定在基类中，具体更新逻辑延迟到子类实现。

数学原理 (Mathematical Theory):
------------------------------
所有TD算法共享的核心结构:
1. 价值函数估计: V(s)或Q(s,a)的表格存储
2. 策略选择: ε-greedy策略实现探索-利用权衡
3. TD更新: δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
4. 训练循环: 采样→更新→记录的标准流程

设计原则:
--------
- 面向接口编程: 使用Protocol定义环境和策略接口
- 开闭原则: 对扩展开放，对修改关闭
- 模板方法: 固定训练流程，变化的是更新规则

复杂度 (Complexity):
-------------------
- 动作选择: O(|A|) - 需要遍历所有动作找最大Q值
- 单步更新: O(1)到O(|S|) - 取决于具体算法
- 存储空间: O(|S|×|A|) - Q表存储
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import (
    Dict, List, Optional, Tuple, Any,
    Protocol, TypeVar, Generic, runtime_checkable
)
import logging

from .config import TDConfig, TrainingMetrics

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 类型变量
State = TypeVar('State')
Action = TypeVar('Action')


@runtime_checkable
class Environment(Protocol[State, Action]):
    """
    强化学习环境协议。

    核心思想 (Core Idea):
    --------------------
    定义环境的最小接口，兼容OpenAI Gym/Gymnasium风格。
    任何实现此协议的类都可以作为TD算法的训练环境。

    接口说明:
    --------
    - reset(): 重置环境到初始状态
    - step(action): 执行动作，返回转移结果
    - action_space: 动作空间，需要有.n属性表示动作数量
    - observation_space: 观测空间

    Example:
        >>> class MyEnv:
        ...     def reset(self): return state, {}
        ...     def step(self, action): return next_state, reward, done, truncated, info
        ...     @property
        ...     def action_space(self): return DiscreteSpace(4)
    """

    def reset(self) -> Tuple[State, Dict[str, Any]]:
        """
        重置环境到初始状态。

        Returns:
            (初始状态, 信息字典)
        """
        ...

    def step(self, action: Action) -> Tuple[State, float, bool, bool, Dict[str, Any]]:
        """
        执行动作，观察环境反馈。

        Args:
            action: 要执行的动作

        Returns:
            (下一状态, 奖励, 是否终止, 是否截断, 信息字典)
        """
        ...

    @property
    def action_space(self) -> Any:
        """动作空间，需要有.n属性。"""
        ...

    @property
    def observation_space(self) -> Any:
        """观测空间。"""
        ...


@runtime_checkable
class Policy(Protocol[State, Action]):
    """
    策略协议。

    核心思想 (Core Idea):
    --------------------
    策略是从状态到动作的映射，可以是确定性的或随机性的。
    本协议定义策略的基本接口。

    接口说明:
    --------
    - __call__(state): 根据状态返回动作
    - action_probabilities(state): 返回动作概率分布
    """

    def __call__(self, state: State) -> Action:
        """
        根据状态选择动作。

        Args:
            state: 当前状态

        Returns:
            选择的动作
        """
        ...

    def action_probabilities(self, state: State) -> Dict[Action, float]:
        """
        返回状态下各动作的概率分布。

        Args:
            state: 当前状态

        Returns:
            动作到概率的映射
        """
        ...


class BaseTDLearner(ABC, Generic[State, Action]):
    """
    时序差分学习算法基类。

    核心思想 (Core Idea):
    --------------------
    提供TD学习算法的通用框架，包括:
    - 价值函数管理（V和Q表）
    - ε-greedy策略实现
    - 标准训练和评估循环
    - 指标记录和日志

    子类只需实现update()方法即可获得完整功能。

    数学原理 (Mathematical Theory):
    ------------------------------
    ε-greedy策略:
        π(a|s) = ε/|A| + (1-ε)·𝟙(a = argmax_a' Q(s,a'))

    该策略以概率ε均匀随机选择动作（探索），
    以概率1-ε选择当前最优动作（利用）。

    设计模式:
    --------
    采用模板方法模式(Template Method Pattern):
    - train()和train_episode()定义算法骨架
    - update()是抽象方法，由子类实现具体更新逻辑
    - 钩子方法on_episode_start/end可被子类覆盖

    复杂度 (Complexity):
    -------------------
    - epsilon_greedy_action: O(|A|)
    - greedy_action: O(|A|)
    - train_episode: O(T × update_complexity)，T为回合长度
    - evaluate: O(N × T)，N为评估回合数

    Example:
        >>> class MySARSA(BaseTDLearner):
        ...     def update(self, state, action, reward, next_state, next_action, done):
        ...         # 实现SARSA更新
        ...         return td_error
    """

    def __init__(self, config: TDConfig) -> None:
        """
        初始化TD学习器。

        Args:
            config: TD学习配置对象，包含所有超参数
        """
        self.config = config

        # 价值函数: 使用defaultdict自动初始化未见过的状态
        self._value_function: Dict[State, float] = defaultdict(
            lambda: config.initial_value
        )
        self._q_function: Dict[Tuple[State, Action], float] = defaultdict(
            lambda: config.initial_value
        )

        # 训练指标
        self.metrics = TrainingMetrics()

        # 动作空间（需要通过set_action_space设置或自动推断）
        self._action_space: Optional[List[Action]] = None

    # =========================================================================
    # 属性访问器
    # =========================================================================

    @property
    def value_function(self) -> Dict[State, float]:
        """
        获取状态价值函数V(s)的副本。

        Returns:
            状态到价值的映射字典
        """
        return dict(self._value_function)

    @property
    def q_function(self) -> Dict[Tuple[State, Action], float]:
        """
        获取动作价值函数Q(s,a)的副本。

        Returns:
            (状态,动作)对到价值的映射字典
        """
        return dict(self._q_function)

    def get_value(self, state: State) -> float:
        """获取状态价值V(s)。"""
        return self._value_function[state]

    def get_q_value(self, state: State, action: Action) -> float:
        """获取动作价值Q(s,a)。"""
        return self._q_function[(state, action)]

    def set_action_space(self, actions: List[Action]) -> None:
        """
        设置动作空间。

        Args:
            actions: 所有可用动作的列表
        """
        self._action_space = actions

    # =========================================================================
    # 策略方法
    # =========================================================================

    def epsilon_greedy_action(self, state: State) -> Action:
        """
        ε-greedy策略选择动作。

        数学原理 (Mathematical Theory):
        ------------------------------
        π(a|s) = ε/|A| + (1-ε)·𝟙(a = argmax_a' Q(s,a'))

        以概率ε随机选择动作实现探索，以概率1-ε选择当前估计的
        最优动作实现利用。当存在多个动作Q值相等时，随机选择一个。

        Args:
            state: 当前状态

        Returns:
            选择的动作

        Raises:
            ValueError: 未设置动作空间

        复杂度: O(|A|)
        """
        if self._action_space is None:
            raise ValueError(
                "未设置动作空间。请调用set_action_space()或使用带action_space.n的环境。"
            )

        # 以ε概率随机探索
        if np.random.random() < self.config.epsilon:
            return np.random.choice(self._action_space)

        # 以1-ε概率贪婪选择
        q_values = [self.get_q_value(state, a) for a in self._action_space]
        max_q = max(q_values)

        # 处理多个最优动作的情况（随机打破平局）
        best_actions = [
            a for a, q in zip(self._action_space, q_values)
            if np.isclose(q, max_q)
        ]
        return np.random.choice(best_actions)

    def greedy_action(self, state: State) -> Action:
        """
        贪婪策略选择动作（用于评估）。

        纯贪婪策略，总是选择Q值最大的动作。

        Args:
            state: 当前状态

        Returns:
            最优动作

        复杂度: O(|A|)
        """
        if self._action_space is None:
            raise ValueError("未设置动作空间")

        q_values = [self.get_q_value(state, a) for a in self._action_space]
        max_q = max(q_values)
        best_actions = [
            a for a, q in zip(self._action_space, q_values)
            if np.isclose(q, max_q)
        ]
        return np.random.choice(best_actions)

    def get_policy(self) -> Dict[State, Action]:
        """
        从Q函数提取贪婪策略。

        Returns:
            状态到最优动作的映射
        """
        policy = {}
        states = set(s for (s, _) in self._q_function.keys())

        for state in states:
            policy[state] = self.greedy_action(state)

        return policy

    # =========================================================================
    # 核心抽象方法
    # =========================================================================

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

        这是模板方法模式中的抽象方法，必须由子类实现。
        不同的TD算法（SARSA、Q-Learning等）的核心区别就在于此方法。

        Args:
            state: 当前状态 S_t
            action: 执行的动作 A_t
            reward: 获得的即时奖励 R_{t+1}
            next_state: 下一状态 S_{t+1}
            next_action: 下一动作 A_{t+1}（SARSA需要，Q-Learning不用）
            done: 是否终止

        Returns:
            TD误差 δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
        """
        pass

    # =========================================================================
    # 钩子方法（可被子类覆盖）
    # =========================================================================

    def on_episode_start(self, episode: int) -> None:
        """
        回合开始时的钩子方法。

        子类可覆盖此方法来执行回合开始时的初始化，
        例如清空资格迹。

        Args:
            episode: 当前回合编号
        """
        pass

    def on_episode_end(self, episode: int, reward: float, length: int) -> None:
        """
        回合结束时的钩子方法。

        Args:
            episode: 当前回合编号
            reward: 回合累积奖励
            length: 回合步数
        """
        pass

    # =========================================================================
    # 训练方法
    # =========================================================================

    def train_episode(
        self,
        env: Environment[State, Action],
        max_steps: int = 10000
    ) -> Tuple[float, int]:
        """
        训练一个完整回合。

        实现标准的TD控制训练循环:
        1. 重置环境获取初始状态
        2. 选择初始动作
        3. 循环: 执行动作→获取反馈→更新Q值→选择下一动作
        4. 记录指标

        Args:
            env: 训练环境
            max_steps: 单回合最大步数限制

        Returns:
            (回合累积奖励, 回合步数)

        复杂度: O(T × update_complexity)，T为实际步数
        """
        state, _ = env.reset()
        action = self.epsilon_greedy_action(state)

        total_reward = 0.0
        td_errors = []

        for step in range(max_steps):
            # 执行动作，观察环境反馈
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 选择下一动作（SARSA需要，Q-Learning可忽略）
            next_action = None if done else self.epsilon_greedy_action(next_state)

            # 执行TD更新（由子类实现）
            td_error = self.update(
                state, action, reward, next_state, next_action, done
            )
            td_errors.append(abs(td_error))

            total_reward += reward

            if done:
                break

            # 状态转移
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
            env: 训练环境
            n_episodes: 训练回合数
            max_steps_per_episode: 每回合最大步数
            log_interval: 日志输出间隔（回合数）
            early_stop_reward: 早停阈值，达到此平均奖励后停止

        Returns:
            训练指标对象

        Example:
            >>> metrics = learner.train(env, n_episodes=500, log_interval=100)
            >>> print(f"最终平均奖励: {np.mean(metrics.episode_rewards[-100:]):.2f}")
        """
        # 自动推断动作空间
        if self._action_space is None:
            if hasattr(env.action_space, 'n'):
                self.set_action_space(list(range(env.action_space.n)))
            else:
                raise ValueError(
                    "无法自动推断动作空间。请手动调用set_action_space()。"
                )

        for episode in range(n_episodes):
            self.on_episode_start(episode)

            reward, steps = self.train_episode(env, max_steps_per_episode)

            self.on_episode_end(episode, reward, steps)

            # 日志输出
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
                            f"早停: 平均奖励 {recent_avg:.2f} >= {early_stop_reward}"
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
        评估当前策略的性能。

        使用纯贪婪策略（不探索）运行多个回合，统计性能。

        Args:
            env: 评估环境
            n_episodes: 评估回合数
            max_steps: 每回合最大步数

        Returns:
            (平均奖励, 奖励标准差)

        Example:
            >>> mean_reward, std_reward = learner.evaluate(env)
            >>> print(f"评估结果: {mean_reward:.2f} ± {std_reward:.2f}")
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

        return float(np.mean(rewards)), float(np.std(rewards))

    def reset(self) -> None:
        """
        重置学习器状态。

        清空Q函数、V函数和训练指标，用于重新开始训练。
        """
        self._value_function.clear()
        self._q_function.clear()
        self.metrics.clear()
