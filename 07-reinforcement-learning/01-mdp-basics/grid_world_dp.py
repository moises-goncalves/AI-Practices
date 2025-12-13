#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网格世界环境与动态规划求解器

本模块实现经典的网格世界 (GridWorld) 强化学习环境，以及基于动态规划的
马尔可夫决策过程 (MDP) 求解算法。

核心算法:
    - 策略评估 (Policy Evaluation): 给定策略，计算状态价值函数
    - 策略改进 (Policy Improvement): 基于价值函数贪心改进策略
    - 策略迭代 (Policy Iteration): 交替执行评估与改进直至收敛
    - 值迭代 (Value Iteration): 直接迭代贝尔曼最优方程求解

理论基础:
    贝尔曼期望方程:
        V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]

    贝尔曼最优方程:
        V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]

References:
    [1] Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 4
    [2] Bellman, R. "Dynamic Programming", Princeton University Press, 1957

Author: Ziming Ding
Date: 2024
License: MIT
"""

from __future__ import annotations

import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Iterator, Callable
import numpy as np

# ============================================================================
# 类型定义
# ============================================================================

State = Tuple[int, int]
Action = str
Policy = Dict[State, Dict[Action, float]]
ValueFunction = Dict[State, float]
QFunction = Dict[State, Dict[Action, float]]
TransitionResult = List[Tuple[State, float, float]]  # [(next_state, prob, reward), ...]


class ActionType(Enum):
    """动作枚举类型"""
    UP = '上'
    DOWN = '下'
    LEFT = '左'
    RIGHT = '右'


@dataclass
class TransitionInfo:
    """状态转移信息"""
    next_state: State
    probability: float
    reward: float


@dataclass
class AlgorithmResult:
    """算法运行结果"""
    policy: Policy
    value_function: ValueFunction
    iterations: int
    converged: bool


# ============================================================================
# 环境抽象基类
# ============================================================================

class MDPEnvironment(ABC):
    """
    马尔可夫决策过程环境抽象基类

    定义了 MDP 环境必须实现的接口，包括状态空间、动作空间、
    转移动力学和奖励函数。
    """

    @property
    @abstractmethod
    def states(self) -> List[State]:
        """返回状态空间"""
        pass

    @property
    @abstractmethod
    def actions(self) -> List[Action]:
        """返回动作空间"""
        pass

    @property
    @abstractmethod
    def terminal_states(self) -> List[State]:
        """返回终止状态集合"""
        pass

    @abstractmethod
    def get_transitions(self, state: State, action: Action) -> TransitionResult:
        """
        获取状态转移分布

        Args:
            state: 当前状态
            action: 执行的动作

        Returns:
            转移结果列表 [(next_state, probability, reward), ...]
        """
        pass

    def is_terminal(self, state: State) -> bool:
        """判断是否为终止状态"""
        return state in self.terminal_states


# ============================================================================
# 网格世界环境实现
# ============================================================================

@dataclass
class GridWorldConfig:
    """网格世界配置参数"""
    size: int = 4
    start: State = (0, 0)
    goal: State = (3, 3)
    obstacles: List[State] = field(default_factory=list)
    slip_probability: float = 0.0
    step_reward: float = -1.0
    goal_reward: float = 100.0
    wall_penalty: float = -1.0


class GridWorld(MDPEnvironment):
    """
    网格世界环境

    标准的强化学习测试环境，智能体在 N×N 网格中从起点导航到目标点。
    支持确定性和随机性状态转移。

    网格示例 (4×4):
        ┌────┬────┬────┬────┐
        │ S  │    │    │    │   S: 起点 (0,0)
        ├────┼────┼────┼────┤   G: 目标 (3,3)
        │    │ X  │    │    │   X: 障碍物
        ├────┼────┼────┼────┤
        │    │    │    │ X  │
        ├────┼────┼────┼────┤
        │    │    │    │ G  │
        └────┴────┴────┴────┘

    Attributes:
        config: 环境配置参数
        _states: 有效状态列表（排除障碍物）
        _action_deltas: 动作到坐标变化的映射
    """

    # 动作到坐标增量的映射
    ACTION_DELTAS: Dict[str, Tuple[int, int]] = {
        '上': (-1, 0),
        '下': (1, 0),
        '左': (0, -1),
        '右': (0, 1)
    }

    # 垂直方向映射（用于随机滑动）
    PERPENDICULAR_ACTIONS: Dict[str, List[str]] = {
        '上': ['左', '右'],
        '下': ['左', '右'],
        '左': ['上', '下'],
        '右': ['上', '下']
    }

    def __init__(self, config: Optional[GridWorldConfig] = None):
        """
        初始化网格世界

        Args:
            config: 环境配置，为 None 时使用默认配置
        """
        self.config = config or GridWorldConfig()
        self._validate_config()
        self._build_state_space()

    def _validate_config(self) -> None:
        """验证配置参数有效性"""
        cfg = self.config

        if cfg.size < 2:
            raise ValueError(f"网格大小必须 >= 2，当前值: {cfg.size}")

        if not self._is_valid_position(cfg.start):
            raise ValueError(f"起点超出边界: {cfg.start}")

        if not self._is_valid_position(cfg.goal):
            raise ValueError(f"目标点超出边界: {cfg.goal}")

        if cfg.start in cfg.obstacles:
            raise ValueError(f"起点不能是障碍物: {cfg.start}")

        if cfg.goal in cfg.obstacles:
            raise ValueError(f"目标点不能是障碍物: {cfg.goal}")

        if not 0.0 <= cfg.slip_probability <= 1.0:
            raise ValueError(f"滑动概率必须在 [0,1] 范围内: {cfg.slip_probability}")

    def _is_valid_position(self, pos: State) -> bool:
        """检查位置是否在网格范围内"""
        return 0 <= pos[0] < self.config.size and 0 <= pos[1] < self.config.size

    def _build_state_space(self) -> None:
        """构建有效状态空间"""
        self._states = [
            (i, j)
            for i in range(self.config.size)
            for j in range(self.config.size)
            if (i, j) not in self.config.obstacles
        ]

    @property
    def states(self) -> List[State]:
        """返回有效状态列表"""
        return self._states.copy()

    @property
    def actions(self) -> List[Action]:
        """返回动作列表"""
        return list(self.ACTION_DELTAS.keys())

    @property
    def terminal_states(self) -> List[State]:
        """返回终止状态"""
        return [self.config.goal]

    @property
    def num_states(self) -> int:
        """状态空间大小"""
        return len(self._states)

    @property
    def num_actions(self) -> int:
        """动作空间大小"""
        return len(self.ACTION_DELTAS)

    def _execute_move(self, state: State, action: Action) -> State:
        """
        执行移动动作（内部方法）

        Args:
            state: 当前状态
            action: 动作

        Returns:
            移动后的状态（考虑边界和障碍物碰撞）
        """
        di, dj = self.ACTION_DELTAS[action]
        next_i = max(0, min(self.config.size - 1, state[0] + di))
        next_j = max(0, min(self.config.size - 1, state[1] + dj))
        next_state = (next_i, next_j)

        # 碰到障碍物则停留原地
        if next_state in self.config.obstacles:
            return state

        return next_state

    def _compute_reward(self, current: State, next_state: State) -> float:
        """
        计算状态转移奖励

        Args:
            current: 当前状态
            next_state: 下一状态

        Returns:
            即时奖励值
        """
        if next_state == self.config.goal:
            return self.config.goal_reward
        return self.config.step_reward

    def get_transitions(self, state: State, action: Action) -> TransitionResult:
        """
        获取状态转移概率分布

        实现确定性或随机性状态转移。随机环境下，智能体有一定概率
        向垂直方向滑动。

        Args:
            state: 当前状态
            action: 执行的动作

        Returns:
            转移结果列表 [(next_state, probability, reward), ...]
        """
        # 终止状态：自循环，零奖励
        if self.is_terminal(state):
            return [(state, 1.0, 0.0)]

        transitions = []

        if self.config.slip_probability == 0.0:
            # 确定性环境
            next_state = self._execute_move(state, action)
            reward = self._compute_reward(state, next_state)
            transitions.append((next_state, 1.0, reward))
        else:
            # 随机环境：主方向 + 滑动方向
            main_prob = 1.0 - self.config.slip_probability
            slip_prob = self.config.slip_probability / 2.0

            # 主方向移动
            main_next = self._execute_move(state, action)
            main_reward = self._compute_reward(state, main_next)
            transitions.append((main_next, main_prob, main_reward))

            # 垂直方向滑动
            for perp_action in self.PERPENDICULAR_ACTIONS[action]:
                perp_next = self._execute_move(state, perp_action)
                perp_reward = self._compute_reward(state, perp_next)
                transitions.append((perp_next, slip_prob, perp_reward))

        return transitions

    def render_policy(self, policy: Policy, stream: Optional[Callable] = None) -> str:
        """
        渲染策略为可视化字符串

        Args:
            policy: 策略字典
            stream: 输出流函数，默认为 print

        Returns:
            渲染后的字符串
        """
        symbols = {'上': '↑', '下': '↓', '左': '←', '右': '→'}
        output = stream or print

        lines = ["\n策略可视化:"]
        lines.append("┌" + "───┬" * (self.config.size - 1) + "───┐")

        for i in range(self.config.size):
            row = "│"
            for j in range(self.config.size):
                if (i, j) == self.config.goal:
                    row += " G │"
                elif (i, j) in self.config.obstacles:
                    row += " X │"
                elif (i, j) in policy:
                    best_action = max(policy[(i, j)], key=policy[(i, j)].get)
                    row += f" {symbols[best_action]} │"
                else:
                    row += "   │"
            lines.append(row)

            if i < self.config.size - 1:
                lines.append("├" + "───┼" * (self.config.size - 1) + "───┤")

        lines.append("└" + "───┴" * (self.config.size - 1) + "───┘")

        result = "\n".join(lines)
        output(result)
        return result

    def render_values(self, V: ValueFunction, stream: Optional[Callable] = None) -> str:
        """
        渲染价值函数为可视化字符串

        Args:
            V: 状态价值函数
            stream: 输出流函数

        Returns:
            渲染后的字符串
        """
        output = stream or print

        lines = ["\n状态价值函数:"]
        lines.append("┌" + "──────┬" * (self.config.size - 1) + "──────┐")

        for i in range(self.config.size):
            row = "│"
            for j in range(self.config.size):
                if (i, j) == self.config.goal:
                    row += "   G  │"
                elif (i, j) in self.config.obstacles:
                    row += "   X  │"
                elif (i, j) in V:
                    row += f"{V[(i,j)]:6.1f}│"
                else:
                    row += "      │"
            lines.append(row)

            if i < self.config.size - 1:
                lines.append("├" + "──────┼" * (self.config.size - 1) + "──────┤")

        lines.append("└" + "──────┴" * (self.config.size - 1) + "──────┘")

        result = "\n".join(lines)
        output(result)
        return result


# ============================================================================
# 动态规划算法
# ============================================================================

class DynamicProgrammingSolver:
    """
    动态规划 MDP 求解器

    实现基于模型的强化学习算法，包括策略评估、策略改进、
    策略迭代和值迭代。

    Attributes:
        env: MDP 环境
        gamma: 折扣因子 γ ∈ [0, 1]
        theta: 收敛阈值
        verbose: 是否输出详细信息
    """

    def __init__(
        self,
        env: MDPEnvironment,
        gamma: float = 0.99,
        theta: float = 1e-6,
        verbose: bool = True
    ):
        """
        初始化求解器

        Args:
            env: MDP 环境实例
            gamma: 折扣因子，控制对未来奖励的重视程度
            theta: 值函数更新的收敛阈值
            verbose: 是否打印中间结果
        """
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"折扣因子必须在 [0,1] 范围内: {gamma}")
        if theta <= 0:
            raise ValueError(f"收敛阈值必须为正数: {theta}")

        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """条件日志输出"""
        if self.verbose:
            print(message)

    def evaluate_policy(
        self,
        policy: Policy,
        max_iterations: int = 10000
    ) -> Tuple[ValueFunction, int]:
        """
        策略评估：计算给定策略的状态价值函数

        通过迭代求解贝尔曼期望方程：
            V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R + γV^π(s')]

        Args:
            policy: 待评估的策略 π(a|s)
            max_iterations: 最大迭代次数

        Returns:
            (V, iterations): 价值函数和迭代次数
        """
        V: ValueFunction = {s: 0.0 for s in self.env.states}

        for iteration in range(1, max_iterations + 1):
            delta = 0.0

            for state in self.env.states:
                if self.env.is_terminal(state):
                    continue

                old_value = V[state]
                new_value = 0.0

                # 贝尔曼期望方程
                for action in self.env.actions:
                    action_prob = policy.get(state, {}).get(action, 0.0)
                    if action_prob > 0:
                        for next_state, trans_prob, reward in self.env.get_transitions(state, action):
                            new_value += action_prob * trans_prob * (
                                reward + self.gamma * V.get(next_state, 0.0)
                            )

                V[state] = new_value
                delta = max(delta, abs(old_value - new_value))

            if delta < self.theta:
                return V, iteration

        return V, max_iterations

    def improve_policy(self, V: ValueFunction) -> Policy:
        """
        策略改进：基于价值函数贪心构造新策略

        对每个状态选择使动作价值最大的动作：
            π'(s) = argmax_a Σ_{s'} P(s'|s,a) [R + γV(s')]

        Args:
            V: 当前状态价值函数

        Returns:
            改进后的确定性策略
        """
        policy: Policy = {}

        for state in self.env.states:
            if self.env.is_terminal(state):
                # 终止状态：均匀随机策略
                policy[state] = {a: 1.0 / len(self.env.actions) for a in self.env.actions}
                continue

            # 计算各动作的 Q 值
            q_values: Dict[Action, float] = {}
            for action in self.env.actions:
                q_val = 0.0
                for next_state, trans_prob, reward in self.env.get_transitions(state, action):
                    q_val += trans_prob * (reward + self.gamma * V.get(next_state, 0.0))
                q_values[action] = q_val

            # 选择最优动作（处理并列情况）
            best_value = max(q_values.values())
            best_actions = [a for a, v in q_values.items() if abs(v - best_value) < 1e-9]

            # 确定性策略：选择第一个最优动作
            policy[state] = {
                a: 1.0 if a == best_actions[0] else 0.0
                for a in self.env.actions
            }

        return policy

    def policy_iteration(self, max_iterations: int = 100) -> AlgorithmResult:
        """
        策略迭代算法

        交替执行策略评估和策略改进，直到策略稳定。

        算法流程：
            1. 初始化随机策略
            2. 策略评估：计算 V^π
            3. 策略改进：π' = greedy(V^π)
            4. 若 π' = π 则停止，否则 π ← π' 并转步骤2

        Args:
            max_iterations: 最大外层迭代次数

        Returns:
            AlgorithmResult 包含最优策略、价值函数和收敛信息
        """
        self._log("=" * 50)
        self._log("策略迭代 (Policy Iteration)")
        self._log("=" * 50)

        # 初始化均匀随机策略
        num_actions = len(self.env.actions)
        policy: Policy = {
            s: {a: 1.0 / num_actions for a in self.env.actions}
            for s in self.env.states
        }

        V: ValueFunction = {}

        for iteration in range(1, max_iterations + 1):
            self._log(f"\n--- 外层迭代 {iteration} ---")

            # 策略评估
            V, eval_iters = self.evaluate_policy(policy)
            self._log(f"策略评估完成，内层迭代: {eval_iters}")

            # 策略改进
            new_policy = self.improve_policy(V)

            # 检查收敛（策略不变）
            if self._policies_equal(policy, new_policy):
                self._log(f"\n策略迭代收敛，总迭代: {iteration}")
                return AlgorithmResult(
                    policy=new_policy,
                    value_function=V,
                    iterations=iteration,
                    converged=True
                )

            policy = new_policy

        self._log(f"\n达到最大迭代次数: {max_iterations}")
        return AlgorithmResult(
            policy=policy,
            value_function=V,
            iterations=max_iterations,
            converged=False
        )

    def value_iteration(self, max_iterations: int = 10000) -> AlgorithmResult:
        """
        值迭代算法

        直接迭代贝尔曼最优方程求解最优价值函数。

        更新规则：
            V(s) ← max_a Σ_{s'} P(s'|s,a) [R + γV(s')]

        Args:
            max_iterations: 最大迭代次数

        Returns:
            AlgorithmResult 包含最优策略、价值函数和收敛信息
        """
        self._log("=" * 50)
        self._log("值迭代 (Value Iteration)")
        self._log("=" * 50)

        V: ValueFunction = {s: 0.0 for s in self.env.states}

        for iteration in range(1, max_iterations + 1):
            delta = 0.0

            for state in self.env.states:
                if self.env.is_terminal(state):
                    continue

                old_value = V[state]

                # 贝尔曼最优方程
                q_values = []
                for action in self.env.actions:
                    q_val = 0.0
                    for next_state, trans_prob, reward in self.env.get_transitions(state, action):
                        q_val += trans_prob * (reward + self.gamma * V.get(next_state, 0.0))
                    q_values.append(q_val)

                V[state] = max(q_values)
                delta = max(delta, abs(old_value - V[state]))

            if delta < self.theta:
                self._log(f"值迭代收敛，迭代次数: {iteration}")
                policy = self.improve_policy(V)
                return AlgorithmResult(
                    policy=policy,
                    value_function=V,
                    iterations=iteration,
                    converged=True
                )

        self._log(f"达到最大迭代次数: {max_iterations}")
        policy = self.improve_policy(V)
        return AlgorithmResult(
            policy=policy,
            value_function=V,
            iterations=max_iterations,
            converged=False
        )

    def compute_q_function(self, V: ValueFunction) -> QFunction:
        """
        根据状态价值函数计算动作价值函数

        Q(s,a) = Σ_{s'} P(s'|s,a) [R + γV(s')]

        Args:
            V: 状态价值函数

        Returns:
            动作价值函数 Q(s, a)
        """
        Q: QFunction = {}

        for state in self.env.states:
            Q[state] = {}
            for action in self.env.actions:
                q_val = 0.0
                for next_state, trans_prob, reward in self.env.get_transitions(state, action):
                    q_val += trans_prob * (reward + self.gamma * V.get(next_state, 0.0))
                Q[state][action] = q_val

        return Q

    @staticmethod
    def _policies_equal(p1: Policy, p2: Policy) -> bool:
        """比较两个策略是否相等"""
        if set(p1.keys()) != set(p2.keys()):
            return False

        for state in p1:
            for action in p1[state]:
                if abs(p1[state].get(action, 0) - p2[state].get(action, 0)) > 1e-9:
                    return False
        return True


# ============================================================================
# 策略执行与评估
# ============================================================================

class PolicyExecutor:
    """
    策略执行器

    在给定环境中执行策略，支持单次和多次回合评估。
    """

    def __init__(self, env: GridWorld, seed: Optional[int] = None):
        """
        初始化执行器

        Args:
            env: 网格世界环境
            seed: 随机数种子
        """
        self.env = env
        self.rng = np.random.default_rng(seed)

    def run_episode(
        self,
        policy: Policy,
        max_steps: int = 100,
        verbose: bool = True
    ) -> Tuple[float, int, List[State]]:
        """
        执行单个回合

        Args:
            policy: 执行的策略
            max_steps: 最大步数
            verbose: 是否输出轨迹

        Returns:
            (total_reward, steps, trajectory): 累积奖励、步数、状态轨迹
        """
        state = self.env.config.start
        total_reward = 0.0
        trajectory = [state]

        if verbose:
            print(f"\n起始状态: {state}")

        for step in range(max_steps):
            if self.env.is_terminal(state):
                if verbose:
                    print(f"到达目标！总步数: {step}, 累积奖励: {total_reward:.1f}")
                return total_reward, step, trajectory

            # 根据策略采样动作
            action_probs = policy[state]
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            action = self.rng.choice(actions, p=probs)

            # 执行动作
            transitions = self.env.get_transitions(state, action)
            trans_probs = [t[1] for t in transitions]
            idx = self.rng.choice(len(transitions), p=trans_probs)
            next_state, _, reward = transitions[idx]

            if verbose:
                print(f"步骤 {step + 1}: {state} --[{action}]--> {next_state}, 奖励: {reward:.1f}")

            total_reward += reward
            state = next_state
            trajectory.append(state)

        if verbose:
            print(f"达到最大步数限制，累积奖励: {total_reward:.1f}")

        return total_reward, max_steps, trajectory

    def evaluate_policy(
        self,
        policy: Policy,
        num_episodes: int = 100,
        max_steps: int = 100
    ) -> Dict[str, float]:
        """
        评估策略性能

        Args:
            policy: 待评估策略
            num_episodes: 评估回合数
            max_steps: 每回合最大步数

        Returns:
            性能统计字典
        """
        rewards = []
        steps_list = []
        successes = 0

        for _ in range(num_episodes):
            reward, steps, trajectory = self.run_episode(policy, max_steps, verbose=False)
            rewards.append(reward)
            steps_list.append(steps)
            if trajectory[-1] == self.env.config.goal:
                successes += 1

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps_list),
            'success_rate': successes / num_episodes
        }


# ============================================================================
# 单元测试
# ============================================================================

class TestGridWorld(unittest.TestCase):
    """网格世界环境单元测试"""

    def setUp(self):
        """测试前初始化"""
        self.config = GridWorldConfig(
            size=4,
            start=(0, 0),
            goal=(3, 3),
            obstacles=[(1, 1)],
            slip_probability=0.0
        )
        self.env = GridWorld(self.config)

    def test_state_space(self):
        """测试状态空间构建"""
        # 4x4 网格减去 1 个障碍物 = 15 个状态
        self.assertEqual(len(self.env.states), 15)
        self.assertNotIn((1, 1), self.env.states)

    def test_action_space(self):
        """测试动作空间"""
        self.assertEqual(len(self.env.actions), 4)
        self.assertIn('上', self.env.actions)

    def test_terminal_state(self):
        """测试终止状态"""
        self.assertTrue(self.env.is_terminal((3, 3)))
        self.assertFalse(self.env.is_terminal((0, 0)))

    def test_deterministic_transition(self):
        """测试确定性状态转移"""
        transitions = self.env.get_transitions((0, 0), '下')
        self.assertEqual(len(transitions), 1)
        next_state, prob, reward = transitions[0]
        self.assertEqual(next_state, (1, 0))
        self.assertEqual(prob, 1.0)

    def test_wall_collision(self):
        """测试边界碰撞"""
        transitions = self.env.get_transitions((0, 0), '上')
        next_state, _, _ = transitions[0]
        self.assertEqual(next_state, (0, 0))  # 停留原地

    def test_obstacle_collision(self):
        """测试障碍物碰撞"""
        transitions = self.env.get_transitions((0, 1), '下')
        next_state, _, _ = transitions[0]
        self.assertEqual(next_state, (0, 1))  # 碰到障碍物停留原地

    def test_goal_reward(self):
        """测试目标奖励"""
        transitions = self.env.get_transitions((3, 2), '右')
        _, _, reward = transitions[0]
        self.assertEqual(reward, 100.0)

    def test_stochastic_transition(self):
        """测试随机状态转移"""
        stochastic_config = GridWorldConfig(slip_probability=0.2)
        stochastic_env = GridWorld(stochastic_config)

        transitions = stochastic_env.get_transitions((1, 1), '右')
        self.assertEqual(len(transitions), 3)  # 主方向 + 两个垂直方向

        total_prob = sum(t[1] for t in transitions)
        self.assertAlmostEqual(total_prob, 1.0)


class TestDynamicProgramming(unittest.TestCase):
    """动态规划算法单元测试"""

    def setUp(self):
        """测试前初始化"""
        self.config = GridWorldConfig(
            size=3,  # 使用小网格加速测试
            start=(0, 0),
            goal=(2, 2),
            obstacles=[],
            slip_probability=0.0
        )
        self.env = GridWorld(self.config)
        self.solver = DynamicProgrammingSolver(self.env, gamma=0.99, verbose=False)

    def test_policy_evaluation(self):
        """测试策略评估"""
        # 均匀随机策略
        policy = {
            s: {a: 0.25 for a in self.env.actions}
            for s in self.env.states
        }

        V, iterations = self.solver.evaluate_policy(policy)

        # 价值函数应该有值
        self.assertTrue(all(isinstance(v, float) for v in V.values()))
        # 目标状态价值为 0
        self.assertEqual(V[(2, 2)], 0.0)
        # 应该收敛
        self.assertLess(iterations, 1000)

    def test_policy_improvement(self):
        """测试策略改进"""
        # 初始化价值函数
        V = {s: 0.0 for s in self.env.states}
        V[(2, 1)] = 50.0  # 目标左边
        V[(1, 2)] = 50.0  # 目标上方

        policy = self.solver.improve_policy(V)

        # 检查策略是确定性的
        for state in self.env.states:
            probs = list(policy[state].values())
            self.assertAlmostEqual(sum(probs), 1.0)

    def test_policy_iteration_convergence(self):
        """测试策略迭代收敛"""
        result = self.solver.policy_iteration()

        self.assertTrue(result.converged)
        self.assertIsNotNone(result.policy)
        self.assertIsNotNone(result.value_function)

    def test_value_iteration_convergence(self):
        """测试值迭代收敛"""
        result = self.solver.value_iteration()

        self.assertTrue(result.converged)
        self.assertIsNotNone(result.policy)
        self.assertIsNotNone(result.value_function)

    def test_optimal_policy_reaches_goal(self):
        """测试最优策略能到达目标"""
        result = self.solver.value_iteration()

        executor = PolicyExecutor(self.env, seed=42)
        stats = executor.evaluate_policy(result.policy, num_episodes=50, max_steps=50)

        # 确定性环境下成功率应为 100%
        self.assertEqual(stats['success_rate'], 1.0)

    def test_algorithms_produce_same_value(self):
        """测试两种算法产生相同的价值函数"""
        result_pi = self.solver.policy_iteration()
        result_vi = self.solver.value_iteration()

        # 价值函数应该近似相等
        for state in self.env.states:
            self.assertAlmostEqual(
                result_pi.value_function[state],
                result_vi.value_function[state],
                places=3
            )


class TestPolicyExecutor(unittest.TestCase):
    """策略执行器单元测试"""

    def setUp(self):
        """测试前初始化"""
        self.config = GridWorldConfig(size=3, goal=(2, 2))
        self.env = GridWorld(self.config)
        self.executor = PolicyExecutor(self.env, seed=42)

    def test_episode_execution(self):
        """测试回合执行"""
        # 简单的向右向下策略
        policy = {
            s: {'下': 0.5, '右': 0.5, '上': 0.0, '左': 0.0}
            for s in self.env.states
        }

        reward, steps, trajectory = self.executor.run_episode(policy, verbose=False)

        self.assertIsInstance(reward, float)
        self.assertGreater(len(trajectory), 0)
        self.assertEqual(trajectory[0], self.env.config.start)


def run_tests():
    """运行所有单元测试"""
    print("=" * 60)
    print("运行单元测试")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestGridWorld))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicProgramming))
    suite.addTests(loader.loadTestsFromTestCase(TestPolicyExecutor))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    # 首先运行单元测试
    if not run_tests():
        print("\n单元测试未通过，请检查代码！")
        return

    print("\n" + "=" * 60)
    print("网格世界动态规划演示")
    print("=" * 60)

    # 创建标准环境
    config = GridWorldConfig(
        size=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=[(1, 1), (2, 3)],
        slip_probability=0.0,
        step_reward=-1.0,
        goal_reward=100.0
    )
    env = GridWorld(config)

    print(f"\n环境参数:")
    print(f"  网格大小: {config.size}×{config.size}")
    print(f"  状态空间: {env.num_states} 个状态")
    print(f"  动作空间: {env.num_actions} 个动作")
    print(f"  障碍物: {config.obstacles}")
    print(f"  折扣因子: γ = 0.99")

    solver = DynamicProgrammingSolver(env, gamma=0.99, theta=1e-6, verbose=True)

    # 策略迭代
    print("\n" + "=" * 60)
    result_pi = solver.policy_iteration()
    env.render_policy(result_pi.policy)
    env.render_values(result_pi.value_function)

    # 值迭代
    print("\n" + "=" * 60)
    result_vi = solver.value_iteration()
    env.render_policy(result_vi.policy)
    env.render_values(result_vi.value_function)

    # 策略评估
    print("\n" + "=" * 60)
    print("策略性能评估")
    print("=" * 60)

    executor = PolicyExecutor(env, seed=42)

    # 展示单次执行
    executor.run_episode(result_vi.policy, verbose=True)

    # 统计评估
    stats = executor.evaluate_policy(result_vi.policy, num_episodes=100)
    print(f"\n100 回合统计:")
    print(f"  平均奖励: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  平均步数: {stats['mean_steps']:.2f}")
    print(f"  成功率: {stats['success_rate']*100:.1f}%")

    # 随机环境测试
    print("\n" + "=" * 60)
    print("随机环境 (滑动概率 20%)")
    print("=" * 60)

    stochastic_config = GridWorldConfig(
        size=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=[(1, 1), (2, 3)],
        slip_probability=0.2
    )
    stochastic_env = GridWorld(stochastic_config)
    stochastic_solver = DynamicProgrammingSolver(stochastic_env, gamma=0.99, verbose=True)

    result_stoch = stochastic_solver.value_iteration()
    stochastic_env.render_policy(result_stoch.policy)
    stochastic_env.render_values(result_stoch.value_function)

    # 随机环境策略评估
    stochastic_executor = PolicyExecutor(stochastic_env, seed=42)
    stochastic_stats = stochastic_executor.evaluate_policy(result_stoch.policy, num_episodes=100)
    print(f"\n随机环境 100 回合统计:")
    print(f"  平均奖励: {stochastic_stats['mean_reward']:.2f} ± {stochastic_stats['std_reward']:.2f}")
    print(f"  平均步数: {stochastic_stats['mean_steps']:.2f}")
    print(f"  成功率: {stochastic_stats['success_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
