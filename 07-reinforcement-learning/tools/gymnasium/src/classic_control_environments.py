#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gymnasium 经典控制环境详解

================================================================================
核心思想 (Core Idea)
================================================================================
经典控制 (Classic Control) 环境是一组基于经典控制理论问题设计的低维环境。
它们具有简单的状态空间和明确的物理意义，是验证强化学习算法的理想测试平台。
这些环境的优势在于训练速度快、可解释性强，适合算法原型开发和教学演示。

================================================================================
数学原理 (Mathematical Theory)
================================================================================
经典控制环境的动力学通常可以用微分方程描述:

**CartPole (倒立摆)**:
    小车运动方程:
    $$\\ddot{x} = \\frac{F + m_p l (\\dot{\\theta}^2 \\sin\\theta - \\ddot{\\theta}\\cos\\theta)}{m_c + m_p}$$

    摆杆运动方程:
    $$\\ddot{\\theta} = \\frac{g\\sin\\theta + \\cos\\theta \\cdot \\frac{-F - m_p l \\dot{\\theta}^2 \\sin\\theta}{m_c + m_p}}{l(\\frac{4}{3} - \\frac{m_p \\cos^2\\theta}{m_c + m_p})}$$

    其中:
    - x: 小车位置
    - θ: 摆杆与垂直方向的夹角
    - F: 施加在小车上的力
    - m_c, m_p: 小车和摆杆质量
    - l: 摆杆半长
    - g: 重力加速度

**MountainCar**:
    位置更新: $x_{t+1} = x_t + v_{t+1}$
    速度更新: $v_{t+1} = v_t + 0.001 \\cdot a - 0.0025 \\cdot \\cos(3x_t)$

    其中 a ∈ {-1, 0, 1} 表示左/不动/右加速。

**Pendulum**:
    $$\\ddot{\\theta} = -\\frac{3g}{2l}\\sin(\\theta + \\pi) + \\frac{3}{ml^2}u$$

    其中 u 是施加的扭矩。

================================================================================
问题背景 (Problem Statement)
================================================================================
这些环境源于控制论中的经典问题:

1. **CartPole**: 欠驱动系统控制。系统自由度大于控制输入数，需要利用
   系统动力学特性实现控制目标。

2. **MountainCar**: 稀疏奖励问题。随机策略几乎无法成功，代理必须
   学会利用动能势能转换的物理规律。

3. **Acrobot**: 双摆控制。只有一个关节有动力，需要学会利用摆动
   产生的角动量。

4. **Pendulum**: 连续控制。需要学习精确的扭矩输出以稳定摆杆。

================================================================================
算法对比 (Comparison)
================================================================================
| 环境           | 状态维度 | 动作空间 | 难度   | 适合算法     |
|----------------|----------|----------|--------|--------------|
| CartPole-v1    | 4        | 离散(2)  | 简单   | DQN, A2C     |
| MountainCar-v0 | 2        | 离散(3)  | 中等   | 需要探索机制 |
| Acrobot-v1     | 6        | 离散(3)  | 中等   | DQN, PPO     |
| Pendulum-v1    | 3        | 连续(1)  | 中等   | DDPG, SAC    |
| MountainCarC-v0| 2        | 连续(1)  | 中等   | DDPG, TD3    |

================================================================================
复杂度 (Complexity)
================================================================================
- 状态空间: O(1) 到 O(n)，所有环境都是低维的
- 动作空间: 离散环境 O(1)，连续环境需要函数逼近
- 训练时间: 通常在数百到数千回合内收敛

================================================================================
算法总结 (Summary)
================================================================================
经典控制环境是强化学习研究的基石:
1. 低维度使得算法快速验证成为可能
2. 明确的物理规律便于理解和调试
3. 成熟的基准性能便于算法比较
4. 为更复杂环境的研究打下基础

Author: Ziming Ding
Date: 2024
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    warnings.warn("gymnasium 未安装")

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyArrow
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
#                           数据结构定义
# ============================================================================

@dataclass
class EnvironmentDescription:
    """
    环境描述数据类

    Attributes:
        name: 环境名称
        env_id: Gymnasium 环境 ID
        state_space: 状态空间描述
        action_space: 动作空间描述
        reward_description: 奖励函数描述
        termination_conditions: 终止条件
        success_threshold: 成功阈值
        physics: 物理参数
    """
    name: str
    env_id: str
    state_space: Dict[str, str]
    action_space: Dict[str, str]
    reward_description: str
    termination_conditions: List[str]
    success_threshold: float
    physics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """
    训练结果数据类

    Attributes:
        episode_rewards: 每回合奖励
        episode_lengths: 每回合长度
        final_policy: 训练后的策略
        training_time: 训练时间
    """
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    final_policy: Optional[Callable] = None
    training_time: float = 0.0

    @property
    def mean_reward(self) -> float:
        """最后 100 回合平均奖励"""
        if len(self.episode_rewards) < 100:
            return np.mean(self.episode_rewards)
        return np.mean(self.episode_rewards[-100:])


# ============================================================================
#                           环境描述
# ============================================================================

CARTPOLE_DESCRIPTION = EnvironmentDescription(
    name="CartPole (倒立摆)",
    env_id="CartPole-v1",
    state_space={
        "cart_position": "小车位置 x ∈ [-4.8, 4.8]",
        "cart_velocity": "小车速度 ẋ ∈ (-∞, +∞)",
        "pole_angle": "摆杆角度 θ ∈ [-0.418, 0.418] rad (约 ±24°)",
        "pole_angular_velocity": "摆杆角速度 θ̇ ∈ (-∞, +∞)"
    },
    action_space={
        "0": "向左推小车",
        "1": "向右推小车"
    },
    reward_description="每步 +1，目标是尽可能长时间保持摆杆直立",
    termination_conditions=[
        "摆杆角度超过 ±12° (约 0.2095 rad)",
        "小车位置超过 ±2.4",
        "回合长度达到 500 步"
    ],
    success_threshold=475.0,
    physics={
        "gravity": 9.8,
        "cart_mass": 1.0,
        "pole_mass": 0.1,
        "pole_length": 0.5,
        "force_magnitude": 10.0
    }
)


MOUNTAIN_CAR_DESCRIPTION = EnvironmentDescription(
    name="MountainCar (爬山车)",
    env_id="MountainCar-v0",
    state_space={
        "position": "位置 x ∈ [-1.2, 0.6]",
        "velocity": "速度 v ∈ [-0.07, 0.07]"
    },
    action_space={
        "0": "向左加速",
        "1": "不加速",
        "2": "向右加速"
    },
    reward_description="每步 -1，到达山顶 (x >= 0.5) 奖励 0",
    termination_conditions=[
        "到达山顶 (position >= 0.5)",
        "回合长度达到 200 步"
    ],
    success_threshold=-110.0,  # 平均 110 步内到达
    physics={
        "min_position": -1.2,
        "max_position": 0.6,
        "max_speed": 0.07,
        "goal_position": 0.5,
        "goal_velocity": 0.0,
        "force": 0.001,
        "gravity": 0.0025
    }
)


ACROBOT_DESCRIPTION = EnvironmentDescription(
    name="Acrobot (双摆)",
    env_id="Acrobot-v1",
    state_space={
        "cos_theta1": "关节1角度的余弦 cos(θ₁)",
        "sin_theta1": "关节1角度的正弦 sin(θ₁)",
        "cos_theta2": "关节2角度的余弦 cos(θ₂)",
        "sin_theta2": "关节2角度的正弦 sin(θ₂)",
        "theta1_dot": "关节1角速度 θ̇₁",
        "theta2_dot": "关节2角速度 θ̇₂"
    },
    action_space={
        "0": "在关节2施加 -1 扭矩",
        "1": "在关节2施加 0 扭矩",
        "2": "在关节2施加 +1 扭矩"
    },
    reward_description="每步 -1，末端超过目标线奖励 0",
    termination_conditions=[
        "末端高度超过目标线",
        "回合长度达到 500 步"
    ],
    success_threshold=-100.0,
    physics={
        "link_length_1": 1.0,
        "link_length_2": 1.0,
        "link_mass_1": 1.0,
        "link_mass_2": 1.0,
        "link_com_pos_1": 0.5,
        "link_com_pos_2": 0.5
    }
)


PENDULUM_DESCRIPTION = EnvironmentDescription(
    name="Pendulum (单摆)",
    env_id="Pendulum-v1",
    state_space={
        "cos_theta": "角度的余弦 cos(θ)",
        "sin_theta": "角度的正弦 sin(θ)",
        "theta_dot": "角速度 θ̇ ∈ [-8, 8]"
    },
    action_space={
        "torque": "施加的扭矩 u ∈ [-2, 2]"
    },
    reward_description="r = -(θ² + 0.1*θ̇² + 0.001*u²), θ ∈ [-π, π]",
    termination_conditions=[
        "回合长度达到 200 步 (无提前终止)"
    ],
    success_threshold=-200.0,
    physics={
        "max_speed": 8.0,
        "max_torque": 2.0,
        "dt": 0.05,
        "gravity": 10.0,
        "mass": 1.0,
        "length": 1.0
    }
)


# ============================================================================
#                           策略实现
# ============================================================================

class BasePolicy(ABC):
    """策略基类"""

    @abstractmethod
    def __call__(self, observation: np.ndarray) -> Union[int, np.ndarray]:
        """根据观测选择动作"""
        pass

    @abstractmethod
    def description(self) -> str:
        """策略描述"""
        pass


class CartPolePolicy(BasePolicy):
    """
    CartPole 环境策略集合

    提供多种不同复杂度的策略实现，用于演示和比较。
    """

    def __init__(self, method: str = "angle"):
        """
        初始化策略

        Parameters
        ----------
        method : str
            策略方法: "random", "angle", "pid", "linear"
        """
        self.method = method

        # PID 控制器参数
        self.kp_theta = 100.0
        self.kd_theta = 20.0
        self.kp_x = 1.0
        self.kd_x = 5.0

        # 线性策略权重 (通过简单优化得到)
        self.linear_weights = np.array([0.1, 0.5, 1.0, 0.5])

    def __call__(self, obs: np.ndarray) -> int:
        """选择动作"""
        if self.method == "random":
            return np.random.randint(2)

        elif self.method == "angle":
            # 简单角度策略
            pole_angle = obs[2]
            return 1 if pole_angle > 0 else 0

        elif self.method == "pid":
            # PID 控制
            x, x_dot, theta, theta_dot = obs

            # 角度 PD
            theta_control = self.kp_theta * theta + self.kd_theta * theta_dot
            # 位置 PD
            x_control = self.kp_x * x + self.kd_x * x_dot

            # 综合控制信号
            u = theta_control + 0.1 * x_control

            return 1 if u > 0 else 0

        elif self.method == "linear":
            # 线性策略
            u = np.dot(self.linear_weights, obs)
            return 1 if u > 0 else 0

        else:
            raise ValueError(f"未知策略方法: {self.method}")

    def description(self) -> str:
        descriptions = {
            "random": "随机策略: 等概率选择左/右",
            "angle": "角度策略: 根据摆杆倾斜方向推动",
            "pid": "PID 控制: 综合角度和位置反馈",
            "linear": "线性策略: 状态的线性组合"
        }
        return descriptions.get(self.method, "未知策略")


class MountainCarPolicy(BasePolicy):
    """
    MountainCar 环境策略

    实现基于物理直觉的动量策略。
    """

    def __init__(self, method: str = "momentum"):
        self.method = method

    def __call__(self, obs: np.ndarray) -> int:
        """选择动作"""
        position, velocity = obs

        if self.method == "random":
            return np.random.randint(3)

        elif self.method == "momentum":
            # 动量策略: 与当前速度方向相同
            if velocity > 0:
                return 2  # 向右
            else:
                return 0  # 向左

        elif self.method == "energy":
            # 能量策略: 考虑势能
            # 在低位置时蓄能，在高位置时冲刺
            if position < -0.5:
                # 在左侧时交替蓄力
                return 2 if velocity > 0 else 0
            elif position > 0.2:
                # 接近目标时全力向右
                return 2
            else:
                # 中间区域跟随速度
                return 2 if velocity > 0 else 0

        else:
            raise ValueError(f"未知策略: {self.method}")

    def description(self) -> str:
        descriptions = {
            "random": "随机策略",
            "momentum": "动量策略: 跟随当前速度方向",
            "energy": "能量策略: 利用势能蓄力"
        }
        return descriptions.get(self.method, "未知策略")


class PendulumPolicy(BasePolicy):
    """
    Pendulum 环境连续控制策略

    实现 PD 控制器和能量控制器。
    """

    def __init__(self, method: str = "pd"):
        self.method = method
        self.kp = 10.0
        self.kd = 2.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """选择连续动作"""
        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)

        if self.method == "random":
            return np.array([np.random.uniform(-2, 2)])

        elif self.method == "pd":
            # PD 控制: u = -Kp*θ - Kd*θ̇
            torque = -self.kp * theta - self.kd * theta_dot
            return np.clip([torque], -2.0, 2.0)

        elif self.method == "energy":
            # 能量成形控制
            # 目标是将系统能量调节到直立位置的能量
            g, l, m = 10.0, 1.0, 1.0

            # 当前能量
            E = 0.5 * m * l**2 * theta_dot**2 - m * g * l * cos_theta
            # 目标能量 (直立位置)
            E_target = m * g * l

            # 能量误差
            E_error = E - E_target

            # 控制律: 当能量不足时加速，接近目标时用PD稳定
            if np.abs(theta) < 0.3:
                # 接近直立，用PD稳定
                torque = -self.kp * theta - self.kd * theta_dot
            else:
                # 能量泵浦
                torque = -5.0 * E_error * theta_dot

            return np.clip([torque], -2.0, 2.0)

        else:
            raise ValueError(f"未知策略: {self.method}")

    def description(self) -> str:
        descriptions = {
            "random": "随机策略",
            "pd": "PD 控制: 比例-微分反馈",
            "energy": "能量控制: 能量成形 + PD 稳定"
        }
        return descriptions.get(self.method, "未知策略")


# ============================================================================
#                           环境演示
# ============================================================================

def print_env_description(desc: EnvironmentDescription) -> None:
    """打印环境描述"""
    print(f"\n{'=' * 70}")
    print(f"{desc.name} ({desc.env_id})")
    print(f"{'=' * 70}")

    print("\n状态空间:")
    for key, value in desc.state_space.items():
        print(f"  - {key}: {value}")

    print("\n动作空间:")
    for key, value in desc.action_space.items():
        print(f"  - {key}: {value}")

    print(f"\n奖励设计:\n  {desc.reward_description}")

    print("\n终止条件:")
    for cond in desc.termination_conditions:
        print(f"  - {cond}")

    print(f"\n成功阈值: {desc.success_threshold}")

    if desc.physics:
        print("\n物理参数:")
        for key, value in desc.physics.items():
            print(f"  - {key}: {value}")


def run_demo(
    env_id: str,
    policy: BasePolicy,
    n_episodes: int = 5,
    max_steps: int = 500,
    render: bool = False,
    seed: Optional[int] = 42
) -> TrainingResult:
    """
    运行策略演示

    Parameters
    ----------
    env_id : str
        环境 ID
    policy : BasePolicy
        策略对象
    n_episodes : int
        演示回合数
    max_steps : int
        最大步数
    render : bool
        是否渲染
    seed : int
        随机种子

    Returns
    -------
    TrainingResult
        演示结果
    """
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装")
        return TrainingResult()

    env = gym.make(env_id, render_mode="human" if render else None)
    result = TrainingResult(final_policy=policy)

    print(f"\n策略: {policy.description()}")
    print(f"运行 {n_episodes} 回合...")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep if seed else None)
        episode_reward = 0.0
        steps = 0

        for step in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        result.episode_rewards.append(episode_reward)
        result.episode_lengths.append(steps)
        print(f"  回合 {ep + 1}: 奖励 = {episode_reward:.1f}, 步数 = {steps}")

    env.close()

    print(f"\n平均奖励: {np.mean(result.episode_rewards):.1f} ± {np.std(result.episode_rewards):.1f}")
    print(f"平均步数: {np.mean(result.episode_lengths):.1f}")

    return result


def compare_policies(
    env_id: str,
    policies: Dict[str, BasePolicy],
    n_episodes: int = 20,
    seed: int = 42
) -> Dict[str, TrainingResult]:
    """
    比较多个策略

    Parameters
    ----------
    env_id : str
        环境 ID
    policies : dict
        策略字典 {名称: 策略对象}
    n_episodes : int
        每个策略运行的回合数
    seed : int
        基础随机种子

    Returns
    -------
    dict
        各策略的结果字典
    """
    if not HAS_GYMNASIUM:
        print("gymnasium 未安装")
        return {}

    results = {}

    print(f"\n{'=' * 70}")
    print(f"策略比较: {env_id}")
    print(f"{'=' * 70}")

    env = gym.make(env_id)

    for name, policy in policies.items():
        print(f"\n[{name}] {policy.description()}")

        result = TrainingResult(final_policy=policy)

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            episode_reward = 0.0
            steps = 0

            while True:
                action = policy(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            result.episode_rewards.append(episode_reward)
            result.episode_lengths.append(steps)

        mean_reward = np.mean(result.episode_rewards)
        std_reward = np.std(result.episode_rewards)
        print(f"  平均奖励: {mean_reward:.1f} ± {std_reward:.1f}")

        results[name] = result

    env.close()

    # 打印比较结果
    print(f"\n{'=' * 70}")
    print("比较结果:")
    print(f"{'=' * 70}")
    print(f"{'策略':<20} {'平均奖励':>15} {'标准差':>12} {'平均步数':>12}")
    print("-" * 60)

    sorted_results = sorted(
        results.items(),
        key=lambda x: np.mean(x[1].episode_rewards),
        reverse=True
    )

    for name, result in sorted_results:
        mean_r = np.mean(result.episode_rewards)
        std_r = np.std(result.episode_rewards)
        mean_l = np.mean(result.episode_lengths)
        print(f"{name:<20} {mean_r:>15.1f} {std_r:>12.1f} {mean_l:>12.1f}")

    return results


# ============================================================================
#                           可视化工具
# ============================================================================

def visualize_cartpole_physics():
    """
    可视化 CartPole 物理系统

    绘制小车-摆杆系统的示意图和受力分析。
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 系统示意图
    ax1 = axes[0]
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1, 3)
    ax1.set_aspect('equal')
    ax1.set_title('CartPole System Diagram', fontsize=14)

    # 绘制地面
    ax1.axhline(y=0, color='brown', linewidth=3)
    ax1.fill_between([-3, 3], [-0.2, -0.2], [0, 0], color='brown', alpha=0.3)

    # 绘制小车
    cart = Rectangle((-0.5, 0), 1.0, 0.4, fill=True, color='blue', alpha=0.7)
    ax1.add_patch(cart)

    # 绘制轮子
    wheel1 = Circle((-0.3, 0), 0.1, fill=True, color='black')
    wheel2 = Circle((0.3, 0), 0.1, fill=True, color='black')
    ax1.add_patch(wheel1)
    ax1.add_patch(wheel2)

    # 绘制摆杆
    theta = 0.3  # 示例角度
    pole_length = 2.0
    pole_x = pole_length * np.sin(theta)
    pole_y = pole_length * np.cos(theta)
    ax1.plot([0, pole_x], [0.4, 0.4 + pole_y], 'r-', linewidth=8)

    # 绘制摆杆端点
    ax1.plot(pole_x, 0.4 + pole_y, 'ro', markersize=20)

    # 标注
    ax1.annotate('Cart (m_c)', (-0.3, 0.2), fontsize=10)
    ax1.annotate('Pole (m_p, l)', (pole_x/2 + 0.2, 0.4 + pole_y/2),
                 fontsize=10, color='red')

    # 绘制角度
    arc_angles = np.linspace(np.pi/2 - theta, np.pi/2, 20)
    arc_r = 0.8
    arc_x = arc_r * np.cos(arc_angles)
    arc_y = 0.4 + arc_r * np.sin(arc_angles)
    ax1.plot(arc_x, arc_y, 'g--', linewidth=2)
    ax1.annotate('θ', (0.3, 1.0), fontsize=14, color='green')

    # 绘制力
    ax1.annotate('', xy=(1.5, 0.2), xytext=(0.5, 0.2),
                 arrowprops=dict(arrowstyle='->', color='orange', lw=3))
    ax1.annotate('F', (1.0, 0.3), fontsize=12, color='orange')

    ax1.set_xlabel('Position (x)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 右图: 状态空间
    ax2 = axes[1]
    ax2.set_title('State Space Visualization', fontsize=14)

    # 状态变量说明
    state_vars = [
        ('x', '小车位置', '[-2.4, 2.4]'),
        ('ẋ', '小车速度', '[-∞, +∞]'),
        ('θ', '摆杆角度', '[-0.21, 0.21] rad'),
        ('θ̇', '摆杆角速度', '[-∞, +∞]')
    ]

    y_positions = [0.8, 0.6, 0.4, 0.2]

    for i, (var, desc, range_str) in enumerate(state_vars):
        ax2.text(0.1, y_positions[i], f'{var}:', fontsize=14,
                 fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.2, y_positions[i], f'{desc}', fontsize=12,
                 transform=ax2.transAxes)
        ax2.text(0.6, y_positions[i], f'{range_str}', fontsize=11,
                 transform=ax2.transAxes, color='gray')

    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('cartpole_physics.png', dpi=150, bbox_inches='tight')
    print("图像已保存: cartpole_physics.png")
    plt.close()


def visualize_mountain_car_landscape():
    """
    可视化 MountainCar 地形

    绘制山谷地形和位置-速度相图。
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 地形
    ax1 = axes[0]

    x = np.linspace(-1.2, 0.6, 200)
    y = np.sin(3 * x) * 0.45 + 0.55

    ax1.plot(x, y, 'b-', linewidth=3, label='Terrain')
    ax1.fill_between(x, 0, y, alpha=0.3, color='green')

    # 标记关键位置
    ax1.axvline(x=-0.5, color='red', linestyle='--', alpha=0.7, label='Start')
    ax1.axvline(x=0.5, color='gold', linestyle='--', alpha=0.7, label='Goal')

    # 绘制小车
    car_x = -0.5
    car_y = np.sin(3 * car_x) * 0.45 + 0.55
    ax1.plot(car_x, car_y + 0.05, 'ro', markersize=15, label='Car')

    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Height', fontsize=12)
    ax1.set_title('MountainCar Terrain', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1.3, 0.7)
    ax1.set_ylim(0, 1.2)

    # 右图: 相图
    ax2 = axes[1]

    # 绘制等能量线
    pos = np.linspace(-1.2, 0.6, 100)
    vel = np.linspace(-0.07, 0.07, 100)
    P, V = np.meshgrid(pos, vel)

    # 势能: 与高度成正比
    potential = np.sin(3 * P) * 0.45 + 0.55
    # 动能: 0.5 * v^2 (忽略质量)
    kinetic = 0.5 * V**2 * 100  # 缩放以便可视化
    # 总能量
    total_energy = potential + kinetic

    contour = ax2.contour(P, V, total_energy, levels=20, cmap='coolwarm')
    ax2.clabel(contour, inline=True, fontsize=8)

    # 标记目标区域
    ax2.axvline(x=0.5, color='gold', linestyle='--', linewidth=2, label='Goal')

    # 标记初始区域
    ax2.plot(-0.5, 0, 'ro', markersize=10, label='Start')

    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Velocity', fontsize=12)
    ax2.set_title('Phase Space (Energy Contours)', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mountaincar_landscape.png', dpi=150, bbox_inches='tight')
    print("图像已保存: mountaincar_landscape.png")
    plt.close()


def plot_policy_comparison(
    results: Dict[str, TrainingResult],
    env_name: str,
    save_path: Optional[str] = None
):
    """
    绘制策略比较图

    Parameters
    ----------
    results : dict
        策略结果字典
    env_name : str
        环境名称
    save_path : str, optional
        保存路径
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.tab10.colors

    # 左图: 奖励分布
    ax1 = axes[0]
    data = [r.episode_rewards for r in results.values()]
    labels = list(results.keys())

    bp = ax1.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title(f'{env_name}: Reward Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图: 平均性能条形图
    ax2 = axes[1]

    means = [np.mean(r.episode_rewards) for r in results.values()]
    stds = [np.std(r.episode_rewards) for r in results.values()]

    x = np.arange(len(labels))
    bars = ax2.bar(x, means, yerr=stds, capsize=5, color=colors[:len(labels)], alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_ylabel('Mean Reward', fontsize=12)
    ax2.set_title(f'{env_name}: Mean Performance', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")

    plt.close()


# ============================================================================
#                           单元测试
# ============================================================================

def run_tests() -> bool:
    """运行单元测试"""
    print("\n" + "=" * 60)
    print("运行单元测试")
    print("=" * 60)

    if not HAS_GYMNASIUM:
        print("gymnasium 未安装，跳过测试")
        return False

    all_passed = True

    # 测试 1: CartPole 策略
    print("\n[测试 1] CartPole 策略...")
    try:
        env = gym.make("CartPole-v1")

        for method in ["random", "angle", "pid", "linear"]:
            policy = CartPolePolicy(method=method)
            obs, _ = env.reset()

            for _ in range(10):
                action = policy(obs)
                assert action in [0, 1], f"无效动作: {action}"
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        env.close()
        print("  [通过] CartPole 策略正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 2: MountainCar 策略
    print("\n[测试 2] MountainCar 策略...")
    try:
        env = gym.make("MountainCar-v0")

        for method in ["random", "momentum", "energy"]:
            policy = MountainCarPolicy(method=method)
            obs, _ = env.reset()

            for _ in range(10):
                action = policy(obs)
                assert action in [0, 1, 2], f"无效动作: {action}"
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        env.close()
        print("  [通过] MountainCar 策略正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 3: Pendulum 策略
    print("\n[测试 3] Pendulum 策略...")
    try:
        env = gym.make("Pendulum-v1")

        for method in ["random", "pd", "energy"]:
            policy = PendulumPolicy(method=method)
            obs, _ = env.reset()

            for _ in range(10):
                action = policy(obs)
                assert action.shape == (1,), f"动作形状错误: {action.shape}"
                assert -2 <= action[0] <= 2, f"动作超出范围: {action}"
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

        env.close()
        print("  [通过] Pendulum 策略正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 4: TrainingResult 数据类
    print("\n[测试 4] TrainingResult 数据类...")
    try:
        result = TrainingResult()

        for i in range(100):
            result.episode_rewards.append(float(i))
            result.episode_lengths.append(i + 10)

        assert len(result.episode_rewards) == 100
        assert result.mean_reward == np.mean(range(100))

        print("  [通过] TrainingResult 正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 测试 5: 环境描述
    print("\n[测试 5] 环境描述...")
    try:
        assert CARTPOLE_DESCRIPTION.env_id == "CartPole-v1"
        assert MOUNTAIN_CAR_DESCRIPTION.env_id == "MountainCar-v0"
        assert PENDULUM_DESCRIPTION.env_id == "Pendulum-v1"
        assert ACROBOT_DESCRIPTION.env_id == "Acrobot-v1"

        print("  [通过] 环境描述正确")
    except Exception as e:
        print(f"  [失败] {e}")
        all_passed = False

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败")
    print("=" * 60)

    return all_passed


# ============================================================================
#                           主程序
# ============================================================================

def main():
    """主程序入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Gymnasium 经典控制环境详解"
    )

    parser.add_argument(
        '--env',
        type=str,
        choices=['cartpole', 'mountaincar', 'acrobot', 'pendulum', 'all'],
        default='all',
        help='选择环境'
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='运行策略演示'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='比较不同策略'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='生成可视化图表'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='运行单元测试'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='演示回合数'
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    # 环境描述映射
    env_descriptions = {
        'cartpole': CARTPOLE_DESCRIPTION,
        'mountaincar': MOUNTAIN_CAR_DESCRIPTION,
        'acrobot': ACROBOT_DESCRIPTION,
        'pendulum': PENDULUM_DESCRIPTION
    }

    # 打印环境描述
    envs_to_show = ['cartpole', 'mountaincar', 'acrobot', 'pendulum'] if args.env == 'all' else [args.env]

    for env_name in envs_to_show:
        print_env_description(env_descriptions[env_name])

    # 运行演示
    if args.demo:
        if args.env in ['cartpole', 'all']:
            print("\n\n" + "#" * 70)
            print("# CartPole 策略演示")
            print("#" * 70)

            policy = CartPolePolicy(method="pid")
            run_demo("CartPole-v1", policy, n_episodes=args.episodes)

        if args.env in ['mountaincar', 'all']:
            print("\n\n" + "#" * 70)
            print("# MountainCar 策略演示")
            print("#" * 70)

            policy = MountainCarPolicy(method="momentum")
            run_demo("MountainCar-v0", policy, n_episodes=args.episodes)

        if args.env in ['pendulum', 'all']:
            print("\n\n" + "#" * 70)
            print("# Pendulum 策略演示")
            print("#" * 70)

            policy = PendulumPolicy(method="energy")
            run_demo("Pendulum-v1", policy, n_episodes=args.episodes)

    # 策略比较
    if args.compare:
        if args.env in ['cartpole', 'all']:
            policies = {
                "Random": CartPolePolicy(method="random"),
                "Angle": CartPolePolicy(method="angle"),
                "PID": CartPolePolicy(method="pid"),
                "Linear": CartPolePolicy(method="linear")
            }
            results = compare_policies("CartPole-v1", policies, n_episodes=20)

            if HAS_MATPLOTLIB:
                plot_policy_comparison(results, "CartPole", "cartpole_comparison.png")

        if args.env in ['mountaincar', 'all']:
            policies = {
                "Random": MountainCarPolicy(method="random"),
                "Momentum": MountainCarPolicy(method="momentum"),
                "Energy": MountainCarPolicy(method="energy")
            }
            results = compare_policies("MountainCar-v0", policies, n_episodes=20)

            if HAS_MATPLOTLIB:
                plot_policy_comparison(results, "MountainCar", "mountaincar_comparison.png")

    # 生成可视化
    if args.visualize:
        print("\n生成可视化图表...")
        visualize_cartpole_physics()
        visualize_mountain_car_landscape()


if __name__ == "__main__":
    main()
