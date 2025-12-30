"""
类型定义模块

定义DQN算法中使用的核心数据结构，包括转移元组和N步转移。
使用NamedTuple实现内存高效且不可变的数据存储。
"""

from typing import Any, NamedTuple
import numpy as np
from numpy.typing import NDArray


# 类型别名
FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.int64]


class Transition(NamedTuple):
    """
    单步转移数据结构
    
    ============================================================
    核心思想 (Core Idea)
    ============================================================
    表示MDP中单步交互的基本数据单元：τ = (s_t, a_t, r_t, s_{t+1}, done_t)
    
    ============================================================
    数学定义 (Mathematical Definition)
    ============================================================
    转移表示MDP的一步：
    
    .. math::
        \\tau_t = (s_t, a_t, r_t, s_{t+1}, d_t)
    
    其中：
    - s_t ∈ S: 当前状态
    - a_t ∈ A: 执行的动作
    - r_t ∈ ℝ: 即时奖励 r(s_t, a_t)
    - s_{t+1} ∈ S: 下一状态，来自转移动态 P(·|s_t, a_t)
    - d_t ∈ {0, 1}: 终止标志
    
    Attributes
    ----------
    state : FloatArray
        当前观测 s_t ∈ ℝ^d，形状 (state_dim,)
    action : int
        离散动作索引 a_t ∈ {0, ..., |A|-1}
    reward : float
        标量即时奖励 r_t
    next_state : FloatArray
        下一观测 s_{t+1} ∈ ℝ^d，形状 (state_dim,)
    done : bool
        回合终止标志（True表示终止状态）
    """
    state: FloatArray
    action: int
    reward: float
    next_state: FloatArray
    done: bool


class NStepTransition(NamedTuple):
    """
    N步转移数据结构
    
    ============================================================
    核心思想 (Core Idea)
    ============================================================
    存储n步累积回报用于多步学习。相比单步TD，n步方法在偏差与方差之间
    提供了更灵活的权衡，通常可以加速学习。
    
    ============================================================
    数学定义 (Mathematical Definition)
    ============================================================
    从时刻t开始的N步回报：
    
    .. math::
        G_t^{(n)} = \\sum_{k=0}^{n-1} \\gamma^k R_{t+k+1} + \\gamma^n V(S_{t+n})
    
    转移存储：
    - 初始状态 s_t 和动作 a_t
    - 累积折扣奖励：Σ_{k=0}^{n-1} γ^k r_{t+k+1}
    - 用于价值估计的bootstrap状态 s_{t+n}
    
    Attributes
    ----------
    state : FloatArray
        n步序列开始时的初始观测 s_t ∈ ℝ^d
    action : int
        在初始状态执行的动作 a_t
    n_step_return : float
        累积n步折扣回报：Σ_{k=0}^{n-1} γ^k r_{t+k+1}
    next_state : FloatArray
        用于bootstrap的状态 s_{t+n}（n步之后）
    done : bool
        如果回合在n步内终止则为True
    n_steps : int
        实际步数（如果回合提前结束可能 < n）
    """
    state: FloatArray
    action: int
    n_step_return: float
    next_state: FloatArray
    done: bool
    n_steps: int
