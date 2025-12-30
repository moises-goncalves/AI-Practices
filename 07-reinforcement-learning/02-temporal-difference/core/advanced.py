"""
高级TD算法模块 (Advanced TD Algorithms)
======================================

核心思想 (Core Idea):
--------------------
本模块包含TD学习的高级变体，解决基础算法的各种局限性:
- Double Q-Learning: 解决最大化偏差（过估计）问题
- N-Step TD: 在TD(0)和Monte Carlo之间权衡
- TD(λ): 通过资格迹统一所有n-step方法
- SARSA(λ): On-Policy版本的TD(λ)
- Watkins Q(λ): Off-Policy安全的TD(λ)

数学原理 (Mathematical Theory):
------------------------------
这些算法代表了TD学习在不同维度的扩展:

1. 偏差-方差权衡维度:
   TD(0) ←→ n-step TD ←→ TD(λ) ←→ Monte Carlo
   (高偏差低方差)              (低偏差高方差)

2. 过估计校正维度:
   Q-Learning → Double Q-Learning
   (有最大化偏差)   (无偏)

3. 信用分配维度:
   单步更新 → 资格迹（多步信用分配）

问题背景 (Problem Statement):
----------------------------
基础TD算法存在各种局限:
- Q-Learning在噪声环境中过估计Q值
- TD(0)只用单步信息，收敛可能较慢
- 信用分配范围有限

本模块的算法针对这些问题提供解决方案。

复杂度 (Complexity):
-------------------
- Double Q-Learning: O(|A|) time, O(2×|S|×|A|) space
- N-Step TD: O(1) time, O(n) buffer space
- TD(λ): O(|S|×|A|) time (需更新所有有资格的状态)
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

from .base import BaseTDLearner, State, Action
from .config import TDConfig, EligibilityTraceType


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
        max_a Q(s', a) 使用同一个Q来选择和评估

        当Q有噪声时: E[max_a Q(s,a)] ≥ max_a E[Q(s,a)]
        这导致系统性的过估计。

    过估计的直觉:
        想象你估计多个随机变量的最大值。
        即使每个估计都是无偏的，max操作会偏向选择
        那些恰好被高估的变量，导致整体过估计。

    Double Q-Learning解决方案:
        以50%概率选择更新Q_A或Q_B:

        更新Q_A时:
            a* = argmax_a Q_A(S', a)           # 用Q_A选择最优动作
            Q_A(S, A) ← Q_A + α[R + γQ_B(S', a*) - Q_A]  # 用Q_B评估

        更新Q_B时:
            a* = argmax_a Q_B(S', a)           # 用Q_B选择最优动作
            Q_B(S, A) ← Q_B + α[R + γQ_A(S', a*) - Q_B]  # 用Q_A评估

    为什么有效:
        关键洞察: E[max(X, Y)] ≥ max(E[X], E[Y]) (Jensen不等式)

        Q_A和Q_B是独立学习的，它们的噪声不相关。
        当用Q_A选择动作时，即使选到了Q_A高估的动作，
        Q_B对该动作的估计（独立噪声）不会同样高估，
        因此期望是无偏的。

    问题背景 (Problem Statement):
    ----------------------------
    在随机环境中，Q-Learning会系统性地过估计Q值。
    经典例子：在一个状态有两个动作，每个动作的真实价值都是0，
    但有随机噪声。Q-Learning会选择估计较高的那个，
    导致该状态的max Q > 0，产生过估计。

    Double Q-Learning通过使用独立的Q表进行选择和评估，
    打破了这种正向偏差，获得无偏的估计。

    算法对比 (Comparison):
    ---------------------
    ┌──────────────────┬────────────┬────────────┬────────────┐
    │      算法        │   偏差     │   方差     │   内存     │
    ├──────────────────┼────────────┼────────────┼────────────┤
    │   Q-Learning     │  过估计    │    中      │    1×      │
    │ Double Q-Learning│   无偏     │    中      │    2×      │
    └──────────────────┴────────────┴────────────┴────────────┘

    复杂度 (Complexity):
    -------------------
    - 时间复杂度: O(|A|) per step
    - 空间复杂度: O(2 × |S| × |A|) for two Q-tables

    算法总结 (Summary):
    -----------------
    Double Q-Learning是Q-Learning的去偏差版本。通过维护两个Q表
    并随机选择哪个用于选择、哪个用于评估，它消除了max操作引入的
    系统性过估计。代价是双倍的内存消耗。这一思想后来被DQN采用
    (Double DQN)，成为深度强化学习的标准技术。

    Example:
        >>> config = TDConfig(alpha=0.5, gamma=0.99, epsilon=0.1)
        >>> double_q = DoubleQLearning(config)
        >>> # 在噪声环境中，Double Q-Learning的估计更准确
    """

    def __init__(self, config: TDConfig) -> None:
        """初始化Double Q-Learning，创建两个独立的Q表。"""
        super().__init__(config)

        # 两个独立的Q表
        self._q_function_a: Dict[Tuple[State, Action], float] = defaultdict(
            lambda: config.initial_value
        )
        self._q_function_b: Dict[Tuple[State, Action], float] = defaultdict(
            lambda: config.initial_value
        )

    def get_q_value(self, state: State, action: Action) -> float:
        """
        获取合并后的Q值（两个Q表的平均）。

        用于策略选择时的Q值评估。
        """
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

        以50%概率选择更新Q_A或Q_B，交叉使用另一个Q表进行评估。

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
        update_a = np.random.random() < 0.5

        if update_a:
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
            # 用另一个Q表评估该动作
            td_target = reward + self.config.gamma * q_eval[(next_state, best_action)]

        td_error = td_target - current_q
        q_update[(state, action)] += self.config.alpha * td_error

        return td_error


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

    对于Q函数（n-step SARSA风格）:
        G_t^{(n)} = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n Q(S_{t+n}, A_{t+n})

    关键洞察:
        当n→∞，G_t^{(n)}变成完整的Monte Carlo回报
        当n=1，G_t^{(n)}就是TD(0)目标
        n-step TD提供了平滑的过渡

    偏差-方差权衡:
        较小的n: 更多自举，偏差大但方差小
        较大的n: 更多实际奖励，偏差小但方差大

    问题背景 (Problem Statement):
    ----------------------------
    TD(0)偏差高、方差低；Monte Carlo偏差低、方差高。
    N-Step TD提供了一种在两者之间平滑过渡的方式，
    允许根据问题特性选择合适的n值。

    实践中，最优n通常在4-10之间，需要针对具体任务调优。

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

    复杂度 (Complexity):
    -------------------
    - 时间复杂度: O(1) per step (摊销)
    - 空间复杂度: O(n) for storing n-step buffer

    算法总结 (Summary):
    -----------------
    N-Step TD通过调整n值在偏差和方差之间权衡。较小的n更新更频繁但偏差大，
    较大的n能利用更多真实奖励信息但需要等待更长时间。
    它是理解TD(λ)的基础——TD(λ)本质上是对所有n-step回报的加权组合。

    Example:
        >>> config = TDConfig(alpha=0.5, gamma=0.99, n_step=3)
        >>> n_step_td = NStepTD(config)
        >>> metrics = n_step_td.train(env, n_episodes=500)
    """

    def __init__(self, config: TDConfig) -> None:
        """初始化N-Step TD。"""
        super().__init__(config)
        # n步经验缓冲区: [(state, action, reward), ...]
        self._buffer: List[Tuple[State, Action, float]] = []
        self._states_buffer: List[State] = []

    def on_episode_start(self, episode: int) -> None:
        """回合开始时清空缓冲区。"""
        self._buffer.clear()
        self._states_buffer.clear()

    def _compute_n_step_return(
        self,
        rewards: List[float],
        final_state: State,
        done: bool
    ) -> float:
        """
        计算n-step回报。

        G_t^{(n)} = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})

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

        # 如果未终止，加上自举项
        if not done:
            # 使用状态的最大Q值作为价值估计
            max_q = max(
                self._q_function[(final_state, a)]
                for a in self._action_space
            ) if self._action_space else 0.0
            n_step_return += discount * max_q

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

        将经验存入缓冲区，当缓冲区满或回合结束时更新Q值。

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
            if self._states_buffer:
                self._states_buffer.pop(0)

        # 回合结束时更新缓冲区中剩余的状态
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

        return td_error


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

    这是所有n-step回报的几何加权平均:
    - G_t^{(1)} 权重: (1-λ)
    - G_t^{(2)} 权重: (1-λ)λ
    - G_t^{(n)} 权重: (1-λ)λ^{n-1}
    - 权重和: (1-λ)(1 + λ + λ² + ...) = 1

    资格迹 (Backward View):
        提供了高效实现λ-回报的方法。

        累积迹 (Accumulating Trace):
            E_t(s) = γλE_{t-1}(s) + 𝟙(S_t = s)
            每次访问状态时迹值累加。

        替换迹 (Replacing Trace):
            E_t(s) = γλE_{t-1}(s) if s ≠ S_t
            E_t(S_t) = 1
            访问时重置为1，避免累积过大。

        荷兰迹 (Dutch Trace):
            E_t(s) = γλE_{t-1}(s) + (1 - αγλE_{t-1}(s))𝟙(S_t = s)
            在函数逼近下有更好的理论保证。

    更新规则:
        δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
        V(s) ← V(s) + αδ_t E_t(s), ∀s

    前向视图与后向视图等价性:
        在离线更新（回合结束后批量更新）下，两种视图产生相同的更新量。
        资格迹提供了在线、增量式的实现。

    问题背景 (Problem Statement):
    ----------------------------
    N-Step TD需要选择特定的n值，不同的n在不同环境中表现差异大。
    TD(λ)通过资格迹实现对所有n的加权组合，由单一参数λ控制:
    - λ=0: 等价于TD(0)，只看一步
    - λ=1: 等价于Monte Carlo，看完整回合
    - 0<λ<1: 两者的混合，通常λ=0.9是好的起点

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

    资格迹类型对比:
    - 累积迹: 经典方法，但在频繁重访时可能不稳定
    - 替换迹: 在部分环境中更稳定，但理论保证较弱
    - 荷兰迹: 推荐用于函数逼近，理论和实践表现都好

    复杂度 (Complexity):
    -------------------
    - 时间复杂度: O(|S|×|A|) per step (需要更新所有有资格的状态)
    - 空间复杂度: O(|S|×|A|) for eligibility traces

    算法总结 (Summary):
    -----------------
    TD(λ)是TD学习的统一框架。通过资格迹，它在每一步将TD误差
    分配给所有最近访问的状态，分配量随时间和λ指数衰减。
    这巧妙地组合了所有n-step方法的优点，用单一参数λ控制权衡。
    在实践中，λ=0.9通常是一个好的起点。

    Example:
        >>> config = TDConfig(alpha=0.1, gamma=0.99, lambda_=0.9)
        >>> td_lambda = TDLambda(config)
        >>> metrics = td_lambda.train(env, n_episodes=500)
    """

    def __init__(self, config: TDConfig) -> None:
        """初始化TD(λ)。"""
        super().__init__(config)
        # 资格迹: (state, action) -> trace value
        self._eligibility_traces: Dict[Tuple[State, Action], float] = defaultdict(float)

    def on_episode_start(self, episode: int) -> None:
        """回合开始时清空资格迹。"""
        self._eligibility_traces.clear()

    def _update_traces(self, state: State, action: Action) -> None:
        """
        更新资格迹。

        根据配置的迹类型执行不同的更新规则。

        Args:
            state: 当前状态
            action: 当前动作
        """
        gamma_lambda = self.config.gamma * self.config.lambda_

        # 衰减所有现有的资格迹
        keys_to_remove = []
        for key in self._eligibility_traces:
            self._eligibility_traces[key] *= gamma_lambda
            # 清除过小的迹以节省内存和计算
            if self._eligibility_traces[key] < 1e-8:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._eligibility_traces[key]

        # 更新当前状态-动作的资格迹
        if self.config.trace_type == EligibilityTraceType.ACCUMULATING:
            # 累积迹: E(s,a) += 1
            self._eligibility_traces[(state, action)] += 1.0

        elif self.config.trace_type == EligibilityTraceType.REPLACING:
            # 替换迹: E(s,a) = 1
            self._eligibility_traces[(state, action)] = 1.0

        elif self.config.trace_type == EligibilityTraceType.DUTCH:
            # 荷兰迹: E(s,a) = (1-α)γλE(s,a) + 1
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

        流程:
        1. 计算TD误差δ
        2. 更新当前状态-动作的资格迹
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
        # 计算TD目标
        if done:
            td_target = reward
        else:
            if next_action is None:
                # Q-Learning风格: 使用max
                max_next_q = max(
                    self._q_function[(next_state, a)]
                    for a in self._action_space
                )
                td_target = reward + self.config.gamma * max_next_q
            else:
                # SARSA风格: 使用实际动作
                td_target = reward + self.config.gamma * self._q_function[(next_state, next_action)]

        # 计算TD误差
        current_q = self._q_function[(state, action)]
        td_error = td_target - current_q

        # 更新资格迹（在计算误差之后）
        self._update_traces(state, action)

        # 使用资格迹更新所有相关的Q值
        for (s, a), trace in self._eligibility_traces.items():
            self._q_function[(s, a)] += self.config.alpha * td_error * trace

        # 回合结束时清空资格迹
        if done:
            self._eligibility_traces.clear()

        return td_error


class SARSALambda(TDLambda):
    """
    SARSA(λ)算法实现。

    核心思想 (Core Idea):
    --------------------
    SARSA(λ)是SARSA与资格迹的结合。它是On-Policy的TD(λ)控制算法，
    使用实际下一动作计算TD目标，同时通过资格迹实现多步信用分配。

    数学原理 (Mathematical Theory):
    ------------------------------
    TD误差 (SARSA风格):
        δ_t = R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)

    资格迹更新:
        E_t(s, a) = γλE_{t-1}(s, a) + 𝟙(S_t=s, A_t=a)

    Q值更新:
        Q(s, a) ← Q(s, a) + αδ_t E_t(s, a), ∀s, a

    与SARSA的关系:
        当λ=0时，退化为SARSA（单步更新）
        当λ=1时，变成完整回合的On-Policy更新

    问题背景 (Problem Statement):
    ----------------------------
    SARSA的更新仅依赖单步信息，信用分配范围有限。
    SARSA(λ)通过资格迹将TD误差传播到所有最近访问的状态-动作对，
    实现更高效的学习，同时保持On-Policy的安全性特点。

    复杂度 (Complexity):
    -------------------
    - 时间复杂度: O(|S|×|A|) per step (更新所有有资格的状态)
    - 空间复杂度: O(|S|×|A|) for eligibility traces

    算法总结 (Summary):
    -----------------
    SARSA(λ)结合了SARSA的On-Policy特性和资格迹的高效信用分配。
    它保持了SARSA的安全性（考虑探索风险），同时通过多步传播加速学习。
    适合需要安全探索且状态空间较大的环境。

    Example:
        >>> config = TDConfig(alpha=0.1, gamma=0.99, lambda_=0.9, epsilon=0.1)
        >>> sarsa_lambda = SARSALambda(config)
        >>> metrics = sarsa_lambda.train(env, n_episodes=500)
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

        强制使用SARSA风格的TD目标（实际下一动作）。

        Args:
            state: 当前状态
            action: 当前动作
            reward: 即时奖励
            next_state: 下一状态
            next_action: 下一动作（SARSA必需）
            done: 是否终止

        Returns:
            TD误差

        Raises:
            ValueError: 当非终止状态缺少next_action时
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


class WatkinsQLambda(TDLambda):
    """
    Watkins's Q(λ)算法实现。

    核心思想 (Core Idea):
    --------------------
    Watkins's Q(λ)是Q-Learning与资格迹的结合，但有一个关键特点：
    当采取非贪婪动作（探索）时，资格迹被清零。这确保了算法在
    Off-Policy设置下的收敛性。

    数学原理 (Mathematical Theory):
    ------------------------------
    TD误差 (Q-Learning风格):
        δ_t = R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)

    资格迹更新 (关键区别):
        如果 A_{t+1} = argmax_a Q(S_{t+1}, a) (贪婪动作):
            E_t(s, a) = γλE_{t-1}(s, a) + 𝟙(S_t=s, A_t=a)
        否则 (探索动作):
            E_t(s, a) = 0  ∀s, a  (清零所有迹!)
            然后 E_t(S_t, A_t) = 1

    为什么清零资格迹:
        Q-Learning的TD目标假设后续动作都是贪婪的。
        当实际采取探索动作时，这个假设被打破。
        如果继续传播TD误差到更早的状态，会引入偏差，
        可能导致不收敛。清零资格迹切断错误的信用分配链。

    问题背景 (Problem Statement):
    ----------------------------
    简单地将资格迹加入Q-Learning会导致在Off-Policy设置下不收敛。
    因为Q-Learning假设目标策略是贪婪的，但资格迹传播的是
    行为策略（含探索）的经验。

    Watkins's Q(λ)通过在探索时切断资格迹来解决这一问题。
    缺点是在高探索率下，资格迹经常被清零，退化为接近Q-Learning。

    算法对比 (Comparison):
    ---------------------
    ┌────────────────┬─────────────────┬─────────────────┐
    │     算法       │   探索时的迹    │    收敛性       │
    ├────────────────┼─────────────────┼─────────────────┤
    │   Q(λ) naive   │     保留        │    不保证       │
    │  Watkins Q(λ)  │     清零        │    保证         │
    │   SARSA(λ)     │     保留        │    保证*        │
    └────────────────┴─────────────────┴─────────────────┘
    * SARSA(λ)是On-Policy的，不存在这个问题

    复杂度 (Complexity):
    -------------------
    - 时间复杂度: O(|S|×|A|) per step (最坏情况)
    - 空间复杂度: O(|S|×|A|) for eligibility traces

    算法总结 (Summary):
    -----------------
    Watkins's Q(λ)在Off-Policy学习中安全地使用资格迹。
    代价是当探索动作发生时，无法利用之前的经验进行信用分配。
    在低ε设置下效果较好，高ε时退化为近似Q-Learning。

    Example:
        >>> config = TDConfig(alpha=0.1, gamma=0.99, lambda_=0.9, epsilon=0.05)
        >>> watkins_q = WatkinsQLambda(config)
        >>> # 低探索率下效果最佳
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

        使用Q-Learning目标，在探索动作时清零资格迹。

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
