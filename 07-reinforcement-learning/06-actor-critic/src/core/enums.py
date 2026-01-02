"""
枚举类型定义模块 (Enumeration Types Module)

本模块定义策略梯度算法中使用的枚举类型，提供类型安全的配置选项。

核心思想 (Core Idea):
    使用枚举替代字符串常量，提供:
    1. 编译时类型检查
    2. IDE自动补全支持
    3. 防止拼写错误
    4. 集中管理配置选项

设计原则:
    - 每个枚举类代表一个配置维度
    - 枚举值使用描述性名称
    - 提供文档说明每个选项的含义和适用场景
"""

from __future__ import annotations

from enum import Enum, auto


class PolicyType(Enum):
    """
    策略网络类型枚举。

    核心思想 (Core Idea):
        不同的动作空间需要不同的策略参数化方式。离散动作使用Categorical分布，
        连续动作使用Gaussian分布，有界连续动作需要额外的squashing变换。

    数学原理 (Mathematical Theory):
        | 类型              | 分布                | 输出              | 适用场景          |
        |-------------------|---------------------|-------------------|-------------------|
        | DISCRETE          | Categorical(logits) | P(a|s)            | 离散动作空间      |
        | CONTINUOUS        | N(μ, σ²)            | μ(s), σ(s)        | 无界连续动作      |
        | SQUASHED_GAUSSIAN | tanh(N(μ, σ²))      | a ∈ [-1, 1]       | 有界连续动作      |

    Members
    -------
    DISCRETE : 离散动作策略
        使用Softmax输出动作概率分布，适用于有限离散动作空间。
        π_θ(a|s) = softmax(f_θ(s))_a

    CONTINUOUS : 连续动作策略
        使用Gaussian分布，输出均值和标准差。
        π_θ(a|s) = N(a | μ_θ(s), σ_θ(s)²)

    SQUASHED_GAUSSIAN : 压缩高斯策略
        Gaussian采样后通过tanh压缩到[-1, 1]，需要Jacobian校正。
        a = tanh(u), u ~ N(μ_θ(s), σ_θ(s)²)
        log π(a|s) = log N(u|μ,σ²) - Σ log(1 - a_i²)
    """

    DISCRETE = auto()
    CONTINUOUS = auto()
    SQUASHED_GAUSSIAN = auto()


class AdvantageEstimator(Enum):
    """
    优势函数估计方法枚举。

    核心思想 (Core Idea):
        优势函数 A(s,a) = Q(s,a) - V(s) 衡量动作相对于平均水平的好坏程度。
        不同估计方法在偏差(bias)和方差(variance)之间权衡。

    数学原理 (Mathematical Theory):
        偏差-方差权衡:
            - 高偏差: 估计系统性偏离真值，但稳定
            - 高方差: 估计波动大，但期望正确

        | 方法          | 偏差   | 方差   | 公式                                    |
        |---------------|--------|--------|----------------------------------------|
        | MONTE_CARLO   | 无     | 高     | A_t = G_t - V(s_t)                     |
        | TD_ERROR      | 高     | 低     | A_t = r_t + γV(s_{t+1}) - V(s_t)       |
        | N_STEP        | 中     | 中     | A_t = G_t^(n) - V(s_t)                 |
        | GAE           | 可调   | 可调   | A_t = Σ(γλ)^l δ_{t+l}                  |

    Members
    -------
    MONTE_CARLO : 蒙特卡洛估计
        使用完整回报 G_t = Σγ^k r_{t+k}
        无偏但高方差，需要完整episode

    TD_ERROR : 时序差分误差
        δ_t = r_t + γV(s_{t+1}) - V(s_t)
        低方差但有偏（依赖V的准确性）

    N_STEP : n步回报
        G_t^(n) = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})
        n越大越接近MC，n=1等于TD(0)

    GAE : 广义优势估计
        A_t^GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
        λ=0 等于TD，λ=1 等于MC
        推荐 λ=0.95 作为默认值
    """

    MONTE_CARLO = auto()
    TD_ERROR = auto()
    N_STEP = auto()
    GAE = auto()


class NetworkArchitecture(Enum):
    """
    网络架构类型枚举。

    核心思想 (Core Idea):
        Actor-Critic方法需要同时学习策略(Actor)和价值函数(Critic)。
        共享网络减少参数但可能导致任务干扰，分离网络更灵活但参数更多。

    架构对比:
        | 架构      | 参数量 | 特征共享 | 训练稳定性 | 适用场景        |
        |-----------|--------|----------|------------|-----------------|
        | SHARED    | ~1.2x  | 是       | 需要调参   | 简单任务        |
        | SEPARATE  | 2x     | 否       | 更稳定     | 复杂任务        |
        | DUAL_HEAD | ~1.5x  | 部分     | 平衡       | 大多数任务      |

    Members
    -------
    SHARED : 完全共享网络
        Actor和Critic共享所有隐藏层，只有输出层不同。
        优点: 参数效率高，特征复用
        缺点: 可能存在梯度干扰

    SEPARATE : 完全分离网络
        Actor和Critic使用独立的网络。
        优点: 独立优化，更稳定
        缺点: 参数量翻倍

    DUAL_HEAD : 双头网络
        共享底层特征提取，分离高层决策。
        优点: 平衡参数效率和独立性
        缺点: 需要设计共享层数
    """

    SHARED = auto()
    SEPARATE = auto()
    DUAL_HEAD = auto()


class OptimizationMethod(Enum):
    """
    策略优化方法枚举。

    核心思想 (Core Idea):
        策略梯度的核心挑战是确定合适的更新步长。过大导致策略崩溃，
        过小导致学习缓慢。不同方法采用不同策略控制更新幅度。

    数学原理 (Mathematical Theory):
        策略更新的一般形式:
            θ_{new} = θ_{old} + α · ∇_θ J(θ)

        关键问题: 如何选择 α 或等效地限制更新幅度?

        | 方法          | 约束方式                    | 特点                    |
        |---------------|----------------------------|-------------------------|
        | VANILLA_PG    | 固定学习率                  | 简单但不稳定            |
        | NATURAL_PG    | Fisher信息矩阵              | 理论优美但计算昂贵      |
        | TRPO          | KL散度硬约束                | 稳定但实现复杂          |
        | PPO_CLIP      | 比率裁剪                    | 简单高效，广泛使用      |
        | PPO_PENALTY   | KL散度软惩罚                | 自适应但需要调参        |

    Members
    -------
    VANILLA_PG : 原始策略梯度
        直接使用梯度上升，依赖学习率调度。

    NATURAL_PG : 自然策略梯度
        使用Fisher信息矩阵进行预条件化:
        θ ← θ + α · F^{-1} · ∇_θ J(θ)

    TRPO : 信任域策略优化
        在KL散度约束下最大化目标:
        max L(θ) s.t. KL(π_old || π_new) ≤ δ

    PPO_CLIP : PPO裁剪版本
        裁剪重要性采样比率:
        L = min(r·A, clip(r, 1-ε, 1+ε)·A)

    PPO_PENALTY : PPO惩罚版本
        添加KL散度惩罚项:
        L = r·A - β·KL(π_old || π_new)
    """

    VANILLA_PG = auto()
    NATURAL_PG = auto()
    TRPO = auto()
    PPO_CLIP = auto()
    PPO_PENALTY = auto()


class ExplorationStrategy(Enum):
    """
    探索策略枚举。

    核心思想 (Core Idea):
        强化学习需要平衡探索(exploration)和利用(exploitation)。
        策略梯度方法通过随机策略自然实现探索，但可能需要额外机制。

    Members
    -------
    ENTROPY_BONUS : 熵正则化
        在目标函数中添加策略熵:
        J(θ) = E[R] + c_ent · H(π)
        鼓励策略保持随机性

    NOISE_INJECTION : 噪声注入
        在动作或参数中添加噪声:
        - 动作噪声: a = π(s) + ε
        - 参数噪声: θ' = θ + ε

    INTRINSIC_REWARD : 内在奖励
        添加基于好奇心或新颖性的奖励:
        r_total = r_ext + β · r_int

    SCHEDULED_EXPLORATION : 计划探索
        随训练进度调整探索程度:
        - 温度退火
        - ε衰减
    """

    ENTROPY_BONUS = auto()
    NOISE_INJECTION = auto()
    INTRINSIC_REWARD = auto()
    SCHEDULED_EXPLORATION = auto()


class ValueTargetType(Enum):
    """
    价值函数训练目标类型枚举。

    核心思想 (Core Idea):
        价值函数的训练目标决定了学习的偏差-方差特性。
        不同目标适用于不同场景。

    Members
    -------
    MC_RETURN : 蒙特卡洛回报
        V_target = G_t = Σγ^k r_{t+k}
        无偏但高方差

    TD_TARGET : TD目标
        V_target = r + γV(s')
        低方差但有偏

    GAE_RETURN : GAE回报
        V_target = A^GAE + V(s)
        平衡偏差和方差

    LAMBDA_RETURN : λ回报
        V_target = (1-λ)Σλ^n G_t^(n)
        等价于GAE的价值目标
    """

    MC_RETURN = auto()
    TD_TARGET = auto()
    GAE_RETURN = auto()
    LAMBDA_RETURN = auto()
