# 时序差分学习 (Temporal Difference Learning)

## 目录

1. [核心概念](#1-核心概念)
2. [数学基础](#2-数学基础)
3. [TD(0)算法](#3-td0算法)
4. [SARSA算法](#4-sarsa算法)
5. [Q-Learning算法](#5-q-learning算法)
6. [Expected SARSA](#6-expected-sarsa)
7. [Double Q-Learning](#7-double-q-learning)
8. [N-Step TD](#8-n-step-td)
9. [TD(λ)与资格迹](#9-tdλ与资格迹)
10. [算法对比与选择](#10-算法对比与选择)
11. [超参数调优](#11-超参数调优)
12. [代码速查](#12-代码速查)

---

## 1. 核心概念

### 1.1 什么是时序差分学习？

时序差分(TD)学习是强化学习中最核心的思想之一，它结合了：
- **蒙特卡洛方法**的采样思想（从经验中学习，无需环境模型）
- **动态规划**的自举思想（用估计值更新估计值）

### 1.2 自举 (Bootstrapping)

> **核心洞察**：用"猜测"来更新"猜测"

TD方法的精髓在于**自举**——不需要等待完整的回报，而是用下一状态的价值估计来更新当前状态的价值估计。

```
传统方法：等到游戏结束，根据最终结果调整估计
TD方法：  走一步，就用新状态的估计来调整旧状态的估计
```

### 1.3 三种学习范式对比

| 方法 | 更新时机 | 使用信息 | 优点 | 缺点 |
|------|----------|----------|------|------|
| **动态规划** | 每步 | 完整环境模型 | 理论完美 | 需要模型 |
| **蒙特卡洛** | 回合结束 | 完整回报 | 无偏 | 高方差、需等待 |
| **时序差分** | 每步 | 单步奖励+估计 | 在线、低方差 | 有偏差 |

---

## 2. 数学基础

### 2.1 价值函数

**状态价值函数** $V^\pi(s)$：
$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s\right]$$

**动作价值函数** $Q^\pi(s, a)$：
$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a\right]$$

### 2.2 Bellman方程

**Bellman期望方程**（评估策略π）：
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')]$$

**Bellman最优方程**（寻找最优策略）：
$$Q^*(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \max_{a'} Q^*(s',a')]$$

### 2.3 TD目标与TD误差

**TD目标 (TD Target)**：
$$\text{TD Target} = R_{t+1} + \gamma V(S_{t+1})$$

**TD误差 (TD Error)**：
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

> TD误差表示"新信息"与"旧估计"的差距，是学习的驱动力。

---

## 3. TD(0)算法

### 3.1 算法描述

TD(0)是最简单的TD方法，用于**策略评估**（给定策略π，估计$V^\pi$）。

**更新规则**：
$$V(S_t) \leftarrow V(S_t) + \alpha \cdot \delta_t$$
$$V(S_t) \leftarrow V(S_t) + \alpha \left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]$$

### 3.2 算法流程

```
输入: 策略π，学习率α，折扣因子γ
初始化: V(s) 任意，V(终止状态) = 0

对于每个回合:
    初始化状态 S
    循环直到 S 是终止状态:
        A ← 根据π选择动作
        执行A，观察 R, S'
        V(S) ← V(S) + α[R + γV(S') - V(S)]
        S ← S'
```

### 3.3 收敛性

在以下条件下，TD(0)以概率1收敛到$V^\pi$：
1. 策略π固定
2. 学习率满足：$\sum_t \alpha_t = \infty$ 且 $\sum_t \alpha_t^2 < \infty$

---

## 4. SARSA算法

### 4.1 算法描述

SARSA是**On-Policy TD控制**算法。名称来源于更新所需的五元组：
**(S**tate, **A**ction, **R**eward, **S**tate, **A**ction)

**更新规则**：
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\right]$$

### 4.2 关键特点

- **On-Policy**：学习的是实际执行的策略（包含探索）
- 使用**实际下一动作** $A_{t+1}$（由当前策略选择）
- 学到的策略考虑了探索时的风险
- 在危险环境中倾向于**保守、安全的策略**

### 4.3 算法流程

```
初始化: Q(s,a) 任意，Q(终止状态, ·) = 0

对于每个回合:
    初始化 S
    A ← ε-greedy(Q, S)

    循环直到 S 是终止状态:
        执行A，观察 R, S'
        A' ← ε-greedy(Q, S')  # 关键：先选择下一动作
        Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
        S ← S', A ← A'
```

---

## 5. Q-Learning算法

### 5.1 算法描述

Q-Learning是**Off-Policy TD控制**算法，直接学习最优动作价值函数$Q^*$。

**更新规则**：
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\right]$$

### 5.2 关键特点

- **Off-Policy**：学习最优策略，与行为策略无关
- 使用**最大Q值** $\max_a Q(S_{t+1}, a)$
- 假设后续动作都是贪婪的
- 学到的是**最优但可能危险的策略**

### 5.3 与SARSA的核心区别

| 特性 | SARSA | Q-Learning |
|------|-------|------------|
| 类型 | On-Policy | Off-Policy |
| TD目标 | $R + \gamma Q(S', A')$ | $R + \gamma \max_a Q(S', a)$ |
| 下一动作 | 实际采样的$A'$ | 假设贪婪选择 |
| 学习目标 | 当前策略的价值 | 最优策略的价值 |
| 安全性 | 考虑探索风险 | 不考虑探索风险 |

### 5.4 Cliff Walking示例

在悬崖行走环境中：
- **Q-Learning**：学到沿悬崖边缘的最短路径（最优但危险）
- **SARSA**：学到远离悬崖的安全路径（次优但安全）

---

## 6. Expected SARSA

### 6.1 算法描述

Expected SARSA使用Q值的**期望**而非单一采样：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \mathbb{E}_\pi[Q(S_{t+1}, A)] - Q(S_t, A_t)\right]$$

对于ε-greedy策略：
$$\mathbb{E}_\pi[Q(S', A)] = \frac{\epsilon}{|A|} \sum_a Q(S', a) + (1-\epsilon) \max_a Q(S', a)$$

### 6.2 优势

- 消除了动作采样带来的**方差**
- 学习更稳定
- 当ε=0时，退化为Q-Learning

---

## 7. Double Q-Learning

### 7.1 最大化偏差问题

Q-Learning的max操作会导致**系统性过估计**：

$$\mathbb{E}[\max_a Q(s,a)] \geq \max_a \mathbb{E}[Q(s,a)]$$

当Q值有噪声时，max总是倾向于选择估计值偏高的动作。

### 7.2 解决方案

维护两个独立的Q表，**解耦选择与评估**：

```
以50%概率:
    a* = argmax_a Q_A(S', a)      # 用Q_A选择
    Q_A(S,A) ← Q_A + α[R + γQ_B(S', a*) - Q_A]  # 用Q_B评估
否则:
    a* = argmax_a Q_B(S', a)      # 用Q_B选择
    Q_B(S,A) ← Q_B + α[R + γQ_A(S', a*) - Q_B]  # 用Q_A评估
```

### 7.3 为什么有效？

用独立的估计器评估选中的动作，避免了同一噪声源的双重影响。

---

## 8. N-Step TD

### 8.1 算法描述

N-Step TD是TD(0)和Monte Carlo的中间方案：

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

### 8.2 不同n值的特性

| n值 | 偏差 | 方差 | 更新延迟 | 等价于 |
|-----|------|------|----------|--------|
| 1 | 高 | 低 | 1步 | TD(0) |
| 5 | 中 | 中 | 5步 | - |
| 100 | 低 | 高 | 100步 | - |
| ∞ | 无 | 高 | 整个回合 | MC |

### 8.3 最优n值

实践中，最优n通常在**4-10**之间，需要根据具体任务调整。

---

## 9. TD(λ)与资格迹

### 9.1 核心思想

TD(λ)不选择特定的n，而是对**所有n-step回报做几何加权平均**：

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

### 9.2 资格迹 (Eligibility Traces)

资格迹提供了高效的实现方式：

**累积迹**：
$$E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbf{1}(S_t = s)$$

**更新规则**：
$$V(s) \leftarrow V(s) + \alpha \delta_t E_t(s), \quad \forall s$$

### 9.3 λ值的意义

| λ值 | 等价于 | 偏差 | 方差 |
|-----|--------|------|------|
| 0 | TD(0) | 高 | 低 |
| 0.5 | 混合 | 中 | 中 |
| 0.9 | 接近MC | 低 | 较高 |
| 1 | MC | 无 | 高 |

### 9.4 资格迹类型

1. **累积迹**：$E(s) \leftarrow \gamma\lambda E(s) + 1$
2. **替换迹**：$E(s) \leftarrow 1$（访问时重置）
3. **荷兰迹**：$E(s) \leftarrow (1-\alpha)\gamma\lambda E(s) + 1$

---

## 10. 算法对比与选择

### 10.1 选择指南

| 场景 | 推荐算法 | 原因 |
|------|----------|------|
| 需要安全探索 | SARSA | 考虑探索风险 |
| 从历史数据学习 | Q-Learning | Off-policy特性 |
| 需要低方差 | Expected SARSA | 消除采样方差 |
| 噪声环境 | Double Q-Learning | 避免过估计 |
| 需要快速信用分配 | TD(λ) | 多步传播 |
| 在线学习 | TD方法 | 每步更新 |
| 回合很短 | MC方法 | 简单无偏 |

### 10.2 偏差-方差权衡

```
高偏差、低方差 ←————————→ 低偏差、高方差
    TD(0)    N-Step TD   TD(λ)    MC
```

### 10.3 On-Policy vs Off-Policy

| 特性 | On-Policy | Off-Policy |
|------|-----------|------------|
| 代表 | SARSA | Q-Learning |
| 数据来源 | 必须是当前策略 | 可以是任意策略 |
| 收敛目标 | 当前策略的价值 | 最优策略的价值 |
| 样本效率 | 较低 | 较高 |
| 稳定性 | 较高 | 可能不稳定 |

---

## 11. 超参数调优

### 11.1 学习率 α

| α值 | 效果 | 建议 |
|-----|------|------|
| 太小 (< 0.01) | 收敛慢 | 需要更多回合 |
| 适中 (0.1-0.5) | 平衡 | 推荐起始值 |
| 太大 (> 0.5) | 震荡 | 可能不收敛 |

**实践建议**：从0.1开始，观察收敛曲线调整

### 11.2 折扣因子 γ

| γ值 | 视野 | 适用场景 |
|-----|------|----------|
| 0 | 只看即时奖励 | 贪婪任务 |
| 0.9 | 中等视野 | 一般任务 |
| 0.99 | 长远视野 | 稀疏奖励 |
| 1 | 无限视野 | 回合制任务 |

**实践建议**：大多数任务用0.99

### 11.3 探索率 ε

| ε值 | 探索程度 | 效果 |
|-----|----------|------|
| 0.01 | 很少探索 | 可能陷入局部最优 |
| 0.1 | 适度探索 | 推荐值 |
| 0.3 | 大量探索 | 学习效率低 |

**实践建议**：
- 初始ε较大（如0.3）
- 随训练逐渐减小
- 最终保持小值（如0.01）

### 11.4 TD(λ)的λ值

| λ值 | 特点 | 适用场景 |
|-----|------|----------|
| 0 | 纯TD | 高噪声环境 |
| 0.5 | 平衡 | 一般情况 |
| 0.9 | 常用起点 | 大多数任务 |
| 1 | 纯MC | 回合短 |

---

## 12. 代码速查

### 12.1 快速开始

```python
from td_algorithms import create_td_learner, TDConfig
from environments import CliffWalkingEnv

# 创建环境
env = CliffWalkingEnv()

# 配置超参数
config = TDConfig(
    alpha=0.5,      # 学习率
    gamma=0.99,     # 折扣因子
    epsilon=0.1,    # 探索率
    lambda_=0.9,    # TD(λ)的λ值
    n_step=3        # N-Step TD的步数
)

# 创建算法
sarsa = create_td_learner('sarsa', config=config)

# 训练
metrics = sarsa.train(
    env,
    n_episodes=500,
    max_steps_per_episode=200,
    log_interval=100
)

# 评估
mean_reward, std_reward = sarsa.evaluate(env, n_episodes=100)
print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
```

### 12.2 支持的算法

```python
# 算法名称 -> 算法类
algorithms = {
    'td0': TD0ValueLearner,      # TD(0)价值预测
    'sarsa': SARSA,               # SARSA
    'expected_sarsa': ExpectedSARSA,  # Expected SARSA
    'q_learning': QLearning,      # Q-Learning
    'double_q': DoubleQLearning,  # Double Q-Learning
    'n_step': NStepTD,            # N-Step TD
    'td_lambda': TDLambda,        # TD(λ)
    'sarsa_lambda': SARSALambda,  # SARSA(λ)
    'watkins_q_lambda': WatkinsQLambda,  # Watkins's Q(λ)
}
```

### 12.3 环境列表

```python
from environments import (
    GridWorld,        # 可配置网格世界
    CliffWalkingEnv,  # 悬崖行走（SARSA vs Q-Learning经典对比）
    WindyGridWorld,   # 有风网格世界
    RandomWalk,       # 随机游走（TD预测验证）
    Blackjack,        # 21点
)
```

### 12.4 可视化工具

```python
from utils import (
    plot_training_curves,    # 训练曲线
    plot_value_heatmap,      # 价值函数热力图
    plot_policy_arrows,      # 策略箭头图
    plot_td_error_analysis,  # TD误差分析
    plot_lambda_comparison,  # λ值对比
)
```

---

## 参考文献

1. **Sutton, R. S., & Barto, A. G. (2018)**. *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

2. **Watkins, C. J. C. D. (1989)**. *Learning from Delayed Rewards*. PhD thesis, Cambridge University.

3. **Rummery, G. A., & Niranjan, M. (1994)**. *On-line Q-learning using connectionist systems*. Technical Report CUED/F-INFENG/TR 166, Cambridge University.

4. **Van Hasselt, H. (2010)**. *Double Q-learning*. Advances in Neural Information Processing Systems.

5. **Sutton, R. S. (1988)**. *Learning to predict by the methods of temporal differences*. Machine Learning, 3(1), 9-44.

---

## 快速记忆卡片

### TD核心公式
```
TD目标:  R + γV(S')
TD误差:  δ = R + γV(S') - V(S)
更新:    V(S) ← V(S) + αδ
```

### SARSA vs Q-Learning
```
SARSA:      Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
Q-Learning: Q(S,A) ← Q(S,A) + α[R + γmax_a Q(S',a) - Q(S,A)]
                                    ↑
                              关键区别：max vs 实际动作
```

### 选择口诀
```
要安全用SARSA，要最优用Q-Learning
怕过估用Double Q，要稳定用Expected
需传播用TD(λ)，回合短用MC
```
