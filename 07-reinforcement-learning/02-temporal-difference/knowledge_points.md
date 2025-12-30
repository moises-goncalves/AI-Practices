# 时序差分学习知识点总结

## 📚 目录

1. [核心概念](#1-核心概念)
2. [TD预测算法](#2-td预测算法)
3. [TD控制算法](#3-td控制算法)
4. [高级TD算法](#4-高级td算法)
5. [算法对比](#5-算法对比)
6. [数学推导](#6-数学推导)
7. [实践指南](#7-实践指南)

---

## 1. 核心概念

### 1.1 什么是时序差分学习？

**TD学习 = 蒙特卡洛采样 + 动态规划自举**

- **从MC继承**: 从经验中学习，不需要环境模型
- **从DP继承**: 使用估计值更新估计值（自举/Bootstrapping）

### 1.2 TD误差（核心驱动力）

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

- $R_{t+1} + \gamma V(S_{t+1})$: TD目标（新证据）
- $V(S_t)$: 当前估计（旧信念）
- $\delta_t$: 预测误差，驱动学习

### 1.3 TD更新规则

$$V(S_t) \leftarrow V(S_t) + \alpha \cdot \delta_t$$

**直觉理解**: 向TD目标方向调整估计值，步长由学习率α控制。

---

## 2. TD预测算法

### 2.1 TD(0) - 单步TD预测

**目标**: 评估给定策略π的价值函数V^π

**更新规则**:
$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

**特点**:
- 每步更新，无需等待回合结束
- 有偏但低方差
- 适合在线学习和连续任务

### 2.2 TD(0) vs Monte Carlo

| 特性 | TD(0) | Monte Carlo |
|------|-------|-------------|
| 偏差 | 有偏 | 无偏 |
| 方差 | 低 | 高 |
| 更新时机 | 每步 | 回合结束 |
| 适用场景 | 在线/连续任务 | 短回合任务 |

---

## 3. TD控制算法

### 3.1 SARSA (On-Policy)

**名称来源**: State-Action-Reward-State-Action

**更新规则**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**关键特点**:
- 使用**实际下一动作**A'计算TD目标
- 学习行为策略（含探索）的价值
- 在危险环境中学到更保守的策略

### 3.2 Q-Learning (Off-Policy)

**更新规则**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**关键特点**:
- 使用**max操作**选择最优动作
- 直接学习最优策略Q*
- 支持经验回放，是DQN的基础

### 3.3 Expected SARSA

**更新规则**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \mathbb{E}_\pi[Q(S_{t+1}, A')] - Q(S_t, A_t)]$$

**期望计算** (ε-greedy):
$$\mathbb{E}_\pi[Q(S', \cdot)] = \frac{\epsilon}{|A|}\sum_a Q(S', a) + (1-\epsilon)\max_a Q(S', a)$$

**特点**: 消除动作采样方差，比SARSA更稳定

### 3.4 悬崖行走案例对比

```
S . . . . . . . . . . G
. . . . . . . . . . . .
. . . . . . . . . . . .
S C C C C C C C C C C G
```

| 算法 | 学到的路径 | 训练奖励 | 评估奖励 |
|------|-----------|---------|---------|
| SARSA | 远离悬崖的安全路径 | 高 | 较低 |
| Q-Learning | 沿悬崖边缘最短路径 | 低 | 最高 |

---

## 4. 高级TD算法

### 4.1 Double Q-Learning

**解决问题**: Q-Learning的最大化偏差

**最大化偏差**:
$$\mathbb{E}[\max_a Q(s,a)] \geq \max_a \mathbb{E}[Q(s,a)]$$

**解决方案**: 维护两个Q表，解耦选择和评估
- Q_A选择动作: $a^* = \arg\max_a Q_A(S', a)$
- Q_B评估价值: $Q_A \leftarrow Q_A + \alpha[R + \gamma Q_B(S', a^*) - Q_A]$

### 4.2 N-Step TD

**n-step回报**:
$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V(S_{t+n})$$

**偏差-方差权衡**:
- n=1: TD(0)，高偏差低方差
- n→∞: MC，低偏差高方差
- 最优n通常在4-10

### 4.3 TD(λ) - 资格迹方法

**λ-回报** (所有n-step回报的几何加权):
$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$$

**资格迹** (后向视图):
$$E_t(s) = \gamma\lambda E_{t-1}(s) + \mathbf{1}[S_t = s]$$

**更新规则**:
$$V(s) \leftarrow V(s) + \alpha\delta_t E_t(s), \quad \forall s$$

**λ的作用**:
- λ=0: TD(0)
- λ=1: Monte Carlo
- λ=0.9: 实践推荐值

### 4.4 资格迹类型

| 类型 | 更新规则 | 特点 |
|------|---------|------|
| 累积迹 | E(s) ← γλE(s) + 1 | 经典方法 |
| 替换迹 | E(s) ← 1 | 避免过度累积 |
| 荷兰迹 | E(s) ← (1-α)γλE(s) + 1 | 函数逼近下更稳定 |

---

## 5. 算法对比

### 5.1 On-Policy vs Off-Policy

| 特性 | On-Policy (SARSA) | Off-Policy (Q-Learning) |
|------|-------------------|------------------------|
| TD目标 | Q(S', A') | max_a Q(S', a) |
| 学习目标 | 行为策略价值 | 最优策略价值 |
| 经验回放 | 不支持 | 支持 |
| 安全性 | 高 | 低 |

### 5.2 算法选择指南

| 场景 | 推荐算法 |
|------|---------|
| 安全重要 | SARSA |
| 需要最优策略 | Q-Learning |
| 稀疏奖励 | TD(λ) 或 SARSA(λ) |
| 过估计问题 | Double Q-Learning |
| 低方差需求 | Expected SARSA |

---

## 6. 数学推导

### 6.1 收敛性条件 (Robbins-Monro)

$$\sum_{t=1}^{\infty}\alpha_t = \infty \quad \text{且} \quad \sum_{t=1}^{\infty}\alpha_t^2 < \infty$$

**直觉**: 学习率要足够大以克服初始条件，又要足够小以最终收敛。

### 6.2 Bellman方程

**状态价值**:
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')]$$

**动作价值**:
$$Q^\pi(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'}\pi(a'|s')Q^\pi(s',a')]$$

**最优Bellman方程**:
$$Q^*(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \max_{a'} Q^*(s',a')]$$

### 6.3 TD(λ)前向后向等价性

**前向视图** (λ-回报):
$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_t^{(n)} + \lambda^{T-t-1}G_t$$

**后向视图** (资格迹):
在离线更新下，两种视图产生相同的总更新量。

---

## 7. 实践指南

### 7.1 超参数调优

| 参数 | 典型范围 | 调优建议 |
|------|---------|---------|
| α (学习率) | 0.01-0.5 | 从0.1开始，观察收敛速度 |
| γ (折扣因子) | 0.9-0.999 | 长期任务用高值 |
| ε (探索率) | 0.01-0.3 | 可用衰减策略 |
| λ (资格迹) | 0.8-0.95 | 0.9是好的起点 |

### 7.2 常见问题与解决

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 不收敛 | α太大 | 降低学习率 |
| 收敛太慢 | α太小 | 提高学习率或用资格迹 |
| 过估计 | Q-Learning偏差 | 使用Double Q-Learning |
| 训练不稳定 | 方差太大 | 使用Expected SARSA |

### 7.3 代码使用示例

```python
from core import TDConfig, create_td_learner
from environments import CliffWalkingEnv

# 创建配置
config = TDConfig(alpha=0.5, gamma=0.99, epsilon=0.1)

# 创建算法
sarsa = create_td_learner('sarsa', config)
qlearn = create_td_learner('q_learning', config)

# 训练
env = CliffWalkingEnv()
sarsa_metrics = sarsa.train(env, n_episodes=500)
qlearn_metrics = qlearn.train(env, n_episodes=500)

# 评估
sarsa_reward, _ = sarsa.evaluate(env, n_episodes=100)
qlearn_reward, _ = qlearn.evaluate(env, n_episodes=100)
```

---

## 📖 参考文献

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
2. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards* (PhD thesis)
3. Van Hasselt, H. (2010). Double Q-learning. *NeurIPS*
4. Singh, S., et al. (2000). Convergence Results for Single-Step On-Policy RL Algorithms

---

*最后更新: 2024年*
