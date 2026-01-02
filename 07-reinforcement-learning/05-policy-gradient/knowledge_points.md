# 策略梯度方法 - 完整知识体系

> 本文档系统整合策略梯度算法的理论基础、核心算法、数学推导与工程实践。

---

## 目录

1. [核心概念](#1-核心概念)
2. [策略梯度定理](#2-策略梯度定理)
3. [REINFORCE算法](#3-reinforce算法)
4. [方差缩减技术](#4-方差缩减技术)
5. [优势函数与GAE](#5-优势函数与gae)
6. [Actor-Critic架构](#6-actor-critic架构)
7. [PPO算法](#7-ppo算法)
8. [连续动作空间](#8-连续动作空间)
9. [工程实践要点](#9-工程实践要点)
10. [算法对比与选择](#10-算法对比与选择)
11. [常见问题与面试要点](#11-常见问题与面试要点)

---

## 1. 核心概念

### 1.1 核心心法

**策略梯度的本质**：直接对策略参数求梯度，沿着期望回报增加的方向更新，用 log 概率加权的回报作为"标签"进行梯度上升。

**策略梯度 = 概率的微积分**

策略梯度方法的本质是一个优雅的想法：**不学习价值函数，直接学习策略，通过梯度上升最大化期望回报**。关键洞察是 Log-Derivative Trick，它将"无法微分的采样"转化为"可微分的期望"。

### 1.2 值方法 vs 策略方法

| 维度 | 值方法 (Value-Based) | 策略方法 (Policy-Based) |
|------|---------------------|------------------------|
| **学习目标** | 动作价值函数 $Q(s,a)$ | 策略函数 $\pi_\theta(a\|s)$ |
| **策略类型** | 隐式（argmax派生） | 显式参数化 |
| **动作空间** | 主要用于离散 | 离散/连续均适用 |
| **探索机制** | $\epsilon$-greedy | 随机策略自带探索 |
| **收敛性** | 可能振荡 | 更平滑的收敛 |
| **代表算法** | DQN, Double DQN | REINFORCE, A2C, PPO |

**数学直觉**：

价值方法的问题：
$$\max_a Q(s,a) \text{ 对连续动作不可微}$$

策略方法的解决：
$$\max_\theta E_{a \sim \pi_\theta}[Q(s,a)] \text{ 可以通过采样估计}$$

### 1.3 策略参数化

**离散动作空间 - Softmax策略：**
$$\pi_\theta(a|s) = \frac{\exp(h(s, a; \theta))}{\sum_{a'} \exp(h(s, a'; \theta))}$$

**连续动作空间 - 高斯策略：**
$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$$

### 1.4 关键术语

- **轨迹 (Trajectory)**: $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$
- **回报 (Return)**: $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$
- **优势 (Advantage)**: $A(s,a) = Q(s,a) - V(s)$
- **熵 (Entropy)**: $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$

---

## 2. 策略梯度定理

### 2.1 目标函数

最大化期望累积回报：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

### 2.2 Log-Derivative Trick（核心洞察）

**问题**：为什么不能直接对采样求导？

**本质**：**期望的梯度 = 梯度的期望**（在特定条件下）

**直接求导（错误）**：
$$\nabla_\theta J = E[\nabla_\theta f(a)] \quad \text{❌ 错误！}$$

**Log-Derivative Trick（正确）**：

$$\nabla_\theta \pi_\theta(a) = \pi_\theta(a) \cdot \nabla_\theta \log \pi_\theta(a)$$

因此：
$$\nabla_\theta J = E_{a \sim \pi_\theta}[f(a) \cdot \nabla_\theta \log \pi_\theta(a)] \quad \text{✓ 正确！}$$

**直觉**：
- $\nabla_\theta \log \pi_\theta(a)$ 指向增加 $\pi_\theta(a)$ 的方向
- 乘以 $f(a)$ 后，高回报的动作被增加，低回报的动作被减少

### 2.3 策略梯度定理 (Sutton et al., 1999)

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi}(s, a)\right]$$

**证明思路：**
1. 对策略轨迹概率求导
2. 利用似然比技巧 (Likelihood Ratio Trick): $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$
3. 消除状态分布梯度依赖

### 2.4 直观理解

- $\nabla_\theta \log \pi_\theta(a|s)$：增加动作概率的方向
- $Q^{\pi}(s, a)$：动作的好坏程度
- **核心思想**：好的动作增加概率，差的动作减少概率

### 2.5 为什么用 log π 而不是 π？

1. **数学原因**：似然比技巧简化梯度计算
2. **数值稳定性**：避免概率连乘导致的数值问题
3. **梯度特性**：$\nabla \log \pi = \nabla \pi / \pi$，自动归一化

---

## 3. REINFORCE算法

### 3.1 算法原理

使用蒙特卡洛回报 $G_t$ 估计 $Q(s,a)$：

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) G_t^{(i)}$$

### 3.2 算法流程

```
Algorithm: REINFORCE
────────────────────────────────────────
1. 初始化策略网络参数 θ
2. for each episode:
   a. 采样轨迹 τ = (s₀,a₀,r₀,...,sₜ)
   b. 计算每步回报 Gₜ = Σₖ γᵏ rₜ₊ₖ
   c. 计算策略梯度 g = Σₜ ∇log π(aₜ|sₜ) · Gₜ
   d. 更新参数 θ ← θ + α · g
```

### 3.3 回报计算实现

```python
def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """从后向前计算折扣回报，时间复杂度 O(T)"""
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)
```

### 3.4 特性分析

| 特性 | 描述 |
|------|------|
| **无偏性** | 使用真实回报，梯度估计无偏 |
| **高方差** | 回报受整条轨迹随机性影响 |
| **样本效率** | 低（需要完整episode） |
| **更新频率** | Episode结束后更新 |

### 3.5 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **梯度方向错误** | 策略变差 | 检查 log 梯度的符号 |
| **方差过高** | 学习不稳定 | 加入基线或使用 GAE |
| **熵消失** | 策略过度确定 | 增加熵系数 |

---

## 4. 方差缩减技术

### 4.1 高方差问题

REINFORCE的梯度方差来源：
1. 策略随机性
2. 环境随机性
3. 回报的累积噪声

### 4.2 基线方法 (Baseline)

引入状态相关基线 $b(s)$：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot (Q(s,a) - b(s))\right]$$

**关键定理**：任意状态相关基线不改变梯度期望：
$$\mathbb{E}_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = 0$$

**证明**：
$$\mathbb{E}_{a}[\nabla_\theta \log \pi \cdot b(s)] = b(s) \sum_a \pi \cdot \frac{\nabla_\theta \pi}{\pi} = b(s) \nabla_\theta \sum_a \pi = b(s) \nabla_\theta 1 = 0$$

### 4.3 最优基线

理论最优基线（最小化方差）：
$$b^*(s) = \frac{\mathbb{E}[(\nabla_\theta \log \pi)^2 \cdot Q]}{\mathbb{E}[(\nabla_\theta \log \pi)^2]}$$

**实践中**：使用状态价值函数 $V(s)$ 作为基线，得到优势函数。

### 4.4 回报标准化

简单有效的方差缩减：
```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

### 4.5 方差缩减效果对比

```
无基线:     方差 ≈ 10000
带基线:     方差 ≈ 100
带标准化:   方差 ≈ 1
```

---

## 5. 优势函数与GAE

### 5.1 优势函数定义

$$A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$$

**三层含义**：
1. **统计含义**：动作相对于平均水平的好坏
2. **方差含义**：减少基线方差
3. **信息论含义**：提取动作的相对价值

**直观含义**：
- $A > 0$：动作优于该状态下的平均水平
- $A < 0$：动作劣于平均水平
- $A = 0$：动作表现等于平均

**数学性质**：
$$E_\pi[A^\pi(s,a)] = 0$$

即优势函数的期望为 0，说明它是"相对"的度量。

### 5.2 TD误差

单步时序差分误差：
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**性质**：$\delta_t$ 是优势函数的无偏估计
$$\mathbb{E}[\delta_t | s_t, a_t] = A^{\pi}(s_t, a_t)$$

### 5.3 N步回报

$$G_t^{(n)} = r_t + \gamma r_{t+1} + \ldots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$$

| N值 | 偏差 | 方差 | 特点 |
|-----|------|------|------|
| N=1 | 高 | 低 | TD(0)，依赖V估计 |
| N=∞ | 0 | 高 | MC，使用真实回报 |
| N=5 | 中 | 中 | 常用折中 |

### 5.4 GAE (Generalized Advantage Estimation)

Schulman et al., 2016 提出的优势估计方法：

$$A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

**展开形式**：
$$A_t^{GAE} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \ldots$$

**递归计算**：
$$A_t = \delta_t + \gamma\lambda A_{t+1}$$

### 5.5 λ参数的作用

| λ值 | 等价形式 | 偏差 | 方差 | 适用场景 |
|-----|----------|------|------|----------|
| λ=0 | TD(0) | 高 | 低 | V估计准确时 |
| λ=1 | MC | 无 | 高 | 短episode |
| λ=0.95 | 加权混合 | 低 | 中 | 通用推荐 |

### 5.6 GAE实现

```python
def compute_gae(rewards, values, next_value, dones, gamma, gae_lambda):
    """计算GAE优势估计。时间复杂度: O(T)"""
    advantages = []
    gae = 0.0
    values = list(values) + [next_value]

    for t in reversed(range(len(rewards))):
        next_val = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages)
```

---

## 6. Actor-Critic架构

### 6.1 架构概述

```
             ┌─────────────────────────────────────┐
             │              State                  │
             └─────────────────────────────────────┘
                             │
                             ▼
             ┌─────────────────────────────────────┐
             │        Shared Feature Net           │
             │        (可选共享层)                   │
             └─────────────────────────────────────┘
                    │                    │
                    ▼                    ▼
        ┌──────────────────┐   ┌──────────────────┐
        │    Actor Head    │   │   Critic Head    │
        │    π(a|s; θ)     │   │    V(s; φ)       │
        └──────────────────┘   └──────────────────┘
                │                       │
                ▼                       ▼
          Policy Loss            Value Loss
         -log π · A              (V - G)²
```

### 6.2 A2C算法

**Advantage Actor-Critic** 同步版本：

```
Algorithm: A2C
────────────────────────────────────────
1. 初始化 Actor θ, Critic φ
2. for each update:
   a. 采集n步经验
   b. 计算GAE优势 Aₜ
   c. 计算目标回报 Gₜ = Aₜ + V(sₜ)
   d. Actor损失: L_π = -E[log π(a|s) · A]
   e. Critic损失: L_V = E[(V(s) - G)²]
   f. 熵奖励: H = E[-log π]
   g. 总损失: L = L_π + c_v·L_V - c_ent·H
   h. 梯度更新 θ, φ
```

### 6.3 损失函数详解

**总损失函数**：
$$\mathcal{L} = \underbrace{-\mathbb{E}[\log \pi(a|s) \cdot A]}_{\text{策略损失}} + \underbrace{c_v \mathbb{E}[(V(s) - G)^2]}_{\text{价值损失}} - \underbrace{c_{ent} H(\pi)}_{\text{熵奖励}}$$

| 组件 | 作用 | 典型系数 |
|------|------|----------|
| 策略损失 | 最大化优势加权对数概率 | 1.0 |
| 价值损失 | 准确估计状态价值 | 0.5 |
| 熵奖励 | 鼓励探索，防止早熟收敛 | 0.01 |

### 6.4 共享网络 vs 分离网络

**共享网络优势**：参数量少、特征复用、训练更稳定

**分离网络优势**：避免梯度干扰、独立学习率、更灵活

**实践建议**：简单任务用共享网络，复杂任务考虑分离。

### 6.5 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **价值函数不准确** | 优势估计有偏 | 增加 Critic 训练 |
| **梯度干扰** | 学习不稳定 | 分离 Actor 和 Critic 网络 |

---

## 7. PPO算法

### 7.1 核心问题

**问题**：大的策略更新会导致灾难性的性能下降：
$$\pi_{\text{old}} \to \pi_{\text{new}} \text{ 差异太大}$$

### 7.2 TRPO 的解决（复杂）

$$\max_\theta E[r_t(\theta) A_t] \quad \text{s.t.} \quad E[\text{KL}(\pi_{\text{old}} || \pi_{\text{new}})] \leq \delta$$

需要共轭梯度和线搜索。

### 7.3 PPO 的解决（简单）

用**裁剪目标**替代硬约束：

$$L^{\text{CLIP}} = E_t[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$$

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是概率比率。

### 7.4 裁剪机制的直觉

```
当 A_t > 0（好动作）：
  ├─ 想增加 π(a|s)，所以 r_t > 1
  ├─ 裁剪防止 r_t 超过 1+ε
  └─ 梯度在 r_t > 1+ε 时为 0（停止增加）

当 A_t < 0（坏动作）：
  ├─ 想减少 π(a|s)，所以 r_t < 1
  ├─ 裁剪防止 r_t 低于 1-ε
  └─ 梯度在 r_t < 1-ε 时为 0（停止减少）
```

### 7.5 关键优势

1. **多 epoch 优化**：可以重用数据多次
2. **简单实现**：只需裁剪，无需二阶优化
3. **稳定性**：裁剪限制了更新幅度

### 7.6 PPO 实现

```python
def ppo_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2):
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    
    # 取较小值（悲观界）
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss
```

### 7.7 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **clip_epsilon 过大** | 更新不足 | 减小 ε（如 0.1） |
| **clip_epsilon 过小** | 更新过度 | 增大 ε（如 0.3） |
| **多 epoch 导致过拟合** | 性能下降 | 减少 epoch 数或增加数据 |

---

## 8. 连续动作空间

### 8.1 高斯策略

$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$$

**网络输出**：
- 均值 $\mu$：线性层直接输出
- 标准差 $\sigma$：可学习参数或网络输出（需正数约束）

### 8.2 重参数化技巧 (Reparameterization Trick)

$$a = \mu_\theta(s) + \sigma_\theta(s) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**优势**：
- 允许梯度通过采样操作传播
- 降低方差
- 适用于确定性策略梯度

### 8.3 Tanh压缩

将无界高斯动作映射到 $[-1, 1]$：
$$a = \tanh(u), \quad u \sim \mathcal{N}(\mu, \sigma^2)$$

**对数概率修正（雅可比行列式）**：
$$\log \pi(a|s) = \log \mathcal{N}(u|\mu,\sigma^2) - \sum_i \log(1 - \tanh^2(u_i))$$

### 8.4 实现细节

```python
class ContinuousPolicy(nn.Module):
    LOG_STD_MIN = -20.0  # 防止std过小
    LOG_STD_MAX = 2.0    # 防止std过大

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)

        # 重参数化采样
        u = dist.rsample()
        action = torch.tanh(u)

        # 修正对数概率
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)

        return action, log_prob
```

---

## 9. 工程实践要点

### 9.1 网络初始化

**正交初始化 (Orthogonal Initialization)**：

```python
def init_weights(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
```

**输出层特殊处理**：
- Actor最后一层：gain=0.01（小初始化，初始策略接近均匀）
- Critic最后一层：gain=1.0

### 9.2 梯度裁剪

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

### 9.3 优势标准化

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**注意**：优势标准化应在detach之后进行。

### 9.4 超参数推荐

| 参数 | 典型范围 | 说明 |
|------|----------|------|
| 学习率(Actor) | 1e-4 ~ 3e-4 | 策略应平滑变化 |
| 学习率(Critic) | 1e-3 ~ 3e-3 | 可比Actor大 |
| γ (折扣因子) | 0.99 | 短任务可用0.95 |
| λ (GAE) | 0.95 | 平衡偏差-方差 |
| 熵系数 | 0.01 | 太大导致随机策略 |
| 梯度裁剪 | 0.5 ~ 1.0 | 防止不稳定 |
| 隐藏层 | 64/128/256 | 任务复杂度决定 |
| PPO clip_epsilon | 0.1 ~ 0.3 | 推荐0.2 |

### 9.5 常见技巧

1. **学习率调度**：余弦退火或线性衰减
2. **奖励裁剪**：限制在[-10, 10]
3. **观测标准化**：Running mean/std
4. **多环境并行**：提高样本效率
5. **早停**：基于验证性能

### 9.6 代码验证清单

```python
# 1. 梯度方向验证
assert (policy_loss.grad > 0 for good_actions)  # 好动作梯度为正
assert (policy_loss.grad < 0 for bad_actions)   # 坏动作梯度为负

# 2. 优势标准化验证
assert abs(advantages.mean()) < 0.01  # 优势应该中心化
assert abs(advantages.std() - 1.0) < 0.1  # 优势应该标准化

# 3. 熵监控
entropy = -(probs * log_probs).sum(dim=1).mean()
assert entropy > min_entropy_threshold  # 防止策略过度确定

# 4. 价值损失监控
assert value_loss < previous_value_loss  # 价值函数应该改进
```

---

## 10. 算法对比与选择

### 10.1 算法演进

```
REINFORCE (1992)
    │
    ├─ 加入基线 ──────────────▶ REINFORCE + Baseline
    │
    ├─ TD替代MC ──────────────▶ Actor-Critic
    │
    ├─ 优势函数 + GAE ────────▶ A2C (2016)
    │
    └─ 加入约束 ──────────────▶ PPO / TRPO (2017)
```

### 10.2 详细对比

| 特性 | REINFORCE | +Baseline | A2C | PPO |
|------|-----------|-----------|-----|-----|
| 优势估计 | $G_t$ | $G_t - V(s)$ | GAE | GAE |
| 更新时机 | Episode结束 | Episode结束 | N步 | N步 |
| 方差 | 高 | 中 | 低 | 低 |
| 偏差 | 无 | 无 | 有（可控） | 有（可控） |
| 样本效率 | 低 | 中 | 高 | 高 |
| 稳定性 | 低 | 中 | 中 | 高 |
| 实现复杂度 | 简单 | 中等 | 中等 | 中等 |
| 适用场景 | 教学/简单任务 | 中等任务 | 生产环境 | 生产环境（推荐） |

### 10.3 算法选择决策树

```
需要什么？
├─ 学习简单任务
│  └─ REINFORCE（教学用）
│
├─ 平衡性能和稳定性
│  └─ A2C（中等任务）
│
├─ 生产环境
│  └─ PPO（推荐）
│     ├─ 离散动作：Softmax 策略
│     └─ 连续动作：高斯策略
│
└─ 理论研究
   └─ TRPO（有理论保证）
```

---

## 11. 常见问题与面试要点

### 11.1 训练问题排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 训练不收敛 | 奖励尺度、学习率、初始化 | 降低学习率、增大熵系数、检查网络 |
| 策略崩溃 | 策略变确定性，熵降为0 | 增大熵系数、减小学习率、用PPO |
| 价值估计不准 | Value loss居高不下 | 增大Critic学习率、增加网络容量 |
| 探索不足 | 陷入局部最优 | 增大熵系数、添加探索噪声 |
| 数值不稳定 | 出现NaN或Inf | 添加epsilon、梯度裁剪、检查除零 |

### 11.2 面试高频问题

**Q1: 策略梯度和值函数方法的本质区别？**
- 值方法：学习"做什么"（最优动作）
- 策略梯度：学习"怎么做"（动作分布）

**Q2: 为什么需要基线？**
减少方差但不改变梯度期望，加速收敛。

**Q3: GAE的λ参数作用？**
平衡偏差和方差：λ=0是TD(0)，λ=1是MC。

**Q4: Actor-Critic比REINFORCE好在哪？**
用价值函数做基线，降低方差，可以N步更新。

**Q5: 连续动作空间的挑战？**
需要参数化分布（如高斯），重参数化技巧允许梯度传播。

**Q6: PPO裁剪如何防止策略崩溃？**
限制概率比率在 $[1-\epsilon, 1+\epsilon]$ 范围内，防止大幅度策略更新。

**Q7: Log-Derivative Trick 的含义？**
将不可微的采样操作转化为可微的期望梯度计算。

---

## 快速复习卡片

| 概念 | 关键词 |
|------|--------|
| 策略梯度 | 直接学习策略，梯度上升 |
| Log-Derivative Trick | $\nabla\log\pi = \nabla\pi/\pi$ |
| REINFORCE | MC回报估计Q，无偏高方差 |
| 基线 | $V(s)$做基线，不改变期望，减少方差 |
| 优势函数 | $A = Q - V$，相对性度量 |
| GAE | TD和MC的指数加权平均，λ控制偏差-方差 |
| Actor-Critic | 策略网络+价值网络，同时学习 |
| A2C | 同步Actor-Critic，用GAE |
| PPO | 裁剪概率比率，稳定更新 |
| 熵正则化 | 鼓励探索，防止早熟收敛 |

---

## 参考文献

1. Williams, R.J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*.
2. Sutton, R.S., McAllester, D., Singh, S., & Mansour, Y. (1999). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS*.
3. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-dimensional continuous control using generalized advantage estimation. *ICLR*.
4. Mnih, V., Badia, A.P., et al. (2016). Asynchronous methods for deep reinforcement learning. *ICML*.
5. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
6. Sutton, R.S., & Barto, A.G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

*最后更新：2026年1月*
