# 神经网络策略：知识芯片

> **从"用猜测更新猜测"到"信任区域"的策略梯度完整体系**

---

## 核心心法（The "Aha!" Moment）

**策略梯度 = 概率的微积分**

策略梯度方法的本质是一个优雅的想法：**不学习价值函数，直接学习策略，通过梯度上升最大化期望回报**。关键洞察是 Log-Derivative Trick，它将"无法微分的采样"转化为"可微分的期望"。

---

## 第一部分：问题本质

### 价值方法 vs 策略方法

**现象**：为什么需要两种完全不同的方法？

**本质**：**学习目标的不同**导致的架构差异。

```
价值方法 (Value-Based)
├─ 学习目标：Q(s,a) 或 V(s)
├─ 动作选择：argmax Q(s,a)（确定性）
├─ 动作空间：离散（标准）
├─ 优势：理论完美，收敛快
└─ 劣势：离散动作，不稳定

策略方法 (Policy-Based)
├─ 学习目标：π_θ(a|s)
├─ 动作选择：从 π 采样（随机）
├─ 动作空间：离散/连续
├─ 优势：连续动作，更平滑
└─ 劣势：高方差，收敛慢
```

**数学直觉**：

价值方法的问题：
$$\max_a Q(s,a) \text{ 对连续动作不可微}$$

策略方法的解决：
$$\max_\theta E_{a \sim \pi_\theta}[Q(s,a)] \text{ 可以通过采样估计}$$

---

## 第二部分：核心概念

### 1. Log-Derivative Trick

#### 深度原理

**现象**：为什么不能直接对采样求导？

**本质**：**期望的梯度 = 梯度的期望**（在特定条件下）

**数学推导**：

给定 $J(\theta) = E_{a \sim \pi_\theta}[f(a)]$

**直接求导（错误）**：
$$\nabla_\theta J = E[\nabla_\theta f(a)] \quad \text{❌ 错误！}$$

**Log-Derivative Trick（正确）**：

$$\nabla_\theta \pi_\theta(a) = \pi_\theta(a) \cdot \nabla_\theta \log \pi_\theta(a)$$

因此：
$$\nabla_\theta J = E_{a \sim \pi_\theta}[f(a) \cdot \nabla_\theta \log \pi_\theta(a)] \quad \text{✓ 正确！}$$

**直觉**：
- $\nabla_\theta \log \pi_\theta(a)$ 指向增加 $\pi_\theta(a)$ 的方向
- 乘以 $f(a)$ 后，高回报的动作被增加，低回报的动作被减少

#### 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **梯度方向错误** | 策略变差 | 检查 log 梯度的符号 |
| **方差过高** | 学习不稳定 | 加入基线或使用 GAE |
| **熵消失** | 策略过度确定 | 增加熵系数 |

---

### 2. 优势函数（Advantage Function）

#### 深度原理

**现象**：为什么需要优势而不是直接用回报？

**本质**：**相对性的度量**减少方差。

**定义**：
$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

**三层含义**：

1. **统计含义**：动作相对于平均水平的好坏
2. **方差含义**：减少基线方差
3. **信息论含义**：提取动作的相对价值

**数学性质**：

$$E_\pi[A^\pi(s,a)] = 0$$

即优势函数的期望为 0，说明它是"相对"的度量。

**为什么减少方差**：

```
无基线：Var[G_t] = 高（完整回报的方差）
有基线：Var[G_t - V(s_t)] = 低（只有残差的方差）

关键：E[∇log π · b(s)] = 0（基线不改变期望）
```

---

## 第三部分：策略梯度定理

### 深度原理

**现象**：为什么策略梯度定理这么重要？

**本质**：**连接目标函数与可执行算法**的桥梁。

**定理陈述**（Sutton et al., 1999）：

对于任何可微策略 $\pi_\theta(a|s)$ 和目标函数 $J(\theta) = E_\tau[R(\tau)]$：

$$\nabla_\theta J(\theta) = E_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^\pi(s_t, a_t)\right]$$

**关键洞察**：

1. 梯度只依赖于 $\log \pi$，不依赖于 $\pi$ 本身
2. 可以用任何 $Q^\pi$ 的估计替代（REINFORCE、Actor-Critic 等）
3. 基线不改变期望，只改变方差

**证明直觉**：

$$\nabla_\theta J = \nabla_\theta E_\tau[\sum_t r_t]$$
$$= E_\tau[\nabla_\theta \log \pi_\theta(\tau) \cdot R(\tau)]$$
$$= E_\tau[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^\pi(s_t, a_t)]$$

---

## 第四部分：核心算法

### 1. REINFORCE

#### 深度原理

**现象**：最简单的策略梯度算法如何工作？

**本质**：**蒙特卡洛估计**的策略梯度。

**算法**：

$$\nabla_\theta J \approx \frac{1}{N} \sum_i \sum_t \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \cdot G_t^i$$

其中 $G_t^i = \sum_{k=0}^{T-t} \gamma^k r_{t+k}^i$ 是蒙特卡洛回报。

**特点**：
- ✅ 无偏估计
- ✅ 实现简单
- ❌ 高方差（完整回报的方差）
- ❌ 需要完整 episode

**为什么高方差**：

```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
      └─ 每一项都有噪声，累积导致高方差
```

#### 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **方差爆炸** | 学习不稳定 | 加入基线 V(s) |
| **需要完整 episode** | 无法在线学习 | 用 Actor-Critic |

---

### 2. Actor-Critic (A2C)

#### 深度原理

**现象**：如何减少 REINFORCE 的方差？

**本质**：**自举**与**基线**的结合。

**架构**：

```
Actor: π_θ(a|s) - 策略网络
Critic: V_φ(s) - 价值网络（基线）

优势估计：A_t = r_t + γV(s_{t+1}) - V(s_t) = δ_t（TD误差）
```

**损失函数**：

$$L = L_{\text{policy}} + c_v L_{\text{value}} - c_{\text{ent}} H(\pi)$$

其中：
- $L_{\text{policy}} = -E[\log \pi(a|s) \cdot A_t]$
- $L_{\text{value}} = E[(V(s) - G_t)^2]$
- $H(\pi) = -E[\pi \log \pi]$（熵正则化）

**为什么有效**：

1. **自举**：用 $V(s')$ 估计未来，减少方差
2. **基线**：$V(s)$ 作为基线，进一步减少方差
3. **在线学习**：每步都能更新，不需要完整 episode

**偏差-方差权衡**：

```
REINFORCE: 无偏但高方差
A2C: 有偏但低方差（偏差来自 V 的估计误差）
```

#### 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **价值函数不准确** | 优势估计有偏 | 增加 Critic 训练 |
| **梯度干扰** | 学习不稳定 | 分离 Actor 和 Critic 网络 |

---

### 3. PPO (Proximal Policy Optimization)

#### 深度原理

**现象**：为什么 A2C 在复杂任务上不稳定？

**本质**：**信任区域**的简化实现。

**问题**：

大的策略更新会导致灾难性的性能下降：
$$\pi_{\text{old}} \to \pi_{\text{new}} \text{ 差异太大}$$

**TRPO 的解决**（复杂）：

$$\max_\theta E[r_t(\theta) A_t] \quad \text{s.t.} \quad E[\text{KL}(\pi_{\text{old}} || \pi_{\text{new}})] \leq \delta$$

需要共轭梯度和线搜索。

**PPO 的解决**（简单）：

用**裁剪目标**替代硬约束：

$$L^{\text{CLIP}} = E_t[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)]$$

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 是概率比率。

**裁剪机制的直觉**：

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

**关键优势**：

1. **多 epoch 优化**：可以重用数据多次
2. **简单实现**：只需裁剪，无需二阶优化
3. **稳定性**：裁剪限制了更新幅度

#### 架构陷阱

| 陷阱 | 症状 | 修复 |
|:----:|:-----|:-----|
| **clip_epsilon 过大** | 更新不足 | 减小 ε（如 0.1） |
| **clip_epsilon 过小** | 更新过度 | 增大 ε（如 0.3） |
| **多 epoch 导致过拟合** | 性能下降 | 减少 epoch 数或增加数据 |

---

## 第五部分：优势估计方法

### 广义优势估计（GAE）

#### 深度原理

**现象**：如何在 TD 和 MC 之间平衡？

**本质**：**几何加权平均**的优势估计。

**定义**：

$$A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

**递归计算**：

$$A_t = \delta_t + \gamma\lambda A_{t+1}$$

**λ 的含义**：

| λ 值 | 等价于 | 偏差 | 方差 |
|:----:|:-----|:-----|:-----|
| 0 | TD(0) | 高 | 低 |
| 0.5 | 平衡 | 中 | 中 |
| 0.95 | 推荐 | 中 | 中 |
| 1 | MC | 无 | 高 |

**为什么有效**：

```
TD(0): A = δ_t（只用一步，低方差但有偏）
MC: A = G_t - V(s)（用完整回报，无偏但高方差）
GAE: A = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...
     └─ 平衡两者，可调的偏差-方差权衡
```

---

## 第六部分：工程实践

### 通用设计模式

#### 模式 1：策略梯度的标准形式

```python
# 通用模式：所有策略梯度算法的核心
def policy_gradient_update(log_probs, advantages, learning_rate):
    # 策略梯度：∇J = E[∇log π · A]
    policy_loss = -(log_probs * advantages).mean()

    # 梯度上升
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
```

#### 模式 2：Actor-Critic 的标准架构

```python
# 通用模式：Actor-Critic 框架
class ActorCritic:
    def __init__(self):
        self.actor = PolicyNetwork()    # π_θ(a|s)
        self.critic = ValueNetwork()    # V_φ(s)

    def compute_advantage(self, rewards, values, gamma=0.99):
        # GAE 计算
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] - values[t]
            gae = delta + gamma * 0.95 * gae
            advantages.insert(0, gae)
        return advantages
```

#### 模式 3：PPO 的裁剪目标

```python
# 通用模式：PPO 裁剪
def ppo_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2):
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)

    # 取较小值（悲观界）
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss
```

### 代码验证清单

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

## 第七部分：算法选择决策树

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

## 第八部分：快速复习清单

### 概念理解

- [ ] Log-Derivative Trick 的含义与推导
- [ ] 策略梯度定理的陈述与证明思路
- [ ] 优势函数的三层含义
- [ ] REINFORCE 为什么高方差
- [ ] Actor-Critic 如何减少方差
- [ ] PPO 裁剪如何防止策略崩溃
- [ ] GAE 如何平衡偏差和方差

### 公式推导

- [ ] 策略梯度定理的完整推导
- [ ] Log-Derivative Trick 的推导
- [ ] 优势函数与基线的关系
- [ ] GAE 的递归形式
- [ ] PPO 裁剪目标的分析

### 代码实现

- [ ] REINFORCE 的正确实现
- [ ] Actor-Critic 的正确实现
- [ ] PPO 裁剪的正确实现
- [ ] GAE 的计算
- [ ] 策略参数化（Softmax、高斯）

### 工程应用

- [ ] 如何选择合适的算法
- [ ] 如何调整超参数（α, γ, λ, ε）
- [ ] 如何处理常见问题（不收敛、策略崩溃等）
- [ ] 如何评估算法性能

---

## 第九部分：迁移学习指南

### 从策略梯度到其他问题

#### 迁移 1：多任务强化学习

**相同点**：策略梯度定理仍然成立

**不同点**：
- 需要任务条件策略 $\pi_\theta(a|s,t)$
- 需要任务间的知识转移

**迁移方法**：
```python
# 单任务
π_θ(a|s)

# 多任务：加入任务条件
π_θ(a|s,t) = softmax(f_θ(s,t))
```

#### 迁移 2：离线强化学习

**相同点**：策略梯度仍然适用

**不同点**：
- 数据来自旧策略，不是当前策略
- 需要处理分布偏移

**迁移方法**：
```python
# 在线：从当前策略采样
# 离线：从固定数据集采样，需要重要性采样修正
importance_weight = π_new(a|s) / π_old(a|s)
```

#### 迁移 3：多智能体强化学习

**相同点**：每个智能体仍然用策略梯度

**不同点**：
- 其他智能体的策略也在变化
- 需要处理非平稳环境

**迁移方法**：
```python
# 单智能体
∇J = E[∇log π · A]

# 多智能体：每个智能体独立更新
∇J_i = E[∇log π_i · A_i]
```

---

## 总结：知识芯片的三个层次

### 第一层：现象理解
- 为什么需要策略梯度？
- 为什么 REINFORCE 高方差？
- 为什么 PPO 比 A2C 稳定？

### 第二层：本质掌握
- Log-Derivative Trick = 期望的梯度
- 优势函数 = 相对性的度量
- PPO 裁剪 = 信任区域的简化
- GAE = 偏差-方差的平衡

### 第三层：工程应用
- 如何选择合适的算法？
- 如何调整超参数？
- 如何处理常见问题？
- 如何扩展到新问题？

**学习路径**：现象 → 本质 → 应用 → 迁移

---

*最后更新：2025-12-20*

*相关代码：`algorithms/` 和 `networks/` 目录*

*交互式实验：修改 `main.py` 中的参数进行验证*
