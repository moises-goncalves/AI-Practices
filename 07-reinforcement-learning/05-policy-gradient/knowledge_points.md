# 策略梯度知识芯片

## 核心心法（The "Aha!" Moment）

**策略梯度的本质：直接对策略参数求梯度，沿着期望回报增加的方向更新，用 log 概率加权的回报作为"标签"进行梯度上升。**

---

## 第一层：问题本质 → 数学建模

### 1.1 为什么需要策略梯度？

| 方法 | 学习目标 | 梯度来源 | 适用场景 |
|------|--------|--------|--------|
| **Q-Learning** | 学习值函数 | 贝尔曼方程 | 离散动作 |
| **策略梯度** | 直接学习策略 | 策略梯度定理 | 连续/离散 |
| **Actor-Critic** | 策略 + 值函数 | 两者结合 | 通用 |

**关键洞察：** Q-Learning 学习"做什么"（最优动作），策略梯度学习"怎么做"（动作分布）。

### 1.2 策略梯度定理的数学推导

**目标函数：**
$$J(\theta) = \mathbb{E}_{\pi_\theta}[G_0] = \mathbb{E}_{s \sim \rho}[V^{\pi_\theta}(s)]$$

其中 $\rho$ 是初始状态分布，$V^{\pi_\theta}(s)$ 是状态价值。

**策略梯度定理（Policy Gradient Theorem）：**
$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)]$$

**证明思路：**
1. 对 $J(\theta)$ 求导
2. 利用 $\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$
3. 得到 log 概率梯度加权的 Q 值

**关键性质：**
- 不需要对环境模型求导
- 只需要对策略网络求导
- 适用于任何可微策略

### 1.3 三大策略梯度算法的本质区别

| 算法 | 目标函数 | 基线 | 方差 | 偏差 |
|------|--------|------|------|------|
| **REINFORCE** | $-\mathbb{E}[\log \pi(a\|s) \cdot G_t]$ | 无 | 高 | 无 |
| **Actor-Critic** | $-\mathbb{E}[\log \pi(a\|s) \cdot A(s,a)]$ | V(s) | 中 | 低 |
| **A2C** | $-\mathbb{E}[\log \pi(a\|s) \cdot A^{GAE}(s,a)]$ | V(s) + GAE | 低 | 低 |

---

## 第二层：核心算法深度解析

### 2.1 REINFORCE（基础策略梯度）

**数学原理：**
$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

其中 $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$ 是蒙特卡洛回报。

**关键特性：**

1. **无偏但高方差**
   - 无偏：$\mathbb{E}[\nabla_\theta \log \pi(a|s) G_t] = \nabla_\theta J(\theta)$
   - 高方差：$G_t$ 是单个轨迹的回报，波动大

2. **回报归一化**
   ```python
   returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
   ```
   - 作用：稳定梯度，加速收敛
   - 原理：不改变梯度方向，只改变幅度

3. **熵正则化**
   ```python
   entropy_loss = -entropy_coeff * entropy.mean()
   ```
   - 作用：鼓励探索，防止策略过早收敛
   - 原理：最大化策略熵 = 最大化不确定性

**实现细节（algorithms/reinforce.py）：**
```python
# 1. 采样轨迹
trajectory = collect_trajectory(env)

# 2. 计算回报
returns = compute_returns(trajectory)

# 3. 计算损失
log_probs, entropy = policy.evaluate(states, actions)
policy_loss = -(log_probs * returns).mean()
entropy_loss = -entropy_coeff * entropy.mean()

# 4. 梯度更新
total_loss = policy_loss + entropy_loss
total_loss.backward()
```

### 2.2 Actor-Critic（降低方差）

**数学原理：**

使用基线（Baseline）减少方差：
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) (Q(s,a) - V(s))]$$
$$= \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) A(s,a)]$$

其中 $A(s,a) = Q(s,a) - V(s)$ 是优势函数。

**关键特性：**

1. **两个网络**
   - Actor（策略网络）：学习 $\pi_\theta(a|s)$
   - Critic（价值网络）：学习 $V_\phi(s)$

2. **TD 误差作为优势**
   ```python
   delta = reward + gamma * V(next_state) - V(state)
   advantage = delta  # 单步 TD 误差
   ```

3. **更新顺序**
   ```python
   # 1. 先更新 Critic（价值网络）
   value_loss = (V(s) - G_t)^2
   value_loss.backward()

   # 2. 再更新 Actor（策略网络）
   advantages = G_t - V(s)  # 使用更新后的 V
   policy_loss = -log_pi * advantages
   policy_loss.backward()
   ```

**方差减少的原理：**
- REINFORCE：$\text{Var}[G_t]$ 很大
- Actor-Critic：$\text{Var}[A(s,a)] = \text{Var}[G_t - V(s)]$ 更小
- 因为 $V(s)$ 捕捉了状态的"平均价值"

### 2.3 A2C（广义优势估计）

**数学原理：**

广义优势估计（GAE）：
$$A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$

其中 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

**参数解释：**
- $\lambda = 0$：只用单步 TD 误差（低方差，高偏差）
- $\lambda = 1$：用完整蒙特卡洛回报（高方差，低偏差）
- $\lambda \in (0,1)$：平衡偏差和方差

**实现细节（algorithms/actor_critic.py）：**
```python
# 后向递推计算 GAE
advantages = np.zeros(T)
gae = 0.0

for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] - values[t]
    gae = delta + gamma * lambda * gae
    advantages[t] = gae
```

**关键优势：**
- 比 REINFORCE 方差低
- 比单步 Actor-Critic 偏差低
- 支持并行环境（多个环境同时收集数据）

---

## 第三层：架构陷阱与工业部署

### 3.1 策略网络的陷阱

**陷阱 1：离散 vs 连续动作**

离散动作：
```python
# 输出 logits，用 softmax 转换为概率
logits = network(state)
dist = Categorical(logits=logits)
action = dist.sample()
```

连续动作：
```python
# 输出均值和标准差
mean = network(state)
std = fixed_std  # 或 exp(log_std)
dist = Normal(mean, std)
action = dist.sample()
```

**陷阱 2：固定方差 vs 学习方差**

固定方差：
- 优点：简单，参数少
- 缺点：探索能力受限

学习方差：
- 优点：自适应探索
- 缺点：需要额外参数，训练不稳定

**陷阱 3：输出层激活函数**

```python
# 连续动作：用 Tanh 限制在 [-1, 1]
output = nn.Tanh()(linear_layer(x))

# 离散动作：无激活（softmax 在分布中处理）
output = linear_layer(x)
```

### 3.2 回报和优势的陷阱

**陷阱 1：回报计算中的终止状态**

```python
# 错误：不处理终止状态
cumulative_return = reward + gamma * cumulative_return

# 正确：终止时不累积
cumulative_return = reward + gamma * cumulative_return * (1 - done)
```

**陷阱 2：优势归一化**

```python
# 必须归一化，否则梯度不稳定
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**陷阱 3：GAE 的 λ 参数**

- λ 过小（< 0.9）：偏差大，收敛慢
- λ 过大（> 0.99）：方差大，训练不稳定
- 推荐：λ = 0.95

### 3.3 训练稳定性的陷阱

**陷阱 1：梯度裁剪**

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
```

- 作用：防止梯度爆炸
- 必要性：策略梯度容易产生大梯度

**陷阱 2：学习率**

- 过大：策略振荡，无法收敛
- 过小：收敛慢，样本浪费
- 推荐：1e-3 到 1e-4

**陷阱 3：熵系数**

- 过大：过度探索，无法收敛到好策略
- 过小：过早收敛，陷入局部最优
- 推荐：0.01 到 0.1

---

## 第四层：前沿演进（SOTA Evolution）

### 4.1 从基础到高级的演变链

```
REINFORCE (1992)
    ↓ [问题：高方差]
Actor-Critic (2000)
    ├─ 用价值函数作基线
    └─ 降低方差
    ↓ [问题：单步 TD 偏差]
A2C (2016)
    ├─ 广义优势估计 (GAE)
    └─ 平衡偏差-方差
    ↓ [问题：策略更新过大]
PPO (2017)
    ├─ 信任域约束
    └─ 稳定性更好
    ↓ [问题：样本效率]
TRPO (2015)
    ├─ 自然梯度
    └─ 理论最优
    ↓ [问题：离散/连续统一]
SAC (2018)
    ├─ 最大熵框架
    └─ 自动温度调节
```

### 4.2 当代前沿方向

| 方向 | 核心创新 | 应用场景 |
|------|--------|--------|
| **PPO** | 信任域 + 裁剪 | 通用，最流行 |
| **TRPO** | 自然梯度 | 理论最优 |
| **SAC** | 最大熵 + 自动温度 | 连续控制 |
| **IMPALA** | 分布式 + 重要性采样 | 大规模并行 |
| **Offline RL** | 从固定数据集学习 | 无环境交互 |

---

## 第五层：交互式思考（Interactive Provocation）

### 问题 1：为什么 Actor-Critic 比 REINFORCE 更好？

**你的任务：** 在相同的环境上运行 REINFORCE 和 Actor-Critic，对比它们的学习曲线。

**实验设计：**
```python
# REINFORCE
reinforce_agent = REINFORCE(policy, learning_rate=1e-3)
reinforce_history = reinforce_agent.train(env, num_episodes=500)

# Actor-Critic
actor_critic_agent = ActorCritic(policy, value_function, learning_rate=1e-3)
ac_history = actor_critic_agent.train(env, num_episodes=500)

# 对比：
# 1. 收敛速度
# 2. 最终性能
# 3. 学习曲线的平滑度
# 4. 方差大小
```

**深度思考：**
- 为什么 Actor-Critic 的学习曲线更平滑？
- 价值函数学到了什么？
- 在什么情况下 REINFORCE 反而更好？

---

### 问题 2：GAE 的 λ 参数影响

**你的任务：** 测试不同的 λ 值，观察收敛行为。

**实验设计：**
```python
lambdas = [0.0, 0.5, 0.9, 0.95, 0.99]

for lam in lambdas:
    agent = A2C(policy, value_function, gae_lambda=lam)
    history = agent.train(env, num_episodes=500)

    # 记录：
    # 1. 收敛代数
    # 2. 最终性能
    # 3. 学习曲线方差
```

**深度思考：**
- λ = 0 时为什么收敛慢？
- λ = 1 时为什么方差大？
- 最优的 λ 是多少？

---

### 问题 3：策略网络架构的影响

**你的任务：** 对比不同的策略网络架构。

**实验设计：**
```python
# 离散动作
discrete_policy = DiscretePolicy(state_dim=4, action_dim=2)

# 连续动作（固定方差）
continuous_policy = ContinuousPolicy(state_dim=4, action_dim=1, std=0.5)

# 连续动作（学习方差）
gaussian_policy = GaussianPolicy(state_dim=4, action_dim=1)

# 对比性能和收敛速度
```

**深度思考：**
- 为什么连续动作需要不同的网络架构？
- 学习方差的优缺点是什么？
- 如何选择合适的架构？

---

## 第六层：通用设计模式（Design Patterns）

### 6.1 策略网络的多态设计

```python
class BasePolicy(ABC, nn.Module):
    @abstractmethod
    def sample(self, state) -> (action, log_prob)
    @abstractmethod
    def evaluate(self, state, action) -> (log_prob, entropy)

class DiscretePolicy(BasePolicy):
    # 离散动作：Categorical 分布

class ContinuousPolicy(BasePolicy):
    # 连续动作：固定方差高斯分布

class GaussianPolicy(BasePolicy):
    # 连续动作：学习方差高斯分布
```

**优势：**
- 统一接口
- 易于切换
- 支持不同动作空间

### 6.2 轨迹缓冲区模式

```python
class TrajectoryBuffer:
    def compute_returns(trajectory) -> returns
    def compute_advantages(trajectory, values) -> advantages
    def get_batch_with_advantages() -> (states, actions, advantages, log_probs)
```

**优势：**
- 集中管理计算逻辑
- 支持 GAE
- 易于扩展

### 6.3 Agent 基类模式

```python
class PolicyGradientAgent(ABC):
    @abstractmethod
    def compute_policy_loss(states, actions, returns, advantages)
    @abstractmethod
    def train_step(states, actions, returns, advantages)
```

**优势：**
- 统一训练接口
- 易于实现新算法
- 代码复用

---

## 第七层：记忆宫殿（Cheat Sheet）

### 快速参考表

| 算法 | 损失函数 | 基线 | 复杂度 | 适用 |
|------|--------|------|-------|------|
| REINFORCE | $-\log\pi \cdot G_t$ | 无 | O(T) | 简单 |
| Actor-Critic | $-\log\pi \cdot A$ | V(s) | O(T) | 中等 |
| A2C | $-\log\pi \cdot A^{GAE}$ | V(s) | O(T) | 通用 |

### 常见参数设置

```python
# 小规模问题
config = {
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'entropy_coeff': 0.01,
    'max_grad_norm': 1.0
}

# 大规模问题
config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'entropy_coeff': 0.001,
    'max_grad_norm': 0.5
}
```

### 调试检查清单

- [ ] 回报是否正确计算？（处理终止状态）
- [ ] 优势是否归一化？
- [ ] 梯度是否裁剪？
- [ ] 学习率是否合理？
- [ ] 熵系数是否合理？
- [ ] 是否在评估时禁用探索？
- [ ] 价值网络是否收敛？
- [ ] 策略是否过早收敛？

---

## 第八层：迁移学习指南

### 如何将策略梯度应用到新场景？

#### 场景 1：离散动作空间

```python
policy = DiscretePolicy(state_dim=obs_dim, action_dim=num_actions)
agent = A2C(policy, value_function)
```

#### 场景 2：连续动作空间

```python
# 固定方差
policy = ContinuousPolicy(state_dim=obs_dim, action_dim=action_dim, std=0.5)

# 学习方差
policy = GaussianPolicy(state_dim=obs_dim, action_dim=action_dim)
```

#### 场景 3：高维状态（图像）

```python
# 使用 CNN 特征提取
class CNNPolicy(BasePolicy):
    def __init__(self):
        self.cnn = CNN()  # 特征提取
        self.policy_head = MLP()  # 策略头
```

#### 场景 4：多任务学习

```python
# 共享特征层，任务特定的策略头
class MultiTaskPolicy(BasePolicy):
    def __init__(self):
        self.shared = MLP()  # 共享特征
        self.task_heads = {task: MLP() for task in tasks}
```

---

## 总结：从代码到直觉

### 为什么策略梯度工作？

1. **直接优化目标**：直接最大化期望回报，不需要中间表示
2. **灵活的策略**：支持任何可微策略（离散、连续、混合）
3. **理论保证**：策略梯度定理提供收敛性保证
4. **样本高效**：Actor-Critic 通过基线减少方差

### 何时使用策略梯度？

- **优势：** 连续动作、复杂策略、理论保证
- **劣势：** 样本效率低、训练不稳定、超参数敏感

### 核心设计哲学

> **"用 log 概率梯度加权的回报直接优化策略参数"**

策略梯度通过计算策略参数对期望回报的梯度，沿着回报增加的方向更新参数。这种直接优化方式比值函数方法更灵活，但需要更多样本。

---

## 参考文献

1. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3-4), 229-256.
2. Konda, V., & Tsitsiklis, J. N. (2000). Actor-Critic Algorithms. SIAM Journal on Control and Optimization, 42(4), 1143-1166.
3. Schulman, J., et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR.
4. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
5. Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML.
