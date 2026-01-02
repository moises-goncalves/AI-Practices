# 策略梯度网络架构与实现细节

> 从网络设计到生产级代码：架构模式与最佳实践

---

## 第一部分：策略网络架构

### 1.1 离散动作策略网络

**架构**：
```
输入 s ∈ ℝ^{d_s}
    ↓
FC(d_s → d_h) + ReLU
    ↓
FC(d_h → d_h) + ReLU
    ↓
FC(d_h → |A|)
    ↓
Softmax
    ↓
输出 π(a|s) ∈ ℝ^{|A|}
```

**代码实现**：
```python
class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        logits = self.net(state)
        return logits

    def sample(self, state):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, state, action):
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
```

**关键点**：
- 输出 logits，不是概率
- Softmax 在分布中处理
- 支持批量操作

### 1.2 连续动作策略网络

**固定方差版本**：
```
输入 s ∈ ℝ^{d_s}
    ↓
FC(d_s → d_h) + ReLU
    ↓
FC(d_h → d_h) + ReLU
    ↓
FC(d_h → |A|)
    ↓
Tanh（限制在 [-1, 1]）
    ↓
输出 μ(s) ∈ ℝ^{|A|}
```

**代码实现**：
```python
class ContinuousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64), std=0.5):
        super().__init__()
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.std = std

    def forward(self, state):
        mean = self.net(state)
        return mean

    def sample(self, state):
        mean = self.forward(state)
        dist = Normal(mean, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, state, action):
        mean = self.forward(state)
        dist = Normal(mean, self.std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
```

**关键点**：
- Tanh 限制输出在 [-1, 1]
- 固定方差简单稳定
- 支持多维动作

### 1.3 学习方差策略网络

**架构**：
```
输入 s ∈ ℝ^{d_s}
    ↓
共享特征层
    ↓
    ├─ 均值头：FC → Tanh → μ(s)
    │
    └─ 方差头：FC → Softplus → σ(s)
    ↓
输出 (μ(s), σ(s))
```

**代码实现**：
```python
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64)):
        super().__init__()

        # 共享特征层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # 均值头
        self.mean_head = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()
        )

        # 方差头
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        features = self.shared(state)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
```

**关键点**：
- 学习方差提高灵活性
- 用 log_std 参数化避免数值问题
- Softplus 确保方差为正

---

## 第二部分：价值网络架构

### 2.1 标准价值网络

**架构**：
```
输入 s ∈ ℝ^{d_s}
    ↓
FC(d_s → d_h) + ReLU
    ↓
FC(d_h → d_h) + ReLU
    ↓
FC(d_h → 1)
    ↓
输出 V(s) ∈ ℝ
```

**代码实现**：
```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        value = self.net(state)
        return value.squeeze(-1)
```

**关键点**：
- 输出单个标量值
- 无输出激活（值无界）
- squeeze 处理维度

### 2.2 对偶价值网络（Dueling Architecture）

**架构**：
```
输入 s ∈ ℝ^{d_s}
    ↓
共享特征层
    ↓
    ├─ V 流：FC → ReLU → FC → V(s)
    │
    └─ A 流：FC → ReLU → FC → A(s,a)
    ↓
聚合：Q(s,a) = V(s) + (A(s,a) - mean(A))
```

**代码实现**：
```python
class DuelingValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64)):
        super().__init__()

        # 共享特征层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # V 流
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

        # A 流
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )

    def forward(self, state):
        features = self.shared(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # 聚合：Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values.squeeze(-1)
```

**关键点**：
- 分离价值和优势流
- 均值约束确保唯一分解
- 提高学习效率

---

## 第三部分：轨迹缓冲区与数据处理

### 3.1 轨迹数据结构

**定义**：
```python
@dataclass
class Trajectory:
    states: np.ndarray          # (T, d_s)
    actions: np.ndarray         # (T,)
    rewards: np.ndarray         # (T,)
    log_probs: np.ndarray       # (T,)
    values: np.ndarray          # (T,)
    dones: np.ndarray           # (T,)
    next_states: np.ndarray     # (T, d_s)
```

### 3.2 回报计算

**蒙特卡洛回报**：
```python
def compute_returns(trajectory, gamma=0.99):
    """计算蒙特卡洛回报"""
    returns = np.zeros(len(trajectory.rewards))
    cumulative_return = 0.0

    for t in reversed(range(len(trajectory.rewards))):
        cumulative_return = trajectory.rewards[t] + gamma * cumulative_return * (1 - trajectory.dones[t])
        returns[t] = cumulative_return

    return returns
```

**关键点**：
- 处理终止状态：乘以 (1 - done)
- 反向遍历计算
- 时间复杂度 O(T)

### 3.3 GAE 计算

**广义优势估计**：
```python
def compute_gae(trajectory, values, gamma=0.99, lambda_=0.95):
    """计算广义优势估计"""
    advantages = np.zeros(len(trajectory.rewards))
    gae = 0.0

    # 添加最后一个值（用于 bootstrap）
    values = np.append(values, 0.0)

    for t in reversed(range(len(trajectory.rewards))):
        delta = trajectory.rewards[t] + gamma * values[t+1] * (1 - trajectory.dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - trajectory.dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns
```

**关键点**：
- 递推计算高效
- 处理终止状态
- 返回优势和回报

### 3.4 优势归一化

**为什么需要**：
- 稳定梯度
- 加速收敛
- 减少超参数敏感性

**实现**：
```python
def normalize_advantages(advantages):
    """归一化优势"""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

---

## 第四部分：训练循环与优化

### 4.1 REINFORCE 训练循环

```python
def train_reinforce(agent, env, num_episodes=100):
    for episode in range(num_episodes):
        # 1. 收集轨迹
        trajectory, episode_return = agent.collect_trajectory(env)

        # 2. 计算回报
        returns = compute_returns(trajectory)

        # 3. 归一化
        returns_normalized = normalize_advantages(returns)

        # 4. 计算损失
        states = torch.FloatTensor(trajectory.states)
        actions = torch.LongTensor(trajectory.actions)
        returns_t = torch.FloatTensor(returns_normalized)

        log_probs, entropy = agent.policy.evaluate(states, actions)
        policy_loss = -(log_probs * returns_t).mean()
        entropy_loss = -agent.entropy_coeff * entropy.mean()

        # 5. 梯度更新
        total_loss = policy_loss + entropy_loss
        agent.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=1.0)
        agent.optimizer.step()
```

### 4.2 Actor-Critic 训练循环

```python
def train_actor_critic(agent, env, num_episodes=100):
    for episode in range(num_episodes):
        # 1. 收集轨迹
        trajectory, episode_return = agent.collect_trajectory(env)

        # 2. 计算优势
        states = torch.FloatTensor(trajectory.states)
        values = agent.value_fn(states).detach().numpy()
        advantages = compute_advantages(trajectory, values)
        advantages_normalized = normalize_advantages(advantages)

        # 3. 计算回报
        returns = advantages + values

        # 4. 更新 Critic
        returns_t = torch.FloatTensor(returns)
        value_pred = agent.value_fn(states)
        value_loss = F.mse_loss(value_pred, returns_t)

        agent.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(agent.value_fn.parameters(), max_norm=1.0)
        agent.value_optimizer.step()

        # 5. 更新 Actor
        actions = torch.LongTensor(trajectory.actions)
        advantages_t = torch.FloatTensor(advantages_normalized)

        log_probs, entropy = agent.policy.evaluate(states, actions)
        policy_loss = -(log_probs * advantages_t).mean()
        entropy_loss = -agent.entropy_coeff * entropy.mean()

        total_loss = policy_loss + entropy_loss
        agent.policy_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=1.0)
        agent.policy_optimizer.step()
```

### 4.3 A2C 训练循环

```python
def train_a2c(agent, env, num_episodes=100):
    for episode in range(num_episodes):
        # 1. 收集轨迹
        trajectory, episode_return = agent.collect_trajectory(env)

        # 2. 计算 GAE
        states = torch.FloatTensor(trajectory.states)
        values = agent.value_fn(states).detach().numpy()
        advantages, returns = compute_gae(trajectory, values)
        advantages_normalized = normalize_advantages(advantages)

        # 3. 批量更新
        states_t = torch.FloatTensor(trajectory.states)
        actions_t = torch.LongTensor(trajectory.actions)
        advantages_t = torch.FloatTensor(advantages_normalized)
        returns_t = torch.FloatTensor(returns)

        # 更新 Critic
        value_pred = agent.value_fn(states_t)
        value_loss = F.mse_loss(value_pred, returns_t)

        agent.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(agent.value_fn.parameters(), max_norm=1.0)
        agent.value_optimizer.step()

        # 更新 Actor
        log_probs, entropy = agent.policy.evaluate(states_t, actions_t)
        policy_loss = -(log_probs * advantages_t).mean()
        entropy_loss = -agent.entropy_coeff * entropy.mean()

        total_loss = policy_loss + entropy_loss
        agent.policy_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=1.0)
        agent.policy_optimizer.step()
```

---

## 第五部分：数值稳定性与优化技巧

### 5.1 梯度裁剪

**为什么需要**：
- 防止梯度爆炸
- 稳定训练
- 提高收敛性

**实现**：
```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**推荐值**：
- 策略梯度：0.5 ~ 1.0
- 价值网络：1.0 ~ 10.0

### 5.2 学习率调度

**固定学习率**：
```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

**衰减学习率**：
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
```

**推荐策略**：
- 早期：常数学习率
- 中期：缓慢衰减
- 晚期：固定小学习率

### 5.3 批归一化与层归一化

**批归一化**：
```python
nn.BatchNorm1d(hidden_dim)
```

**层归一化**：
```python
nn.LayerNorm(hidden_dim)
```

**在 RL 中的注意**：
- 批归一化可能改变数据分布
- 层归一化更稳定
- 推荐：使用层归一化或不用

---

## 第六部分：记忆宫殿（设计模式）

### 6.1 通用策略网络接口

```python
class BasePolicy(ABC, nn.Module):
    @abstractmethod
    def sample(self, state) -> Tuple[Tensor, Tensor]:
        """采样动作和 log_prob"""
        pass

    @abstractmethod
    def evaluate(self, state, action) -> Tuple[Tensor, Tensor]:
        """计算 log_prob 和熵"""
        pass
```

### 6.2 通用价值网络接口

```python
class BaseValueFunction(ABC, nn.Module):
    @abstractmethod
    def forward(self, state) -> Tensor:
        """计算状态价值"""
        pass
```

### 6.3 通用 Agent 接口

```python
class PolicyGradientAgent(ABC):
    @abstractmethod
    def collect_trajectory(self, env) -> Trajectory:
        """收集轨迹"""
        pass

    @abstractmethod
    def train_step(self, trajectory) -> Dict:
        """单步训练"""
        pass
```

---

## 核心心法

**网络设计的三个原则**：
1. **架构简洁**：两层隐层通常足够
2. **初始化正确**：正交初始化 + 合适的增益
3. **数值稳定**：梯度裁剪、学习率调度、归一化

---

[返回上级](../README.md)
