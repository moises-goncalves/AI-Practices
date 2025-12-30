# 缓冲区与采样知识芯片

> 从均匀采样到优先级采样：样本效率的核心

---

## 第一部分：经验回放的本质

### 1.1 为什么需要经验回放？

**问题的本质**：
- 在线学习：每个转移只用一次
- 时序相关性：$\text{Cov}(x_t, x_{t+1}) \neq 0$
- SGD 假设：需要 i.i.d. 样本

**后果**：
- 训练不稳定
- 样本效率低
- 收敛缓慢

**解决方案**：经验回放
$$D = \{(s_t, a_t, r_t, s_{t+1}, d_t)\}_{t=1}^{N}$$
$$\text{Batch} \sim \text{Uniform}(D)$$

### 1.2 深度原理

**为什么有效**？

1. **打破时序相关性**
   - 均匀采样 → 样本独立
   - SGD 的 i.i.d. 假设成立
   - 梯度无偏

2. **提高样本效率**
   - 每个转移重复使用
   - 样本效率提升 3-5 倍
   - 减少环境交互

3. **稳定训练**
   - 梯度方差降低
   - 收敛更稳定
   - 超参数更鲁棒

**数学证明**（简化）：
$$\mathbb{E}[\nabla_\theta L] = \mathbb{E}_{(s,a,r,s') \sim D}[\nabla_\theta L(s,a,r,s')]$$

只有当 $(s,a,r,s')$ i.i.d. 时，梯度才无偏。

---

## 第二部分：均匀回放缓冲区

### 2.1 架构设计

**数据结构**：
```python
class Transition:
    state: np.ndarray        # s_t
    action: int              # a_t
    reward: float            # r_t
    next_state: np.ndarray   # s_{t+1}
    done: bool               # d_t
```

**缓冲区实现**：
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self._buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buffer.append(Transition(...))

    def sample(self, batch_size):
        batch = random.sample(self._buffer, batch_size)
        return states, actions, rewards, next_states, dones
```

### 2.2 复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| push() | $O(1)$ 摊销 | $O(1)$ |
| sample(N) | $O(N)$ | $O(N)$ |
| 总体 | - | $O(\text{capacity} \times d_s)$ |

**内存估算**：
- 状态维度：$d_s = 4$（CartPole）
- 缓冲区大小：$10^5$
- 每个转移：$4 \times 4 + 1 + 1 + 4 + 1 = 25$ 字节
- 总内存：$25 \times 10^5 = 2.5$ MB

### 2.3 架构陷阱

**陷阱1**：缓冲区大小选择
- 太小：样本多样性不足，过拟合
- 太大：内存浪费，采样变慢
- 推荐：$10^4 - 10^6$（根据环境复杂度）

**陷阱2**：FIFO 替换策略
- 问题：旧数据被覆盖，可能丢失重要转移
- 解决：使用优先级采样（PER）

**陷阱3**：采样偏差
- 问题：最后添加的转移采样概率高
- 解决：充分混合，使用大 batch size

---

## 第三部分：优先级经验回放（PER）

### 3.1 为什么需要优先级采样？

**问题的本质**：
- 均匀采样浪费低 TD 误差的转移
- 高 TD 误差的转移包含更多学习信号
- 应该优先采样

**深度原理**：
$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \epsilon$$

其中：
- $\delta_i = r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)$：TD 误差
- $\alpha \in [0, 1]$：优先级强度
- $\epsilon > 0$：数值稳定性常数

### 3.2 重要性采样修正

**问题**：优先级采样改变了数据分布

**后果**：梯度有偏，收敛性无保证

**解决方案**：重要性采样权重
$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta / \max_j w_j$$

**参数含义**：
- $\beta \in [0, 1]$：IS 修正强度
- $\beta = 0$：无修正（有偏）
- $\beta = 1$：完全修正（无偏）
- 推荐：从 0.4 线性退火到 1.0

**数学原理**：
$$\mathbb{E}[w_i \cdot L_i] = \mathbb{E}_{\text{uniform}}[L_i]$$

### 3.3 参数选择

**α 的含义**：
- $\alpha = 0$：均匀采样
- $\alpha = 1$：完全优先级
- 推荐：$\alpha = 0.6$

**β 的含义**：
- $\beta = 0$：无修正（有偏）
- $\beta = 1$：完全修正（无偏）
- 推荐：从 0.4 线性退火到 1.0

**ε 的含义**：
- 防止零优先级
- 推荐：$\epsilon = 1e-6$

### 3.4 架构陷阱

**陷阱1**：$\alpha$ 选择不当
- 太小（$\alpha \approx 0$）：接近均匀采样，无优先级效果
- 太大（$\alpha \approx 1$）：过度优先级，忽视低误差转移
- 推荐：$\alpha = 0.6$

**陷阱2**：$\beta$ 退火速度
- 太快：早期有偏差，收敛性差
- 太慢：晚期仍有偏差
- 推荐：线性退火，$\beta_{\text{start}} = 0.4$，$\beta_{\text{end}} = 1.0$

**陷阱3**：优先级更新延迟
- 问题：TD 误差计算后才更新优先级
- 后果：采样分布与实际误差不同步
- 解决：每次更新后立即更新优先级

### 3.5 实现优化

**朴素实现**：
```python
# 采样：O(capacity + batch_size)
priorities = self._priorities[:buffer_len]
probs = priorities ** self._alpha
probs = probs / probs.sum()
indices = np.random.choice(buffer_len, batch_size, p=probs)
```

**生产级实现**：使用 Sum-Tree
```
# 采样：O(log capacity)
# 更新：O(log capacity)
```

### 3.6 SumTree 数据结构

**目的**：高效采样和更新

**结构**：
- 叶节点：优先级值
- 内部节点：子节点和
- 根节点：总优先级

**操作复杂度**：
- 采样：$O(\log N)$
- 更新：$O(\log N)$

**vs 朴素实现**：
- 朴素：$O(N)$ 采样
- SumTree：$O(\log N)$ 采样

**SumTree 实现**：
```python
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def add(self, priority, data):
        """添加转移"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity

    def update(self, idx, priority):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def sample(self, batch_size):
        """采样批次"""
        batch_idx = []
        priorities = []
        for _ in range(batch_size):
            v = np.random.uniform(0, self.tree[0])
            idx = self._retrieve(0, v)
            batch_idx.append(idx)
            priorities.append(self.tree[idx + self.capacity - 1])
        return batch_idx, priorities

    def _retrieve(self, idx, v):
        """从根节点遍历到叶节点"""
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if v <= self.tree[left]:
            return self._retrieve(left, v)
        else:
            return self._retrieve(right, v - self.tree[left])
```

### 3.7 性能提升

**Atari 基准**：
- 标准 DQN：79%
- PER：141%（+79%）

---

## 第四部分：N-step 缓冲区

### 4.1 为什么需要 N-step？

**1-step TD**：
$$y_t = r_t + \gamma V(s_{t+1})$$
- 低方差（只依赖一步）
- 高偏差（依赖 V 的准确性）

**Monte Carlo**：
$$y_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$$
- 无偏（完整轨迹）
- 高方差（依赖所有未来奖励）

**最优**：
- 权衡偏差和方差
- 选择合适的 n

### 4.2 N-step 回报

**定义**：
$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$$

**特殊情况**：
- $n = 1$：1-step TD
- $n = \infty$：Monte Carlo

### 4.3 偏差-方差分析

**偏差**：
$$\text{Bias}[G_t^{(n)}] \propto \gamma^n \text{Bias}[V(s_{t+n})]$$

**方差**：
$$\text{Var}[G_t^{(n)}] \propto \sum_{k=0}^{n-1} \gamma^{2k} \text{Var}[r_{t+k}]$$

**权衡**：
- 增大 n：偏差减小，方差增大
- 减小 n：偏差增大，方差减小

### 4.4 最优 n 值

**经验规则**：
- $n = 1$：简单环境
- $n = 3-5$：中等环境（推荐）
- $n = 10+$：复杂环境

**Rainbow 使用**：$n = 3$

### 4.5 实现细节

**缓冲区**：需要存储 n 步转移

**计算**：
```python
G = 0
for k in range(n):
    G += gamma**k * rewards[t+k]
G += gamma**n * V(states[t+n])
```

**终止状态处理**：
```python
# 轨迹可能在 n 步内结束
for k in range(n):
    if dones[t+k]:
        # 提前截断
        G = sum(gamma**i * rewards[t+i] for i in range(k+1))
        break
else:
    # 完整 n 步
    G = sum(gamma**k * rewards[t+k] for k in range(n))
    G += gamma**n * V(states[t+n])
```

### 4.6 架构陷阱

**陷阱1**：n 值选择
- 太小：偏差大
- 太大：方差大
- 推荐：$n = 3$

**陷阱2**：缓冲区大小
- 需要存储 n 步转移
- 内存开销：$O(n \times \text{buffer\_size})$
- 推荐：$n = 3$

**陷阱3**：终止状态处理
- 问题：轨迹可能在 n 步内结束
- 解决：提前截断，调整 $\gamma$

### 4.7 性能提升

**Atari 基准**：
- 标准 DQN：79%
- N-step DQN：~110%（+39%）

---

## 第五部分：采样策略对比

### 5.1 对比矩阵

| 特性 | 均匀回放 | 优先级回放 | N-step 缓冲 |
|------|---------|-----------|-----------|
| 采样方式 | 均匀随机 | 按 TD 误差 | 顺序（轨迹） |
| 时间复杂度 | $O(N)$ | $O(\log N)$ | $O(1)$ |
| 空间复杂度 | $O(N)$ | $O(N)$ | $O(n \times N)$ |
| 适用算法 | DQN | DQN/Rainbow | DQN/Rainbow |
| 样本效率 | 基准 | 2-3 倍 | 1.5 倍 |
| 实现复杂度 | 简单 | 中等 | 简单 |

### 5.2 选择指南

**使用均匀回放**：
- 算法：DQN, Double DQN, Dueling DQN
- 环境：简单，样本充足
- 优点：实现简单，稳定

**使用优先级回放**：
- 算法：Rainbow, 高级 DQN
- 环境：复杂，样本稀缺
- 优点：样本效率高，收敛快

**使用 N-step 缓冲**：
- 算法：Rainbow, 所有 DQN 变体
- 环境：任何
- 优点：偏差-方差平衡，性能提升

---

## 第六部分：架构陷阱与调试

### 6.1 常见问题

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| 样本多样性不足 | 缓冲区太小 | 增大 capacity |
| 内存溢出 | 缓冲区太大 | 减小 capacity 或分布式存储 |
| 采样偏差 | batch_size 太小 | 增大 batch_size |
| PER 不稳定 | α 或 β 选择不当 | 调整参数，线性退火 β |
| N-step 方差高 | n 太大 | 减小 n（推荐 3） |
| N-step 偏差高 | n 太小 | 增大 n（推荐 3） |

### 6.2 调试技巧

**1. 检查采样分布**
```python
# 均匀回放
indices = []
for _ in range(10000):
    batch = buffer.sample(1)
    indices.append(batch[0])
plt.hist(indices, bins=100)  # 应该均匀
```

**2. 监控优先级**
```python
# PER
priorities = buffer._priorities[:len(buffer)]
print(f"Priority stats: min={priorities.min()}, max={priorities.max()}, mean={priorities.mean()}")
```

**3. 验证 N-step 计算**
```python
# N-step 缓冲
returns, advantages = buffer.compute_nstep(last_value=0.0)
print(f"Return stats: mean={returns.mean()}, std={returns.std()}")
# 应该接近 0 均值，单位方差
```

---

## 第七部分：记忆宫殿（设计模式）

### 7.1 通用缓冲区接口

```python
class Buffer:
    def push(self, state, action, reward, next_state, done):
        """添加转移"""
        pass

    def sample(self, batch_size):
        """采样批次"""
        pass

    def is_ready(self, batch_size):
        """检查是否准备好"""
        return len(self) >= batch_size

    def __len__(self):
        """返回缓冲区大小"""
        pass
```

### 7.2 采样流程模板

```python
# 训练循环
while not done:
    # 1. 采样
    if buffer.is_ready(batch_size):
        batch = buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch

        # 2. 计算目标
        targets = compute_targets(batch)

        # 3. 更新网络
        loss = network(states, targets)
        optimizer.step()

    # 4. 添加新转移
    next_state, reward, done, _ = env.step(action)
    buffer.push(state, action, reward, next_state, done)
```

---

## 第八部分：前沿演进

### 8.1 最新研究方向

**2020-2024**：
- 自适应优先级调整
- 与分布式 RL 的结合
- 多步回报与 PER 的交互
- 离线 RL 中的缓冲区设计

### 8.2 离线 RL 的特殊考虑

**问题**：
- 缓冲区是固定的（不再添加新数据）
- 分布外动作可能导致过估计

**解决方案**：
- 约束学习（Conservative Q-Learning）
- 行为克隆正则化
- 不确定性估计

---

## 核心心法

**缓冲区设计的三个原则**：
1. 均匀回放：简单稳定，适合 off-policy
2. 优先级回放：高效率，适合复杂环境
3. N-step 缓冲：偏差-方差平衡，适合所有算法

**记住这三句话**：
1. 经验回放打破时序相关性，稳定训练
2. 优先级采样提升样本效率，加速收敛
3. N-step 学习权衡偏差与方差，改进性能

---

[返回上级](../README.md)
