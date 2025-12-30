# 网络架构知识芯片

> 从标准 MLP 到 Rainbow 网络：架构设计与初始化

---

## 第一部分：基础网络架构

### 1.1 标准 DQN 网络

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
输出 Q(s, ·) ∈ ℝ^{|A|}
```

**设计原则**：
- 两个隐层：足够表达能力，避免过度参数化
- ReLU 激活：稀疏表示，计算高效
- 无输出激活：Q 值无界

**复杂度分析**：
- 参数数量：$d_s \cdot d_h + d_h^2 + d_h \cdot |A|$
- 前向传播：$O(d_s \cdot d_h + d_h^2 + d_h \cdot |A|)$
- 反向传播：$O(d_s \cdot d_h + d_h^2 + d_h \cdot |A|)$

### 1.2 权重初始化的深度原理

**问题**：
- 初始权重太小 → 梯度消失
- 初始权重太大 → 梯度爆炸

**解决方案**：正交初始化

**正交初始化的数学原理**：
$$W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}, \quad W^T W = I$$

**为什么有效**？
- 保范性：$\|Wx\|_2 = \|x\|_2$
- 梯度流：$\|\nabla_x L\|_2 = \|\nabla_W L\|_2$
- 深层网络中梯度不消失也不爆炸

**增益缩放**：
$$\text{Var}[y] = \text{gain}^2 \times \text{Var}[x]$$

**推荐增益值**：

| 激活函数 | 增益 | 原因 |
|---------|------|------|
| ReLU | $\sqrt{2}$ | He 初始化等价 |
| Tanh | $\sqrt{5/3} \approx 1.29$ | Sigmoid 族 |
| 线性 | 1.0 | 无激活 |
| 策略输出 | 0.01 | 初始动作小 |
| 价值输出 | 1.0 | 标准初始化 |

**代码模式**：
```python
def init_weights(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
```

### 1.3 架构陷阱

**陷阱1**：网络容量不足
- 问题：$d_h$ 太小 → 无法学习复杂 Q 函数
- 后果：收敛到次优策略
- 解决：$d_h = 128-512$（根据环境复杂度）

**陷阱2**：过拟合
- 问题：$d_h$ 太大 → 过拟合训练数据
- 后果：泛化性差
- 解决：使用 dropout 或早停

**陷阱3**：激活函数选择
- ReLU：稀疏，快速，但可能死亡
- Tanh：平滑，但计算慢
- 推荐：ReLU（RL 中标准）

---

## 第二部分：Dueling 网络架构

### 2.1 架构设计

**结构**：
```
输入 s ∈ ℝ^{d_s}
    ↓
共享特征层：FC(d_s → d_h) + ReLU
    ↓
    ├─ V 流：FC(d_h → d_h) + ReLU → FC(d_h → 1)
    │
    └─ A 流：FC(d_h → d_h) + ReLU → FC(d_h → |A|)
    ↓
聚合：Q(s,a) = V(s) + (A(s,a) - mean(A))
    ↓
输出 Q(s, ·) ∈ ℝ^{|A|}
```

### 2.2 聚合函数的实现

**标准聚合**：
```python
value = self.value_stream(features)  # (batch, 1)
advantage = self.advantage_stream(features)  # (batch, |A|)

# 聚合：Q = V + (A - mean(A))
q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
```

**为什么要减去均值**？
- 确保唯一分解
- 稳定优化
- 改进收敛性

### 2.3 参数数量对比

| 方面 | DQN | Dueling DQN |
|------|-----|------------|
| 参数数量 | $d_h \times (d_s + \|A\|)$ | $d_h \times (d_s + d_h + \|A\| + 1)$ |
| 参数增长 | ~1.5 倍 | ~1.5 倍 |
| 计算开销 | 基准 | +5-10% |
| 样本效率 | 基准 | $\|A\|$ 倍（V 流） |

### 2.4 初始化策略

**共享层**：
```python
self.shared.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
```

**V 流**：
```python
init_weights(self.value_stream, gain=1.0)  # 标准初始化
```

**A 流**：
```python
init_weights(self.advantage_stream, gain=0.01)  # 小增益
```

**原因**：
- V 流：无特殊约束
- A 流：初始优势应接近 0

---

## 第三部分：Noisy 网络架构

### 3.1 Noisy Linear 层

**标准 Linear 层**：
$$y = Wx + b$$

**Noisy Linear 层**：
$$y = (\mu^w + \sigma^w \odot \varepsilon^w) x + (\mu^b + \sigma^b \odot \varepsilon^b)$$

其中：
- $\mu^w, \mu^b$：可学习的均值
- $\sigma^w, \sigma^b$：可学习的噪声尺度
- $\varepsilon^w, \varepsilon^b$：随机噪声 $\sim \mathcal{N}(0, 1)$

### 3.2 因式分解噪声（高效参数化）

**朴素方法**：
- 参数数量：$O(pq)$（$p \times q$ 权重矩阵）
- 计算复杂度：高

**因式分解方法**：
$$\varepsilon_{ij} = f(\varepsilon_i) \cdot f(\varepsilon_j), \quad f(x) = \text{sign}(x)\sqrt{|x|}$$

**优势**：
- 参数数量：$O(p + q)$（降低 $|A|$ 倍）
- 计算复杂度：$O(p + q)$
- 性能无损失

**代码实现**：
```python
def factorized_noise(size):
    """生成因式分解噪声"""
    x = torch.randn(size)
    return torch.sign(x) * torch.sqrt(torch.abs(x))

# 使用
noise_in = factorized_noise(input_size)
noise_out = factorized_noise(output_size)
eps = torch.outer(noise_out, noise_in)
```

### 3.3 Noisy 网络架构

**结构**：
```
输入 s ∈ ℝ^{d_s}
    ↓
标准 FC(d_s → d_h) + ReLU
    ↓
Noisy FC(d_h → d_h) + ReLU
    ↓
Noisy FC(d_h → |A|)
    ↓
输出 Q(s, ·) ∈ ℝ^{|A|}
```

**为什么只在最后两层用 Noisy**？
- 前层：特征提取，不需要噪声
- 后层：决策层，需要参数化探索

### 3.4 噪声采样频率

**选项1**：每步采样
```python
def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.noisy_fc2(x)  # 采样新噪声
    x = F.relu(x)
    x = self.noisy_fc3(x)  # 采样新噪声
    return x
```

**选项2**：每 episode 采样
```python
def reset_noise(self):
    """在 episode 开始时采样一次"""
    self.noisy_fc2.sample_noise()
    self.noisy_fc3.sample_noise()
```

**推荐**：每步采样（更好的探索）

### 3.5 σ 初始化

**推荐值**：$\sigma_0 = 0.5$

**原因**：
- 太大：初期过度探索
- 太小：初期探索不足
- 0.5：平衡

**代码**：
```python
self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
```

---

## 第四部分：Categorical 网络架构 (C51)

### 4.1 网络结构

**标准 DQN**：
```
输入 s ∈ ℝ^{d_s}
    ↓
FC(d_s → d_h) + ReLU
    ↓
FC(d_h → d_h) + ReLU
    ↓
FC(d_h → |A|)
    ↓
输出 Q(s, ·) ∈ ℝ^{|A|}
```

**Categorical DQN**：
```
输入 s ∈ ℝ^{d_s}
    ↓
FC(d_s → d_h) + ReLU
    ↓
FC(d_h → d_h) + ReLU
    ↓
FC(d_h → |A| × N)
    ↓
Reshape: (batch, |A|, N)
    ↓
Softmax over atoms
    ↓
输出 Z(s, ·) ∈ ℝ^{|A| × N}
```

### 4.2 输出解释

**标准 DQN**：
- 输出：$Q(s, a)$ 标量
- 含义：动作 a 的期望价值

**Categorical DQN**：
- 输出：$p_i(s, a)$ 概率向量
- 含义：动作 a 的回报分布

**转换**：
```python
# 从分布计算期望
q_values = (probabilities * support).sum(dim=-1)  # (batch, |A|)
```

### 4.3 支撑点设计

**固定支撑点**：
$$z_i = V_{\min} + i \cdot \Delta z, \quad \Delta z = \frac{V_{\max} - V_{\min}}{N - 1}$$

**推荐参数**：

| 参数 | 推荐值 | 范围 |
|------|--------|------|
| N (atoms) | 51 | 21 ~ 101 |
| $V_{\min}$ | -10 | -20 ~ -5 |
| $V_{\max}$ | 10 | 5 ~ 20 |

**代码**：
```python
self.support = torch.linspace(v_min, v_max, num_atoms)
self.delta_z = (v_max - v_min) / (num_atoms - 1)
```

### 4.4 投影操作的实现

**目标**：将 $r + \gamma z_j$ 投影到支撑点

**步骤**：
1. 计算 $\mathcal{T}z_j = r + \gamma z_j$
2. 裁剪到 $[V_{\min}, V_{\max}]$
3. 线性插值到最近的支撑点

**代码**：
```python
def project_distribution(self, rewards, next_probs, dones):
    """投影分布到支撑点"""
    batch_size = rewards.shape[0]

    # 计算投影目标
    Tz = rewards + self.gamma * self.support * (1 - dones)
    Tz = Tz.clamp(self.v_min, self.v_max)

    # 计算投影索引
    b = (Tz - self.v_min) / self.delta_z
    l = b.floor().long()
    u = b.ceil().long()

    # 线性插值
    ml = (u.float() - b) * next_probs
    mu = (b - l.float()) * next_probs

    # 累积到目标分布
    m = torch.zeros_like(next_probs)
    for i in range(batch_size):
        m[i].scatter_add_(0, l[i], ml[i])
        m[i].scatter_add_(0, u[i], mu[i])

    return m
```

### 4.5 损失函数

**KL 散度**：
```python
loss = F.kl_div(
    F.log_softmax(logits, dim=-1),
    target_distribution,
    reduction='batchmean'
)
```

**为什么用 KL 而不是 MSE**？
- KL：分布匹配
- MSE：点匹配
- KL 更适合概率分布

---

## 第五部分：Rainbow 网络架构

### 5.1 完整架构

**Rainbow 网络 = Dueling + Noisy + Categorical**

**结构**：
```
输入 s ∈ ℝ^{d_s}
    ↓
共享特征层：FC(d_s → d_h) + ReLU
    ↓
    ├─ V 流：Noisy FC(d_h → d_h) + ReLU → Noisy FC(d_h → 1)
    │
    └─ A 流：Noisy FC(d_h → d_h) + ReLU → Noisy FC(d_h → |A| × N)
    ↓
聚合：Q(s,a) = V(s) + (A(s,a) - mean(A))
    ↓
Softmax over atoms
    ↓
输出 Z(s, ·) ∈ ℝ^{|A| × N}
```

### 5.2 参数数量

**对比**：

| 网络 | 参数数量 | 相对大小 |
|------|---------|---------|
| DQN | $d_h(d_s + \|A\|)$ | 基准 |
| Dueling | $d_h(d_s + d_h + \|A\| + 1)$ | 1.5 倍 |
| Noisy | $2 \times d_h(d_s + \|A\|)$ | 2 倍 |
| C51 | $d_h(d_s + \|A\| \times N)$ | $N$ 倍 |
| Rainbow | $2 \times d_h(d_s + d_h + \|A\| \times N + 1)$ | $2N$ 倍 |

### 5.3 计算开销

**前向传播时间**：

| 网络 | 相对时间 |
|------|---------|
| DQN | 1.0x |
| Dueling | 1.1x |
| Noisy | 1.5x |
| C51 | 1.2x |
| Rainbow | 2.0-2.5x |

### 5.4 内存开销

**GPU 内存**：

| 网络 | 相对内存 |
|------|---------|
| DQN | 1.0x |
| Dueling | 1.2x |
| Noisy | 1.5x |
| C51 | 1.3x |
| Rainbow | 1.8-2.0x |

---

## 第六部分：数值稳定性技巧

### 6.1 梯度裁剪

**问题**：梯度爆炸导致参数更新过大

**解决方案**：
```python
nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
```

**推荐值**：
- DQN：10.0
- PPO：0.5
- A2C：0.5

### 6.2 学习率调度

**问题**：固定学习率可能不适应训练阶段

**解决方案**：
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
```

**推荐策略**：
- 早期：常数学习率
- 中期：缓慢衰减
- 晚期：固定小学习率

### 6.3 优化器选择

**对比分析**：

| 优化器 | 优点 | 缺点 | 推荐场景 |
|------|------|------|---------|
| SGD | 简单，稳定 | 收敛慢 | 基准 |
| Adam | 自适应，快速 | 可能不稳定 | 大多数 RL |
| RMSprop | 平衡 | 超参数多 | 旧代码 |

**推荐**：Adam with $\epsilon = 1e-5$（RL 中的标准）

### 6.4 批归一化的注意事项

**在 RL 中要谨慎**：
- BN 改变了数据分布
- 在 off-policy 学习中可能导致偏差
- 推荐：使用 Layer Normalization 或不用

---

## 第七部分：架构陷阱与调试

### 7.1 常见问题

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| 梯度消失 | 初始化不当 | 使用正交初始化 |
| 梯度爆炸 | 学习率太大 | 梯度裁剪，降低 LR |
| Q 值发散 | 目标网络更新频率不当 | 调整 target_update_freq |
| 策略退化 | 熵正则不足 | 增大 entropy_coef |
| 过拟合 | 网络容量过大 | 减小 hidden_dim 或加 dropout |

### 7.2 调试技巧

**1. 检查激活值分布**
```python
for name, param in network.named_parameters():
    print(f"{name}: mean={param.mean():.4f}, std={param.std():.4f}")
```

**2. 监控梯度范数**
```python
total_norm = 0
for p in network.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"Gradient norm: {total_norm:.4f}")
```

**3. 可视化网络输出**
```python
import matplotlib.pyplot as plt
q_values = network(state_batch).detach().cpu().numpy()
plt.hist(q_values.flatten(), bins=50)
plt.show()
```

---

## 第八部分：记忆宫殿（设计模式）

### 8.1 通用网络模板

```python
class RLNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # 特征提取
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 任务特定头
        self.head = nn.Linear(hidden_dim, action_dim)

        # 初始化
        self.feature.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        init_weights(self.head, gain=0.01)

    def forward(self, x):
        features = self.feature(x)
        return self.head(features)
```

### 8.2 初始化检查清单

- [ ] 使用正交初始化
- [ ] 为不同层设置不同增益
- [ ] 检查激活值范围
- [ ] 验证梯度流向
- [ ] 监控初始损失

---

## 核心心法

**网络设计的三个原则**：
1. 初始化决定收敛速度（正交 + 增益）
2. 架构决定学习效率（共享 vs 分离）
3. 激活函数决定稳定性（ReLU vs Tanh）

---

[返回上级](../README.md)
