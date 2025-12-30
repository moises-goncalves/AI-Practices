# 策略梯度高级知识芯片

> 从基础算法到生产级实现：深度原理、架构陷阱与前沿演进

---

## 第一部分：策略梯度的数学基础

### 1.1 策略梯度定理的完整推导

**目标函数**：
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{s_0}[V^{\pi_\theta}(s_0)]$$

其中 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ 是轨迹。

**策略梯度定理（Policy Gradient Theorem）**：
$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)]$$

**完整推导**：

1. **对目标函数求导**：
$$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{s_0}[V^{\pi_\theta}(s_0)]$$

2. **展开价值函数**：
$$V^{\pi_\theta}(s) = \mathbb{E}_{a \sim \pi_\theta}[\sum_{t=0}^\infty \gamma^t r_t | s_0 = s]$$

3. **利用 log-trick**：
$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$$

4. **最终得到**：
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)]$$

**关键性质**：
- 不需要对环境模型求导
- 只需要对策略网络求导
- 适用于任何可微策略

### 1.2 无偏性与方差分析

**无偏性证明**：
$$\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) G_t] = \nabla_\theta J(\theta)$$

其中 $G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}$ 是蒙特卡洛回报。

**方差分析**：
$$\text{Var}[\nabla_\theta \log \pi_\theta(a|s) G_t] = \mathbb{E}[(\nabla_\theta \log \pi_\theta(a|s) G_t)^2] - (\nabla_\theta J(\theta))^2$$

**方差来源**：
- $G_t$ 的高方差：单个轨迹的回报波动大
- 策略梯度的高方差：导致训练不稳定

**方差缩减方法**：
1. 基线减法：$A(s,a) = Q(s,a) - V(s)$
2. GAE：加权组合 n-step returns
3. 重要性采样：off-policy 学习

---

## 第二部分：三大核心算法的深度对比

### 2.1 REINFORCE vs Actor-Critic vs A2C

**对比矩阵**：

| 维度 | REINFORCE | Actor-Critic | A2C |
|------|-----------|--------------|-----|
| **目标函数** | $-\mathbb{E}[\log \pi \cdot G_t]$ | $-\mathbb{E}[\log \pi \cdot A]$ | $-\mathbb{E}[\log \pi \cdot A^{GAE}]$ |
| **基线** | 无 | V(s) | V(s) + GAE |
| **方差** | 高 | 中 | 低 |
| **偏差** | 无 | 低 | 低 |
| **收敛速度** | 慢 | 中 | 快 |
| **样本复杂度** | $O(1/\epsilon^2)$ | $O(1/\epsilon)$ | $O(1/\epsilon)$ |
| **实现复杂度** | 简单 | 中等 | 复杂 |
| **并行支持** | 否 | 否 | 是 |

### 2.2 方差-偏差权衡的深度分析

**REINFORCE**：
- 无偏：$\mathbb{E}[\nabla_\theta \log \pi \cdot G_t] = \nabla_\theta J(\theta)$
- 高方差：$\text{Var}[G_t]$ 很大
- 收敛慢：需要大量样本

**Actor-Critic**：
- 低偏差：$V(s)$ 是 $Q(s,a)$ 的无偏估计
- 中方差：$\text{Var}[A(s,a)] < \text{Var}[G_t]$
- 收敛快：需要较少样本

**A2C**：
- 低偏差：GAE 平衡偏差和方差
- 低方差：加权组合 n-step returns
- 最快收敛：最高样本效率

### 2.3 GAE 的 λ 参数详解

**GAE 公式**：
$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$

其中 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

**λ 的含义**：
- $\lambda = 0$：纯 TD（$\hat{A}_t = \delta_t$）
  - 低方差，高偏差
  - 收敛快，但可能不准确

- $\lambda = 1$：纯 MC（$\hat{A}_t = R_t - V(s_t)$）
  - 无偏，高方差
  - 收敛慢，但最终准确

- $\lambda = 0.95$：最优平衡（推荐）
  - 中等方差，低偏差
  - 快速且准确的收敛

**递推计算**（高效）：
```python
advantages = np.zeros(T)
gae = 0.0

for t in reversed(range(T)):
    delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
    gae = delta + gamma * lambda * (1 - dones[t]) * gae
    advantages[t] = gae
```

---

## 第三部分：架构陷阱与工业部署

### 3.1 策略网络的陷阱

**陷阱1：离散 vs 连续动作**

离散动作：
```python
# 输出 logits，用 softmax 转换为概率
logits = network(state)
dist = Categorical(logits=logits)
action = dist.sample()
log_prob = dist.log_prob(action)
```

连续动作：
```python
# 输出均值和标准差
mean = network(state)
std = fixed_std  # 或 exp(log_std)
dist = Normal(mean, std)
action = dist.sample()
log_prob = dist.log_prob(action)
```

**陷阱2：固定方差 vs 学习方差**

固定方差：
- 优点：简单，参数少，训练稳定
- 缺点：探索能力受限，无法自适应

学习方差：
- 优点：自适应探索，更灵活
- 缺点：需要额外参数，训练不稳定

**陷阱3：输出层激活函数**

```python
# 连续动作：用 Tanh 限制在 [-1, 1]
output = nn.Tanh()(linear_layer(x))

# 离散动作：无激活（softmax 在分布中处理）
output = linear_layer(x)
```

### 3.2 回报和优势的陷阱

**陷阱1：回报计算中的终止状态**

```python
# 错误：不处理终止状态
cumulative_return = reward + gamma * cumulative_return

# 正确：终止时不累积
cumulative_return = reward + gamma * cumulative_return * (1 - done)
```

**陷阱2：优势归一化**

```python
# 必须归一化，否则梯度不稳定
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**陷阱3：GAE 的 λ 参数**

- λ 过小（< 0.9）：偏差大，收敛慢
- λ 过大（> 0.99）：方差大，训练不稳定
- 推荐：λ = 0.95

### 3.3 训练稳定性的陷阱

**陷阱1：梯度裁剪**

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
```

- 作用：防止梯度爆炸
- 必要性：策略梯度容易产生大梯度

**陷阱2：学习率**

- 过大：策略振荡，无法收敛
- 过小：收敛慢，样本浪费
- 推荐：1e-3 到 1e-4

**陷阱3：熵系数**

- 过大：过度探索，无法收敛到好策略
- 过小：过早收敛，陷入局部最优
- 推荐：0.01 到 0.1

---

## 第四部分：前沿演进与高级算法

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

### 4.2 PPO 的信任域约束

**问题**：策略更新太大导致崩溃

**解决方案**：限制策略变化幅度

**PPO-Clip 目标**：
$$L^{CLIP} = \mathbb{E}_t[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)]$$

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是策略比率。

**裁剪的效果**：
- 当 $\hat{A}_t > 0$（好动作）：阻止概率过度增加
- 当 $\hat{A}_t < 0$（坏动作）：阻止概率过度减少

### 4.3 SAC 的最大熵框架

**目标**：最大化期望回报和策略熵

$$J(\theta) = \mathbb{E}[R(\tau) + \alpha H[\pi_\theta]]$$

其中 $H[\pi_\theta] = -\mathbb{E}[\log \pi_\theta(a|s)]$ 是策略熵。

**自动温度调节**：
$$\alpha^* = \arg\min_\alpha \mathbb{E}[-\alpha \log \pi_\theta(a|s) - \alpha H_{\text{target}}]$$

---

## 第五部分：实践指南与调试

### 5.1 超参数选择

**关键超参数**（按重要性）：

| 参数 | 推荐值 | 范围 | 敏感性 |
|------|--------|------|--------|
| 学习率 | 1e-3 | 1e-5 ~ 1e-2 | 高 |
| γ | 0.99 | 0.95 ~ 0.999 | 中 |
| λ (GAE) | 0.95 | 0.9 ~ 0.99 | 中 |
| 熵系数 | 0.01 | 0.001 ~ 0.1 | 中 |
| 梯度裁剪 | 1.0 | 0.5 ~ 10.0 | 低 |

### 5.2 常见问题排查

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| 奖励不增长 | 探索不足 | 增大熵系数 |
| 奖励波动大 | 学习率太大 | 降低学习率 |
| 训练崩溃 | 梯度爆炸 | 梯度裁剪，检查奖励尺度 |
| 收敛慢 | 学习率太小 | 增大学习率 |
| 过拟合 | 网络容量过大 | 减小隐层宽度 |

### 5.3 调试检查清单

- [ ] 回报是否正确计算？（处理终止状态）
- [ ] 优势是否归一化？
- [ ] 梯度是否裁剪？
- [ ] 学习率是否合理？
- [ ] 熵系数是否合理？
- [ ] 是否在评估时禁用探索？
- [ ] 价值网络是否收敛？
- [ ] 策略是否过早收敛？

---

## 第六部分：生产级实现要点

### 6.1 代码质量标准

**必须项**：
- 完整的类型注解
- 详细的文档字符串
- 参数验证
- 错误处理
- 梯度裁剪
- 模型保存/加载

**推荐项**：
- 单元测试
- 集成测试
- 性能基准
- 可视化工具
- 日志记录

### 6.2 性能优化

**时间复杂度**：
- REINFORCE：O(T) 每个 episode
- Actor-Critic：O(T) 每个 step
- A2C：O(N*T) 每个 batch

**空间复杂度**：
- REINFORCE：O(T) 存储轨迹
- Actor-Critic：O(1) 流式处理
- A2C：O(N*T) 存储所有轨迹

**优化技巧**：
- 批量处理
- 并行环境
- GPU 加速
- 内存池

---

## 第七部分：记忆宫殿（快速参考）

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

---

## 核心心法

**策略梯度的三个层次**：
1. **基础**：REINFORCE（高方差）
2. **改进**：Actor-Critic（方差缩减）
3. **生产级**：A2C（最优平衡）

**记住这三句话**：
1. 策略梯度定理：用 log 概率梯度加权的回报直接优化策略
2. 方差缩减：基线和 GAE 是关键
3. 稳定性：梯度裁剪、学习率、熵系数很重要

---

[返回上级](../README.md)
