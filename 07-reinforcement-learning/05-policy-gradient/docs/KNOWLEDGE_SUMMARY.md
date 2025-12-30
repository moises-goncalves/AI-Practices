# Policy Gradient Methods: Knowledge Summary

## 核心概念总结

### 1. 策略梯度定理 (Policy Gradient Theorem)

**数学表达式：**
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) Q(s,a)]$$

**核心思想：**
- 期望回报的梯度可以表示为得分函数和动作价值函数的乘积的期望
- 得分函数 $\nabla_\theta \log \pi_\theta(a|s)$ 指向增加策略概率的方向
- 动作价值函数 $Q(s,a)$ 提供信号强度

**关键洞察：**
这个定理使我们能够直接对策略进行梯度上升，而不需要显式的价值函数。

---

## 算法对比

### REINFORCE
**特点：**
- 使用完整轨迹回报作为信号
- 高方差，低偏差
- 简单易实现
- 收敛速度慢

**更新规则：**
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) G_t$$

**适用场景：**
- 教学用途
- 简单环境
- 样本充足的情况

---

### Actor-Critic
**特点：**
- 结合策略梯度（Actor）和价值函数学习（Critic）
- 中等方差，中等偏差
- 使用时间差分（TD）学习
- 收敛速度中等

**核心公式：**
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a|s) \delta_t$$

**优势函数：**
$$A(s,a) = Q(s,a) - V(s) \approx \delta_t$$

**适用场景：**
- 标准强化学习问题
- 需要平衡方差和偏差
- 连续动作空间

---

### A2C (Advantage Actor-Critic)
**特点：**
- 使用广义优势估计（GAE）
- 批量更新
- 支持并行环境
- 低方差，中等偏差

**GAE公式：**
$$A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_t^{(l)}$$

**高效计算：**
$$A_t = \delta_t + (\gamma\lambda) A_{t+1}$$

**适用场景：**
- 需要高样本效率
- 可用并行计算
- 大规模训练

---

## 关键技术

### 1. 基线函数 (Baseline)

**作用：**
减少方差而不引入偏差

**数学证明：**
$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) (Q(s,a) - b(s))]$$

基线项 $b(s)$ 的期望为零，因此不改变梯度期望值。

**最优基线：**
$$b^*(s) = V(s) = \mathbb{E}[G_t | s]$$

**方差减少效果：**
- 无基线：$\text{Var}(G_t)$ 很大
- 有基线：$\text{Var}(G_t - V(s))$ 显著降低

---

### 2. 优势函数 (Advantage Function)

**定义：**
$$A(s,a) = Q(s,a) - V(s)$$

**含义：**
- 衡量动作 $a$ 相对于平均动作的相对质量
- 正值：该动作优于平均
- 负值：该动作劣于平均

**估计方法：**
- **1步TD**：$A_t^{(1)} = r_t + \gamma V(s_{t+1}) - V(s_t)$（低方差，高偏差）
- **n步TD**：$A_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n}) - V(s_t)$（中等）
- **蒙特卡洛**：$A_t^{(\infty)} = G_t - V(s_t)$（高方差，低偏差）
- **GAE**：$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_t^{(l)}$（可调）

---

### 3. 广义优势估计 (GAE)

**参数 λ 的作用：**
- $\lambda = 0$：仅使用1步TD（低方差，高偏差）
- $\lambda = 1$：使用完整蒙特卡洛回报（高方差，低偏差）
- $0 < \lambda < 1$：在两者之间权衡（推荐 $\lambda = 0.95$）

**高效计算算法：**
```python
advantages = np.zeros(len(rewards))
gae = 0.0
for t in reversed(range(len(rewards))):
    delta = rewards[t] + gamma * values[t+1] - values[t]
    gae = delta + gamma * lambda_ * gae
    advantages[t] = gae
```

**时间复杂度：** O(T)，其中 T 是轨迹长度

---

## 方差-偏差权衡

### 理论分析

| 方法 | 方差 | 偏差 | 收敛速度 | 样本效率 |
|------|------|------|---------|---------|
| REINFORCE | 高 | 低 | 慢 | 低 |
| Actor-Critic | 中 | 中 | 中 | 中 |
| A2C (GAE) | 低 | 中 | 快 | 高 |

### 实践建议

1. **高方差问题**：
   - 增加 λ（更接近蒙特卡洛）
   - 增加批量大小
   - 使用更好的价值函数

2. **高偏差问题**：
   - 减少 λ（更接近1步TD）
   - 改进价值函数训练
   - 增加熵正则化

---

## 实现技巧

### 1. 优势归一化

**公式：**
$$A_{norm} = \frac{A - \text{mean}(A)}{\text{std}(A) + \epsilon}$$

**好处：**
- 稳定训练
- 学习率对奖励尺度不敏感
- 改善梯度流

### 2. 回报归一化

**公式：**
$$G_{norm} = \frac{G - \text{mean}(G)}{\text{std}(G) + \epsilon}$$

**应用：**
用于价值函数训练

### 3. 梯度裁剪

**公式：**
$$\nabla \leftarrow \text{clip}(\nabla, -\text{max\_norm}, \text{max\_norm})$$

**作用：**
防止大的梯度更新导致训练不稳定

### 4. 熵正则化

**公式：**
$$L_{total} = L_{policy} + L_{value} - \beta H(\pi)$$

其中 $H(\pi) = -\mathbb{E}[\log \pi]$ 是策略熵

**作用：**
鼓励探索，防止策略过早收敛

---

## 算法选择指南

### 何时使用 REINFORCE
- ✓ 学习和理解基础概念
- ✓ 简单环境（CartPole等）
- ✓ 样本充足
- ✗ 需要快速收敛
- ✗ 高维问题

### 何时使用 Actor-Critic
- ✓ 标准强化学习问题
- ✓ 需要平衡方差和偏差
- ✓ 连续动作空间
- ✓ 中等复杂度环境
- ✗ 需要最高样本效率

### 何时使用 A2C
- ✓ 需要高样本效率
- ✓ 可用并行计算
- ✓ 大规模训练
- ✓ 复杂环境
- ✗ 单机单环境

---

## 常见问题与解决方案

### 问题1：训练不稳定
**原因：**
- 学习率过高
- 价值函数不准确
- 方差过高

**解决方案：**
- 降低学习率
- 增加价值函数训练步数
- 增加 λ 值（使用更多蒙特卡洛）
- 增加批量大小

### 问题2：收敛速度慢
**原因：**
- 方差过高
- 样本效率低
- 探索不足

**解决方案：**
- 减少 λ 值（使用更多TD）
- 增加熵正则化系数
- 使用更好的网络架构
- 增加并行环境数

### 问题3：策略过早收敛
**原因：**
- 探索不足
- 熵衰减过快

**解决方案：**
- 增加初始熵正则化系数
- 使用更大的初始标准差（连续动作）
- 增加ε-贪心探索

---

## 数学基础回顾

### 期望值性质
$$\mathbb{E}[f(s) \nabla_\theta \log \pi_\theta(a|s)] = 0$$

这是因为 $\nabla_\theta \log \pi_\theta(a|s)$ 的期望为零。

### 对数导数技巧
$$\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

这使我们能够计算不可微分函数的梯度。

### 折扣因子的作用
$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

- $\gamma$ 接近1：考虑长期回报
- $\gamma$ 接近0：只关注即时回报

---

## 复杂度分析

### 时间复杂度
- **REINFORCE**：O(T) 每个episode，其中 T 是episode长度
- **Actor-Critic**：O(T) 每个step
- **A2C**：O(N*T) 每个batch，其中 N 是并行环境数

### 空间复杂度
- **REINFORCE**：O(T) 存储轨迹
- **Actor-Critic**：O(1) 每个step（流式处理）
- **A2C**：O(N*T) 存储所有轨迹

### 样本复杂度
- **REINFORCE**：O(1/ε²) 达到ε-最优策略
- **Actor-Critic**：O(1/ε)
- **A2C**：O(1/ε) 但常数更小

---

## 扩展阅读

### 相关论文
1. **REINFORCE**: Williams (1992) - "Simple Statistical Gradient-Following Algorithms"
2. **Actor-Critic**: Konda & Tsitsiklis (2000) - "Actor-Critic Algorithms"
3. **GAE**: Schulman et al. (2015) - "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
4. **A3C**: Mnih et al. (2016) - "Asynchronous Methods for Deep Reinforcement Learning"
5. **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"

### 进阶主题
- 信任域方法（TRPO, PPO）
- 离策略方法（SAC, TD3）
- 模型基础强化学习
- 多任务学习
- 元学习

---

## 实现检查清单

### 基础实现
- [ ] 策略网络（离散/连续动作）
- [ ] 价值网络
- [ ] 轨迹收集
- [ ] 回报计算
- [ ] 策略损失函数
- [ ] 价值损失函数

### 优化技巧
- [ ] 优势归一化
- [ ] 回报归一化
- [ ] 梯度裁剪
- [ ] 熵正则化
- [ ] 学习率调度

### 调试和监控
- [ ] 记录episode回报
- [ ] 监控策略熵
- [ ] 跟踪损失函数
- [ ] 可视化学习曲线
- [ ] 验证梯度流

---

## 总结

策略梯度方法是强化学习中最重要的算法族之一。从简单的REINFORCE到复杂的A2C，每种方法都在方差-偏差权衡中找到了不同的平衡点。

**关键要点：**
1. 策略梯度定理提供了理论基础
2. 基线函数是减少方差的关键
3. GAE提供了灵活的优势估计
4. 实现细节（归一化、裁剪等）对性能至关重要
5. 选择合适的算法取决于具体问题

通过理解这些方法的原理和权衡，你可以为不同的问题选择合适的算法，并有效地调整超参数以获得最佳性能。
