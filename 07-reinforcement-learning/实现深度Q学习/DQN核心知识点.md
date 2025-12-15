# Deep Q-Network 核心知识点

> 快速复习指南 | 面试准备 | 概念速查

---

## 一、DQN 是什么？

### 一句话定义
**DQN = Q-Learning + 深度神经网络 + 经验回放 + 目标网络**

### 解决的问题
| 传统 Q-Learning 的问题 | DQN 的解决方案 |
|------------------------|----------------|
| Q 表无法存储高维状态 | 用神经网络逼近 Q 函数 |
| 连续样本高度相关 | 经验回放打破相关性 |
| TD 目标不稳定 | 目标网络固定目标 |

### 核心公式

**Q-Learning 更新**：
$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

**DQN 损失函数**：
$$L(\theta) = \mathbb{E}\left[ \left( \underbrace{r + \gamma \max_{a'} Q(s',a';\theta^-)}_{\text{TD 目标 } y} - Q(s,a;\theta) \right)^2 \right]$$

---

## 二、两大核心创新

### 1. 经验回放 (Experience Replay)

**为什么需要？**
- 在线学习：$(s_1,a_1,r_1,s_2) \to (s_2,a_2,r_2,s_3) \to \cdots$
- 连续样本高度相关 → 违反 SGD 的 i.i.d. 假设

**怎么做？**
```
存储：buffer.push(s, a, r, s', done)
采样：batch = buffer.sample(batch_size)  # 均匀随机
```

**三大好处**：
1. ✅ 打破时序相关性
2. ✅ 样本可重复使用（数据高效）
3. ✅ 批量梯度方差更低

---

### 2. 目标网络 (Target Network)

**为什么需要？**
- 标准 Q-Learning：$y = r + \gamma \max_{a'} Q(s',a';\theta)$
- 参数 $\theta$ 更新 → 目标 $y$ 变化 → **追逐移动目标**

**怎么做？**
```
y = r + γ max_a' Q(s', a'; θ⁻)   # θ⁻ 是冻结的目标网络

每 C 步：θ⁻ ← θ  # 硬更新
或每步：θ⁻ ← τθ + (1-τ)θ⁻  # 软更新
```

**效果**：固定回归目标，训练更稳定

---

## 三、重要算法变体

### 1. Double DQN

**解决问题**：标准 DQN 过估计 Q 值

**原因**：$\mathbb{E}[\max_a Q] \geq \max_a \mathbb{E}[Q]$（Jensen 不等式）

**解决方案**：解耦动作选择与评估
$$y = r + \gamma Q\left(s', \underbrace{\arg\max_{a'} Q(s',a';\theta)}_{\text{在线网络选择}}, \underbrace{\theta^-}_{\text{目标网络评估}}\right)$$

**记忆口诀**：**选用在线，评用目标**

---

### 2. Dueling DQN

**核心思想**：分解 Q 函数
$$Q(s,a) = \underbrace{V(s)}_{\text{状态价值}} + \underbrace{A(s,a)}_{\text{动作优势}} - \frac{1}{|A|}\sum_{a'} A(s,a')$$

**直觉理解**：
- $V(s)$：这个状态本身有多好
- $A(s,a)$：这个动作比平均好多少

**网络结构**：
```
Input → 共享层 → ┬→ Value Stream  → V(s)   ─┬→ Q(s,a)
                 └→ Advantage Stream → A(s,a) ─┘
```

**为什么有效**：$V(s)$ 从所有动作学习，收敛更快

---

### 3. 优先经验回放 (PER)

**核心思想**：TD 误差大的样本更重要

**采样概率**：
$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \epsilon$$

**重要性采样权重**（纠正偏差）：
$$w_i = (N \cdot P(i))^{-\beta}$$

**参数**：
- $\alpha$：0=均匀，1=完全贪心
- $\beta$：0→1 随训练增加

---

## 四、关键超参数速查

| 超参数 | 典型值 | 作用 |
|--------|--------|------|
| $\alpha$ (学习率) | 1e-4 ~ 1e-3 | 更新步长 |
| $\gamma$ (折扣因子) | 0.99 | 未来奖励权重 |
| $\epsilon$ (探索率) | 1.0 → 0.01 | 随机探索概率 |
| buffer_size | 10K ~ 1M | 回放缓冲容量 |
| batch_size | 32 ~ 64 | 每批样本数 |
| target_update | 100 ~ 10K | 目标网络同步频率 |

---

## 五、ε-Greedy 探索策略

**策略定义**：
$$\pi(a|s) = \begin{cases} 1-\epsilon+\frac{\epsilon}{|A|} & a = \arg\max Q(s,a) \\ \frac{\epsilon}{|A|} & \text{otherwise} \end{cases}$$

**衰减策略**：
$$\epsilon_t = \epsilon_{\text{end}} + (\epsilon_{\text{start}} - \epsilon_{\text{end}}) \cdot \max(0, 1-\frac{t}{T})$$

**直觉**：
- 初期：多探索（$\epsilon$ 大）
- 后期：多利用（$\epsilon$ 小）

---

## 六、DQN vs 其他算法

### DQN vs 表格 Q-Learning

| 特性 | 表格 Q-Learning | DQN |
|------|-----------------|-----|
| 状态空间 | 离散/小规模 | 连续/高维 |
| 存储 | Q 表 | 神经网络 |
| 泛化 | 无 | 自动泛化 |
| 样本效率 | 低 | 高（经验回放） |

### DQN vs Policy Gradient

| 特性 | DQN | Policy Gradient |
|------|-----|-----------------|
| 输出 | Q 值 | 动作概率 |
| 策略类型 | 离策略 | 在策略 |
| 动作空间 | 离散 | 离散/连续 |
| 样本效率 | 高 | 低 |
| 方差 | 低 | 高 |

---

## 七、常见面试问题

### Q1: 为什么 DQN 需要经验回放？
**答**：打破样本相关性，提高数据效率，降低梯度方差。

### Q2: 目标网络的作用是什么？
**答**：固定 TD 目标，避免追逐移动目标导致的训练不稳定。

### Q3: Double DQN 如何解决过估计？
**答**：用在线网络选动作，用目标网络评价值，解耦避免 max 偏差累积。

### Q4: Dueling DQN 的优势是什么？
**答**：分离 V 和 A，V 从所有动作学习，在动作影响小的状态收敛更快。

### Q5: DQN 为什么不能处理连续动作？
**答**：需要对所有动作求 max，连续空间无法枚举。解决方案：DDPG/TD3/SAC。

---

## 八、复杂度总结

**空间复杂度**：$O(N \cdot d + |\theta|)$
- $N$：缓冲区大小
- $d$：状态维度
- $|\theta|$：网络参数量

**时间复杂度（每步）**：$O(B \cdot |\theta|)$
- $B$：批大小

**样本复杂度**：$\tilde{O}\left(\frac{|S||A|}{(1-\gamma)^4 \epsilon^2}\right)$

---

## 九、一图总结

```
┌─────────────────────────────────────────────────────────────┐
│                        DQN 算法流程                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Environment ←──────────────────────┐                      │
│       │                              │                      │
│       ↓ state                        │ action               │
│   ┌─────────┐    ε-greedy    ┌───────┴───────┐             │
│   │ Q-Net θ │───────────────→│ Select Action │             │
│   └────┬────┘                └───────────────┘             │
│        │                                                    │
│        ↓ Q values                                           │
│   ┌─────────────┐                                           │
│   │ Replay      │←── (s, a, r, s', done)                   │
│   │ Buffer D    │                                           │
│   └──────┬──────┘                                           │
│          │ sample batch                                     │
│          ↓                                                  │
│   ┌─────────────┐    y = r + γ max Q(s';θ⁻)                │
│   │ Target Net  │─────────────────┐                        │
│   │     θ⁻      │                 │                        │
│   └─────────────┘                 ↓                        │
│          ↑               Loss = (y - Q(s,a;θ))²            │
│          │                        │                        │
│     sync every C steps            ↓                        │
│          │               θ ← θ - α∇L                       │
│          └────────────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 十、快速记忆卡片

| 概念 | 关键词 |
|------|--------|
| DQN | 神经网络逼近 Q 函数 |
| 经验回放 | 打破相关性，重复利用 |
| 目标网络 | 固定目标，稳定训练 |
| Double DQN | 选用在线，评用目标 |
| Dueling DQN | V + A 分解，快收敛 |
| PER | TD 误差大，优先采样 |
| ε-Greedy | 探索衰减，利用增加 |
