# Q-Learning 详解

> 无模型强化学习的基石——从理论到实践

---

## 一、从动态规划到无模型学习

### 1.1 动态规划的局限性

上一节我们学习了动态规划 (DP) 求解 MDP，它需要：

1. **完整的环境模型** $P(s'|s,a)$ 和 $R(s,a,s')$
2. **遍历所有状态和动作**

**现实问题**:
- 转移概率通常未知（如何知道开车时每个动作的精确后果？）
- 状态空间可能巨大甚至连续（围棋有 $10^{170}$ 种状态）

### 1.2 无模型强化学习

**核心思想**: 通过与环境**交互采样**学习最优策略，无需知道环境模型。

```
┌─────────────────────────────────────────────────────────────┐
│                    强化学习方法分类                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐           ┌─────────────────────────────┐  │
│  │  基于模型   │           │        无模型 (Model-Free)   │  │
│  │ (Model-Based)│          ├──────────────┬──────────────┤  │
│  │             │           │  基于价值    │  基于策略     │  │
│  │ • 动态规划   │           │ (Value-Based)│(Policy-Based)│  │
│  │ • 模型预测控制│          │              │              │  │
│  │ • Dyna-Q    │           │ • Q-Learning │ • REINFORCE  │  │
│  └─────────────┘           │ • SARSA      │ • Actor-Critic│  │
│                            │ • DQN        │ • PPO        │  │
│                            └──────────────┴──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、时序差分学习 (TD Learning)

### 2.1 蒙特卡洛 vs 时序差分

**蒙特卡洛方法**（Monte Carlo, MC）：

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]$$

- 需要等待回合结束才能更新
- $G_t$ 是完整回合的实际累积回报

**时序差分方法**（Temporal Difference, TD）：

$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

- 每一步都可以更新
- 使用**自举** (Bootstrapping)：用估计值更新估计值

### 2.2 TD 目标与 TD 误差

**TD 目标** (TD Target):

$$\text{TD Target} = R_{t+1} + \gamma V(S_{t+1})$$

**TD 误差** (TD Error):

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**直觉理解**：
- TD 误差 > 0：实际比预期好，增大 $V(S_t)$
- TD 误差 < 0：实际比预期差，减小 $V(S_t)$
- TD 误差 = 0：预测准确，价值收敛

---

## 三、Q-Learning 算法

### 3.1 算法原理

Q-Learning 是一种**离策略** (Off-Policy) TD 控制算法：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$

**关键特性**:
- 直接学习最优动作价值函数 $Q^*$
- 使用 $\max$ 操作，与当前行为策略无关
- 保证收敛到最优 Q 函数（在一定条件下）

### 3.2 算法流程

```
算法: Q-Learning

输入: 状态空间 S, 动作空间 A, 学习率 α, 折扣因子 γ, 探索率 ε
输出: 最优 Q 函数

1. 初始化 Q(s, a) = 0，对于所有 s ∈ S, a ∈ A
2. 对于每个回合:
   a. 初始化状态 S
   b. 重复 (对于回合中的每一步):
      i.   使用 ε-greedy 从 Q 选择动作 A
      ii.  执行动作 A，观察奖励 R 和下一状态 S'
      iii. Q(S, A) ← Q(S, A) + α[R + γ max_a Q(S', a) - Q(S, A)]
      iv.  S ← S'
   c. 直到 S 是终止状态
3. 返回 Q
```

---

## 四、探索与利用 (Exploration vs Exploitation)

### 4.1 困境

- **利用 (Exploitation)**: 选择当前已知最优的动作
- **探索 (Exploration)**: 尝试新动作，可能发现更好的策略

**过度利用**: 可能错过最优解
**过度探索**: 无法收敛，浪费资源

### 4.2 常用策略

#### ε-Greedy

以概率 ε 随机选择动作，否则选择最优动作。

**特点**: 简单有效，但探索是均匀随机的

#### Softmax (Boltzmann)

$$P(a|s) = \frac{\exp(Q(s,a) / \tau)}{\sum_{a'} \exp(Q(s,a') / \tau)}$$

**特点**: 根据 Q 值大小分配概率，$\tau$ 越小越贪心

#### UCB (Upper Confidence Bound)

$$A_t = \arg\max_a \left[ Q(s, a) + c \sqrt{\frac{\ln t}{N(s, a)}} \right]$$

**特点**: 平衡价值估计和不确定性

### 4.3 策略比较

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| ε-Greedy | 简单 | 探索均匀 | 大多数场景 |
| Softmax | 考虑 Q 值差异 | 需调参 | Q 值有意义差异时 |
| UCB | 理论保证 | 计算开销 | 多臂老虎机 |

---

## 五、SARSA 算法

### 5.1 On-Policy vs Off-Policy

| 特性 | Q-Learning (Off-Policy) | SARSA (On-Policy) |
|------|------------------------|-------------------|
| 更新目标 | $\max_a Q(S', a)$ | $Q(S', A')$ |
| 学习的策略 | 最优策略 | 当前行为策略 |
| 探索影响 | 不影响学习目标 | 直接影响学习 |
| 收敛性 | 收敛到 $Q^*$ | 收敛到 $Q^\pi$ |

### 5.2 SARSA 算法

**State-Action-Reward-State-Action**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

### 5.3 悬崖行走示例

```
┌────────────────────────────────────────┐
│ S                                     G │
│ ████████████████████████████████████████│  (悬崖)
└────────────────────────────────────────┘

S: 起点, G: 目标, █: 悬崖 (掉下去 -100 奖励)
```

**行为差异**:
- **Q-Learning**: 学习靠近悬崖的最短路径（因为更新不考虑探索）
- **SARSA**: 学习更安全的远离悬崖路径（考虑探索时可能掉下去）

---

## 六、高级技巧

### 6.1 Double Q-Learning

解决 Q-Learning 的过估计问题，通过解耦动作选择和价值评估：

- 维护两个 Q 表 $Q_1$ 和 $Q_2$
- 更新 $Q_1$ 时：用 $Q_1$ 选择动作，用 $Q_2$ 评估
- 更新 $Q_2$ 时：用 $Q_2$ 选择动作，用 $Q_1$ 评估

### 6.2 学习率调度

基于访问次数的衰减学习率：

$$\alpha(s,a) = \frac{1}{1 + N(s,a)}$$

---

## 七、本模块内容

### 交互式 Notebook

| 文件 | 内容 | 难度 |
|------|------|------|
| 01-Q-Learning-Fundamentals.ipynb | Q-Learning 基础与实现 | 入门 |
| 02-SARSA-Comparison.ipynb | SARSA 与算法对比 | 进阶 |
| 03-Advanced-Techniques.ipynb | 高级技巧与实战 | 进阶 |

### 核心代码模块

`q_learning_sarsa.py` 提供以下功能：

- `QLearningAgent`: Q-Learning 智能体（支持 Double Q-Learning）
- `SARSAAgent`: SARSA 智能体
- `ExpectedSARSAAgent`: Expected SARSA 智能体
- `CliffWalkingEnv`: 悬崖行走环境
- `train_q_learning()`: Q-Learning 训练函数
- `train_sarsa()`: SARSA 训练函数
- 可视化工具函数

---

## 八、总结

### 8.1 核心公式

| 算法 | 更新公式 | 类型 |
|------|----------|------|
| TD(0) | $V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]$ | 策略评估 |
| Q-Learning | $Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \max_a Q(S',a) - Q(S,A)]$ | Off-Policy |
| SARSA | $Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma Q(S',A') - Q(S,A)]$ | On-Policy |

### 8.2 超参数选择

| 参数 | 典型值 | 说明 |
|------|--------|------|
| $\alpha$ (学习率) | 0.1 ~ 0.5 | 较大值加速学习，但可能不稳定 |
| $\gamma$ (折扣因子) | 0.99 | 接近 1 重视长期奖励 |
| $\epsilon$ (探索率) | 1.0 → 0.01 | 从探索逐渐转向利用 |
| 衰减率 | 0.99 ~ 0.999 | 控制 ε 下降速度 |

### 8.3 局限性与展望

**表格型 Q-Learning 的局限**:
- 状态空间必须离散且有限
- 无法处理连续状态（如图像输入）
- 无法泛化到未见过的状态

**解决方案** → **深度 Q 网络 (DQN)**：用神经网络近似 Q 函数

---

## 参考资料

1. Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 6
2. Watkins, "Q-Learning", Machine Learning, 1992
3. Van Hasselt, "Double Q-learning", NeurIPS, 2010
4. OpenAI Gymnasium Documentation

---

[返回上级](../README.md)
