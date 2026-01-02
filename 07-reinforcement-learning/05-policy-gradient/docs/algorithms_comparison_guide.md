# 策略梯度算法对比与应用指南

> 从理论到实践：如何选择和应用策略梯度方法

---

## 第一部分：算法全景对比

### 1.1 功能对比矩阵

| 维度 | REINFORCE | Actor-Critic | A2C | PPO | SAC |
|------|-----------|--------------|-----|-----|-----|
| **方差** | 高 | 中 | 低 | 低 | 低 |
| **偏差** | 无 | 低 | 低 | 低 | 低 |
| **收敛速度** | 慢 | 中 | 快 | 快 | 快 |
| **样本复杂度** | $O(1/\epsilon^2)$ | $O(1/\epsilon)$ | $O(1/\epsilon)$ | $O(1/\epsilon)$ | $O(1/\epsilon)$ |
| **实现复杂度** | 简单 | 中等 | 中等 | 复杂 | 复杂 |
| **并行支持** | 否 | 否 | 是 | 是 | 是 |
| **离散动作** | ✓ | ✓ | ✓ | ✓ | ✗ |
| **连续动作** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **稳定性** | 低 | 中 | 高 | 高 | 高 |

### 1.2 性能对比（基准环境）

**CartPole-v1**（简单环境）：

| 算法 | 收敛代数 | 最终性能 | 学习曲线平滑度 |
|------|---------|---------|--------------|
| REINFORCE | 300+ | 500 | 低 |
| Actor-Critic | 150-200 | 500 | 中 |
| A2C | 100-150 | 500 | 高 |
| PPO | 80-120 | 500 | 高 |

**MuJoCo（复杂环境）**：

| 算法 | 样本效率 | 最终性能 | 稳定性 |
|------|---------|---------|--------|
| REINFORCE | 低 | 中 | 低 |
| Actor-Critic | 中 | 中 | 中 |
| A2C | 中 | 高 | 高 |
| PPO | 高 | 高 | 高 |
| SAC | 高 | 高 | 高 |

### 1.3 计算开销对比

**时间复杂度**：

| 算法 | 每 episode | 每 step | 备注 |
|------|-----------|---------|------|
| REINFORCE | O(T) | - | 需要完整轨迹 |
| Actor-Critic | O(T) | O(1) | 流式处理 |
| A2C | O(N*T) | - | 批量处理 |
| PPO | O(N*T*K) | - | K 个 epoch |
| SAC | O(N*T) | - | 连续动作 |

**空间复杂度**：

| 算法 | 存储需求 | 备注 |
|------|---------|------|
| REINFORCE | O(T) | 存储完整轨迹 |
| Actor-Critic | O(1) | 流式处理 |
| A2C | O(N*T) | 存储所有轨迹 |
| PPO | O(N*T) | 存储所有轨迹 |
| SAC | O(N*T) | 存储所有轨迹 |

---

## 第二部分：算法选择决策树

### 2.1 基于环境复杂度

```
环境复杂度
  │
  ├─ 简单（CartPole, Acrobot）
  │  └─ REINFORCE 或 Actor-Critic
  │
  ├─ 中等（Atari 简单游戏）
  │  └─ A2C 或 PPO
  │
  └─ 复杂（MuJoCo, Atari 困难游戏）
     └─ PPO 或 SAC
```

### 2.2 基于动作空间

```
动作空间
  │
  ├─ 离散
  │  ├─ 简单：REINFORCE
  │  ├─ 中等：Actor-Critic
  │  └─ 复杂：A2C 或 PPO
  │
  └─ 连续
     ├─ 简单：Actor-Critic
     ├─ 中等：A2C 或 PPO
     └─ 复杂：PPO 或 SAC
```

### 2.3 基于计算资源

```
计算资源
  │
  ├─ 受限（移动设备、边缘计算）
  │  └─ REINFORCE 或 Actor-Critic
  │
  ├─ 中等（单 GPU）
  │  └─ A2C 或 PPO
  │
  └─ 充足（多 GPU、集群）
     └─ PPO 或 SAC
```

### 2.4 基于优先级

```
优先级
  │
  ├─ 性能优先
  │  └─ PPO 或 SAC
  │
  ├─ 稳定性优先
  │  └─ A2C 或 PPO
  │
  ├─ 效率优先
  │  └─ Actor-Critic
  │
  └─ 简单性优先
     └─ REINFORCE
```

---

## 第三部分：详细应用指南

### 3.1 场景1：入门学习

**目标**：理解策略梯度基础

**推荐**：REINFORCE

**原因**：
- 实现简单
- 易于理解
- 计算开销小

**超参数**：
```python
config = {
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'entropy_coeff': 0.01,
    'max_grad_norm': 1.0
}
```

**代码示例**：
```python
from algorithms import REINFORCE
from networks import DiscretePolicy

policy = DiscretePolicy(state_dim=4, action_dim=2)
agent = REINFORCE(policy, learning_rate=1e-3)

history = agent.train(env, num_episodes=200)
```

### 3.2 场景2：中等难度环境

**目标**：平衡性能和效率

**推荐**：A2C

**原因**：
- 性能显著
- 样本效率高
- 实现相对简单

**超参数**：
```python
config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'entropy_coeff': 0.01,
    'max_grad_norm': 0.5
}
```

**代码示例**：
```python
from algorithms import A2C
from networks import DiscretePolicy, ValueNetwork

policy = DiscretePolicy(state_dim=84*84*4, action_dim=18)
value_fn = ValueNetwork(state_dim=84*84*4)
agent = A2C(policy, value_fn, learning_rate=3e-4, gae_lambda=0.95)

history = agent.train(env, num_episodes=500)
```

### 3.3 场景3：追求最优性能

**目标**：最大化性能

**推荐**：PPO

**原因**：
- 性能最优
- 稳定性最好
- 已验证有效

**超参数**：
```python
config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coeff': 0.01,
    'max_grad_norm': 0.5,
    'n_epochs': 10
}
```

**代码示例**：
```python
from algorithms import PPO
from networks import DiscretePolicy, ValueNetwork

policy = DiscretePolicy(state_dim=84*84*4, action_dim=18)
value_fn = ValueNetwork(state_dim=84*84*4)
agent = PPO(policy, value_fn, learning_rate=3e-4, clip_epsilon=0.2)

history = agent.train(env, num_episodes=1000)
```

### 3.4 场景4：连续控制

**目标**：处理连续动作空间

**推荐**：PPO 或 SAC

**原因**：
- PPO：通用，稳定
- SAC：最大熵，自动温度调节

**超参数（PPO）**：
```python
config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coeff': 0.001,  # 连续动作用较小值
    'max_grad_norm': 0.5
}
```

**代码示例**：
```python
from algorithms import PPO
from networks import GaussianPolicy, ValueNetwork

policy = GaussianPolicy(state_dim=17, action_dim=6)
value_fn = ValueNetwork(state_dim=17)
agent = PPO(policy, value_fn, learning_rate=3e-4)

history = agent.train(env, num_episodes=1000)
```

### 3.5 场景5：计算受限

**目标**：在有限资源下最大化性能

**推荐**：Actor-Critic

**原因**：
- 计算开销小
- 性能可接受
- 实现简单

**超参数**：
```python
config = {
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'entropy_coeff': 0.01,
    'max_grad_norm': 1.0
}
```

**代码示例**：
```python
from algorithms import ActorCritic
from networks import DiscretePolicy, ValueNetwork

policy = DiscretePolicy(state_dim=4, action_dim=2, hidden_dims=(32, 32))
value_fn = ValueNetwork(state_dim=4, hidden_dims=(32, 32))
agent = ActorCritic(policy, value_fn, learning_rate=1e-3)

history = agent.train(env, num_episodes=200)
```

---

## 第四部分：实践技巧与调试

### 4.1 超参数调优指南

**关键超参数**（按重要性）：

| 参数 | 推荐值 | 范围 | 敏感性 | 调整方向 |
|------|--------|------|--------|---------|
| 学习率 | 1e-3 | 1e-5 ~ 1e-2 | 高 | 奖励不增长→增大；波动大→减小 |
| γ | 0.99 | 0.95 ~ 0.999 | 中 | 长期任务→增大；短期→减小 |
| λ (GAE) | 0.95 | 0.9 ~ 0.99 | 中 | 方差大→减小；偏差大→增大 |
| 熵系数 | 0.01 | 0.001 ~ 0.1 | 中 | 探索不足→增大；过度探索→减小 |
| 梯度裁剪 | 1.0 | 0.5 ~ 10.0 | 低 | 梯度爆炸→减小 |

### 4.2 常见问题排查

| 现象 | 原因 | 解决方案 | 优先级 |
|------|------|---------|--------|
| 奖励不增长 | 探索不足 | 增大熵系数 | 高 |
| 奖励波动大 | 学习率太大 | 降低学习率 | 高 |
| 训练崩溃 | 梯度爆炸 | 梯度裁剪，检查奖励尺度 | 高 |
| 收敛慢 | 学习率太小 | 增大学习率 | 中 |
| 过拟合 | 网络容量过大 | 减小隐层宽度 | 中 |
| 价值网络不收敛 | 学习率不匹配 | 调整价值网络学习率 | 中 |

### 4.3 调试检查清单

**数据处理**：
- [ ] 回报是否正确计算？（处理终止状态）
- [ ] 优势是否归一化？
- [ ] 是否处理了 NaN 和 Inf？

**网络训练**：
- [ ] 梯度是否裁剪？
- [ ] 学习率是否合理？
- [ ] 熵系数是否合理？
- [ ] 是否在评估时禁用探索？

**模型验证**：
- [ ] 价值网络是否收敛？
- [ ] 策略是否过早收敛？
- [ ] 损失曲线是否平滑？

---

## 第五部分：性能基准

### 5.1 CartPole-v1 基准

**环境**：
- 状态维度：4
- 动作维度：2（离散）
- 目标奖励：500

**性能对比**：

| 算法 | 收敛代数 | 最终性能 | 学习曲线 |
|------|---------|---------|---------|
| REINFORCE | 300-400 | 500 | 波动大 |
| Actor-Critic | 150-200 | 500 | 波动中 |
| A2C | 100-150 | 500 | 平滑 |
| PPO | 80-120 | 500 | 平滑 |

### 5.2 MuJoCo 基准

**环境**：HalfCheetah-v2

**性能对比**：

| 算法 | 样本数 | 最终性能 | 稳定性 |
|------|--------|---------|--------|
| REINFORCE | 1M+ | 低 | 低 |
| Actor-Critic | 500K | 中 | 中 |
| A2C | 300K | 高 | 高 |
| PPO | 200K | 高 | 高 |
| SAC | 200K | 高 | 高 |

---

## 第六部分：前沿研究方向

### 6.1 最新改进（2020-2024）

**PPO 的改进**：
- PPO-Lagrangian：用拉格朗日乘数替代固定 ε
- PPO-Penalty：用 KL 惩罚替代裁剪

**SAC 的改进**：
- SAC-X：多任务学习
- SAC-Offline：离线强化学习

**新方向**：
- 与 Transformer 的结合
- 多任务强化学习
- 从人类反馈学习（RLHF）

### 6.2 离线强化学习

**问题**：从固定数据集学习

**解决方案**：
- Conservative Q-Learning (CQL)
- 行为克隆正则化
- 不确定性估计

---

## 第七部分：快速参考

### 快速选择表

| 需求 | 推荐算法 | 理由 |
|------|---------|------|
| 最简单 | REINFORCE | 实现简单，易理解 |
| 最快 | A2C | 收敛快，样本效率高 |
| 最稳定 | PPO | 稳定性最好 |
| 最高效 | PPO | 性能最优 |
| 连续控制 | PPO/SAC | 都很好，SAC 更灵活 |
| 计算受限 | Actor-Critic | 开销小 |

### 超参数速查表

```python
# 简单环境（CartPole）
simple_config = {
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'entropy_coeff': 0.01,
    'max_grad_norm': 1.0
}

# 中等环境（Atari）
medium_config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'entropy_coeff': 0.01,
    'max_grad_norm': 0.5
}

# 复杂环境（MuJoCo）
complex_config = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'entropy_coeff': 0.001,
    'max_grad_norm': 0.5
}
```

---

## 核心心法

**策略梯度方法的三个层次**：
1. **基础**：REINFORCE（高方差）
2. **改进**：Actor-Critic（方差缩减）
3. **生产级**：A2C/PPO（最优平衡）

**记住这三句话**：
1. 没有银弹：选择最适合你的环境和资源的算法
2. 超参数很关键：花时间调优超参数比选择算法更重要
3. 从简到复：掌握 REINFORCE 后再学习高级算法

---

[返回上级](../README.md)
