# Policy Gradient Methods: Research-Grade Implementation

## 项目概述

这是一个研究级的策略梯度方法实现，包含完整的算法实现、高质量的文档和教学资源。所有代码遵循Google风格指南，适合直接用于顶会论文发表和工业生产环境部署。

## 项目结构

```
策略梯度/
├── __init__.py                          # 包初始化
├── core/                                # 核心模块
│   ├── __init__.py
│   ├── base.py                          # 基类定义（PolicyGradientAgent, BasePolicy, BaseValueFunction）
│   └── trajectory.py                    # 轨迹缓冲区和数据处理
├── algorithms/                          # 算法实现
│   ├── __init__.py
│   ├── reinforce.py                     # REINFORCE算法
│   └── actor_critic.py                  # Actor-Critic和A2C算法
├── networks/                            # 神经网络架构
│   ├── __init__.py
│   ├── policy_networks.py               # 策略网络（离散、连续、高斯）
│   └── value_networks.py                # 价值网络（标准、对偶）
├── utils/                               # 工具函数
│   ├── __init__.py
│   ├── training_utils.py                # 训练工具（GAE、回报计算等）
│   └── evaluation_utils.py              # 评估工具
├── buffers/                             # 缓冲区
│   ├── __init__.py
│   └── replay_buffer.py                 # 经验回放缓冲区
├── tests/                               # 测试模块
│   ├── __init__.py
│   ├── test_algorithms.py               # 算法测试
│   └── test_networks.py                 # 网络测试
├── notebooks/                           # Jupyter教程
│   ├── 01_policy_gradient_fundamentals.ipynb
│   ├── 02_reinforce_algorithm.ipynb
│   ├── 03_actor_critic_methods.ipynb
│   └── 04_advanced_topics_gae_a2c.ipynb
├── docs/                                # 文档
│   └── KNOWLEDGE_SUMMARY.md             # 知识点总结
└── run_tests.py                         # 测试运行脚本
```

## 核心特性

### 1. 完整的算法实现

#### REINFORCE
- 基础策略梯度算法
- 使用完整轨迹回报
- 支持熵正则化
- 可选的价值函数基线

```python
from 策略梯度.algorithms import REINFORCE
from 策略梯度.networks import DiscretePolicy, ValueNetwork

policy = DiscretePolicy(state_dim=4, action_dim=2)
value_fn = ValueNetwork(state_dim=4)

agent = REINFORCE(
    policy=policy,
    value_function=value_fn,
    learning_rate=1e-3,
    gamma=0.99,
    entropy_coeff=0.01
)

history = agent.train(env, num_episodes=100)
```

#### Actor-Critic
- 结合策略梯度和价值函数学习
- 使用时间差分（TD）学习
- 显著降低方差
- 更快的收敛速度

```python
from 策略梯度.algorithms import ActorCritic

agent = ActorCritic(
    policy=policy,
    value_function=value_fn,
    learning_rate=1e-3,
    gamma=0.99
)

history = agent.train(env, num_episodes=100)
```

#### A2C (Advantage Actor-Critic)
- 使用广义优势估计（GAE）
- 支持批量更新
- 支持并行环境
- 最高的样本效率

```python
from 策略梯度.algorithms import A2C

agent = A2C(
    policy=policy,
    value_function=value_fn,
    learning_rate=1e-3,
    gamma=0.99,
    gae_lambda=0.95
)

history = agent.train(env, num_episodes=100)
```

### 2. 灵活的网络架构

#### 策略网络
- **DiscretePolicy**: 离散动作空间
- **ContinuousPolicy**: 连续动作空间（固定方差）
- **GaussianPolicy**: 高斯策略（可学习方差）

#### 价值网络
- **ValueNetwork**: 标准价值网络
- **DuelingValueNetwork**: 对偶架构（分离价值和优势流）

### 3. 高质量的文档

#### Jupyter Notebooks
1. **01_policy_gradient_fundamentals.ipynb**
   - 策略梯度定理
   - 基线函数的作用
   - 方差减少原理

2. **02_reinforce_algorithm.ipynb**
   - REINFORCE算法详解
   - 实现细节
   - 训练循环

3. **03_actor_critic_methods.ipynb**
   - Actor-Critic架构
   - 时间差分学习
   - 方差-偏差权衡

4. **04_advanced_topics_gae_a2c.ipynb**
   - 广义优势估计（GAE）
   - A2C算法
   - 实践技巧

#### 知识点总结
- 完整的数学推导
- 算法对比分析
- 实现技巧
- 常见问题解决方案

## 数学基础

### 策略梯度定理

$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) Q(s,a)]$$

### 优势函数

$$A(s,a) = Q(s,a) - V(s)$$

### 广义优势估计（GAE）

$$A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_t^{(l)}$$

其中 $\delta_t^{(l)} = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是时间差分误差。

## 使用示例

### 基础训练

```python
import gymnasium as gym
from 策略梯度.algorithms import ActorCritic
from 策略梯度.networks import DiscretePolicy, ValueNetwork

# 创建环境
env = gym.make("CartPole-v1")

# 创建网络
policy = DiscretePolicy(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_dims=(64, 64)
)

value_fn = ValueNetwork(
    state_dim=env.observation_space.shape[0],
    hidden_dims=(64, 64)
)

# 创建代理
agent = ActorCritic(
    policy=policy,
    value_function=value_fn,
    learning_rate=1e-3,
    gamma=0.99,
    entropy_coeff=0.01
)

# 训练
history = agent.train(
    env,
    num_episodes=200,
    max_steps=500,
    eval_interval=10
)

# 保存模型
agent.save("actor_critic_model.pt")
```

### 评估策略

```python
from 策略梯度.utils import evaluate_policy

metrics = evaluate_policy(
    agent,
    env,
    num_episodes=10,
    max_steps=500
)

print(f"Mean Return: {metrics['mean_return']:.2f}")
print(f"Std Return: {metrics['std_return']:.2f}")
```

### 自定义训练循环

```python
import torch

# 收集轨迹
trajectory, episode_return = agent.collect_trajectory(env, max_steps=500)

# 转换为张量
states = torch.FloatTensor(trajectory.states)
actions = torch.FloatTensor(trajectory.actions)
returns = torch.FloatTensor(agent.trajectory_buffer.compute_returns(trajectory))

# 训练步骤
metrics = agent.train_step(states, actions, returns)
```

## 性能特性

### 时间复杂度
- **REINFORCE**: O(T) 每个episode
- **Actor-Critic**: O(T) 每个step
- **A2C**: O(N*T) 每个batch

### 空间复杂度
- **REINFORCE**: O(T) 存储轨迹
- **Actor-Critic**: O(1) 流式处理
- **A2C**: O(N*T) 存储所有轨迹

### 样本复杂度
- **REINFORCE**: O(1/ε²)
- **Actor-Critic**: O(1/ε)
- **A2C**: O(1/ε) 但常数更小

## 实现亮点

### 1. 研究级代码质量
- 严格遵循Google风格指南
- 完整的类型注解
- 详细的文档字符串
- 模块化和解耦的设计

### 2. 深度文档
- 每个核心算法包含：
  - 核心思想（Core Idea）
  - 数学原理（Mathematical Theory）
  - 问题背景（Problem Statement）
  - 算法对比（Comparison）
  - 复杂度分析（Complexity）
  - 算法总结（Summary）

### 3. 教学资源
- 4个高质量的Jupyter Notebooks
- 每个notebook包含：
  - 理论讲解
  - 数学推导
  - 代码示例
  - 可视化分析

### 4. 生产就绪
- 完整的错误处理
- 梯度裁剪
- 参数验证
- 模型保存/加载

## 算法对比

| 特性 | REINFORCE | Actor-Critic | A2C |
|------|-----------|--------------|-----|
| 方差 | 高 | 中 | 低 |
| 偏差 | 低 | 中 | 中 |
| 收敛速度 | 慢 | 中 | 快 |
| 样本效率 | 低 | 中 | 高 |
| 实现复杂度 | 简单 | 中等 | 复杂 |
| 并行支持 | 否 | 否 | 是 |

## 扩展方向

### 已实现
- ✓ REINFORCE
- ✓ Actor-Critic
- ✓ A2C with GAE
- ✓ 离散和连续动作空间
- ✓ 经验回放缓冲区
- ✓ 优先级回放

### 可扩展
- [ ] PPO (Proximal Policy Optimization)
- [ ] TRPO (Trust Region Policy Optimization)
- [ ] SAC (Soft Actor-Critic)
- [ ] TD3 (Twin Delayed DDPG)
- [ ] 多任务学习
- [ ] 元学习

## 依赖项

```
torch>=1.9.0
numpy>=1.19.0
gymnasium>=0.26.0
matplotlib>=3.3.0
```

## 安装

```bash
# 克隆或下载项目
cd 策略梯度

# 安装依赖
pip install torch numpy gymnasium matplotlib

# 运行测试
python run_tests.py
```

## 快速开始

### 1. 学习基础概念
```bash
# 打开Jupyter Notebook
jupyter notebook notebooks/01_policy_gradient_fundamentals.ipynb
```

### 2. 实现REINFORCE
```bash
jupyter notebook notebooks/02_reinforce_algorithm.ipynb
```

### 3. 学习Actor-Critic
```bash
jupyter notebook notebooks/03_actor_critic_methods.ipynb
```

### 4. 高级主题
```bash
jupyter notebook notebooks/04_advanced_topics_gae_a2c.ipynb
```

### 5. 查看知识总结
```bash
cat docs/KNOWLEDGE_SUMMARY.md
```

## 论文参考

1. **REINFORCE**: Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine Learning, 8(3-4), 229-256.

2. **Actor-Critic**: Konda, V., & Tsitsiklis, J. N. (2000). "Actor-Critic Algorithms." SIAM Journal on Control and Optimization, 42(4), 1143-1166.

3. **GAE**: Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv preprint arXiv:1506.02438.

4. **A3C**: Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). "Asynchronous Methods for Deep Reinforcement Learning." ICML.

5. **PPO**: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347.

## 许可证

本项目用于教学和研究目的。

## 贡献

欢迎提交问题和改进建议！

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送Pull Request
- 查看文档和教程

---

**最后更新**: 2024年12月

**版本**: 1.0.0

**状态**: 生产就绪 ✓
