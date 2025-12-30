# 深度Q学习(DQN)实现

生产级Deep Q-Network实现，支持多种算法变体。

## 特性

- **标准DQN**: 基础深度Q学习实现
- **Double DQN**: 解决过估计问题
- **Dueling DQN**: 价值-优势分解架构
- **优先经验回放(PER)**: 高效样本采样

## 快速开始

```python
from agent import create_dqn_agent

# 创建agent
agent = create_dqn_agent(
    state_dim=4,
    action_dim=2,
    double_dqn=True,
    dueling=True,
)

# 训练循环
state = env.reset()
action = agent.select_action(state)
next_state, reward, done, _ = env.step(action)
loss = agent.train_step(state, action, reward, next_state, done)
```

## 目录结构

```
实现深度Q学习/
├── agent.py             # DQN Agent实现
├── train.py             # 训练脚本
├── core/                # 核心配置和类型
├── buffers/             # 经验回放缓冲区
├── networks/            # 神经网络架构
├── utils/               # 工具函数
├── notebooks/           # Jupyter教程
└── knowledge_points.md  # 知识点总结
```

## 命令行训练

```bash
python train.py --env CartPole-v1 --episodes 300 --double --dueling
```

## 依赖

```bash
pip install torch numpy gymnasium matplotlib
```

## 参考文献

1. Mnih et al. (2015). Human-level control through deep RL. Nature.
2. van Hasselt et al. (2016). Deep RL with Double Q-learning. AAAI.
3. Wang et al. (2016). Dueling Network Architectures. ICML.
