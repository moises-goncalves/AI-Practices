# Flappy Bird DQN - 深度Q网络实战项目

**难度**: ⭐⭐⭐☆☆ (中级)

## 项目简介

本项目使用Deep Q-Network (DQN)算法训练AI玩Flappy Bird游戏。通过这个项目，你将深入理解DQN的核心原理，并亲手实现一个能够自主学习的游戏AI。

### 学习目标

- 理解DQN算法的核心思想和数学原理
- 掌握经验回放(Experience Replay)技术
- 学习epsilon-greedy探索策略
- 实践图像预处理和状态表示
- 了解强化学习的训练调试技巧

## 理论背景

### 什么是DQN？

DQN (Deep Q-Network) 是DeepMind在2013年提出的算法，首次成功将深度学习与强化学习结合，在Atari游戏上达到人类水平。

**核心思想**：用神经网络近似Q函数

```
传统Q-Learning: Q表格 (状态数 × 动作数)
    ↓ 问题：状态空间太大（如图像）
DQN: 神经网络 Q(s,a;θ) 
    ↓ 解决：可以处理高维输入
```

### Q-Learning回顾

Q值表示在状态s下执行动作a的期望累积奖励：

```
Q(s,a) = r + γ * max Q(s',a')
         ↑       ↑
      即时奖励  未来奖励的折扣
```

### DQN的两大创新

**1. 经验回放 (Experience Replay)**

```python
# 问题：连续样本高度相关，训练不稳定
样本1 → 样本2 → 样本3  (时间相关)

# 解决：存储经验，随机采样
经验池: [(s1,a1,r1,s1'), (s2,a2,r2,s2'), ...]
训练时: 随机抽取batch，打破相关性
```

**2. 目标网络 (Target Network)**

```python
# 问题：目标值不断变化，训练震荡
target = r + γ * max Q(s',a'; θ)  # θ在变
                              ↑
# 解决：使用固定的目标网络
target = r + γ * max Q(s',a'; θ-)  # θ-定期更新
```

## 项目结构

```
01-flappy-bird-dqn/
├── README.md              # 项目说明（本文件）
├── requirements.txt       # 依赖包
├── train.py              # 训练脚本
├── play.py               # 测试脚本
├── assets/               # 游戏资源
│   └── sprites/          # 图片素材
├── src/
│   ├── __init__.py
│   ├── dqn.py            # DQN网络定义
│   ├── game_env.py       # 游戏环境封装
│   ├── trainer.py        # 训练器
│   └── utils.py          # 工具函数
├── models/               # 保存的模型
└── tests/
    └── test_dqn.py       # 单元测试
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行测试

```bash
# 确保代码正常工作
python -m pytest tests/ -v
```

### 3. 开始训练

```bash
# 默认参数训练（完整训练需要较长时间）
python train.py

# 快速测试（减少迭代次数）
python train.py --num_iters 10000

# 自定义参数
python train.py --lr 1e-5 --batch_size 64 --num_iters 500000
```

### 4. 测试模型

```bash
python play.py --model models/dqn_final.pth --episodes 5
```

## 代码详解

### DQN网络结构

```python
输入: 4帧游戏画面 (4, 84, 84)
      ↓
Conv1: 32个8x8卷积核, 步长4 → (32, 20, 20)
      ↓
Conv2: 64个4x4卷积核, 步长2 → (64, 9, 9)
      ↓
Conv3: 64个3x3卷积核, 步长1 → (64, 7, 7)
      ↓
Flatten: 64*7*7 = 3136
      ↓
FC1: 3136 → 512
      ↓
FC2: 512 → 2 (动作数)
      ↓
输出: [Q(不跳), Q(跳跃)]
```

### 训练流程

```
1. 初始化
   - 创建DQN网络
   - 创建经验回放缓冲区
   - 设置epsilon=0.1（探索率）

2. 每一步
   a. 观察当前状态s（4帧堆叠）
   b. epsilon-greedy选择动作
      - 概率epsilon: 随机动作
      - 概率1-epsilon: argmax Q(s,a)
   c. 执行动作，获得(r, s', done)
   d. 存入经验池
   e. 从经验池采样batch
   f. 计算TD目标: y = r + γ*max Q(s',a')
   g. 更新网络: minimize (Q(s,a) - y)²
   h. 衰减epsilon

3. 重复直到收敛
```

### 奖励设计

```python
reward = {
    '存活':     +0.1,   # 鼓励存活
    '通过管道': +1.0,   # 主要目标
    '碰撞死亡': -1.0    # 惩罚
}
```

## 超参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| learning_rate | 1e-6 | 学习率，太大会震荡 |
| gamma | 0.99 | 折扣因子，越大越重视未来 |
| epsilon_start | 0.1 | 初始探索率 |
| epsilon_end | 1e-4 | 最终探索率 |
| batch_size | 32 | 批次大小 |
| replay_size | 50000 | 经验池容量 |

## 训练技巧

### 1. 观察训练曲线

```
好的迹象：
- Loss逐渐下降并稳定
- 平均奖励逐渐上升
- 分数逐渐提高

坏的迹象：
- Loss剧烈震荡 → 降低学习率
- 奖励不增长 → 检查奖励设计
- 总是选同一动作 → 增加探索
```

### 2. 常见问题

**Q: 训练很慢怎么办？**
A: 使用GPU训练，或减少replay_size

**Q: 模型不收敛？**
A: 尝试降低学习率，增加探索率

**Q: 分数上不去？**
A: 检查奖励设计，确保正负奖励平衡

## 进阶优化

学完基础版本后，可以尝试：

1. **Double DQN**: 解决Q值过估计
2. **Dueling DQN**: 分离状态价值和优势函数
3. **Prioritized Replay**: 优先采样重要经验
4. **Rainbow**: 结合多种改进技术

## 参考资料

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - DQN原始论文
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Nature论文

## 致谢

本项目基于以下开源项目改编：
- [Flappy-bird-deep-Q-learning-pytorch](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch) by Viet Nguyen

感谢原作者的开源贡献！

---

**预计学习时间**: 1-2周
**前置知识**: Python基础、PyTorch基础、强化学习基础概念
