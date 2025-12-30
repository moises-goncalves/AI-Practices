<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg" alt="Tests">
</p>

<h1 align="center">Flappy Bird DQN</h1>

<p align="center">
  <b>使用深度Q网络训练AI玩Flappy Bird游戏</b>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> •
  <a href="#项目结构">项目结构</a> •
  <a href="#训练指南">训练指南</a> •
  <a href="#算法原理">算法原理</a> •
  <a href="#常见问题">FAQ</a>
</p>

---

## 项目简介

本项目使用 **Deep Q-Network (DQN)** 算法训练AI玩经典的Flappy Bird游戏。DQN是DeepMind在2013年提出的里程碑式算法，首次成功将深度学习与强化学习结合，在Atari游戏上达到人类水平。

### 学习目标

通过本项目，你将掌握：

| 知识点 | 说明 |
|:-------|:-----|
| DQN算法原理 | Q-Learning + 深度神经网络 |
| 经验回放 | Experience Replay机制 |
| 目标网络 | Target Network稳定训练 |
| 图像预处理 | 游戏画面状态表示 |
| 探索策略 | Epsilon-Greedy策略 |

---

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，GPU加速)

### 三步运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证环境
python -m unittest tests.test_dqn -v

# 3. 开始训练
python train.py --num_iters 100000
```

### 详细安装步骤

<details>
<summary><b>点击展开完整安装指南</b></summary>

#### 1. 检查Python版本

```bash
python --version  # 需要 3.8+
```

#### 2. 创建虚拟环境（推荐）

```bash
# 创建
python -m venv venv

# 激活
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate.bat     # Windows CMD
venv\Scripts\Activate.ps1     # Windows PowerShell
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt

# 国内用户使用镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 4. 验证安装

```bash
python -m unittest tests.test_dqn -v
# 应该看到 11 tests passed
```

</details>

---

## 项目结构

```
01-flappy-bird-dqn/
├── README.md                 # 项目文档
├── requirements.txt          # 依赖列表
├── train.py                  # 训练入口
├── play.py                   # 模型测试
├── assets/                   # 游戏资源
│   └── sprites/              # 图片素材
├── src/                      # 核心代码
│   ├── __init__.py
│   ├── dqn.py               # DQN网络定义
│   ├── game_env.py          # 游戏环境封装
│   ├── trainer.py           # 训练逻辑
│   └── utils.py             # 工具函数
├── models/                   # 模型存储
└── tests/                    # 单元测试
    └── test_dqn.py
```

---

## 训练指南

### 基础训练

```bash
# 快速测试（验证代码正常）
python train.py --num_iters 1000

# 标准训练（约4-8小时）
python train.py --num_iters 500000

# GPU训练（约1-2小时）
python train.py --num_iters 500000  # 自动检测GPU
```

### 参数说明

| 参数 | 默认值 | 说明 | 建议范围 |
|:-----|:------:|:-----|:--------:|
| `--num_iters` | 500000 | 训练步数 | 500K-2M |
| `--lr` | 1e-6 | 学习率 | 1e-7 ~ 1e-5 |
| `--batch_size` | 32 | 批次大小 | 32-128 |
| `--gamma` | 0.99 | 折扣因子 | 0.95-0.99 |
| `--epsilon_start` | 0.1 | 初始探索率 | 0.1-1.0 |
| `--replay_size` | 50000 | 经验池大小 | 10K-100K |
| `--save_interval` | 100000 | 保存间隔 | - |

### 恢复训练

```bash
python train.py --resume models/dqn_checkpoint.pth --num_iters 1000000
```

### 测试模型

```bash
python play.py --model models/dqn_final.pth --episodes 10
```

---

## 算法原理

### DQN网络架构

```
输入: 游戏画面 (4, 84, 84) - 4帧堆叠
      │
      ▼
┌─────────────────────────────────┐
│  Conv2D(32, 8×8, stride=4)     │  → (32, 20, 20)
│  ReLU                          │
├─────────────────────────────────┤
│  Conv2D(64, 4×4, stride=2)     │  → (64, 9, 9)
│  ReLU                          │
├─────────────────────────────────┤
│  Conv2D(64, 3×3, stride=1)     │  → (64, 7, 7)
│  ReLU                          │
├─────────────────────────────────┤
│  Flatten                       │  → 3136
├─────────────────────────────────┤
│  Linear(3136, 512)             │
│  ReLU                          │
├─────────────────────────────────┤
│  Linear(512, 2)                │  → Q值: [不跳, 跳跃]
└─────────────────────────────────┘
```

### 核心创新

#### 1. 经验回放 (Experience Replay)

```python
# 存储经验
replay_buffer.push(state, action, reward, next_state, done)

# 随机采样训练
batch = replay_buffer.sample(batch_size)
```

**作用**：打破样本时间相关性，提高数据利用效率

#### 2. 目标网络 (Target Network)

```python
# 计算目标Q值（使用旧网络）
target_q = reward + gamma * target_net(next_state).max()

# 定期同步
target_net.load_state_dict(policy_net.state_dict())
```

**作用**：稳定训练过程，避免目标值震荡

### 奖励设计

| 事件 | 奖励 | 说明 |
|:-----|:----:|:-----|
| 存活 | +0.1 | 鼓励持续飞行 |
| 通过管道 | +1.0 | 主要目标奖励 |
| 碰撞死亡 | -1.0 | 惩罚失败 |

---

## 训练监控

### 关键指标

训练过程中关注以下指标：

| 指标 | 含义 | 期望趋势 |
|:-----|:-----|:--------:|
| Loss | 损失值 | ↓ 下降 |
| Epsilon | 探索率 | ↓ 下降 |
| Score | 游戏得分 | ↑ 上升 |
| Avg Reward | 平均奖励 | ↑ 上升 |

### 训练曲线参考

```
训练初期 (0-100K):    Loss震荡，Score低
训练中期 (100K-300K): Loss下降，Score逐渐提升
训练后期 (300K+):     Loss稳定，Score持续提高
```

---

## 常见问题

<details>
<summary><b>Q: 游戏窗口没有弹出？</b></summary>

```bash
# 重新安装pygame
pip uninstall pygame && pip install pygame

# 远程服务器需设置显示
export DISPLAY=:0  # Linux
```
</details>

<details>
<summary><b>Q: 报错 "No module named xxx"？</b></summary>

```bash
# 确保激活虚拟环境
source venv/bin/activate
pip install -r requirements.txt
```
</details>

<details>
<summary><b>Q: 训练很慢怎么办？</b></summary>

1. 使用GPU训练
2. 减少 `--replay_size`
3. 增加 `--batch_size`（显存允许时）
</details>

<details>
<summary><b>Q: 模型效果不好？</b></summary>

1. 增加训练步数：`--num_iters 1000000`
2. 调整学习率：`--lr 1e-5` 或 `--lr 1e-7`
3. 增加探索：`--epsilon_start 0.2`
</details>

---

## 进阶优化

完成基础版本后，可尝试以下改进：

| 算法 | 改进点 | 难度 |
|:-----|:-------|:----:|
| Double DQN | 解决Q值过估计 | ⭐⭐ |
| Dueling DQN | 分离状态/动作价值 | ⭐⭐ |
| Prioritized Replay | 优先采样重要经验 | ⭐⭐⭐ |
| Rainbow | 综合多种改进 | ⭐⭐⭐⭐ |

---

## 参考资料

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - DQN原始论文
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Nature论文
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

---

## 致谢

本项目基于 [Flappy-bird-deep-Q-learning-pytorch](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch) 改编，感谢原作者 **Viet Nguyen** 的开源贡献！

---

<p align="center">
  <b>预计学习时间: 1-2周 | 前置知识: Python基础</b>
</p>
