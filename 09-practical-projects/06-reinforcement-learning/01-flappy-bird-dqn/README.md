# Flappy Bird DQN - 深度Q网络实战项目

**难度**: ⭐⭐⭐☆☆ (中级)

---

## 写在前面：给初学者的话

如果你是第一次接触强化学习项目，不用担心！本教程会手把手带你完成整个过程。即使你之前没有运行过类似项目，只要按照步骤操作，就能成功训练出一个会玩Flappy Bird的AI。

**你需要准备的**：
- 一台电脑（Windows/Mac/Linux都可以）
- Python 3.8或更高版本
- 基本的命令行操作知识（会cd、会输入命令就行）

---

## 项目简介

本项目使用Deep Q-Network (DQN)算法训练AI玩Flappy Bird游戏。通过这个项目，你将深入理解DQN的核心原理，并亲手实现一个能够自主学习的游戏AI。

### 学习目标

- 理解DQN算法的核心思想和数学原理
- 掌握经验回放(Experience Replay)技术
- 学习epsilon-greedy探索策略
- 实践图像预处理和状态表示
- 了解强化学习的训练调试技巧

---

## 第一步：环境准备（超详细版）

### 1.1 检查Python版本

首先，打开你的终端（Windows叫命令提示符或PowerShell，Mac/Linux叫Terminal）。

```bash
# 输入以下命令检查Python版本
python --version
# 或者
python3 --version
```

你应该看到类似 `Python 3.8.x` 或更高版本的输出。如果没有安装Python，请先去 [python.org](https://www.python.org/downloads/) 下载安装。

### 1.2 进入项目目录

```bash
# 假设你的项目在这个路径，请根据实际情况修改
cd /path/to/AI-Practices/09-practical-projects/06-reinforcement-learning/01-flappy-bird-dqn

# 查看当前目录下的文件，确认进入正确
ls  # Mac/Linux
dir # Windows
```

你应该能看到 `train.py`、`play.py`、`requirements.txt` 等文件。

### 1.3 创建虚拟环境（强烈推荐）

虚拟环境可以隔离项目依赖，避免与其他项目冲突。

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac:
source venv/bin/activate
# Windows CMD:
venv\Scripts\activate.bat
# Windows PowerShell:
venv\Scripts\Activate.ps1
```

激活成功后，你的命令行前面会出现 `(venv)` 标志。

### 1.4 安装依赖包

```bash
# 安装项目所需的所有依赖
pip install -r requirements.txt
```

这会安装以下包：
- `torch`: PyTorch深度学习框架
- `numpy`: 数值计算库
- `pygame`: 游戏开发库（用于运行Flappy Bird）
- `opencv-python`: 图像处理库

**常见问题**：
- 如果安装很慢，可以使用国内镜像：
  ```bash
  pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```
- 如果报错找不到pip，试试 `python -m pip install ...`

### 1.5 验证安装

```bash
# 运行单元测试，确保一切正常
python -m unittest tests.test_dqn -v
```

如果看到 `OK` 和所有测试通过，说明环境配置成功！

---

## 第二步：理解项目结构

```
01-flappy-bird-dqn/
├── README.md              # 你正在看的这个文件
├── requirements.txt       # 依赖包列表
├── train.py              # 训练脚本（主程序）
├── play.py               # 测试脚本（用训练好的模型玩游戏）
├── assets/               # 游戏资源
│   └── sprites/          # 小鸟、管道等图片
├── src/                  # 源代码目录
│   ├── __init__.py       # 包初始化文件
│   ├── dqn.py            # DQN神经网络定义
│   ├── game_env.py       # Flappy Bird游戏环境
│   ├── trainer.py        # 训练器（核心训练逻辑）
│   └── utils.py          # 工具函数（图像处理等）
├── models/               # 训练好的模型保存在这里
└── tests/
    └── test_dqn.py       # 单元测试
```

---

## 第三步：开始训练

### 3.1 快速测试（推荐先做这个）

第一次运行，建议先用少量迭代测试代码是否正常：

```bash
python train.py --num_iters 1000
```

你会看到：
1. 一个游戏窗口弹出，显示Flappy Bird游戏
2. 小鸟开始自动玩游戏（一开始会很菜，经常撞管道）
3. 终端输出训练信息

**预期输出**：
```
==================================================
Flappy Bird DQN Training
==================================================
Training iterations: 1000
Learning rate: 1e-06
Batch size: 32
==================================================
Iter: 0, Loss: 0.0000, Epsilon: 0.1000
Iter: 1000, Loss: 0.0234, Epsilon: 0.0999
```

### 3.2 正式训练

确认代码正常后，开始正式训练：

```bash
# 基础训练（约需要几小时到一天）
python train.py --num_iters 500000

# 如果你有GPU，训练会快很多
# 代码会自动检测并使用GPU
```

**训练参数说明**：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--num_iters` | 训练步数 | 500000-2000000 |
| `--lr` | 学习率 | 1e-6（默认） |
| `--batch_size` | 批次大小 | 32（默认） |
| `--save_interval` | 多少步保存一次模型 | 100000（默认） |

**训练时间参考**：
- CPU训练：500000步约需要4-8小时
- GPU训练：500000步约需要1-2小时

### 3.3 中断和恢复训练

如果需要中断训练，按 `Ctrl+C`，模型会自动保存。

恢复训练：
```bash
python train.py --resume models/dqn_interrupted.pth --num_iters 1000000
```

---

## 第四步：测试训练好的模型

训练完成后，用训练好的模型玩游戏：

```bash
# 使用最终模型
python play.py --model models/dqn_final.pth --episodes 10

# 或使用中间保存的模型
python play.py --model models/dqn_500000.pth --episodes 5
```

你会看到AI控制的小鸟自动躲避管道，分数会比训练初期高很多！

---

## 理论背景（可选阅读）

### 什么是DQN？

DQN (Deep Q-Network) 是DeepMind在2013年提出的算法，首次成功将深度学习与强化学习结合。

**核心思想**：用神经网络来估计每个动作的价值（Q值）

```
传统Q-Learning: 用表格存储Q值
    ↓ 问题：状态太多（图像有无数种可能）
DQN: 用神经网络预测Q值
    ↓ 解决：可以处理任意复杂的状态
```

### DQN的两大创新

**1. 经验回放 (Experience Replay)**
- 把玩游戏的经历存起来
- 训练时随机抽取，避免连续样本的相关性

**2. 目标网络 (Target Network)**
- 用一个"旧"网络计算目标值
- 避免训练时目标不断变化导致的不稳定

### 奖励设计

```python
reward = {
    '存活':     +0.1,   # 每活一帧给一点奖励
    '通过管道': +1.0,   # 成功通过管道，大奖励
    '碰撞死亡': -1.0    # 撞到东西，惩罚
}
```

---

## 常见问题解答

### Q1: 游戏窗口没有弹出？

**可能原因**：pygame没有正确安装或显示问题

**解决方法**：
```bash
# 重新安装pygame
pip uninstall pygame
pip install pygame

# 如果是远程服务器，需要设置显示
export DISPLAY=:0  # Linux
```

### Q2: 报错 "No module named 'xxx'"？

**解决方法**：
```bash
# 确保激活了虚拟环境
source venv/bin/activate  # Linux/Mac
# 然后重新安装依赖
pip install -r requirements.txt
```

### Q3: 训练很慢怎么办？

**解决方法**：
1. 使用GPU（如果有的话）
2. 减少 `replay_size` 参数
3. 增加 `batch_size`（如果显存够）

### Q4: 模型训练后效果不好？

**可能原因和解决方法**：
1. 训练步数不够 → 增加 `--num_iters`
2. 学习率不合适 → 尝试 `--lr 1e-5` 或 `--lr 1e-7`
3. 探索不足 → 增加 `--epsilon_start`

### Q5: 如何查看训练进度？

训练过程中会输出：
- `Iter`: 当前迭代次数
- `Loss`: 损失值（应该逐渐下降）
- `Epsilon`: 探索率（逐渐减小）
- `Episode`: 完成的游戏局数
- `Score`: 游戏得分

---

## 进阶学习

完成基础版本后，可以尝试：

1. **Double DQN**: 解决Q值过估计问题
2. **Dueling DQN**: 分离状态价值和动作优势
3. **Prioritized Experience Replay**: 优先学习重要经验

---

## 参考资料

- [DQN原始论文](https://arxiv.org/abs/1312.5602)
- [PyTorch官方教程](https://pytorch.org/tutorials/)

## 致谢

本项目基于 [Flappy-bird-deep-Q-learning-pytorch](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch) 改编，感谢原作者 Viet Nguyen 的开源贡献！

---

**预计学习时间**: 1-2周
**前置知识**: Python基础
