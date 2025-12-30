# Chrome Dino DQN - 恐龙跳跃游戏AI

**难度**: ⭐⭐⭐☆☆ (中级)

## 项目简介

本项目使用DQN算法训练AI玩Chrome浏览器的恐龙跳跃游戏。提供两种运行模式：浏览器模式（控制真实游戏）和模拟器模式（用于快速测试）。

### 学习目标

- 掌握DQN在实际游戏中的应用
- 学习Selenium自动化控制浏览器
- 理解图像预处理和状态表示
- 实践强化学习的训练流程

## 游戏说明

Chrome Dino是Chrome浏览器断网时出现的小游戏：
- 恐龙自动向前跑
- 按空格键跳跃躲避障碍物
- 碰到障碍物游戏结束
- 目标：尽可能跑得更远

## 项目结构

```
02-dino-run-dqn/
├── README.md
├── requirements.txt
├── train.py              # 训练脚本
├── src/
│   ├── __init__.py
│   ├── dqn.py            # DQN网络和智能体
│   ├── game_env.py       # 游戏环境（浏览器/模拟器）
│   └── utils.py          # 工具函数
├── models/               # 保存的模型
└── tests/
    └── test_dino.py      # 单元测试
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行测试

```bash
python -m pytest tests/ -v
```

### 3. 训练模型

**模拟器模式（推荐先用这个测试）：**
```bash
python train.py --mode simulator --episodes 100
```

**浏览器模式（需要ChromeDriver）：**
```bash
# 先下载ChromeDriver: https://chromedriver.chromium.org/
python train.py --mode browser --chrome_driver /path/to/chromedriver
```

## 两种模式对比

| 特性 | 模拟器模式 | 浏览器模式 |
|------|-----------|-----------|
| 速度 | 快 | 慢 |
| 依赖 | 无需浏览器 | 需要Chrome+ChromeDriver |
| 用途 | 测试代码 | 真实训练 |
| 画面 | 简化 | 真实游戏 |

## 核心代码说明

### DQN网络

```python
# 输入: 4帧游戏画面 (80x80x4)
# 输出: 2个动作的Q值 [不跳, 跳跃]

Conv2D(32, 8x8, stride=4) → MaxPool
Conv2D(64, 4x4, stride=2) → MaxPool
Conv2D(64, 3x3, stride=1) → MaxPool
Flatten → Dense(512) → Dense(2)
```

### 训练流程

```
1. 获取游戏画面
2. 预处理：灰度化 → 缩放 → 归一化
3. 堆叠4帧作为状态
4. epsilon-greedy选择动作
5. 执行动作，获取奖励
6. 存入经验池
7. 采样训练
8. 重复
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --mode | simulator | 运行模式 |
| --episodes | 1000 | 训练回合数 |
| --lr | 1e-4 | 学习率 |
| --gamma | 0.99 | 折扣因子 |
| --epsilon | 0.1 | 初始探索率 |
| --batch_size | 32 | 批次大小 |

## 常见问题

**Q: 浏览器模式报错？**
A: 确保ChromeDriver版本与Chrome浏览器版本匹配

**Q: 训练很慢？**
A: 先用模拟器模式验证代码，再用浏览器模式正式训练

**Q: 模型不收敛？**
A: 尝试调整学习率，增加训练回合数

## 致谢

本项目参考了以下开源项目：
- [DinoRunTutorial](https://github.com/Paperspace/DinoRunTutorial) by Paperspace

---

**预计学习时间**: 1-2周
**前置知识**: Python、TensorFlow/Keras基础、强化学习基础
