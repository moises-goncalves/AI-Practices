<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.0+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg" alt="Tests">
</p>

<h1 align="center">Chrome Dino DQN</h1>

<p align="center">
  <b>使用深度Q网络训练AI玩Chrome恐龙跳跃游戏</b>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> •
  <a href="#运行模式">运行模式</a> •
  <a href="#训练指南">训练指南</a> •
  <a href="#算法原理">算法原理</a> •
  <a href="#常见问题">FAQ</a>
</p>

---

## 项目简介

本项目使用 **Deep Q-Network (DQN)** 算法训练AI玩Chrome浏览器的恐龙跳跃游戏。项目提供两种运行模式：**模拟器模式**（快速学习）和**浏览器模式**（真实环境），非常适合强化学习入门学习。

### 项目特色

| 特性 | 说明 |
|:-----|:-----|
| 双模式支持 | 模拟器模式 + 浏览器模式 |
| TensorFlow实现 | 使用Keras构建DQN网络 |
| 浏览器自动化 | Selenium控制真实游戏 |
| 完整测试 | 11个单元测试覆盖 |

---

## 快速开始

### 环境要求

- Python 3.8+
- Chrome浏览器（浏览器模式需要）

### 三步运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证环境
python -m unittest tests.test_dino -v

# 3. 开始训练（模拟器模式）
python train.py --mode simulator --episodes 100
```

<details>
<summary><b>点击展开完整安装指南</b></summary>

#### 1. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate.bat     # Windows
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt

# 国内镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 3. 验证安装

```bash
python -m unittest tests.test_dino -v
```

</details>

---

## 项目结构

```
02-dino-run-dqn/
├── README.md                 # 项目文档
├── requirements.txt          # 依赖列表
├── train.py                  # 训练入口
├── src/                      # 核心代码
│   ├── __init__.py
│   ├── dqn.py               # DQN网络和智能体
│   ├── game_env.py          # 游戏环境封装
│   └── utils.py             # 图像处理工具
├── models/                   # 模型存储
└── tests/                    # 单元测试
    └── test_dino.py
```

---

## 运行模式

### 模式对比

| 特性 | 模拟器模式 | 浏览器模式 |
|:-----|:----------:|:----------:|
| 速度 | 快 | 慢 |
| 依赖 | 无需浏览器 | Chrome + ChromeDriver |
| 画面 | 简化逻辑 | 真实游戏 |
| 推荐 | 初学者首选 | 进阶使用 |

### 模拟器模式（推荐）

```bash
# 快速测试
python train.py --mode simulator --episodes 10

# 正式训练
python train.py --mode simulator --episodes 1000
```

### 浏览器模式

<details>
<summary><b>点击展开浏览器模式配置</b></summary>

#### 1. 确认Chrome版本

打开Chrome，访问 `chrome://version`，记录版本号

#### 2. 下载ChromeDriver

访问 https://chromedriver.chromium.org/downloads 下载对应版本

#### 3. 开始训练

```bash
python train.py --mode browser --chrome_driver /path/to/chromedriver --episodes 100
```

</details>

---

## 训练指南

### 参数说明

| 参数 | 默认值 | 说明 | 建议范围 |
|:-----|:------:|:-----|:--------:|
| `--mode` | simulator | 运行模式 | simulator/browser |
| `--episodes` | 1000 | 训练回合数 | 500-5000 |
| `--lr` | 1e-4 | 学习率 | 1e-5 ~ 1e-3 |
| `--gamma` | 0.99 | 折扣因子 | 0.95-0.99 |
| `--epsilon` | 0.1 | 初始探索率 | 0.1-0.3 |
| `--batch_size` | 32 | 批次大小 | 32-128 |
| `--save_interval` | 100 | 保存间隔 | - |

### 训练监控

| 指标 | 含义 | 期望趋势 |
|:-----|:-----|:--------:|
| Score | 当前得分 | 上升 |
| Avg | 平均得分 | 上升 |
| Epsilon | 探索率 | 下降 |

---

## 算法原理

### DQN网络架构

```
输入: 4帧游戏画面 (80, 80, 4)
      │
      ▼
┌─────────────────────────────────┐
│  Conv2D(32, 8×8) + MaxPool     │
│  ReLU                          │
├─────────────────────────────────┤
│  Conv2D(64, 4×4) + MaxPool     │
│  ReLU                          │
├─────────────────────────────────┤
│  Conv2D(64, 3×3) + MaxPool     │
│  ReLU                          │
├─────────────────────────────────┤
│  Flatten → Dense(512) → ReLU   │
├─────────────────────────────────┤
│  Dense(2)                      │
│  输出: [不跳Q值, 跳跃Q值]        │
└─────────────────────────────────┘
```

### 训练流程

```
┌────────────────────────────────────────┐
│  1. 获取游戏画面                        │
│  2. 图像预处理（灰度化→缩放→归一化）     │
│  3. 堆叠4帧作为状态                     │
│  4. DQN预测Q值，选择动作                │
│  5. 执行动作（跳/不跳）                 │
│  6. 获取奖励（存活+0.1，死亡-1）        │
│  7. 存入经验池                         │
│  8. 采样训练网络                        │
│  9. 循环                               │
└────────────────────────────────────────┘
```

---

## 常见问题

<details>
<summary><b>Q: 报错 "No module named cv2"？</b></summary>

```bash
pip install opencv-python
```
</details>

<details>
<summary><b>Q: ChromeDriver版本不匹配？</b></summary>

确保ChromeDriver版本与Chrome浏览器版本一致，访问 `chrome://version` 查看
</details>

<details>
<summary><b>Q: TensorFlow GPU警告？</b></summary>

```bash
export TF_CPP_MIN_LOG_LEVEL=2  # Linux/Mac
set TF_CPP_MIN_LOG_LEVEL=2     # Windows
```
</details>

<details>
<summary><b>Q: 训练效果不好？</b></summary>

1. 增加回合数：`--episodes 2000`
2. 调整学习率：`--lr 1e-5`
3. 增加探索：`--epsilon 0.2`
</details>

---

## 进阶优化

| 方向 | 说明 | 难度 |
|:-----|:-----|:----:|
| Double DQN | 解决Q值过估计 | ⭐⭐ |
| Dueling DQN | 分离状态/动作价值 | ⭐⭐ |
| 奖励塑形 | 优化奖励函数设计 | ⭐⭐ |

---

## 致谢

本项目参考 [DinoRunTutorial](https://github.com/Paperspace/DinoRunTutorial) by **Paperspace**，感谢原作者的开源贡献！

---

<p align="center">
  <b>预计学习时间: 1-2周 | 前置知识: Python基础</b>
</p>
