# Chrome Dino DQN - 恐龙跳跃游戏AI

**难度**: ⭐⭐⭐☆☆ (中级)

---

## 写在前面：给初学者的话

这是一个非常有趣的项目！你将训练AI来玩Chrome浏览器断网时出现的小恐龙游戏。本项目提供两种模式：
- **模拟器模式**：不需要浏览器，适合快速测试和学习
- **浏览器模式**：控制真实的Chrome浏览器玩游戏

建议初学者先从模拟器模式开始，熟悉后再尝试浏览器模式。

---

## 第一步：环境准备（超详细版）

### 1.1 检查Python版本

打开终端（Windows: CMD/PowerShell，Mac/Linux: Terminal）：

```bash
python --version
# 或
python3 --version
```

需要Python 3.8或更高版本。

### 1.2 进入项目目录

```bash
# 根据你的实际路径修改
cd /path/to/AI-Practices/09-practical-projects/06-reinforcement-learning/02-dino-run-dqn

# 确认目录正确
ls  # Mac/Linux
dir # Windows
# 应该能看到 train.py, requirements.txt 等文件
```

### 1.3 创建虚拟环境

```bash
# 创建
python -m venv venv

# 激活
# Linux/Mac:
source venv/bin/activate
# Windows CMD:
venv\Scripts\activate.bat
# Windows PowerShell:
venv\Scripts\Activate.ps1

# 成功后命令行前面会显示 (venv)
```

### 1.4 安装依赖

```bash
pip install -r requirements.txt

# 如果速度慢，使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**安装的主要包**：
- `tensorflow`: 深度学习框架
- `numpy`: 数值计算
- `opencv-python`: 图像处理
- `Pillow`: 图像处理
- `selenium`: 浏览器自动化（浏览器模式需要）

### 1.5 验证安装

```bash
python -m unittest tests.test_dino -v
```

看到 `OK` 表示环境配置成功！

---

## 第二步：理解项目结构

```
02-dino-run-dqn/
├── README.md              # 本文件
├── requirements.txt       # 依赖列表
├── train.py              # 训练脚本
├── src/
│   ├── __init__.py
│   ├── dqn.py            # DQN网络和智能体
│   ├── game_env.py       # 游戏环境（模拟器+浏览器）
│   └── utils.py          # 图像处理工具
├── models/               # 模型保存目录
└── tests/
    └── test_dino.py      # 单元测试
```

---

## 第三步：使用模拟器模式训练（推荐新手）

模拟器模式不需要浏览器，运行速度快，非常适合学习和测试。

### 3.1 快速测试

```bash
# 先用少量回合测试代码是否正常
python train.py --mode simulator --episodes 10
```

**预期输出**：
```
==================================================
Chrome Dino DQN Training
Mode: simulator
Episodes: 10
==================================================
Episode 1/10, Score: 3, Avg: 3.0, Epsilon: 0.0999
Episode 2/10, Score: 5, Avg: 4.0, Epsilon: 0.0998
...
```

### 3.2 正式训练

```bash
# 训练500回合（约需10-30分钟）
python train.py --mode simulator --episodes 500

# 训练更多回合以获得更好效果
python train.py --mode simulator --episodes 2000
```

### 3.3 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | simulator | 运行模式：simulator或browser |
| `--episodes` | 1000 | 训练回合数 |
| `--lr` | 1e-4 | 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--epsilon` | 0.1 | 初始探索率 |
| `--batch_size` | 32 | 批次大小 |
| `--save_interval` | 100 | 每多少回合保存模型 |

### 3.4 观察训练效果

训练过程中关注这些指标：
- **Score**: 当前回合得分（越高越好）
- **Avg**: 最近10回合平均分（应该逐渐上升）
- **Epsilon**: 探索率（逐渐下降）

---

## 第四步：使用浏览器模式（进阶）

浏览器模式会打开真实的Chrome浏览器，让AI玩真正的恐龙游戏。

### 4.1 准备工作

**步骤1：确认Chrome浏览器版本**
1. 打开Chrome浏览器
2. 地址栏输入 `chrome://version`
3. 记下版本号（如 `120.0.6099.109`）

**步骤2：下载对应版本的ChromeDriver**
1. 访问 https://chromedriver.chromium.org/downloads
2. 下载与你Chrome版本匹配的ChromeDriver
3. 解压到一个你记得的位置

**步骤3：验证ChromeDriver**
```bash
# 替换为你的ChromeDriver路径
/path/to/chromedriver --version
```

### 4.2 开始训练

```bash
python train.py --mode browser --chrome_driver /path/to/chromedriver --episodes 100
```

你会看到：
1. Chrome浏览器自动打开
2. 自动进入恐龙游戏页面
3. AI开始自动玩游戏

**注意**：浏览器模式比模拟器慢很多，建议先用模拟器模式验证代码。

---

## 第五步：测试训练好的模型

```bash
# 使用模拟器模式测试
python train.py --mode simulator --episodes 10
# 观察Score是否比训练前高
```

---

## 两种模式对比

| 特性 | 模拟器模式 | 浏览器模式 |
|------|-----------|-----------|
| 速度 | 快（推荐学习用） | 慢 |
| 依赖 | 无需额外软件 | 需要Chrome+ChromeDriver |
| 画面 | 简化的游戏逻辑 | 真实游戏画面 |
| 用途 | 学习、测试、快速实验 | 真实环境训练 |
| 推荐 | 初学者首选 | 进阶使用 |

---

## 核心代码解析

### DQN网络结构

```python
输入: 4帧游戏画面堆叠 (80, 80, 4)
      ↓
Conv2D(32, 8x8) → MaxPool → ReLU
      ↓
Conv2D(64, 4x4) → MaxPool → ReLU
      ↓
Conv2D(64, 3x3) → MaxPool → ReLU
      ↓
Flatten → Dense(512) → ReLU
      ↓
Dense(2) → 输出: [不跳的Q值, 跳跃的Q值]
```

### 训练流程图解

```
┌─────────────────────────────────────────────────┐
│  1. 获取游戏画面                                  │
│     ↓                                           │
│  2. 图像预处理（灰度化→缩放→归一化）              │
│     ↓                                           │
│  3. 堆叠最近4帧作为状态                          │
│     ↓                                           │
│  4. DQN网络预测Q值，选择动作                     │
│     ↓                                           │
│  5. 执行动作（跳或不跳）                         │
│     ↓                                           │
│  6. 获取奖励（存活+0.1，死亡-1）                 │
│     ↓                                           │
│  7. 存入经验池                                   │
│     ↓                                           │
│  8. 从经验池采样，训练网络                       │
│     ↓                                           │
│  9. 回到步骤1                                    │
└─────────────────────────────────────────────────┘
```

---

## 常见问题解答

### Q1: 模拟器模式报错 "No module named cv2"

```bash
pip install opencv-python
```

### Q2: 浏览器模式报错 "ChromeDriver版本不匹配"

确保ChromeDriver版本与Chrome浏览器版本一致。访问 `chrome://version` 查看浏览器版本。

### Q3: 浏览器模式报错 "找不到ChromeDriver"

```bash
# 使用完整路径
python train.py --mode browser --chrome_driver /full/path/to/chromedriver
```

### Q4: TensorFlow报GPU相关警告

这些警告通常可以忽略，不影响训练。如果想消除：
```bash
export TF_CPP_MIN_LOG_LEVEL=2  # Linux/Mac
set TF_CPP_MIN_LOG_LEVEL=2     # Windows
```

### Q5: 训练效果不好怎么办？

1. 增加训练回合数：`--episodes 2000`
2. 调整学习率：`--lr 1e-5` 或 `--lr 5e-4`
3. 增加探索：`--epsilon 0.2`

### Q6: 如何中断训练？

按 `Ctrl+C`，模型会自动保存到 `models/` 目录。

---

## 进阶学习建议

1. **理解代码**：阅读 `src/dqn.py` 了解DQN实现
2. **修改奖励**：尝试修改 `src/game_env.py` 中的奖励设计
3. **调参实验**：尝试不同的超参数组合
4. **算法改进**：实现Double DQN或Dueling DQN

---

## 致谢

本项目参考了 [DinoRunTutorial](https://github.com/Paperspace/DinoRunTutorial) by Paperspace，感谢原作者的开源贡献！

---

**预计学习时间**: 1-2周
**前置知识**: Python基础
