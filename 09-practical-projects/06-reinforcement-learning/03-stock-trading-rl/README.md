# 股票交易强化学习 - FinRL实战项目

**难度**: ⭐⭐⭐⭐☆ (中高级)

---

## 写在前面：给初学者的话

这是一个将强化学习应用于金融领域的项目。通过这个项目，你将学会如何让AI自动学习股票交易策略。

**重要提示**：本项目仅供学习研究，不构成任何投资建议！

**你需要准备的**：
- Python 3.8或更高版本
- 基本的命令行操作知识
- 对股票交易有基本了解（知道买入、卖出、持有即可）

---

## 第一步：环境准备（超详细版）

### 1.1 检查Python版本

```bash
python --version
# 需要 Python 3.8+
```

### 1.2 进入项目目录

```bash
cd /path/to/AI-Practices/09-practical-projects/06-reinforcement-learning/03-stock-trading-rl

# 确认目录正确
ls  # Mac/Linux
dir # Windows
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
```

### 1.4 安装依赖

```bash
pip install -r requirements.txt

# 国内用户使用镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**安装的主要包**：
- `torch`: PyTorch深度学习框架
- `numpy`: 数值计算
- `pandas`: 数据处理
- `yfinance`: 下载股票数据（可选）
- `matplotlib`: 绑图（可选）

### 1.5 验证安装

```bash
python -m unittest tests.test_trading -v
```

看到 `OK` 表示环境配置成功！

---

## 第二步：理解项目结构

```
03-stock-trading-rl/
├── README.md              # 本文件
├── requirements.txt       # 依赖列表
├── train.py              # 训练脚本
├── src/
│   ├── __init__.py
│   ├── env.py            # 交易环境
│   ├── agents.py         # DQN和A2C智能体
│   └── data.py           # 数据处理和技术指标
├── models/               # 模型保存目录
└── tests/
    └── test_trading.py   # 单元测试
```

---

## 第三步：理解核心概念

### 3.1 交易环境

把股票交易想象成一个游戏：

```
状态（State）= 你看到的信息
├── 账户余额（还有多少钱）
├── 持仓数量（持有多少股票）
├── 当前股价
└── 技术指标（MA、RSI、MACD等）

动作（Action）= 你能做的操作
├── 0: 持有（什么都不做）
├── 1: 买入（用钱买股票）
└── 2: 卖出（把股票换成钱）

奖励（Reward）= 操作的结果
└── 资产变化 = 当前总资产 - 上一步总资产
```

### 3.2 技术指标简介

| 指标 | 全称 | 作用 |
|------|------|------|
| MA | Moving Average | 移动平均线，判断趋势 |
| RSI | Relative Strength Index | 相对强弱指数，判断超买超卖 |
| MACD | Moving Average Convergence Divergence | 判断趋势和动量 |

### 3.3 两种算法

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| DQN | 稳定，使用经验回放 | 通用，推荐新手 |
| A2C | 在线学习，收敛快 | 需要快速适应的场景 |

---

## 第四步：开始训练

### 4.1 快速测试（推荐先做）

```bash
# 用少量回合测试代码是否正常
python train.py --agent dqn --episodes 10
```

**预期输出**：
```
==================================================
Stock Trading RL Training
Agent: DQN
Episodes: 10
Initial Balance: 100000
==================================================
Episode 10/10, Reward: 0.15, Avg: 0.12, Final Asset: 100234.56
```

### 4.2 使用DQN训练

```bash
# 基础训练（约需5-15分钟）
python train.py --agent dqn --episodes 500

# 更多回合以获得更好效果
python train.py --agent dqn --episodes 2000
```

### 4.3 使用A2C训练

```bash
python train.py --agent a2c --episodes 500
```

### 4.4 自定义参数

```bash
# 调整学习率和初始资金
python train.py --agent dqn --episodes 1000 --lr 0.0005 --initial_balance 50000
```

---

## 第五步：参数详解

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `--agent` | dqn | 算法类型 | dqn 或 a2c |
| `--episodes` | 500 | 训练回合数 | 500-5000 |
| `--lr` | 1e-3 | 学习率 | 1e-4 到 1e-2 |
| `--gamma` | 0.99 | 折扣因子 | 0.95-0.99 |
| `--initial_balance` | 100000 | 初始资金 | 根据需要设置 |
| `--save_path` | models | 模型保存目录 | 任意路径 |
| `--data_file` | None | 自定义数据文件 | CSV文件路径 |

---

## 第六步：使用自定义数据

### 6.1 数据格式要求

CSV文件需要包含以下列：
```
date,open,high,low,close,volume
2020-01-02,100.5,101.2,99.8,100.8,1000000
2020-01-03,100.8,102.0,100.5,101.5,1200000
...
```

### 6.2 使用自定义数据训练

```bash
python train.py --data_file /path/to/your/stock_data.csv --episodes 500
```

### 6.3 下载真实股票数据（可选）

如果安装了yfinance，可以在Python中下载：
```python
from src.data import download_stock_data
df = download_stock_data('AAPL', '2020-01-01', '2023-12-31', 'data/aapl.csv')
```

---

## 第七步：理解训练输出

训练过程中会显示：
```
Episode 100/500, Reward: 0.25, Avg: 0.18, Final Asset: 102345.67
```

| 指标 | 含义 | 期望趋势 |
|------|------|----------|
| Episode | 当前回合/总回合 | - |
| Reward | 本回合累计奖励 | 逐渐上升 |
| Avg | 最近10回合平均奖励 | 逐渐上升 |
| Final Asset | 回合结束时总资产 | 大于初始资金 |

---

## 第八步：评估模型

训练结束后会自动在测试集上评估：
```
Test Results: Final Asset: 108234.56, Profit: 8234.56, ROI: 8.23%
```

| 指标 | 含义 |
|------|------|
| Final Asset | 最终总资产 |
| Profit | 盈利金额 |
| ROI | 投资回报率 |

---

## 常见问题解答

### Q1: 报错 "No module named torch"

```bash
pip install torch
```

### Q2: 训练效果不好（一直亏钱）

1. 增加训练回合：`--episodes 2000`
2. 调整学习率：`--lr 5e-4`
3. 检查数据质量

### Q3: 如何保存和加载模型？

模型会自动保存到 `models/` 目录：
- `dqn_best.pth`: 最佳模型
- `dqn_final.pth`: 最终模型

### Q4: 可以用于实盘交易吗？

**强烈不建议！** 本项目仅供学习，原因：
1. 模拟环境与真实市场有差距
2. 没有考虑滑点、流动性等因素
3. 历史表现不代表未来收益

### Q5: 如何中断训练？

按 `Ctrl+C`，模型会自动保存。

---

## 代码结构解析

### 交易环境 (env.py)

```python
class StockTradingEnv:
    def reset(self):
        # 重置环境，返回初始状态
        
    def step(self, action):
        # 执行动作，返回：新状态、奖励、是否结束、额外信息
```

### DQN智能体 (agents.py)

```python
class DQNAgent:
    def select_action(self, state):
        # 根据状态选择动作（探索或利用）
        
    def train(self):
        # 从经验池采样，更新网络
```

---

## 进阶学习建议

1. **理解代码**：仔细阅读 `src/` 下的代码
2. **修改奖励函数**：尝试不同的奖励设计
3. **添加更多指标**：在 `data.py` 中添加新的技术指标
4. **尝试其他算法**：实现PPO、SAC等算法

---

## 致谢

本项目参考了 [FinRL-Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials) by AI4Finance Foundation，感谢原作者的开源贡献！

---

**预计学习时间**: 2-3周
**前置知识**: Python基础、对股票交易有基本了解
