<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg" alt="Tests">
</p>

<h1 align="center">Stock Trading RL</h1>

<p align="center">
  <b>使用强化学习算法学习股票交易策略</b>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> •
  <a href="#算法介绍">算法介绍</a> •
  <a href="#训练指南">训练指南</a> •
  <a href="#技术指标">技术指标</a> •
  <a href="#常见问题">FAQ</a>
</p>

---

## 项目简介

本项目使用 **DQN** 和 **A2C** 两种强化学习算法学习股票交易策略。通过自定义交易环境和技术指标，训练AI自动做出买入、卖出、持有决策。

> **风险提示**：本项目仅供学习研究，不构成任何投资建议！

### 项目特色

| 特性 | 说明 |
|:-----|:-----|
| 双算法支持 | DQN + A2C 算法对比 |
| 自定义环境 | 基于Gym接口的交易环境 |
| 技术指标 | MA、RSI、MACD等指标支持 |
| 完整测试 | 11个单元测试覆盖 |

---

## 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选)

### 三步运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证环境
python -m unittest tests.test_trading -v

# 3. 开始训练
python train.py --agent dqn --episodes 100
```

---

## 项目结构

```
03-stock-trading-rl/
├── README.md                 # 项目文档
├── requirements.txt          # 依赖列表
├── train.py                  # 训练入口
├── src/                      # 核心代码
│   ├── __init__.py
│   ├── env.py               # 交易环境
│   ├── agents.py            # DQN/A2C智能体
│   └── data.py              # 数据处理
├── models/                   # 模型存储
└── tests/                    # 单元测试
    └── test_trading.py
```

---

## 算法介绍

### DQN vs A2C

| 特性 | DQN | A2C |
|:-----|:---:|:---:|
| 类型 | 值函数方法 | 策略梯度方法 |
| 经验回放 | 需要 | 不需要 |
| 学习方式 | 离线 | 在线 |
| 稳定性 | 高 | 中 |
| 收敛速度 | 慢 | 快 |
| 推荐场景 | 通用 | 快速适应 |

### 交易环境设计

```
状态空间 (State)
├── 账户余额 (归一化)
├── 持仓数量 (归一化)
├── 当前股价 (归一化)
└── 技术指标 (MA, RSI, MACD)

动作空间 (Action)
├── 0: 持有 (Hold)
├── 1: 买入 (Buy)
└── 2: 卖出 (Sell)

奖励函数 (Reward)
└── 资产变化 = 当前总资产 - 上一步总资产
```

---

## 训练指南

### 使用DQN训练

```bash
# 快速测试
python train.py --agent dqn --episodes 10

# 正式训练
python train.py --agent dqn --episodes 1000
```

### 使用A2C训练

```bash
python train.py --agent a2c --episodes 1000
```

### 参数说明

| 参数 | 默认值 | 说明 | 建议范围 |
|:-----|:------:|:-----|:--------:|
| `--agent` | dqn | 算法类型 | dqn/a2c |
| `--episodes` | 500 | 训练回合数 | 500-5000 |
| `--lr` | 1e-3 | 学习率 | 1e-4 ~ 1e-2 |
| `--gamma` | 0.99 | 折扣因子 | 0.95-0.99 |
| `--initial_balance` | 100000 | 初始资金 | - |
| `--data_file` | None | 自定义数据 | CSV路径 |

---

## 技术指标

| 指标 | 全称 | 作用 |
|:-----|:-----|:-----|
| MA5/MA20 | Moving Average | 短期/长期趋势 |
| RSI | Relative Strength Index | 超买超卖判断 |
| MACD | Moving Average Convergence Divergence | 趋势和动量 |

---

## 自定义数据

### 数据格式

```csv
date,open,high,low,close,volume
2020-01-02,100.5,101.2,99.8,100.8,1000000
2020-01-03,100.8,102.0,100.5,101.5,1200000
```

### 使用自定义数据

```bash
python train.py --data_file /path/to/stock_data.csv --episodes 500
```

---

## 训练监控

| 指标 | 含义 | 期望趋势 |
|:-----|:-----|:--------:|
| Reward | 累计奖励 | 上升 |
| Final Asset | 最终资产 | > 初始资金 |
| ROI | 投资回报率 | 正值 |

---

## 常见问题

<details>
<summary><b>Q: 训练效果不好（一直亏钱）？</b></summary>

1. 增加训练回合：`--episodes 2000`
2. 调整学习率：`--lr 5e-4`
3. 检查数据质量
</details>

<details>
<summary><b>Q: 可以用于实盘交易吗？</b></summary>

**强烈不建议！** 原因：
- 模拟环境与真实市场有差距
- 未考虑滑点、流动性等因素
- 历史表现不代表未来收益
</details>

<details>
<summary><b>Q: 如何保存和加载模型？</b></summary>

模型自动保存到 `models/` 目录：
- `dqn_best.pth`: 最佳模型
- `dqn_final.pth`: 最终模型
</details>

---

## 进阶优化

| 方向 | 说明 | 难度 |
|:-----|:-----|:----:|
| PPO算法 | 更稳定的策略梯度 | ⭐⭐⭐ |
| 多股票组合 | 投资组合优化 | ⭐⭐⭐ |
| 更多指标 | 布林带、KDJ等 | ⭐⭐ |

---

## 致谢

本项目参考 [FinRL-Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials) by **AI4Finance Foundation**，感谢原作者的开源贡献！

---

<p align="center">
  <b>预计学习时间: 2-3周 | 前置知识: Python基础、股票交易基础</b>
</p>
