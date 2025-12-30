# 股票交易强化学习 - FinRL实战项目

**难度**: ⭐⭐⭐⭐☆ (中高级)

## 项目简介

本项目使用强化学习算法进行股票交易策略学习。实现了DQN和A2C两种算法，支持自定义股票数据和技术指标。

### 学习目标

- 理解强化学习在金融领域的应用
- 掌握交易环境的设计和奖励函数设计
- 学习DQN和A2C算法的实现
- 实践技术指标的计算和使用

## 项目结构

```
03-stock-trading-rl/
├── README.md
├── requirements.txt
├── train.py              # 训练脚本
├── src/
│   ├── __init__.py
│   ├── env.py            # 交易环境
│   ├── agents.py         # RL智能体
│   └── data.py           # 数据处理
├── models/               # 保存的模型
└── tests/
    └── test_trading.py   # 单元测试
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

```bash
# 使用DQN（默认）
python train.py --agent dqn --episodes 500

# 使用A2C
python train.py --agent a2c --episodes 500

# 使用自定义数据
python train.py --data_file data/stock.csv
```

## 核心概念

### 交易环境

```
状态空间:
- 账户余额（归一化）
- 持仓数量（归一化）
- 当前价格（归一化）
- 技术指标（MA, RSI, MACD等）

动作空间:
- 0: 持有
- 1: 买入
- 2: 卖出

奖励设计:
- 基于资产变化的即时奖励
- reward = (当前资产 - 上一资产) * 缩放因子
```

### 技术指标

| 指标 | 说明 | 用途 |
|------|------|------|
| MA5/MA20 | 移动平均线 | 趋势判断 |
| RSI | 相对强弱指数 | 超买超卖 |
| MACD | 指数平滑异同 | 趋势和动量 |

## 算法对比

| 算法 | 动作类型 | 特点 |
|------|----------|------|
| DQN | 离散 | 稳定，适合简单策略 |
| A2C | 离散 | 在线学习，收敛快 |

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --agent | dqn | 算法类型 |
| --episodes | 500 | 训练回合 |
| --lr | 1e-3 | 学习率 |
| --gamma | 0.99 | 折扣因子 |
| --initial_balance | 100000 | 初始资金 |

## 注意事项

1. **风险提示**: 本项目仅供学习，不构成投资建议
2. **数据质量**: 真实交易需要高质量的历史数据
3. **过拟合**: 注意在测试集上验证策略

## 致谢

本项目参考了以下开源项目：
- [FinRL-Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials) by AI4Finance Foundation

---

**预计学习时间**: 2-3周
**前置知识**: Python、PyTorch、强化学习基础、金融基础知识
