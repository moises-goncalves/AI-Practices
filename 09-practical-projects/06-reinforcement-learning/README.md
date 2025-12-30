# 06-Reinforcement Learning | 强化学习实战项目

> 从游戏AI到金融交易的强化学习实战

---

## 项目列表

| 项目 | 难度 | 算法 | 说明 |
|------|------|------|------|
| [01-flappy-bird-dqn](./01-flappy-bird-dqn) | ⭐⭐⭐ | DQN | Flappy Bird游戏AI |
| [02-dino-run-dqn](./02-dino-run-dqn) | ⭐⭐⭐ | DQN | Chrome恐龙游戏AI |
| [03-stock-trading-rl](./03-stock-trading-rl) | ⭐⭐⭐⭐ | DQN/A2C | 股票交易策略 |

---

## 学习路径

```
入门: Flappy Bird DQN (理解DQN基础)
     ↓
进阶: Chrome Dino DQN (浏览器自动化 + RL)
     ↓
高级: 股票交易RL (金融应用 + 多算法)
```

---

## 核心知识点

### DQN (Deep Q-Network)

```
核心思想: 用神经网络近似Q函数

关键技术:
1. 经验回放 - 打破样本相关性
2. 目标网络 - 稳定训练
3. epsilon-greedy - 探索与利用平衡
```

### A2C (Advantage Actor-Critic)

```
核心思想: 同时学习策略和价值函数

Actor: 输出动作概率
Critic: 评估状态价值
Advantage: A(s,a) = Q(s,a) - V(s)
```

---

## 环境配置

每个项目都有独立的requirements.txt，建议使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 安装依赖
cd 01-flappy-bird-dqn
pip install -r requirements.txt
```

---

## 运行测试

```bash
# 测试所有项目
python -m pytest 01-flappy-bird-dqn/tests/ -v
python -m pytest 02-dino-run-dqn/tests/ -v
python -m pytest 03-stock-trading-rl/tests/ -v
```

---

## 致谢

本项目参考了以下优秀的开源项目：

- [Flappy-bird-deep-Q-learning-pytorch](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch) by Viet Nguyen
- [DinoRunTutorial](https://github.com/Paperspace/DinoRunTutorial) by Paperspace
- [FinRL-Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials) by AI4Finance Foundation

感谢这些项目的作者们的开源贡献！

---

[返回主页](../README.md) | [强化学习理论](../../07-reinforcement-learning/README.md)
