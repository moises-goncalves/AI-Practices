# 奖励优化 (Reward Optimization)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **从稀疏奖励困境到密集学习信号的系统性突破**

本模块实现了强化学习中四大核心奖励优化技术，涵盖理论推导、工程实现与交互式教程。

---

## 目录结构

```
06-reward-optimization/
├── README.md                          # 本文件
├── __init__.py                        # 模块导出
├── 知识点.md                          # 核心知识点总结
├── knowledge_points_detailed.md       # 详细知识芯片
│
├── reward_shaping.py                  # 势能基奖励塑形 (PBRS)
├── inverse_rl.py                      # 逆强化学习 (IRL/GAIL/AIRL)
├── curiosity_driven.py                # 好奇心驱动探索 (ICM/RND)
├── hindsight_experience_replay.py     # 后见经验回放 (HER)
│
├── core/                              # 核心算法实现
│   ├── reward_shaping.py
│   ├── inverse_rl.py
│   ├── curiosity.py
│   └── hindsight.py
│
├── networks/                          # 神经网络组件
│   ├── feature_encoders.py            # 特征编码器
│   ├── dynamics_models.py             # 动力学模型
│   ├── discriminators.py              # 判别器网络
│   └── value_networks.py              # 价值网络
│
├── buffers/                           # 经验回放缓冲区
│   ├── replay_buffer.py               # 标准回放
│   ├── goal_buffer.py                 # 目标条件回放
│   └── demonstration_buffer.py        # 演示数据缓冲
│
├── utils/                             # 工具函数
│   ├── metrics.py                     # 评估指标
│   └── visualization.py               # 可视化工具
│
└── notebooks/                         # 交互式教程
    ├── 01-Reward-Shaping-Fundamentals.ipynb
    ├── 02-Inverse-RL-Tutorial.ipynb
    ├── 03-Curiosity-Driven-Exploration.ipynb
    └── 04-HER-Tutorial.ipynb
```

---

## 核心算法

### 1. 势能基奖励塑形 (PBRS)

**核心定理** (Ng et al., 1999):

$$R'(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

保证最优策略不变的同时提供密集学习信号。

```python
from reward_shaping import DistanceBasedShaper, ShapedRewardConfig

shaper = DistanceBasedShaper(
    goal_position=np.array([10.0, 10.0]),
    config=ShapedRewardConfig(discount_factor=0.99)
)
shaped_reward = shaper.shape_reward(state, action, next_state, reward, done)
```

### 2. 逆强化学习 (IRL)

从专家演示中学习奖励函数，支持 Max-Margin、MaxEnt、Deep IRL 和 GAIL。

```python
from inverse_rl import MaxEntropyIRL, IRLConfig

irl = MaxEntropyIRL(
    config=IRLConfig(feature_dim=10),
    feature_extractor=extractor
)
reward_weights = irl.fit(demonstrations)
```

### 3. 好奇心驱动探索 (ICM/RND)

通过预测误差生成内在动机奖励：

$$r_i = \eta \cdot \|f(s') - \hat{f}(s, a)\|^2$$

```python
from curiosity_driven import IntrinsicCuriosityModule, CuriosityConfig

icm = IntrinsicCuriosityModule(
    observation_dim=obs_dim,
    action_dim=action_dim,
    config=CuriosityConfig(intrinsic_reward_scale=0.01)
)
intrinsic_reward = icm.compute_intrinsic_reward(state, action, next_state)
```

### 4. 后见经验回放 (HER)

将失败轨迹重标注为成功，大幅提升样本效率：

```python
from hindsight_experience_replay import HindsightExperienceReplay, HERConfig

her = HindsightExperienceReplay(
    buffer=buffer,
    config=HERConfig(replay_k=4, strategy=GoalSelectionStrategy.FUTURE)
)
her.store_transition(state, action, reward, next_state, done, desired_goal, achieved_goal)
```

---

## 方法选择指南

```
是否有专家演示？
├── 是 → IRL/GAIL
└── 否 → 是目标条件任务？
          ├── 是 → HER
          └── 否 → 探索困难？
                    ├── 是 → ICM/RND
                    └── 否 → PBRS
```

| 方法 | 需要演示 | 策略不变性 | 适用场景 |
|------|:--------:|:----------:|----------|
| PBRS | ✗ | ✓ | 目标明确，有领域知识 |
| IRL  | ✓ | N/A | 有专家演示 |
| ICM/RND | ✗ | ✗ | 探索困难环境 |
| HER  | ✗ | ✓ | 目标条件任务 |

---

## 快速开始

```bash
# 运行单元测试
python reward_shaping.py
python inverse_rl.py
python curiosity_driven.py
python hindsight_experience_replay.py

# 启动 Jupyter 教程
jupyter notebook notebooks/
```

---

## 参考文献

1. Ng, A.Y. et al. (1999). *Policy invariance under reward transformations*. ICML.
2. Ziebart, B.D. et al. (2008). *Maximum entropy inverse reinforcement learning*. AAAI.
3. Pathak, D. et al. (2017). *Curiosity-driven exploration*. ICML.
4. Andrychowicz, M. et al. (2017). *Hindsight experience replay*. NeurIPS.
5. Ho, J. & Ermon, S. (2016). *Generative adversarial imitation learning*. NeurIPS.

---

[← 返回强化学习主目录](../README.md)
