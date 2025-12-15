# OpenAI Gymnasium 强化学习环境接口指南

本模块提供 Gymnasium (原 OpenAI Gym) 强化学习环境接口的完整教程，包括环境使用、
包装器应用和经典控制任务详解。

## 目录

```
openai-gym介绍/
├── README.md                         # 本文件
├── gym_introduction.py               # Gymnasium 核心接口教程
├── classic_control_environments.py   # 经典控制环境详解
└── environment_wrappers.py           # 环境包装器工具集
```

## 核心概念

### Gymnasium 简介

Gymnasium 是强化学习研究的标准化环境接口，定义了智能体与环境交互的统一 API：

```python
import gymnasium as gym

# 创建环境
env = gym.make("CartPole-v1")

# 重置环境
observation, info = env.reset()

# 交互循环
for _ in range(1000):
    action = env.action_space.sample()  # 随机动作
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### 核心 API

| 方法 | 描述 | 返回值 |
|------|------|--------|
| `env.reset()` | 重置环境到初始状态 | `(observation, info)` |
| `env.step(action)` | 执行动作 | `(obs, reward, terminated, truncated, info)` |
| `env.render()` | 渲染环境 | `None` 或 `np.ndarray` |
| `env.close()` | 释放资源 | `None` |

### 空间类型

Gymnasium 定义了多种空间类型：

- **Discrete(n)**: 离散空间 {0, 1, ..., n-1}
- **Box(low, high, shape)**: 连续空间 [low, high]^shape
- **MultiDiscrete(nvec)**: 多离散空间
- **MultiBinary(n)**: 多二值空间
- **Tuple, Dict**: 组合空间

## 模块说明

### 1. gym_introduction.py

核心接口教程，包含：

- 空间分析工具 (`analyze_space`, `get_obs_dim`, `get_action_dim`)
- 环境规格提取 (`get_env_spec`, `EnvironmentSpec`)
- 回合执行工具 (`run_episode`, `evaluate_policy`)
- 观测归一化器 (`ObservationNormalizer`)
- 奖励缩放器 (`RewardScaler`)

**运行示例：**

```bash
# 运行所有演示
python gym_introduction.py

# 运行特定演示
python gym_introduction.py --demo cartpole
python gym_introduction.py --demo mountaincar
python gym_introduction.py --demo pendulum

# 运行单元测试
python gym_introduction.py --test
```

### 2. classic_control_environments.py

经典控制环境详解，包含：

- **CartPole**: 倒立摆平衡任务
- **MountainCar**: 爬山车任务
- **Acrobot**: 双摆控制任务
- **Pendulum**: 单摆连续控制任务

每个环境提供：
- 完整的数学描述和物理参数
- 多种策略实现（随机、规则、PID 等）
- 策略比较工具

**运行示例：**

```bash
# 查看所有环境描述
python classic_control_environments.py

# 运行策略演示
python classic_control_environments.py --demo --episodes 10

# 比较不同策略
python classic_control_environments.py --compare

# 生成可视化图表
python classic_control_environments.py --visualize

# 运行单元测试
python classic_control_environments.py --test
```

### 3. environment_wrappers.py

环境包装器工具集，包含：

**观测包装器：**
- `NormalizeObservationWrapper`: 在线观测归一化
- `FrameStackWrapper`: 帧堆叠（用于部分可观测环境）
- `FlattenObservationWrapper`: 观测展平

**动作包装器：**
- `ClipActionWrapper`: 动作裁剪
- `RescaleActionWrapper`: 动作重缩放 ([-1,1] ↔ [low,high])
- `StickyActionWrapper`: 粘性动作（随机重复）

**奖励包装器：**
- `NormalizeRewardWrapper`: 奖励归一化
- `ClipRewardWrapper`: 奖励裁剪
- `SignRewardWrapper`: 符号奖励

**通用包装器：**
- `TimeLimitWrapper`: 时间限制
- `EpisodeStatisticsWrapper`: 回合统计
- `ActionRepeatWrapper`: 动作重复

**便捷工厂函数：**

```python
from environment_wrappers import make_wrapped_env

env = make_wrapped_env(
    "Pendulum-v1",
    normalize_obs=True,
    normalize_reward=True,
    clip_action=True,
    record_stats=True
)
```

**运行示例：**

```bash
# 运行演示
python environment_wrappers.py --demo

# 运行单元测试
python environment_wrappers.py --test
```

## 安装依赖

```bash
# 基础安装
pip install gymnasium numpy

# 经典控制环境
pip install gymnasium[classic-control]

# 可视化支持
pip install matplotlib

# 完整安装
pip install gymnasium[all] matplotlib
```

## 快速开始

### 示例 1: 基础环境交互

```python
from gym_introduction import run_episode, evaluate_policy
import gymnasium as gym

# 创建环境
env = gym.make("CartPole-v1")

# 定义策略
def simple_policy(obs):
    return 1 if obs[2] > 0 else 0  # 根据角度选择动作

# 运行单个回合
result = run_episode(env, simple_policy)
print(f"回合奖励: {result.total_reward}")

# 评估策略
stats = evaluate_policy(env, simple_policy, n_episodes=100)
print(f"平均奖励: {stats['mean_reward']:.1f} ± {stats['std_reward']:.1f}")
```

### 示例 2: 使用包装器

```python
from environment_wrappers import (
    NormalizeObservationWrapper,
    EpisodeStatisticsWrapper,
    make_wrapped_env
)
import gymnasium as gym

# 方式 1: 手动包装
env = gym.make("Pendulum-v1")
env = EpisodeStatisticsWrapper(env)
env = NormalizeObservationWrapper(env)

# 方式 2: 使用工厂函数
env = make_wrapped_env(
    "Pendulum-v1",
    normalize_obs=True,
    normalize_reward=True
)
```

### 示例 3: 策略比较

```python
from classic_control_environments import (
    CartPolePolicy,
    compare_policies
)

policies = {
    "Random": CartPolePolicy(method="random"),
    "Angle": CartPolePolicy(method="angle"),
    "PID": CartPolePolicy(method="pid")
}

results = compare_policies("CartPole-v1", policies, n_episodes=50)
```

## 环境列表

### 经典控制 (Classic Control)

| 环境 ID | 状态维度 | 动作空间 | 最大步数 |
|---------|----------|----------|----------|
| CartPole-v1 | 4 | Discrete(2) | 500 |
| MountainCar-v0 | 2 | Discrete(3) | 200 |
| MountainCarContinuous-v0 | 2 | Box(1) | 999 |
| Acrobot-v1 | 6 | Discrete(3) | 500 |
| Pendulum-v1 | 3 | Box(1) | 200 |
| LunarLander-v2 | 8 | Discrete(4) | 1000 |

### 环境创建参数

```python
# 指定渲染模式
env = gym.make("CartPole-v1", render_mode="human")

# 修改最大步数
env = gym.make("CartPole-v1", max_episode_steps=1000)
```

## 数学基础

### 马尔可夫决策过程 (MDP)

$$MDP = (S, A, P, R, \gamma)$$

- $S$: 状态空间
- $A$: 动作空间
- $P(s'|s,a)$: 转移概率
- $R(s,a,s')$: 奖励函数
- $\gamma$: 折扣因子

### 期望回报

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### 观测归一化

$$\tilde{s} = \frac{s - \mu}{\sigma + \epsilon}$$

使用 Welford 算法在线估计 $\mu$ 和 $\sigma$。

## 最佳实践

1. **始终使用 `env.close()`**: 释放渲染资源
2. **使用随机种子**: `env.reset(seed=42)` 确保可复现
3. **检查终止条件**: 区分 `terminated`（任务完成）和 `truncated`（超时）
4. **归一化观测**: 对神经网络策略至关重要
5. **记录统计信息**: 使用 `EpisodeStatisticsWrapper` 监控训练
6. **并行采样**: 使用向量化环境提高效率

## 参考文献

1. Brockman, G. et al. (2016). OpenAI Gym. arXiv:1606.01540
2. Towers, M. et al. (2023). Gymnasium: A Standard Interface for RL Environments
3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction

## 许可证

MIT License
