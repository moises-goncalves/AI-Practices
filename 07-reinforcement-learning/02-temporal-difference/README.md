# 时序差分学习 (Temporal Difference Learning)

研究级TD学习算法实现，包含完整的理论文档和实验环境。

## 目录结构

```
时序差分学习/
├── core/                 # 核心算法
│   ├── config.py        # 配置类
│   ├── base.py          # 基类
│   ├── td_prediction.py # TD(0)预测
│   ├── td_control.py    # SARSA, Q-Learning, Expected SARSA
│   ├── advanced.py      # Double Q, N-Step, TD(λ)
│   └── factory.py       # 工厂函数
├── environments/         # 测试环境
│   ├── grid_world.py    # 网格世界
│   ├── cliff_walking.py # 悬崖行走
│   ├── windy_grid.py    # 有风网格
│   └── random_walk.py   # 随机行走
├── utils/               # 工具函数
├── tests/               # 单元测试
├── notebooks/           # Jupyter教程
├── docs/                # 文档
├── knowledge_points.md  # 知识点总结
└── main.py              # CLI入口
```

## 快速开始

```python
from core import TDConfig, create_td_learner
from environments import CliffWalkingEnv

# 创建并训练
config = TDConfig(alpha=0.5, gamma=0.99, epsilon=0.1)
learner = create_td_learner('sarsa', config)
metrics = learner.train(CliffWalkingEnv(), n_episodes=500)

# 评估
reward, std = learner.evaluate(CliffWalkingEnv(), n_episodes=100)
```

## 命令行使用

```bash
python main.py --algorithm sarsa --env cliff_walking --episodes 500
python main.py --compare sarsa q_learning --env cliff_walking
```

## 支持的算法

| 算法 | 类型 | 说明 |
|------|------|------|
| TD(0) | 预测 | 单步TD预测 |
| SARSA | On-Policy | 安全的TD控制 |
| Q-Learning | Off-Policy | 最优策略学习 |
| Expected SARSA | On-Policy | 低方差SARSA |
| Double Q-Learning | Off-Policy | 消除过估计 |
| N-Step TD | 混合 | 偏差-方差权衡 |
| TD(λ) | 混合 | 资格迹方法 |
