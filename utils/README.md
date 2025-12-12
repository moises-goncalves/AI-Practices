# utils | 工具模块

> 通用工具函数与可视化组件

---

## 目录结构

```
utils/
├── common.py         # 通用工具：随机种子、设备检测、计时器
├── visualization.py  # 可视化：训练曲线、混淆矩阵、特征图
├── paths.py          # 路径管理：跨平台路径处理
└── metrics/          # 评估指标：自定义评估函数
```

---

## 快速使用

```python
from utils import set_seed, get_data_path, plot_training_history

# 设置随机种子
set_seed(42)

# 获取数据路径
data_path = get_data_path('mnist')

# 绘制训练曲线
plot_training_history(history.history)
```

---

## 核心功能

| 模块 | 功能 |
|------|------|
| `set_seed()` | 设置全局随机种子 |
| `get_device()` | 检测 GPU/CPU |
| `get_data_path()` | 跨平台数据路径 |
| `plot_training_history()` | 训练曲线可视化 |
| `plot_confusion_matrix()` | 混淆矩阵可视化 |

---

[返回主页](../README.md)
