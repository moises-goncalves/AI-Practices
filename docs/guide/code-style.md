# 代码风格指南

本文档定义了 AI-Practices 项目的代码风格和最佳实践。

## 🐍 Python 代码规范

### 基本规则

遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范：

1. **缩进**: 使用 4 个空格
2. **行宽**: 最多 79 个字符（文档字符串 72 个字符）
3. **空行**: 函数和类之间 2 个空行，方法之间 1 个空行
4. **导入**: 每个导入占一行，按标准库、第三方库、本地库分组

### 导入规范

```python
# 正确的导入顺序
# 1. 标准库
import os
import sys
from typing import List, Tuple

# 2. 第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 3. 本地模块
from utils import data_loader
from models import create_model
```

### 变量声明

```python
# 好的例子
learning_rate = 0.001
num_epochs = 100
batch_size = 32
model_name = 'resnet50'

# 避免
lr = 0.001  # 除非是公认的缩写
e = 100
bs = 32
mn = 'resnet50'
```

### 函数定义

```python
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    verbose: bool = True
) -> Tuple[tf.keras.Model, dict]:
    """
    训练神经网络模型

    参数:
        X_train: 训练数据，shape (n_samples, n_features)
        y_train: 训练标签，shape (n_samples,)
        epochs: 训练轮数，默认 100
        batch_size: 批次大小，默认 32
        learning_rate: 学习率，默认 0.001
        verbose: 是否打印训练信息，默认 True

    返回:
        model: 训练好的模型
        history: 包含训练历史的字典

    示例:
        >>> X_train = np.random.rand(1000, 10)
        >>> y_train = np.random.randint(0, 2, 1000)
        >>> model, history = train_model(X_train, y_train)
    """
    model = create_model(X_train.shape[1])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1 if verbose else 0
    )

    return model, history.history
```

### 类定义

```python
class NeuralNetwork:
    """
    自定义神经网络类

    属性:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        activation: 激活函数名称

    方法:
        build(): 构建模型
        train(): 训练模型
        predict(): 进行预测
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu'
    ):
        """
        初始化神经网络

        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表，如 [64, 32]
            output_dim: 输出维度
            activation: 激活函数，默认 'relu'
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.model = None
```

## 📓 Jupyter Notebook 规范

### Notebook 结构

每个 notebook 应遵循以下结构：

```python
# ============================================================
# 文件名: linear_regression_tutorial.ipynb
# 描述: 线性回归算法的完整教程
# 作者: Your Name
# 日期: 2024-01-01
# ============================================================
```

#### 1. 标题和简介

```markdown
# 线性回归教程

## 📚 学习目标

通过本教程，你将学会：
- 理解线性回归的数学原理
- 使用 NumPy 实现线性回归
- 使用 Scikit-learn 快速构建模型
- 评估模型性能

## 📋 前置知识

- Python 基础
- NumPy 基础
- 线性代数基础
```

#### 2. 导入库

```python
# ============================================================
# 导入必要的库
# ============================================================

# 数值计算
import numpy as np
import pandas as pd

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 机器学习
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置
np.random.seed(42)  # 设置随机种子以确保可重复性

# 显示设置
%matplotlib inline
%config InlineBackend.figure_format = 'retina'  # 高清图像

print("所有库导入成功！")
```

#### 3. 代码块注释

```python
# === 数据预处理 ===
# 1. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

## 📛 命名约定

### 变量命名

```python
# 使用小写字母和下划线
learning_rate = 0.001
num_epochs = 100
train_data = load_data()

# 常量使用大写字母
MAX_ITERATIONS = 1000
DEFAULT_BATCH_SIZE = 32
PI = 3.14159
```

### 函数命名

```python
# 使用小写字母和下划线，动词开头
def calculate_accuracy(y_true, y_pred):
    pass

def load_dataset(file_path):
    pass

def preprocess_text(text):
    pass
```

### 类命名

```python
# 使用驼峰命名法
class NeuralNetwork:
    pass

class DataLoader:
    pass

class ModelTrainer:
    pass
```

### 文件命名

```python
# Notebook 文件
linear_regression_tutorial.ipynb
cnn_image_classification.ipynb
lstm_text_generation.ipynb

# Python 脚本
data_preprocessing.py
model_utils.py
evaluation_metrics.py
```

## 💬 注释规范

### 单行注释

```python
# 正确：注释说明为什么这样做
learning_rate = 0.001  # 使用较小的学习率以确保收敛稳定

# 错误：注释只是重复代码
x = 5  # 设置 x 为 5
```

### 多行注释

```python
# 正确：解释复杂逻辑
# 使用 Adam 优化器因为它结合了动量和自适应学习率
# 这对于深度神经网络训练特别有效
# 参考：https://arxiv.org/abs/1412.6980
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

### TODO 注释

```python
# TODO: 添加数据增强
# TODO(username): 实现学习率调度器
# FIXME: 修复维度不匹配问题
# NOTE: 这里需要足够的内存
```

## 📖 文档字符串

### 函数文档字符串

使用 Google 风格：

```python
def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 100,
    batch_size: int = 32
) -> Tuple[tf.keras.Model, dict]:
    """
    训练神经网络模型

    该函数使用提供的训练数据训练神经网络，支持验证集和早停机制。

    Args:
        X_train: 训练特征，shape 为 (n_samples, n_features)
        y_train: 训练标签，shape 为 (n_samples,) 或 (n_samples, n_classes)
        X_val: 验证特征，可选
        y_val: 验证标签，可选
        epochs: 训练轮数，默认 100
        batch_size: 批次大小，默认 32

    Returns:
        model: 训练好的 Keras 模型
        history: 包含训练历史的字典

    Raises:
        ValueError: 如果 X_train 和 y_train 的样本数不匹配

    Examples:
        >>> X_train = np.random.rand(1000, 10)
        >>> y_train = np.random.randint(0, 2, 1000)
        >>> model, history = train_neural_network(X_train, y_train)
    """
    pass
```

## ✨ 最佳实践

### 1. 代码组织

```python
# 将相关功能分组
# === 配置参数 ===
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# === 数据加载 ===
def load_data():
    pass

# === 模型定义 ===
def create_model():
    pass

# === 训练流程 ===
def train():
    pass

# === 主程序 ===
if __name__ == '__main__':
    main()
```

### 2. 避免魔法数字

```python
# 错误：使用魔法数字
model.add(Dense(64))
optimizer = Adam(0.001)

# 正确：使用命名常量
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001

model.add(Dense(HIDDEN_SIZE))
optimizer = Adam(LEARNING_RATE)
```

### 3. 类型提示

```python
from typing import List, Tuple, Optional, Union

def process_batch(
    batch: np.ndarray,
    labels: np.ndarray,
    augment: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """处理一个批次的数据"""
    pass
```

## 🔍 代码审查清单

在提交代码前，检查以下项目：

- [ ] 代码遵循 PEP 8 规范
- [ ] 所有函数和类都有文档字符串
- [ ] 变量命名清晰且有意义
- [ ] 添加了必要的注释
- [ ] 没有魔法数字
- [ ] 代码可以正常运行
- [ ] Notebook 单元格可以顺序执行
- [ ] 图表清晰且有标题和标签
- [ ] 没有调试代码（print 语句除外）
- [ ] 导入语句按规范排序

---

遵循这些规范将使你的代码更加专业和易于维护！
