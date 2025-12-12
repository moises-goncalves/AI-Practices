# 代码风格指南

本文档定义了 AI-Practices 项目的代码风格和最佳实践。

## 📋 目录

- [Python代码规范](#python代码规范)
- [Jupyter Notebook规范](#jupyter-notebook规范)
- [命名约定](#命名约定)
- [注释规范](#注释规范)
- [文档字符串](#文档字符串)
- [最佳实践](#最佳实践)

## 🐍 Python代码规范

### 基本规则

遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范：

1. **缩进**: 使用4个空格
2. **行宽**: 最多79个字符（文档字符串72个字符）
3. **空行**: 函数和类之间2个空行，方法之间1个空行
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
        epochs: 训练轮数，默认100
        batch_size: 批次大小，默认32
        learning_rate: 学习率，默认0.001
        verbose: 是否打印训练信息，默认True

    返回:
        model: 训练好的模型
        history: 包含训练历史的字典

    示例:
        >>> X_train = np.random.rand(1000, 10)
        >>> y_train = np.random.randint(0, 2, 1000)
        >>> model, history = train_model(X_train, y_train)
    """
    # 函数实现
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
            hidden_dims: 隐藏层维度列表，如[64, 32]
            output_dim: 输出维度
            activation: 激活函数，默认'relu'
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.model = None

    def build(self) -> tf.keras.Model:
        """构建模型架构"""
        model = tf.keras.Sequential()

        # 输入层
        model.add(tf.keras.layers.Dense(
            self.hidden_dims[0],
            activation=self.activation,
            input_shape=(self.input_dim,)
        ))

        # 隐藏层
        for dim in self.hidden_dims[1:]:
            model.add(tf.keras.layers.Dense(dim, activation=self.activation))

        # 输出层
        model.add(tf.keras.layers.Dense(self.output_dim, activation='softmax'))

        self.model = model
        return model
```

## 📓 Jupyter Notebook规范

### Notebook结构

每个notebook应遵循以下结构：

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
- 使用NumPy实现线性回归
- 使用Scikit-learn快速构建模型
- 评估模型性能

## 📋 前置知识

- Python基础
- NumPy基础
- 线性代数基础

## ⏱️ 预计时间

30-45分钟
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
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 显示设置
%matplotlib inline
%config InlineBackend.figure_format = 'retina'  # 高清图像

print("所有库导入成功！")
```

#### 3. 理论背景

```markdown
## 📖 理论背景

### 什么是线性回归？

线性回归是一种用于建立变量之间线性关系的统计方法。

### 数学公式

对于单变量线性回归：
$$y = wx + b$$

其中：
- $y$ 是预测值
- $x$ 是输入特征
- $w$ 是权重（斜率）
- $b$ 是偏置（截距）

### 损失函数

使用均方误差(MSE)作为损失函数：
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

### 优化方法

1. **正规方程法**：直接计算最优解
2. **梯度下降法**：迭代优化

> 💡 **提示**: 当特征数量较少时，正规方程法更快；特征数量很大时，梯度下降法更适合。
```

#### 4. 数据准备

```python
# ============================================================
# 数据准备
# ============================================================

def generate_linear_data(n_samples=100, noise=0.1):
    """
    生成线性回归的模拟数据

    参数:
        n_samples: 样本数量
        noise: 噪声水平

    返回:
        X: 特征矩阵
        y: 目标变量
    """
    X = 2 * np.random.rand(n_samples, 1)
    y = 4 + 3 * X + noise * np.random.randn(n_samples, 1)
    return X, y

# 生成数据
X, y = generate_linear_data(n_samples=100, noise=0.5)

print(f"数据形状 - X: {X.shape}, y: {y.shape}")
print(f"X范围: [{X.min():.2f}, {X.max():.2f}]")
print(f"y范围: [{y.min():.2f}, {y.max():.2f}]")
```

#### 5. 数据可视化

```python
# ============================================================
# 数据可视化
# ============================================================

plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('生成的线性数据', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print("✓ 数据可视化完成")
```

#### 6. 模型实现

```python
# ============================================================
# 方法1: 使用正规方程
# ============================================================

# 添加偏置项
X_b = np.c_[np.ones((len(X), 1)), X]  # 添加 x0 = 1

# 计算最优参数: θ = (X^T * X)^(-1) * X^T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("方法1: 正规方程")
print(f"截距 (b): {theta_best[0][0]:.4f}")
print(f"斜率 (w): {theta_best[1][0]:.4f}")
print()

# ============================================================
# 方法2: 使用Scikit-learn
# ============================================================

model = LinearRegression()
model.fit(X, y)

print("方法2: Scikit-learn")
print(f"截距 (b): {model.intercept_[0]:.4f}")
print(f"斜率 (w): {model.coef_[0][0]:.4f}")
```

#### 7. 结果可视化

```python
# ============================================================
# 结果可视化
# ============================================================

# 生成预测点
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pred_manual = X_new_b.dot(theta_best)
y_pred_sklearn = model.predict(X_new)

# 绘图
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 左图：正规方程结果
axes[0].scatter(X, y, alpha=0.6, s=50, edgecolors='k', linewidth=0.5, label='数据点')
axes[0].plot(X_new, y_pred_manual, 'r-', linewidth=2, label='拟合线')
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title('方法1: 正规方程', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：Scikit-learn结果
axes[1].scatter(X, y, alpha=0.6, s=50, edgecolors='k', linewidth=0.5, label='数据点')
axes[1].plot(X_new, y_pred_sklearn, 'b-', linewidth=2, label='拟合线')
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('y', fontsize=12)
axes[1].set_title('方法2: Scikit-learn', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ 结果可视化完成")
```

#### 8. 模型评估

```python
# ============================================================
# 模型评估
# ============================================================

# 计算预测值
y_pred = model.predict(X)

# 计算评估指标
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("模型评估结果:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"R² 分数: {r2:.4f}")

# 可视化残差
residuals = y - y_pred

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 残差图
axes[0].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('预测值', fontsize=12)
axes[0].set_ylabel('残差', fontsize=12)
axes[0].set_title('残差图', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 残差分布
axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('残差', fontsize=12)
axes[1].set_ylabel('频数', fontsize=12)
axes[1].set_title('残差分布', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

#### 9. 总结

```markdown
## 📝 总结

### 关键要点

1. ✅ 线性回归用于建立变量之间的线性关系
2. ✅ 可以使用正规方程或梯度下降法求解
3. ✅ Scikit-learn提供了简单易用的API
4. ✅ 模型评估使用MSE、RMSE和R²等指标

### 下一步

- 学习多元线性回归
- 了解正则化方法（Ridge、Lasso）
- 探索非线性回归模型

### 练习题

1. 尝试使用多个特征进行回归
2. 实现梯度下降算法
3. 比较不同正则化方法的效果

## 📚 参考资料

- [Scikit-learn文档](https://scikit-learn.org/stable/)
- [线性回归数学推导](https://example.com)
```

### Markdown单元格规范

#### 标题层次

```markdown
# 一级标题（章节标题）

## 二级标题（主要部分）

### 三级标题（子部分）

#### 四级标题（详细说明）
```

#### 强调和提示

```markdown
**重要概念加粗**

*斜体用于强调*

> 💡 **提示**: 这是一个有用的提示

> ⚠️ **注意**: 这需要特别注意

> ✅ **最佳实践**: 推荐的做法

> ❌ **避免**: 不推荐的做法
```

#### 代码块

```markdown
行内代码：使用 `model.fit()` 训练模型

代码块：
\```python
import numpy as np
X = np.array([[1, 2], [3, 4]])
\```
```

#### 列表

```markdown
有序列表：
1. 第一步
2. 第二步
3. 第三步

无序列表：
- 选项A
- 选项B
- 选项C

任务列表：
- [x] 已完成任务
- [ ] 待完成任务
```

#### 数学公式

```markdown
行内公式：损失函数 $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

独立公式：
$$
\theta = (X^TX)^{-1}X^Ty
$$

多行公式：
$$
\begin{aligned}
y &= wx + b \\
L &= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\end{aligned}
$$
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
# Notebook文件
linear_regression_tutorial.ipynb
cnn_image_classification.ipynb
lstm_text_generation.ipynb

# Python脚本
data_preprocessing.py
model_utils.py
evaluation_metrics.py

# Markdown文档
决策树算法详解.md
Keras使用指南.md
```

## 💬 注释规范

### 单行注释

```python
# 正确：注释说明为什么这样做
learning_rate = 0.001  # 使用较小的学习率以确保收敛稳定

# 错误：注释只是重复代码
x = 5  # 设置x为5
```

### 多行注释

```python
# 正确：解释复杂逻辑
# 使用Adam优化器因为它结合了动量和自适应学习率
# 这对于深度神经网络训练特别有效
# 参考：https://arxiv.org/abs/1412.6980
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

### 代码块注释

```python
# === 数据预处理 ===
# 1. 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 3. 转换为TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(32)
```

### TODO注释

```python
# TODO: 添加数据增强
# TODO(username): 实现学习率调度器
# FIXME: 修复维度不匹配问题
# NOTE: 这里需要足够的内存
```

## 📖 文档字符串

### 函数文档字符串

使用Google风格：

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

    该函数使用提供的训练数据训练神经网络，支持验证集
    和早停机制。

    Args:
        X_train: 训练特征，shape为(n_samples, n_features)
        y_train: 训练标签，shape为(n_samples,)或(n_samples, n_classes)
        X_val: 验证特征，可选
        y_val: 验证标签，可选
        epochs: 训练轮数，默认100
        batch_size: 批次大小，默认32

    Returns:
        model: 训练好的Keras模型
        history: 包含训练历史的字典，键包括:
            - 'loss': 训练损失
            - 'accuracy': 训练准确率
            - 'val_loss': 验证损失（如果提供验证集）
            - 'val_accuracy': 验证准确率（如果提供验证集）

    Raises:
        ValueError: 如果X_train和y_train的样本数不匹配
        ValueError: 如果提供X_val但未提供y_val

    Examples:
        >>> X_train = np.random.rand(1000, 10)
        >>> y_train = np.random.randint(0, 2, 1000)
        >>> model, history = train_neural_network(X_train, y_train)
        >>> print(f"最终准确率: {history['accuracy'][-1]:.4f}")

    Note:
        - 建议提供验证集以监控过拟合
        - 对于大型数据集，考虑使用生成器
    """
    # 函数实现
    pass
```

### 类文档字符串

```python
class ConvolutionalNeuralNetwork:
    """
    卷积神经网络实现

    该类提供了构建和训练CNN的完整功能，适用于图像
    分类任务。

    Attributes:
        input_shape: 输入图像形状，如(28, 28, 1)
        num_classes: 分类数量
        conv_layers: 卷积层配置列表
        dense_layers: 全连接层配置列表
        model: Keras模型实例

    Methods:
        build(): 构建模型架构
        compile(): 编译模型
        train(): 训练模型
        evaluate(): 评估模型
        predict(): 进行预测

    Example:
        >>> cnn = ConvolutionalNeuralNetwork(
        ...     input_shape=(28, 28, 1),
        ...     num_classes=10
        ... )
        >>> cnn.build()
        >>> cnn.compile()
        >>> history = cnn.train(X_train, y_train, epochs=10)
    """

    def __init__(self, input_shape, num_classes):
        """
        初始化CNN

        Args:
            input_shape: 输入形状，如(height, width, channels)
            num_classes: 输出类别数
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

def preprocess_data():
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

### 2. 魔法数字

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

### 3. 错误处理

```python
def load_dataset(file_path):
    """加载数据集"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"读取文件失败: {e}")

    if data.empty:
        raise ValueError("数据集为空")

    return data
```

### 4. 类型提示

```python
from typing import List, Tuple, Optional, Union

def process_batch(
    batch: np.ndarray,
    labels: np.ndarray,
    augment: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """处理一个批次的数据"""
    pass

def create_layers(
    layer_sizes: List[int],
    activation: str = 'relu'
) -> List[tf.keras.layers.Layer]:
    """创建层列表"""
    pass
```

### 5. 上下文管理器

```python
# 使用with语句管理资源
with open('data.txt', 'r') as f:
    data = f.read()

# 使用TensorFlow的上下文管理器
with tf.device('/GPU:0'):
    model.fit(X_train, y_train)
```

### 6. 列表推导式

```python
# 好的例子
squares = [x**2 for x in range(10)]
even_numbers = [x for x in numbers if x % 2 == 0]

# 避免过于复杂的推导式
# 如果逻辑复杂，使用传统循环
```

### 7. 代码复用

```python
# 避免重复代码
def evaluate_model(model, X, y, name):
    """评估模型并打印结果"""
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"{name} 准确率: {accuracy:.4f}")
    return accuracy

# 使用
train_acc = evaluate_model(model, X_train, y_train, "训练集")
test_acc = evaluate_model(model, X_test, y_test, "测试集")
```

## 🔍 代码审查清单

在提交代码前，检查以下项目：

- [ ] 代码遵循PEP 8规范
- [ ] 所有函数和类都有文档字符串
- [ ] 变量命名清晰且有意义
- [ ] 添加了必要的注释
- [ ] 没有魔法数字
- [ ] 代码可以正常运行
- [ ] Notebook单元格可以顺序执行
- [ ] 图表清晰且有标题和标签
- [ ] 没有调试代码（print语句除外）
- [ ] 导入语句按规范排序

---

遵循这些规范将使你的代码更加专业和易于维护！
