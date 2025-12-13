# 02 - 神经网络与深度学习

> 🧠 **模块目标**: 掌握深度学习核心技术，从框架使用到自定义模型实现

## 模块概览

本模块深入介绍神经网络和深度学习的核心概念，从 Keras 入门到自定义训练循环。

| 信息 | 详情 |
|:-----|:-----|
| **难度** | ⭐⭐ 中级 |
| **预计时长** | 2-3 周 |
| **Notebooks** | 15+ |
| **前置要求** | 01-Foundations |

## 子模块

### 01 - Keras Introduction | Keras 入门

快速上手 Keras 高级 API。

**核心内容**:
- Sequential API
- 层、激活函数、优化器
- 模型编译与训练
- 回调函数基础

**关键技术**: `Sequential`, `Dense`, `Compile`, `Fit`

---

### 02 - Training Deep Networks | 深度网络训练

掌握深度网络训练的关键技巧。

**核心内容**:
- 权重初始化策略
- 批归一化 (Batch Normalization)
- Dropout 正则化
- 梯度消失/爆炸问题
- 学习率调度

**关键技术**: `BatchNorm`, `Dropout`, `Learning Rate Schedule`

---

### 03 - Custom Models & Training | 自定义模型

从零构建自定义模型和训练循环。

**核心内容**:
- 自定义层 (Custom Layer)
- 自定义损失函数
- 自定义评估指标
- tf.GradientTape 训练循环

**关键技术**: `tf.keras.Model`, `GradientTape`, `Custom Layer`

---

### 04 - Data Loading & Preprocessing | 数据管道

构建高效的数据加载管道。

**核心内容**:
- tf.data API
- TFRecord 格式
- 数据增强
- 混合精度训练

**关键技术**: `tf.data`, `TFRecord`, `Albumentations`, `Mixed Precision`

## 技术栈

```
tensorflow >= 2.13.0
keras >= 2.13.0
albumentations >= 1.3.0
tensorboard >= 2.13.0
```

## 核心代码示例

### Sequential API

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 自定义训练循环

```python
@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

## 下一步

完成本模块后，根据你的兴趣选择：

- 👁️ [03 - Computer Vision](/modules/03-computer-vision) - 计算机视觉方向
- 📝 [04 - Sequence Models](/modules/04-sequence-models) - NLP/序列建模方向
