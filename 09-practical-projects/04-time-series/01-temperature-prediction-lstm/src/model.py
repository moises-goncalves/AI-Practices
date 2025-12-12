"""
LSTM温度预测模型

实现三种LSTM架构：
1. 简单LSTM（单层）- 适用于快速实验
2. 堆叠LSTM（多层）- 适用于复杂时序模式
3. GRU模型 - 参数更少，训练更快

每个模型都有详细的架构说明和参数选择依据。
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TemperatureLSTMPredictor:
    """
    LSTM温度预测器

    基于LSTM的时间序列预测模型，用于预测未来时段的温度。

    特点：
        - 支持多变量输入（温度、湿度、气压等）
        - 捕获长期时间依赖关系
        - 提供三种架构选择（simple/stacked/gru）

    架构说明：
        - simple: 单层LSTM，适合简单任务
        - stacked: 多层LSTM，学习多层次时序特征
        - gru: GRU变体，参数更少，训练更快
    """

    def __init__(self,
                 input_shape,
                 forecast_horizon=24,
                 model_type='stacked',
                 **kwargs):
        """
        初始化预测器

        Args:
            input_shape: 输入形状 (lookback, num_features)
            forecast_horizon: 预测范围（小时）
            model_type: 模型类型 ('simple', 'stacked', 'gru')
            **kwargs: 其他参数
        """
        self.input_shape = input_shape
        self.forecast_horizon = forecast_horizon
        self.model_type = model_type

        # 根据模型类型设置参数
        self.config = self._get_model_config(model_type)
        self.config.update(kwargs)

        # 创建模型
        self.model = self._build_model()

    def _get_model_config(self, model_type):
        """
        获取模型配置

        Args:
            model_type: 模型类型

        Returns:
            配置字典
        """
        configs = {
            'simple': {
                # 简单LSTM配置（单层）
                # 适用场景：快速实验、简单时序模式、资源有限
                'lstm_units': [64],
                'dropout': 0.2,
                'dense_units': [32],
            },

            'stacked': {
                # 堆叠LSTM配置（多层）
                # 适用场景：复杂时序模式、多层次特征
                #
                # 层次递减设计：
                #   第1层(128): 学习低级时间特征（小时级波动）
                #   第2层(64):  学习中级时间特征（日级变化）
                #   第3层(32):  学习高级时间特征（周级趋势）
                #
                # 递减原因：特征逐层抽象，高层需要的容量更小
                'lstm_units': [128, 64, 32],
                'dropout': 0.3,
                'dense_units': [64, 32],
            },

            'gru': {
                # GRU配置（对比LSTM）
                #
                # GRU vs LSTM：
                #   GRU: 2个门（更新门、重置门），参数少30%
                #   LSTM: 3个门（遗忘门、输入门、输出门）
                #
                # 选择建议：
                #   数据量小 → GRU（不易过拟合）
                #   数据量大 → LSTM（表达能力强）
                #   快速训练 → GRU
                'gru_units': [128, 64],
                'dropout': 0.3,
                'dense_units': [32],
            }
        }

        return configs.get(model_type, configs['stacked'])

    def _build_model(self):
        """
        构建模型

        Returns:
            Keras模型
        """
        # 输入层，形状(batch, lookback, num_features)
        # 例如：(32, 168, 5) = 32个样本，168小时，5个特征
        inputs = layers.Input(shape=self.input_shape, name='input')

        x = inputs

        # LSTM/GRU层
        if self.model_type == 'gru':
            # GRU模型
            gru_units = self.config['gru_units']

            for i, units in enumerate(gru_units):
                # return_sequences: True表示返回所有时间步（用于堆叠）
                #                   False表示只返回最后一个时间步
                return_sequences = (i < len(gru_units) - 1)

                x = layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config['dropout'],
                    recurrent_dropout=0.0,
                    name=f'gru_{i+1}'
                )(x)

        else:
            # LSTM模型（simple或stacked）
            lstm_units = self.config['lstm_units']

            for i, units in enumerate(lstm_units):
                # return_sequences设置：
                #   最后一层: False（只返回最后的输出）
                #   其他层: True（返回序列给下一层）
                return_sequences = (i < len(lstm_units) - 1)

                x = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config['dropout'],
                    recurrent_dropout=0.0,
                    name=f'lstm_{i+1}'
                )(x)

                # 中间层添加额外的Dropout
                if i < len(lstm_units) - 1:
                    x = layers.Dropout(self.config['dropout'], name=f'dropout_{i+1}')(x)

        # 全连接层
        # 作用：整合LSTM/GRU输出并映射到预测范围
        dense_units = self.config['dense_units']

        for i, units in enumerate(dense_units):
            x = layers.Dense(
                units,
                activation='relu',
                name=f'dense_{i+1}'
            )(x)

            x = layers.Dropout(self.config['dropout'], name=f'dense_dropout_{i+1}')(x)

        # 输出层（线性激活，适用于回归任务）
        # 温度可以是任意实数，因此不使用激活函数
        outputs = layers.Dense(
            self.forecast_horizon,
            activation='linear',
            name='output'
        )(x)

        # 创建模型
        model = keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=f'temperature_lstm_{self.model_type}'
        )

        return model

    def compile_model(self, learning_rate=0.001, loss='mse'):
        """
        编译模型

        优化器选择：Adam
            - 自适应学习率
            - 对超参数不敏感
            - 时序预测的常用选择

        损失函数选择：
            - MSE (Mean Squared Error): 对大误差惩罚更重
            - MAE (Mean Absolute Error): 对异常值更鲁棒

        Args:
            learning_rate: 学习率
            loss: 损失函数
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # 评估指标
        metrics = [
            'mae',
            'mse',
            keras.metrics.RootMeanSquaredError(name='rmse')
        ]

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train(self, X_train, y_train, X_val, y_val,
              epochs=50, batch_size=32, learning_rate=0.001,
              callbacks=None, verbose=1):
        """
        训练模型

        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据
            y_val: 验证标签
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            callbacks: 回调函数列表
            verbose: 详细程度

        Returns:
            训练历史
        """
        # 编译模型
        self.compile_model(learning_rate=learning_rate)

        # 训练
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X):
        """
        预测

        Args:
            X: 输入数据

        Returns:
            预测结果
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        评估模型

        Args:
            X: 测试数据
            y: 测试标签

        Returns:
            评估指标字典
        """
        results = self.model.evaluate(X, y, verbose=0)

        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value

        return metrics

    def calculate_metrics(self, y_true, y_pred):
        """
        计算详细评估指标

        Args:
            y_true: 真实值
            y_pred: 预测值

        Returns:
            指标字典，包含MAE、MSE、RMSE、MAPE
        """
        # 展平数组（如果是多步预测）
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        metrics = {
            'mae': mean_absolute_error(y_true_flat, y_pred_flat),
            'mse': mean_squared_error(y_true_flat, y_pred_flat),
            'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        }

        # 计算MAPE（平均绝对百分比误差），避免除以0
        mask = y_true_flat != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
            metrics['mape'] = mape

        return metrics

    def save_model(self, filepath):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        self.model.save(filepath)
        print(f"模型已保存: {filepath}")

    def load_model(self, filepath):
        """
        加载模型

        Args:
            filepath: 模型路径
        """
        self.model = keras.models.load_model(filepath)
        print(f"模型已加载: {filepath}")

    def summary(self):
        """打印模型摘要"""
        self.model.summary()


if __name__ == '__main__':
    """
    测试模型
    """
    print("="*60)
    print("LSTM温度预测模型测试")
    print("="*60)

    # 测试参数
    lookback = 168  # 7天
    num_features = 5
    forecast_horizon = 24  # 预测24小时
    batch_size = 32

    # 输入形状
    input_shape = (lookback, num_features)

    # 创建随机数据
    X_train = np.random.randn(1000, lookback, num_features)
    y_train = np.random.randn(1000, forecast_horizon)
    X_val = np.random.randn(200, lookback, num_features)
    y_val = np.random.randn(200, forecast_horizon)

    # 测试三种模型
    for model_type in ['simple', 'stacked', 'gru']:
        print(f"\n{'='*60}")
        print(f"测试 {model_type} 模型")
        print(f"{'='*60}")

        # 创建模型
        predictor = TemperatureLSTMPredictor(
            input_shape=input_shape,
            forecast_horizon=forecast_horizon,
            model_type=model_type
        )

        # 打印摘要
        print(f"\n模型结构:")
        predictor.summary()

        # 训练
        print(f"\n训练模型...")
        history = predictor.train(
            X_train, y_train,
            X_val, y_val,
            epochs=2,
            batch_size=batch_size,
            verbose=0
        )

        # 评估
        metrics = predictor.evaluate(X_val, y_val)
        print(f"\n验证集性能:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # 预测
        predictions = predictor.predict(X_val[:5])
        print(f"\n预测形状: {predictions.shape}")
        print(f"预测示例: {predictions[0][:5]}")

    print("\n所有测试通过！")
