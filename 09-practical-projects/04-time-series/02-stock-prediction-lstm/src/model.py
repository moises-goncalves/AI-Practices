"""
LSTM-based stock prediction models.

This module implements:
1. Basic LSTM model
2. LSTM with Attention mechanism
3. Multi-task learning model (price + trend prediction)
4. Attention weight visualization

The attention mechanism helps the model focus on important time steps,
improving prediction accuracy and model interpretability.
"""

import logging
from typing import Tuple, Dict, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttentionLayer(layers.Layer):
    """
    Self-attention layer for sequence modeling.

    Computes attention weights for each time step, allowing the model
    to focus on important historical points (e.g., earnings reports).

    Process:
    1. Calculate attention scores using learned weights
    2. Apply softmax to get normalized weights
    3. Compute weighted sum of inputs
    """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build layer parameters.

        Args:
            input_shape: (batch, time_steps, features)
        """
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch, time_steps, features)

        Returns:
            context: Context vector (batch, features)
            attention_weights: Attention weights (batch, time_steps)
        """
        # Step 1: Compute attention scores
        # Score indicates the importance of each time step
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)

        # Step 2: Normalize with softmax
        # Converts scores to probability distribution
        attention_weights = tf.nn.softmax(e, axis=1)

        # Step 3: Weighted sum
        # Focus on important time steps with higher weights
        context = x * attention_weights
        context = tf.reduce_sum(context, axis=1)

        return context, tf.squeeze(attention_weights, -1)


class StockLSTMPredictor:
    """
    LSTM-based stock price predictor.

    Supports three model types:
    1. lstm_basic: Standard LSTM architecture
    2. lstm_attention: LSTM with attention mechanism
    3. multitask: Joint prediction of price (regression) and trend (classification)

    The attention mechanism improves accuracy by automatically learning
    which historical time points are most relevant for prediction.
    """

    def __init__(self, input_shape: Tuple[int, int],
                 model_type: str = 'lstm_attention', **kwargs):
        """
        Initialize predictor.

        Args:
            input_shape: (time_steps, features)
            model_type: Model architecture type
            **kwargs: Additional model parameters
        """
        self.input_shape = input_shape
        self.model_type = model_type

        self.config = self._get_model_config(model_type)
        self.config.update(kwargs)

        self.model = self._build_model()

    def _get_model_config(self, model_type: str) -> Dict:
        """Get default model configuration."""
        configs = {
            'lstm_basic': {
                'lstm_units': [128, 64],
                'dropout': 0.2,
                'dense_units': [32],
            },

            'lstm_attention': {
                'lstm_units': [128, 64],
                'dropout': 0.3,
                'use_attention': True,
                'dense_units': [64, 32],
            },

            'multitask': {
                'lstm_units': [128, 64],
                'dropout': 0.3,
                'use_attention': True,
                'multitask': True,
                'dense_units': [64, 32],
            }
        }

        return configs.get(model_type, configs['lstm_attention'])

    def _build_model(self) -> keras.Model:
        """Build model architecture."""
        inputs = layers.Input(shape=self.input_shape, name='input')

        x = inputs

        # LSTM layers
        lstm_units = self.config['lstm_units']

        for i, units in enumerate(lstm_units):
            return_sequences = (i < len(lstm_units) - 1) or self.config.get('use_attention', False)

            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config['dropout'],
                name=f'lstm_{i+1}'
            )(x)

            if i < len(lstm_units) - 1:
                x = layers.Dropout(self.config['dropout'], name=f'dropout_{i+1}')(x)

        # Attention layer (optional)
        if self.config.get('use_attention', False):
            context, attention_weights = AttentionLayer(name='attention')(x)
            x = context

            # Store attention model for visualization
            self.attention_model = keras.Model(inputs=inputs, outputs=attention_weights)

        # Dense layers
        dense_units = self.config['dense_units']

        for i, units in enumerate(dense_units):
            x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
            x = layers.Dropout(self.config['dropout'], name=f'dense_dropout_{i+1}')(x)

        # Output layer(s)
        if self.config.get('multitask', False):
            # Multi-task: Price (regression) + Trend (classification)
            price_output = layers.Dense(1, activation='linear', name='price')(x)
            trend_output = layers.Dense(1, activation='sigmoid', name='trend')(x)

            model = keras.Model(
                inputs=inputs,
                outputs=[price_output, trend_output],
                name='stock_lstm_multitask'
            )

        else:
            # Single-task: Price prediction only
            output = layers.Dense(1, activation='linear', name='price')(x)

            model = keras.Model(
                inputs=inputs,
                outputs=output,
                name=f'stock_lstm_{self.model_type}'
            )

        return model

    def compile_model(self, learning_rate: float = 0.001) -> None:
        """Compile model with optimizer and loss functions."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        if self.config.get('multitask', False):
            # Multi-task learning
            self.model.compile(
                optimizer=optimizer,
                loss={
                    'price': 'mse',
                    'trend': 'binary_crossentropy'
                },
                loss_weights={
                    'price': 1.0,
                    'trend': 0.5
                },
                metrics={
                    'price': ['mae', keras.metrics.RootMeanSquaredError(name='rmse')],
                    'trend': ['accuracy']
                }
            )
        else:
            # Single-task learning
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
            )

    def train(self, X_train, y_train, X_val, y_val,
              epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001,
              callbacks = None, verbose: int = 1) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels (single or multi-task)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            callbacks: Training callbacks
            verbose: Verbosity level

        Returns:
            Training history
        """
        self.compile_model(learning_rate=learning_rate)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            For single-task: price predictions
            For multi-task: (price predictions, trend predictions)
        """
        predictions = self.model.predict(X)

        if self.config.get('multitask', False):
            price_pred, trend_pred = predictions
            return price_pred.flatten(), (trend_pred > 0.5).astype(int).flatten()
        else:
            return predictions.flatten()

    def predict_with_attention(self, X: np.ndarray) -> Tuple:
        """
        Predict and return attention weights for visualization.

        Args:
            X: Input features

        Returns:
            predictions: Model predictions
            attention_weights: Attention weights for each sample

        Raises:
            ValueError: If model doesn't use attention
        """
        if not self.config.get('use_attention', False):
            raise ValueError("Model does not use attention mechanism")

        predictions = self.model.predict(X)
        attention_weights = self.attention_model.predict(X)

        return predictions, attention_weights

    def evaluate(self, X: np.ndarray, y) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Input features
            y: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        results = self.model.evaluate(X, y, verbose=0)

        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = value

        return metrics

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate detailed evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics (MAE, MSE, RMSE, MAPE, direction accuracy)
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        }

        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = mape

        # Direction accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_accuracy = accuracy_score(true_direction, pred_direction)
            metrics['direction_accuracy'] = direction_accuracy

        return metrics

    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        self.model.save(filepath)
        logger.info(f"Model saved: {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logger.info(f"Model loaded: {filepath}")

    def summary(self) -> None:
        """Print model architecture summary."""
        self.model.summary()


if __name__ == '__main__':
    """Test model architectures."""
    print("="*60)
    print("LSTM Stock Prediction Model Test")
    print("="*60)

    # Test parameters
    time_steps = 60
    n_features = 20
    batch_size = 32

    input_shape = (time_steps, n_features)

    # Generate random test data
    X_train = np.random.randn(1000, time_steps, n_features)
    y_price_train = np.random.randn(1000)
    y_trend_train = np.random.randint(0, 2, 1000)

    X_val = np.random.randn(200, time_steps, n_features)
    y_price_val = np.random.randn(200)
    y_trend_val = np.random.randint(0, 2, 200)

    # Test all three model types
    for model_type in ['lstm_basic', 'lstm_attention', 'multitask']:
        print(f"\n{'='*60}")
        print(f"Testing {model_type} model")
        print(f"{'='*60}")

        # Create model
        predictor = StockLSTMPredictor(
            input_shape=input_shape,
            model_type=model_type
        )

        # Print architecture
        print(f"\nModel architecture:")
        predictor.summary()

        # Prepare training data
        if model_type == 'multitask':
            y_train = {'price': y_price_train, 'trend': y_trend_train}
            y_val = {'price': y_price_val, 'trend': y_trend_val}
        else:
            y_train = y_price_train
            y_val = y_price_val

        # Train
        print(f"\nTraining model...")
        history = predictor.train(
            X_train, y_train,
            X_val, y_val,
            epochs=2,
            batch_size=batch_size,
            verbose=0
        )

        # Evaluate
        metrics = predictor.evaluate(X_val, y_val)
        print(f"\nValidation performance:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

        # Predict
        if model_type == 'multitask':
            price_pred, trend_pred = predictor.predict(X_val[:5])
            print(f"\nPrediction examples:")
            print(f"  Prices: {price_pred}")
            print(f"  Trends: {trend_pred}")
        else:
            predictions = predictor.predict(X_val[:5])
            print(f"\nPrediction examples: {predictions}")

        # Test attention weights
        if model_type in ['lstm_attention', 'multitask']:
            print(f"\nTesting attention weights...")
            _, attention_weights = predictor.predict_with_attention(X_val[:1])
            print(f"  Attention shape: {attention_weights.shape}")
            print(f"  Attention example: {attention_weights[0][:10]}")

    logger.info("All tests passed!")
