"""
Stock data processing module.

This module handles:
1. Loading stock data
2. Calculating technical indicators
3. Creating sliding window sequences
4. Data normalization

Technical indicators provide additional information beyond raw price data,
including trend, momentum, and volatility signals that help improve prediction accuracy.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculator for common stock technical indicators.

    Technical indicators augment raw price data with derived signals
    that capture trends, momentum, and volatility patterns.
    """

    @staticmethod
    def calculate_ma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        Calculate Moving Average indicators.

        Moving averages smooth price fluctuations and identify trend direction.
        Short-term MA crossing above long-term MA often signals upward momentum.

        Args:
            df: DataFrame with price data
            periods: List of window sizes for moving averages

        Returns:
            DataFrame with MA columns added
        """
        for period in periods:
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).

        RSI measures the magnitude of recent price changes to evaluate
        overbought or oversold conditions. Range: 0-100.
        - RSI > 70: Potentially overbought
        - RSI < 30: Potentially oversold

        Formula: RSI = 100 - (100 / (1 + RS))
                 RS = Average Gain / Average Loss

        Args:
            df: DataFrame with price data
            period: Calculation window (typically 14 days)

        Returns:
            DataFrame with RSI column added
        """
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD tracks the relationship between two moving averages.
        Crossovers between MACD and signal line indicate potential trend changes.

        Components:
        - MACD Line: EMA(12) - EMA(26)
        - Signal Line: EMA(MACD, 9)
        - Histogram: MACD - Signal

        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD columns added
        """
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()

        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Bollinger Bands measure volatility and provide dynamic support/resistance levels.

        Formula:
        - Middle Band: SMA(20)
        - Upper Band: Middle + (2 * std_dev)
        - Lower Band: Middle - (2 * std_dev)

        Price touching upper band may indicate overbought conditions,
        while touching lower band may indicate oversold conditions.

        Args:
            df: DataFrame with price data
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            DataFrame with Bollinger Band columns added
        """
        df['BB_middle'] = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()

        df['BB_upper'] = df['BB_middle'] + (std * std_dev)
        df['BB_lower'] = df['BB_middle'] - (std * std_dev)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility. Higher ATR indicates higher volatility.

        Formula:
        True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        ATR = MA(True Range, period)

        Args:
            df: DataFrame with OHLC data
            period: Calculation window

        Returns:
            DataFrame with ATR column added
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period).mean()
        return df

    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume (OBV).

        OBV cumulates volume based on price direction:
        - Price up: OBV += Volume
        - Price down: OBV -= Volume

        Rising OBV suggests accumulation (buying pressure),
        while falling OBV suggests distribution (selling pressure).

        Args:
            df: DataFrame with Close and Volume data

        Returns:
            DataFrame with OBV column added
        """
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])

        df['OBV'] = obv
        return df

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all indicator columns added
        """
        logger.info("Calculating technical indicators...")

        df = TechnicalIndicators.calculate_ma(df)
        logger.info("  Moving Averages (MA) calculated")

        df = TechnicalIndicators.calculate_rsi(df)
        logger.info("  Relative Strength Index (RSI) calculated")

        df = TechnicalIndicators.calculate_macd(df)
        logger.info("  MACD calculated")

        df = TechnicalIndicators.calculate_bollinger_bands(df)
        logger.info("  Bollinger Bands calculated")

        df = TechnicalIndicators.calculate_atr(df)
        logger.info("  Average True Range (ATR) calculated")

        df = TechnicalIndicators.calculate_obv(df)
        logger.info("  On-Balance Volume (OBV) calculated")

        return df


class StockDataProcessor:
    """
    Processor for stock time series data.

    Handles the complete data pipeline:
    - Loading raw stock data
    - Computing technical indicators
    - Creating sliding windows for LSTM input
    - Normalizing features and targets
    """

    def __init__(self, data_path: str, target_column: str = 'Close'):
        """
        Initialize data processor.

        Args:
            data_path: Path to CSV file with stock data
            target_column: Column name for prediction target
        """
        self.data_path = data_path
        self.target_column = target_column

        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self.df = None
        self.feature_names = None

    def load_data(self) -> pd.DataFrame:
        """
        Load stock data from CSV file.

        Returns:
            DataFrame with loaded data
        """
        logger.info("Loading stock data...")

        self.df = pd.read_csv(self.data_path)

        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df.set_index('Date', inplace=True)

        self.df.sort_index(inplace=True)

        logger.info(f"  Shape: {self.df.shape}")
        logger.info(f"  Date range: {self.df.index[0]} to {self.df.index[-1]}")
        logger.info(f"  Columns: {self.df.columns.tolist()}")

        return self.df

    def create_features(self) -> pd.DataFrame:
        """
        Create feature set including technical indicators.

        Returns:
            DataFrame with all features
        """
        logger.info("Creating features...")

        self.df = TechnicalIndicators.calculate_all_indicators(self.df)

        logger.info(f"Removing NaN values...")
        logger.info(f"  Before: {len(self.df)}")
        self.df = self.df.dropna()
        logger.info(f"  After: {len(self.df)}")

        base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        indicator_features = [col for col in self.df.columns
                            if col not in base_features and col != self.target_column]

        self.feature_names = base_features + indicator_features

        logger.info(f"Feature counts:")
        logger.info(f"  Base features: {len(base_features)}")
        logger.info(f"  Technical indicators: {len(indicator_features)}")
        logger.info(f"  Total features: {len(self.feature_names)}")

        return self.df[self.feature_names]

    def normalize_data(self, X: np.ndarray, y: np.ndarray,
                      train_split: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize features and targets using MinMaxScaler.

        Important: Scaler is fit only on training data to prevent data leakage.

        Args:
            X: Feature array
            y: Target array
            train_split: Training data proportion

        Returns:
            Normalized X and y arrays
        """
        logger.info("Normalizing data...")

        train_size = int(len(X) * train_split)

        self.feature_scaler.fit(X[:train_size])
        self.target_scaler.fit(y[:train_size].reshape(-1, 1))

        X_normalized = self.feature_scaler.transform(X)
        y_normalized = self.target_scaler.transform(y.reshape(-1, 1)).flatten()

        logger.info(f"  Feature range: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
        logger.info(f"  Target range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")

        return X_normalized, y_normalized

    def create_sequences(self, X: np.ndarray, y: np.ndarray,
                        lookback: int = 60, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM input.

        For multi-task learning, also generates trend labels (up/down).

        Args:
            X: Feature array
            y: Target array
            lookback: Number of historical time steps
            forecast_horizon: Number of steps ahead to predict

        Returns:
            X_seq: Input sequences (batch, lookback, features)
            y_seq: Price targets (batch,)
            y_trend: Trend labels (batch,) - 1 for up, 0 for down
        """
        logger.info("Creating sliding window sequences...")
        logger.info(f"  Lookback window: {lookback} days")
        logger.info(f"  Forecast horizon: {forecast_horizon} days")

        X_seq, y_seq, y_trend = [], [], []

        for i in range(lookback, len(X) - forecast_horizon + 1):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i+forecast_horizon-1])

            current_price = y[i-1]
            future_price = y[i+forecast_horizon-1]
            trend = 1 if future_price > current_price else 0
            y_trend.append(trend)

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        y_trend = np.array(y_trend)

        logger.info(f"  Generated sequences: {len(X_seq)}")
        logger.info(f"  Input shape: {X_seq.shape}")
        logger.info(f"  Price target shape: {y_seq.shape}")
        logger.info(f"  Trend target shape: {y_trend.shape}")
        logger.info(f"  Trend distribution: Up={y_trend.sum()}, Down={len(y_trend)-y_trend.sum()}")

        return X_seq, y_seq, y_trend

    def split_data(self, X: np.ndarray, y_price: np.ndarray, y_trend: np.ndarray,
                   train_split: float = 0.7, val_split: float = 0.15) -> Tuple:
        """
        Split data into train/validation/test sets while preserving time order.

        Args:
            X: Input sequences
            y_price: Price targets
            y_trend: Trend targets
            train_split: Training data proportion
            val_split: Validation data proportion

        Returns:
            Training, validation, and test tuples (X, y_price, y_trend)
        """
        logger.info("Splitting data...")

        n_samples = len(X)
        train_size = int(n_samples * train_split)
        val_size = int(n_samples * val_split)

        X_train = X[:train_size]
        y_price_train = y_price[:train_size]
        y_trend_train = y_trend[:train_size]

        X_val = X[train_size:train_size+val_size]
        y_price_val = y_price[train_size:train_size+val_size]
        y_trend_val = y_trend[train_size:train_size+val_size]

        X_test = X[train_size+val_size:]
        y_price_test = y_price[train_size+val_size:]
        y_trend_test = y_trend[train_size+val_size:]

        logger.info(f"  Training set: {X_train.shape}")
        logger.info(f"  Validation set: {X_val.shape}")
        logger.info(f"  Test set: {X_test.shape}")

        return (X_train, y_price_train, y_trend_train), \
               (X_val, y_price_val, y_trend_val), \
               (X_test, y_price_test, y_trend_test)

    def inverse_transform_price(self, y: np.ndarray) -> np.ndarray:
        """Reverse normalization to get original price scale."""
        return self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    def save_processor(self, filepath: str) -> None:
        """Save scaler and feature configuration."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler,
                'feature_names': self.feature_names
            }, f)
        logger.info(f"Processor saved: {filepath}")

    def load_processor(self, filepath: str) -> None:
        """Load saved scaler and feature configuration."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.feature_scaler = data['feature_scaler']
            self.target_scaler = data['target_scaler']
            self.feature_names = data['feature_names']
        logger.info(f"Processor loaded: {filepath}")


def prepare_stock_data(
    data_path: str,
    lookback: int = 60,
    forecast_horizon: int = 1,
    train_split: float = 0.7,
    val_split: float = 0.15
) -> Tuple:
    """
    Complete data preparation pipeline.

    Args:
        data_path: Path to stock data CSV file
        lookback: Lookback window size
        forecast_horizon: Prediction horizon
        train_split: Training data proportion
        val_split: Validation data proportion

    Returns:
        train_data, val_data, test_data, processor
    """
    print("="*60)
    print("Stock Data Preparation")
    print("="*60)

    processor = StockDataProcessor(data_path)

    processor.load_data()

    X = processor.create_features()
    y = processor.df[processor.target_column].values

    X_normalized, y_normalized = processor.normalize_data(X.values, y, train_split)

    X_seq, y_price, y_trend = processor.create_sequences(
        X_normalized, y_normalized,
        lookback, forecast_horizon
    )

    train_data, val_data, test_data = processor.split_data(
        X_seq, y_price, y_trend,
        train_split, val_split
    )

    logger.info("Data preparation complete!")

    return train_data, val_data, test_data, processor


if __name__ == '__main__':
    """Test data processing module."""
    print("="*60)
    print("Data Processing Module Test")
    print("="*60)

    logger.info("Creating synthetic stock data...")
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)

    price = 100
    prices = [price]
    for _ in range(499):
        change = np.random.randn() * 2
        price = max(price + change, 50)
        prices.append(price)

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 500)
    })

    temp_path = 'temp_stock_data.csv'
    df.to_csv(temp_path, index=False)

    try:
        train_data, val_data, test_data, processor = prepare_stock_data(
            temp_path,
            lookback=30,
            forecast_horizon=1
        )

        logger.info("Data processing test passed!")

    finally:
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

    logger.info("All tests passed!")
