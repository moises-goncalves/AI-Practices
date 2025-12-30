"""
数据处理模块

提供股票数据下载和预处理功能。
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def download_stock_data(ticker, start_date, end_date, save_path=None):
    """
    下载股票数据
    
    Args:
        ticker: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        save_path: 保存路径（可选）
    
    Returns:
        DataFrame
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        
        if save_path:
            df.to_csv(save_path, index=False)
        
        return df
    except ImportError:
        print("yfinance not installed. Using sample data.")
        return generate_sample_data(start_date, end_date)


def generate_sample_data(start_date, end_date, initial_price=100):
    """
    生成模拟股票数据（用于测试）
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        initial_price: 初始价格
    
    Returns:
        DataFrame
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days = (end - start).days
    
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)
    prices = initial_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, days)),
        'low': prices * (1 - np.random.uniform(0, 0.02, days)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days)
    })
    
    return df


def add_technical_indicators(df):
    """
    添加技术指标
    
    Args:
        df: 股票数据DataFrame
    
    Returns:
        添加指标后的DataFrame
    """
    df = df.copy()
    
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    df = df.dropna().reset_index(drop=True)
    
    for col in ['ma5', 'ma20', 'rsi', 'macd', 'macd_signal', 'volatility']:
        if col in df.columns:
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    
    return df


def split_data(df, train_ratio=0.8):
    """
    划分训练集和测试集
    
    Args:
        df: 数据DataFrame
        train_ratio: 训练集比例
    
    Returns:
        train_df, test_df
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df
