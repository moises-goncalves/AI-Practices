#!/usr/bin/env python
"""
Comprehensive test suite for stock prediction project.

Tests all modules with minimal parameters to ensure functionality.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import prepare_stock_data, StockDataProcessor
from src.model import StockLSTMPredictor


def create_test_data(filepath='data/test_stock_data.csv', n_samples=200):
    """Create synthetic stock data for testing."""
    print("\n" + "="*60)
    print("Creating test data...")
    print("="*60)

    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')
    np.random.seed(42)

    price = 100
    prices = [price]
    for _ in range(n_samples-1):
        change = np.random.randn() * 2
        price = max(price + change, 50)
        prices.append(price)

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, n_samples)
    })

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    print(f"Test data created: {filepath}")
    print(f"  Samples: {len(df)}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

    return filepath


def test_data_processing(data_path):
    """Test data processing module."""
    print("\n" + "="*60)
    print("Test 1: Data Processing")
    print("="*60)

    try:
        train_data, val_data, test_data, processor = prepare_stock_data(
            data_path=str(data_path),
            lookback=20,
            forecast_horizon=1,
            train_split=0.7,
            val_split=0.15
        )

        X_train, y_price_train, y_trend_train = train_data
        X_val, y_price_val, y_trend_val = val_data
        X_test, y_price_test, y_trend_test = test_data

        print(f"\n[PASS] Data processing successful")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")

        return True, (train_data, val_data, test_data, processor)

    except Exception as e:
        print(f"\n[FAIL] Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_creation(input_shape):
    """Test model creation for all architectures."""
    print("\n" + "="*60)
    print("Test 2: Model Creation")
    print("="*60)

    results = []

    for model_type in ['lstm_basic', 'lstm_attention', 'multitask']:
        try:
            predictor = StockLSTMPredictor(
                input_shape=input_shape,
                model_type=model_type
            )

            param_count = predictor.model.count_params()
            print(f"\n[PASS] {model_type} model created")
            print(f"  Parameters: {param_count:,}")

            results.append(True)

        except Exception as e:
            print(f"\n[FAIL] {model_type} model creation failed: {e}")
            results.append(False)

    return all(results)


def test_model_training(train_data, val_data):
    """Test model training with minimal epochs."""
    print("\n" + "="*60)
    print("Test 3: Model Training (Quick)")
    print("="*60)

    X_train, y_price_train, y_trend_train = train_data
    X_val, y_price_val, y_trend_val = val_data

    input_shape = (X_train.shape[1], X_train.shape[2])

    results = []

    for model_type in ['lstm_basic', 'lstm_attention', 'multitask']:
        try:
            print(f"\nTraining {model_type}...")

            predictor = StockLSTMPredictor(
                input_shape=input_shape,
                model_type=model_type
            )

            # Prepare training data
            if model_type == 'multitask':
                y_train = {'price': y_price_train, 'trend': y_trend_train}
                y_val = {'price': y_price_val, 'trend': y_trend_val}
            else:
                y_train = y_price_train
                y_val = y_price_val

            # Prepare subset for quick testing
            X_train_sub = X_train[:50]
            X_val_sub = X_val[:20] if len(X_val) >= 20 else X_val

            if model_type == 'multitask':
                y_train_sub = {'price': y_price_train[:50], 'trend': y_trend_train[:50]}
                y_val_sub = {'price': y_price_val[:len(X_val_sub)], 'trend': y_trend_val[:len(X_val_sub)]}
            else:
                y_train_sub = y_price_train[:50]
                y_val_sub = y_price_val[:len(X_val_sub)]

            # Train for just 2 epochs
            history = predictor.train(
                X_train_sub, y_train_sub,
                X_val_sub, y_val_sub,
                epochs=2,
                batch_size=16,
                verbose=0
            )

            # Evaluate
            metrics = predictor.evaluate(X_val_sub, y_val_sub)

            print(f"[PASS] {model_type} training completed")
            print(f"  Final loss: {metrics.get('loss', 'N/A'):.4f}")

            results.append(True)

        except Exception as e:
            print(f"[FAIL] {model_type} training failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    return all(results)


def test_prediction(train_data, val_data):
    """Test prediction functionality."""
    print("\n" + "="*60)
    print("Test 4: Model Prediction")
    print("="*60)

    X_train, y_price_train, y_trend_train = train_data
    X_val, y_price_val, y_trend_val = val_data

    input_shape = (X_train.shape[1], X_train.shape[2])

    results = []

    for model_type in ['lstm_attention', 'multitask']:
        try:
            predictor = StockLSTMPredictor(
                input_shape=input_shape,
                model_type=model_type
            )

            # Make predictions
            if model_type == 'multitask':
                price_pred, trend_pred = predictor.predict(X_val[:5])
                print(f"\n[PASS] {model_type} prediction successful")
                print(f"  Price predictions: {price_pred[:3]}")
                print(f"  Trend predictions: {trend_pred[:3]}")
            else:
                predictions = predictor.predict(X_val[:5])
                print(f"\n[PASS] {model_type} prediction successful")
                print(f"  Predictions: {predictions[:3]}")

            # Test attention weights
            if model_type in ['lstm_attention', 'multitask']:
                _, attention_weights = predictor.predict_with_attention(X_val[:2])
                print(f"  Attention weights shape: {attention_weights.shape}")

            results.append(True)

        except Exception as e:
            print(f"\n[FAIL] {model_type} prediction failed: {e}")
            results.append(False)

    return all(results)


def main():
    """Run all tests."""
    print("="*60)
    print("Stock Prediction Project - Comprehensive Test Suite")
    print("="*60)

    test_results = []

    # Test 0: Create test data
    data_path = create_test_data()
    test_results.append(True)

    # Test 1: Data processing
    success, data = test_data_processing(data_path)
    test_results.append(success)

    if not success:
        print("\n[ERROR] Cannot proceed without successful data processing")
        return 1

    train_data, val_data, test_data, processor = data

    # Test 2: Model creation
    input_shape = (train_data[0].shape[1], train_data[0].shape[2])
    success = test_model_creation(input_shape)
    test_results.append(success)

    # Test 3: Model training
    success = test_model_training(train_data, val_data)
    test_results.append(success)

    # Test 4: Prediction
    success = test_prediction(train_data, val_data)
    test_results.append(success)

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    test_names = [
        "Test Data Creation",
        "Data Processing",
        "Model Creation",
        "Model Training",
        "Model Prediction"
    ]

    for name, result in zip(test_names, test_results):
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    total_passed = sum(test_results)
    total_tests = len(test_results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    # Clean up
    if data_path.exists():
        os.remove(data_path)
        print(f"\nCleaned up test data: {data_path}")

    if all(test_results):
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        return 1


if __name__ == '__main__':
    exit(main())
