"""
数据预处理脚本 - 降噪处理

本脚本的主要功能：
1. 对类别特征进行编码转换
2. 对数值特征进行精度降噪（乘以100后取整）
3. 转换为feather格式以提高后续读取速度

技术原理：
- 降噪目的：减少数值精度可以：
  1. 降低过拟合风险（去除无意义的小数位）
  2. 减少内存占用
  3. 加快模型训练速度
- Feather格式：Apache Arrow的列式存储格式，比CSV快10-100倍

数据规模：
- 训练数据：约16GB CSV -> 约4GB Feather
- 测试数据：约33GB CSV -> 约8GB Feather
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def denoise(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据降噪处理

    处理步骤：
    1. 类别特征编码：将字符串类型的类别特征转换为整数
    2. 数值特征降噪：乘以100后向下取整，保留两位小数精度

    Args:
        df: 原始数据DataFrame

    Returns:
        处理后的DataFrame

    技术说明：
        - D_63, D_64是类别特征，需要手动编码
        - 其他数值特征通过*100降低精度，减少噪声
        - np.int8可节省内存，取值范围[-128, 127]足够
    """
    # 类别特征编码：D_63
    # 原始值：CR, XZ, XM, CO, CL, XL
    df['D_63'] = df['D_63'].apply(
        lambda t: {'CR': 0, 'XZ': 1, 'XM': 2, 'CO': 3, 'CL': 4, 'XL': 5}[t]
    ).astype(np.int8)

    # 类别特征编码：D_64
    # 原始值：O, -1, R, U, NaN
    df['D_64'] = df['D_64'].apply(
        lambda t: {np.nan: -1, 'O': 0, '-1': 1, 'R': 2, 'U': 3}[t]
    ).astype(np.int8)

    # 数值特征降噪：乘以100后取整
    # 例如：0.123456 -> 12.3456 -> 12（int）
    for col in tqdm(df.columns, desc="降噪处理"):
        if col not in ['customer_ID', 'S_2', 'D_63', 'D_64']:
            df[col] = np.floor(df[col] * 100)

    return df


if __name__ == '__main__':
    print("=" * 60)
    print("Step 1: 数据降噪处理")
    print("=" * 60)

    # 处理训练数据
    print("\n处理训练数据...")
    train = pd.read_csv('./input/train_data.csv')
    print(f"训练数据原始大小: {train.shape}")
    train = denoise(train)
    train.to_feather('./input/train.feather')
    print(f"训练数据保存完成: ./input/train.feather")
    del train

    # 处理测试数据
    print("\n处理测试数据...")
    test = pd.read_csv('./input/test_data.csv')
    print(f"测试数据原始大小: {test.shape}")
    test = denoise(test)
    test.to_feather('./input/test.feather')
    print(f"测试数据保存完成: ./input/test.feather")
    del test

    print("\n" + "=" * 60)
    print("Step 1 完成：数据降噪处理完毕")
    print("=" * 60)
