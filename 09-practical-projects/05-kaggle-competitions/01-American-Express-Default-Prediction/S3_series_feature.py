"""
序列特征提取脚本

本脚本基于降噪后的数据，使用LightGBM模型生成序列级别的特征。
这些特征将作为神经网络模型的输入。

主要步骤：
1. 读取降噪后的数据
2. 训练LightGBM模型并生成OOF预测
3. 将OOF预测作为序列特征保存

技术要点：
- 使用GroupKFold确保同一客户的数据不会同时出现在训练集和验证集
- OOF预测可以作为强特征输入到后续模型
"""

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import random
import time
import datetime
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from utils import *

# 配置参数
root = args.root
seed = args.seed

print("=" * 60)
print("Step 3: 序列特征提取")
print("=" * 60)

# 加载数据
print("\n加载数据...")
train = pd.read_feather(f'./input/train.feather')
test = pd.read_feather(f'./input/test.feather')

def one_hot_encoding(df: pd.DataFrame, cols: list, is_drop: bool = True) -> pd.DataFrame:
    """
    One-hot编码类别特征

    Args:
        df: 输入DataFrame
        cols: 需要编码的列名列表
        is_drop: 是否删除原始列

    Returns:
        编码后的DataFrame
    """
    for col in cols:
        print(f'One-hot编码特征: {col}')
        dummies = pd.get_dummies(pd.Series(df[col]), prefix=f'oneHot_{col}')
        df = pd.concat([df, dummies], axis=1)
    if is_drop:
        df.drop(cols, axis=1, inplace=True)
    return df


# 类别特征列表
cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
                "D_126", "D_63", "D_64", "D_66", "D_68"]
eps = 1e-3

# 合并标签
print("合并训练标签...")
train_y = pd.read_csv(f'{root}/train_labels.csv')
train = train.merge(train_y, how='left', on=id_name)

print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

# LightGBM配置
# 使用DART boosting以减少过拟合
lgb_config = {
    'lgb_params': {
        'objective': 'binary',              # 二分类任务
        'metric': 'binary_logloss',         # 评估指标
        'boosting': 'dart',                 # DART: Dropouts meet Multiple Additive Regression Trees
        'max_depth': -1,                    # 不限制树深度
        'num_leaves': 64,                   # 叶子节点数量
        'learning_rate': 0.035,             # 学习率
        'bagging_freq': 5,                  # 每5轮进行一次bagging
        'bagging_fraction': 0.7,            # 采样70%的数据
        'feature_fraction': 0.7,            # 采样70%的特征
        'min_data_in_leaf': 256,            # 叶子节点最小样本数
        'max_bin': 63,                      # 特征分桶数量
        'min_data_in_bin': 256,             # 每个桶的最小样本数
        'tree_learner': 'serial',           # 串行学习
        'boost_from_average': 'false',      # 不从平均值开始
        'lambda_l1': 0.1,                   # L1正则化
        'lambda_l2': 30,                    # L2正则化
        'num_threads': 24,                  # 线程数
        'verbosity': 1,
    },
    'feature_name': [col for col in train.columns if col not in [id_name, label_name, 'S_2']],
    'rounds': 4500,                         # 最大迭代轮数
    'early_stopping_rounds': 100,           # 早停轮数
    'verbose_eval': 50,                     # 每50轮输出一次
    'folds': 5,                             # 5折交叉验证
    'seed': seed
}

print("\n开始训练LightGBM模型（用于生成序列特征）...")
print("注意：使用GroupKFold确保同一客户的数据不会泄露")

# 训练模型并生成OOF预测
Lgb_train_and_predict(
    train, test, lgb_config,
    gkf=True,  # 使用GroupKFold
    aug=None,
    run_id='LGB_with_series_feature'
)

print("\n" + "=" * 60)
print("Step 3 完成：序列特征提取完毕")
print("=" * 60)
