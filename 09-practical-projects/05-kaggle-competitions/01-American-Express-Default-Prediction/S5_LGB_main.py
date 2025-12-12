"""
LightGBM主训练脚本

本脚本使用所有工程特征训练最终的LightGBM模型。
这是整个流程中最关键的步骤之一。

训练策略：
1. 加载所有特征（手动特征 + 序列特征）
2. 使用5折交叉验证训练
3. 生成OOF预测和测试集预测

模型说明：
- 本脚本训练两个模型：
  1. 不包含序列OOF特征的模型
  2. 包含序列OOF特征的模型（性能更好）
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
from model import *

# 配置参数
root = args.root
seed = args.seed

print("=" * 60)
print("Step 5: LightGBM主模型训练")
print("=" * 60)

# 加载所有特征
print("\n加载合并后的特征...")
df = pd.read_feather(f'{root}/all_feature.feather')

# 加载标签并拆分训练/测试集
print("拆分训练集和测试集...")
train_y = pd.read_csv(f'{root}/train_labels.csv')
train = df[:train_y.shape[0]].copy()
train['target'] = train_y['target']
test = df[train_y.shape[0]:].reset_index(drop=True)
del df
gc.collect()

print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

# LightGBM配置
# 相比S3，这里的feature_fraction更小（0.05 vs 0.7），因为特征更多
lgb_config = {
    'lgb_params': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'dart',
        'max_depth': -1,
        'num_leaves': 64,
        'learning_rate': 0.035,
        'bagging_freq': 5,
        'bagging_fraction': 0.75,
        'feature_fraction': 0.05,           # 只使用5%的特征，防止过拟合
        'min_data_in_leaf': 256,
        'max_bin': 63,
        'min_data_in_bin': 256,
        'tree_learner': 'serial',
        'boost_from_average': 'false',
        'lambda_l1': 0.1,
        'lambda_l2': 30,
        'num_threads': 24,
        'verbosity': 1,
    },
    'feature_name': [],  # 稍后指定
    'rounds': 4500,
    'early_stopping_rounds': 100,
    'verbose_eval': 50,
    'folds': 5,
    'seed': seed
}

# 训练模型1：不包含序列OOF特征
print("\n" + "=" * 60)
print("训练模型1: 仅使用手动特征")
print("=" * 60)
lgb_config_1 = lgb_config.copy()
lgb_config_1['feature_name'] = [
    col for col in train.columns
    if col not in [id_name, label_name, 'S_2']
    and 'target' not in col  # 排除序列OOF特征
]
print(f"特征数量: {len(lgb_config_1['feature_name'])}")

Lgb_train_and_predict(
    train, test, lgb_config_1,
    aug=None,
    run_id='LGB_with_manual_feature'
)

# 训练模型2：包含序列OOF特征
print("\n" + "=" * 60)
print("训练模型2: 使用所有特征（包含序列OOF）")
print("=" * 60)
lgb_config_2 = lgb_config.copy()
lgb_config_2['feature_name'] = [
    col for col in train.columns
    if col not in [id_name, label_name, 'S_2']
]
print(f"特征数量: {len(lgb_config_2['feature_name'])}")

Lgb_train_and_predict(
    train, test, lgb_config_2,
    aug=None,
    run_id='LGB_with_manual_feature_and_series_oof'
)

print("\n" + "=" * 60)
print("Step 5 完成：LightGBM训练完毕")
print("=" * 60)
