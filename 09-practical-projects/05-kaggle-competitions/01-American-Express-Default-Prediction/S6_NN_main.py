"""
神经网络训练脚本

本脚本训练基于GRU的深度学习模型，用于处理时间序列数据。

模型特点：
1. 使用双向GRU处理客户的历史交易序列
2. 结合聚合特征提升性能
3. 支持多GPU训练

训练说明：
- 需要GPU支持（推荐）
- 使用--do_train启用训练
- 训练时间约4-6小时（取决于GPU性能）
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

from utils import *
from model import *

# 配置参数
root = args.root
seed = args.seed

print("=" * 60)
print("Step 6: 神经网络模型训练")
print("=" * 60)

# 加载序列数据和特征
print("\n加载数据...")
df = pd.read_feather('./input/nn_series.feather')
y = pd.read_csv('./input/train_labels.csv')
f = pd.read_feather('./input/nn_all_feature.feather')

# 构建序列索引
print("构建序列索引...")
df['idx'] = df.index
series_idx = df.groupby('customer_ID', sort=False).idx.agg(['min', 'max'])
series_idx['feature_idx'] = np.arange(len(series_idx))
df = df.drop(['idx'], axis=1)

print(f"序列数据形状: {df.shape}")
print(f"特征数据形状: {f.shape}")
print(f"标签数据形状: {y.shape}")
# 神经网络配置
nn_config = {
    'id_name': id_name,
    'feature_name': [],
    'label_name': label_name,
    'obj_max': 1,                    # 最大化目标（Amex指标越大越好）
    'epochs': 10,                    # 训练轮数
    'smoothing': 0.001,              # 标签平滑（可选）
    'clipnorm': 1,                   # 梯度裁剪范数
    'patience': 100,                 # 早停耐心值
    'lr': 3e-4,                      # 学习率（由scheduler控制）
    'batch_size': 256,               # 批次大小
    'folds': 5,                      # 交叉验证折数
    'seed': seed,
    'remark': args.remark
}

# 训练模型1：仅使用序列特征
print("\n" + "=" * 60)
print("训练神经网络模型1: 仅使用序列特征")
print("=" * 60)
NN_train_and_predict(
    train=[df, f, y, series_idx.values[:y.shape[0]]],
    test=[df, f, series_idx.values[y.shape[0]:]],
    model_class=Amodel,
    config=nn_config,
    use_series_oof=False,  # 不使用聚合特征分支
    run_id='NN_with_series_feature'
)

# 训练模型2：使用序列特征 + 聚合特征
print("\n" + "=" * 60)
print("训练神经网络模型2: 使用序列特征 + 聚合特征")
print("=" * 60)
NN_train_and_predict(
    train=[df, f, y, series_idx.values[:y.shape[0]]],
    test=[df, f, series_idx.values[y.shape[0]:]],
    model_class=Amodel,
    config=nn_config,
    use_series_oof=True,  # 使用聚合特征分支
    run_id='NN_with_series_and_all_feature'
)

print("\n" + "=" * 60)
print("Step 6 完成：神经网络训练完毕")
print("=" * 60)
