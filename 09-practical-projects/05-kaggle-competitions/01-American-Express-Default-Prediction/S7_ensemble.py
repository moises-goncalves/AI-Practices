"""
模型集成脚本

本脚本将多个模型的预测结果进行加权融合，生成最终提交文件。

集成策略：
- 使用加权平均融合4个模型的预测
- 权重根据各模型的验证集性能确定
- 模型多样性：LightGBM（2个）+ 神经网络（2个）

模型列表：
1. LGB_manual (30%): 仅手动特征的LightGBM
2. LGB_manual_series_oof (35%): 包含序列OOF的LightGBM
3. NN_series (15%): 仅序列特征的神经网络
4. NN_series_all (20%): 序列+聚合特征的神经网络
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

print("=" * 60)
print("Step 7: 模型集成")
print("=" * 60)

# 加载各模型的预测结果
print("\n加载模型预测...")
p0 = pd.read_csv('./output/LGB_with_manual_feature/submission.csv.zip')
print(f"模型1加载完成: {p0.shape}")

p1 = pd.read_csv('./output/LGB_with_manual_feature_and_series_oof/submission.csv.zip')
print(f"模型2加载完成: {p1.shape}")

p2 = pd.read_csv('./output/NN_with_series_feature/submission.csv.zip')
print(f"模型3加载完成: {p2.shape}")

p3 = pd.read_csv('./output/NN_with_series_and_all_feature/submission.csv.zip')
print(f"模型4加载完成: {p3.shape}")

# 加权融合
# 权重说明：
# - p0 (30%): LGB基础模型
# - p1 (35%): LGB最强模型（包含序列OOF）
# - p2 (15%): NN序列模型
# - p3 (20%): NN完整模型
print("\n进行加权融合...")
print("权重分配: [0.30, 0.35, 0.15, 0.20]")
p0['prediction'] = (
    p0['prediction'] * 0.30 +
    p1['prediction'] * 0.35 +
    p2['prediction'] * 0.15 +
    p3['prediction'] * 0.20
)

# 保存最终结果
output_path = './output/final_submission.csv.zip'
p0.to_csv(output_path, index=False, compression='zip')
print(f"\n最终提交文件已保存: {output_path}")

# 输出统计信息
print(f"\n预测统计:")
print(f"  - 样本数量: {len(p0)}")
print(f"  - 预测均值: {p0['prediction'].mean():.6f}")
print(f"  - 预测标准差: {p0['prediction'].std():.6f}")
print(f"  - 预测最小值: {p0['prediction'].min():.6f}")
print(f"  - 预测最大值: {p0['prediction'].max():.6f}")

print("\n" + "=" * 60)
print("Step 7 完成：模型集成完毕")
print("=" * 60)
print("\n全部流程执行完毕！")
print(f"最终提交文件位置: {output_path}")
