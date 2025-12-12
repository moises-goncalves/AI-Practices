"""
路径配置模块
=============

本模块定义了项目中使用的所有数据和结果目录路径。

目录说明：
    - DATA_KAGGLE_DIR: Kaggle竞赛原始数据目录，包含train/test图像和标注CSV文件
    - DATA_PROCESSED_DIR: 预处理后的数据目录，包含处理后的numpy数组和修正的标注
    - RESULT_DIR: 训练结果输出目录，包含模型权重、日志和验证结果

配置方法：
    1. 下载竞赛数据并解压到指定目录
    2. 修改下方路径为实际的完整路径（不要使用相对路径）
    3. 确保目录存在且有读写权限

示例：
    Linux: '/home/username/data/rsna-2024-lumbar-spine'
    Windows: 'C:/Users/username/data/rsna-2024-lumbar-spine'
"""

import os
from pathlib import Path

# ===== 请在此处设置您的路径 =====
# 方案1：使用环境变量（推荐）
DATA_KAGGLE_DIR = os.environ.get(
    'RSNA_DATA_DIR',
    f'/home/{os.getenv("USER", "user")}/data/kaggle/rsna-2024-lumbar-spine-degenerative-classification'
)
DATA_PROCESSED_DIR = os.environ.get(
    'RSNA_PROCESSED_DIR',
    f'/home/{os.getenv("USER", "user")}/data/kaggle/rsna2024-processed'
)
RESULT_DIR = os.environ.get(
    'RSNA_RESULT_DIR',
    f'/home/{os.getenv("USER", "user")}/results/rsna2024-lumbar-spine'
)

# 方案2：直接设置路径（如果环境变量未设置）
# 请取消注释并修改为您的实际路径
# DATA_KAGGLE_DIR    = '/your/path/to/rsna-2024-lumbar-spine-degenerative-classification'
# DATA_PROCESSED_DIR = '/your/path/to/processed/data'
# RESULT_DIR         = '/your/path/to/results'

# ===== 路径验证和创建 =====
def validate_and_create_dirs():
    """
    验证并创建必要的目录

    该函数会：
    1. 检查DATA_KAGGLE_DIR是否存在（必须手动创建并放置数据）
    2. 自动创建DATA_PROCESSED_DIR和RESULT_DIR（如果不存在）
    3. 打印路径信息供用户确认
    """
    print("\n" + "="*60)
    print("RSNA 2024 Lumbar Spine - 路径配置")
    print("="*60)

    # 检查Kaggle数据目录
    if not os.path.exists(DATA_KAGGLE_DIR):
        print(f"⚠️  警告: Kaggle数据目录不存在: {DATA_KAGGLE_DIR}")
        print("   请确保已下载并解压竞赛数据到该目录")
    else:
        print(f"✓ Kaggle数据目录: {DATA_KAGGLE_DIR}")

    # 创建处理数据目录
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    print(f"✓ 预处理数据目录: {DATA_PROCESSED_DIR}")

    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"✓ 结果输出目录: {RESULT_DIR}")

    print("="*60 + "\n")

# 在模块导入时自动验证路径
if __name__ != '__main__':
    # 只在作为模块导入时验证，避免在直接运行时重复输出
    pass  # 如需自动验证，取消注释下一行
    # validate_and_create_dirs()
