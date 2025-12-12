# American Express Default Prediction

信用违约预测竞赛解决方案 - 使用LightGBM和深度学习的混合模型方法

## 项目概述

本项目实现了一个完整的信用违约预测系统，结合了传统机器学习（LightGBM）和深度学习（GRU）的优势。主要特点：

- **数据规模**：处理约458K训练样本，每个样本包含13个月的历史交易数据
- **特征维度**：190个原始特征，经过工程后生成6000+衍生特征
- **模型架构**：LightGBM + GRU混合模型集成
- **评估指标**：自定义Amex指标（Gini系数 + Top-4%捕获率）

## 环境要求

### 系统要求
- Python 3.7+
- 内存：至少32GB RAM（推荐64GB）
- 存储：至少100GB可用空间
- GPU：可选，用于神经网络训练（需CUDA支持）

### 依赖安装

```bash
# 创建虚拟环境（推荐）
conda create -n amex python=3.8
conda activate amex

# 安装依赖
pip install -r requirements.txt
```

## 项目结构

```
.
├── input/                  # 数据目录（需自行准备）
│   ├── train_data.csv
│   ├── train_labels.csv
│   └── test_data.csv
├── output/                 # 输出目录（自动创建）
├── S1_denoise.py          # 步骤1：数据降噪
├── S2_manual_feature.py   # 步骤2：手动特征工程
├── S3_series_feature.py   # 步骤3：序列特征提取
├── S4_feature_combined.py # 步骤4：特征合并
├── S5_LGB_main.py         # 步骤5：LightGBM训练
├── S6_NN_main.py          # 步骤6：神经网络训练
├── S7_ensemble.py         # 步骤7：模型集成
├── model.py               # 神经网络模型定义
├── scheduler.py           # 学习率调度器
├── utils.py               # 工具函数
├── run.sh                 # 一键运行脚本
└── requirements.txt       # Python依赖
```

## 快速开始

### 1. 数据准备

将数据文件放置到`input/`目录：
```bash
mkdir -p input
# 将以下文件复制到input目录：
# - train_data.csv
# - train_labels.csv
# - test_data.csv
```

### 2. 运行训练

#### 方式1：一键运行（推荐）
```bash
bash run.sh
```

#### 方式2：分步运行
```bash
# 步骤1：数据降噪（约10分钟）
python S1_denoise.py

# 步骤2：手动特征工程（约30分钟）
python S2_manual_feature.py

# 步骤3：序列特征提取（约15分钟）
python S3_series_feature.py

# 步骤4：特征合并（约5分钟）
python S4_feature_combined.py

# 步骤5：LightGBM训练（约2小时）
python S5_LGB_main.py

# 步骤6：神经网络训练（约4小时，需GPU）
CUDA_VISIBLE_DEVICES=0 python S6_NN_main.py --do_train --batch_size 512

# 步骤7：模型集成
python S7_ensemble.py
```

### 3. 获取结果

最终预测结果保存在：
```
output/final_submission.csv.zip
```

## 技术方案

### 特征工程

#### 1. 聚合统计特征
对每个客户的时间序列数据计算：
- 统计量：mean, std, min, max, sum, last
- 差分特征：last - first, max - min
- 排序特征：时间维度排序、全局排序

#### 2. 时间窗口特征
- 全时间段（13个月）
- 最近6个月
- 最近3个月

#### 3. 类别特征处理
- One-hot编码
- 类别统计特征
- 类别组合特征

### 模型架构

#### LightGBM模型
```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'dart',
    'num_leaves': 64,
    'learning_rate': 0.035,
    'bagging_fraction': 0.75,
    'feature_fraction': 0.05,
    # ... 其他参数见S5_LGB_main.py
}
```

#### 神经网络模型
- **序列编码器**：双向GRU (hidden_dim=128)
- **特征编码器**：多层MLP
- **融合策略**：Concat + MLP
- **输出层**：Sigmoid激活

### 训练策略

1. **交叉验证**：5折分层交叉验证
2. **早停**：监控验证集指标，100轮无提升则停止
3. **学习率调度**：阶段性衰减（0.001 -> 0.0001 -> 0.00001）
4. **数据增强**：支持自定义增强策略
5. **模型集成**：加权平均融合多个模型

## 评估指标

项目使用Amex自定义指标：

```python
Metric = 0.5 * (Normalized_Gini + Top4_Capture_Rate)
```

其中：
- **Normalized_Gini**：归一化的Gini系数
- **Top4_Capture_Rate**：在加权样本Top 4%中的违约捕获率
- **权重策略**：负样本（未违约）权重为20

## 性能优化

### 内存优化
- 使用`float32`代替`float64`
- 使用`category`类型存储类别特征
- 分块读取大文件
- 及时释放不需要的变量

### 计算优化
- 多进程特征工程（16核并行）
- Feather格式快速I/O
- GPU加速神经网络训练
- 混合精度训练（可选）

## 结果复现

为确保结果可复现：
1. 设置随机种子（默认42）
2. 使用确定性算法
3. 保存完整的训练日志
4. 备份训练代码到输出目录

注意：不同硬件环境可能导致轻微的性能差异。

## 常见问题

### Q1: 内存不足怎么办？
- 减少并行进程数（修改S2中的`n_cpu`）
- 减少batch_size（S6中的`--batch_size`参数）
- 使用更积极的特征选择

### Q2: GPU显存不足？
- 减小batch_size
- 减小hidden_dim
- 使用梯度累积

### Q3: 训练时间太长？
- 减少交叉验证折数
- 降低LightGBM的rounds
- 减少神经网络的epochs

## 许可证

本项目遵循MIT许可证。

## 参考资料

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Time Series Feature Engineering](https://www.kaggle.com/c/amex-default-prediction/discussion)
