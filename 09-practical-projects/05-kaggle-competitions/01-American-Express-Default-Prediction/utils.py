"""
American Express Default Prediction - 工具函数模块

本模块包含了违约预测项目的核心工具函数，包括：
1. 数据处理和特征工程
2. LightGBM模型训练和预测
3. 神经网络模型训练和预测
4. 评估指标计算
5. 实验管理和日志记录

技术要点：
- 自定义Amex评估指标（结合Gini系数和Top-4%捕获率）
- 分层交叉验证确保样本分布一致性
- 混合精度训练加速神经网络训练
- 序列数据的动态padding和批处理
"""

import pandas as pd
import numpy as np
import os
import random
import datetime
from contextlib import contextmanager
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Any

from sklearn.metrics import roc_auc_score, mean_squared_error, average_precision_score, log_loss
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler

import torch.cuda.amp as amp

from scheduler import *

import argparse

# ===================== 命令行参数配置 =====================
parser = argparse.ArgumentParser(description='American Express Default Prediction')
parser.add_argument("--root", type=str, default='./input/',
                    help='数据根目录路径')
parser.add_argument("--save_dir", type=str, default='tmp',
                    help='模型保存目录名称')
parser.add_argument("--use_apm", action='store_true', default=False,
                    help='是否使用自动混合精度训练')
parser.add_argument("--num_workers", type=int, default=16,
                    help='数据加载的工作进程数')
parser.add_argument("--do_train", action='store_true', default=False,
                    help='是否执行训练')
parser.add_argument("--test", action='store_true', default=False,
                    help='是否执行测试')
parser.add_argument("--seed", type=int, default=42,
                    help='随机种子，确保实验可复现')
parser.add_argument("--remark", type=str, default='',
                    help='实验备注信息')

args, unknown = parser.parse_known_args()


def Seed_everything(seed: int = 42) -> None:
    """
    设置所有随机种子以确保实验的可复现性

    Args:
        seed: 随机种子值

    Note:
        设置随机种子对于深度学习实验的可复现性至关重要，但需要注意：
        1. 某些CUDA操作仍然是不确定的
        2. 多GPU训练时结果可能略有不同
        3. 不同硬件平台结果可能存在微小差异
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU场景
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 禁用自动优化以确保确定性


# 初始化随机种子
Seed_everything(args.seed)

# ===================== 全局常量定义 =====================
id_name = 'customer_ID'      # 客户ID列名
label_name = 'target'         # 目标标签列名

# 创建输出目录
os.makedirs('./output/', exist_ok=True)

# GPU配置
gpus = list(range(torch.cuda.device_count()))
print(f'检测到可用GPU数量: {len(gpus)}, GPU IDs: {gpus}')

@contextmanager
def Timer(title: str):
    """
    计时器上下文管理器，用于测量代码块执行时间

    Args:
        title: 计时任务的标题

    Example:
        with Timer("数据加载"):
            df = pd.read_csv("data.csv")
    """
    t0 = datetime.datetime.now()
    yield
    elapsed = (datetime.datetime.now() - t0).seconds
    print(f"{title} - 完成，耗时 {elapsed}秒")


def amex_metric_mod(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算American Express定制的评估指标

    该指标是Gini系数和Top-4%捕获率的加权组合，专门设计用于信用违约预测。
    核心思想：
    1. 对负样本（未违约）赋予20倍权重，体现业务中对假阴性的高度关注
    2. 评估模型在高风险区间（Top 4%）的捕获能力
    3. 使用标准化的Gini系数衡量整体排序能力

    Args:
        y_true: 真实标签，0表示未违约，1表示违约
        y_pred: 预测概率

    Returns:
        评估分数，范围[0, 1]，越大越好

    数学原理：
        Metric = 0.5 * (Normalized_Gini + Top4_Capture_Rate)

        其中：
        - Normalized_Gini = Gini(model) / Gini(perfect)
        - Top4_Capture_Rate: 在加权样本的Top 4%中捕获的违约比例

    Note:
        这是Kaggle竞赛的官方评估指标，反映了实际业务中的成本权衡
    """
    # 构建预测-标签对，并按预测概率降序排列
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]

    # 对负样本赋予20倍权重（业务中假阴性代价更高）
    weights = np.where(labels[:, 0] == 0, 20, 1)

    # 计算Top 4%的捕获率
    # 选取加权样本累积和不超过总权重4%的样本
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:, 0]) / np.sum(labels[:, 0])

    # 计算归一化的Gini系数
    gini = [0, 0]
    for i in [1, 0]:  # i=1: 按预测排序; i=0: 按真实标签排序（完美模型）
        labels_sorted = np.transpose(np.array([y_true, y_pred]))
        labels_sorted = labels_sorted[labels_sorted[:, i].argsort()[::-1]]

        weight = np.where(labels_sorted[:, 0] == 0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))  # 随机模型的累积曲线

        total_pos = np.sum(labels_sorted[:, 0] * weight)
        cum_pos_found = np.cumsum(labels_sorted[:, 0] * weight)  # 实际累积正样本
        lorentz = cum_pos_found / total_pos  # Lorenz曲线

        # Gini系数 = 2 * (AUC - 0.5) = Lorenz曲线与随机线之间的面积
        gini[i] = np.sum((lorentz - weight_random) * weight)

    # 返回归一化Gini和Top-4%捕获率的平均值
    return 0.5 * (gini[1] / gini[0] + top_four)


def Metric(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    评估指标的简化接口

    Args:
        labels: 真实标签
        preds: 预测值

    Returns:
        评估分数
    """
    return amex_metric_mod(labels, preds)


def Write_log(logFile, text: str, isPrint: bool = True) -> None:
    """
    写入日志文件并可选择打印到控制台

    Args:
        logFile: 打开的日志文件对象
        text: 要写入的文本
        isPrint: 是否同时打印到控制台

    Note:
        使用buffering=1参数打开文件可以实现行缓冲，
        确保日志实时写入，即使程序意外终止也能保留日志
    """
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    logFile.flush()  # 强制刷新缓冲区


def Lgb_train_and_predict(
    train: Optional[pd.DataFrame],
    test: Optional[pd.DataFrame],
    config: Dict[str, Any],
    gkf: bool = False,
    aug: Optional[pd.DataFrame] = None,
    output_root: str = './output/',
    run_id: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Tuple[float, float]]:
    """
    使用LightGBM进行训练和预测

    该函数实现了完整的LightGBM模型训练流程，包括：
    1. 交叉验证训练
    2. OOF（Out-Of-Fold）预测生成
    3. 测试集预测
    4. 特征重要性分析
    5. 实验日志记录

    Args:
        train: 训练数据，包含特征和标签
        test: 测试数据，仅包含特征
        config: 配置字典，包含以下键：
            - lgb_params: LightGBM参数字典
            - feature_name: 特征列名列表
            - rounds: 最大训练轮数
            - early_stopping_rounds: 早停轮数
            - verbose_eval: 日志输出间隔
            - folds: 交叉验证折数
            - seed: 随机种子
        gkf: 是否使用GroupKFold（基于customer_ID分组）
        aug: 数据增强样本
        output_root: 输出根目录
        run_id: 实验运行ID，如果为None则自动生成

    Returns:
        (oof_df, submission_df, (mean_metric, global_metric))
        - oof_df: Out-Of-Fold预测结果
        - submission_df: 测试集预测结果
        - mean_metric: 各折平均指标
        - global_metric: 全局指标

    技术要点：
        1. GroupKFold vs StratifiedKFold:
           - GroupKFold: 确保同一客户的所有时间点数据不会同时出现在训练集和验证集中
           - StratifiedKFold: 仅保证标签分布均衡
        2. OOF预测: 每个样本的预测来自于它未参与训练的模型，避免过拟合
        3. 特征重要性: 同时计算gain和split两种重要性，全面评估特征贡献
    """
    # 生成实验ID和输出路径
    if not run_id:
        run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root + run_id + '/'):
            time.sleep(1)
            run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_root + f'{args.save_dir}/'
    else:
        output_path = output_root + run_id + '/'

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # 备份代码以便复现
    os.system(f'cp ./*.py {output_path}')
    os.system(f'cp ./*.sh {output_path}')

    # 设置随机种子
    config['lgb_params']['seed'] = config['seed']

    oof, sub = None, None
    mean_valid_metric, global_valid_metric = 0.0, 0.0

    # ========== 训练阶段 ==========
    if train is not None:
        log = open(output_path + '/train.log', 'w', buffering=1)
        log.write(str(config) + '\n')

        features = config['feature_name']
        params = config['lgb_params']
        rounds = config['rounds']
        verbose = config['verbose_eval']
        early_stopping_rounds = config['early_stopping_rounds']
        folds = config['folds']
        seed = config['seed']

        # 初始化OOF预测
        oof = train[[id_name]].copy()
        oof[label_name] = 0.0

        all_valid_metric = []
        feature_importance = []

        # 配置交叉验证策略
        if gkf:
            # GroupKFold: 按customer_ID分组，避免数据泄露
            tmp = train[[id_name, label_name]].drop_duplicates(id_name).reset_index(drop=True)
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            split = skf.split(tmp, tmp[label_name])

            # 将user-level的划分映射到row-level
            new_split = []
            for trn_index, val_index in split:
                trn_uids = tmp.loc[trn_index, id_name].values
                val_uids = tmp.loc[val_index, id_name].values
                train_idx = train.loc[train[id_name].isin(trn_uids)].index
                valid_idx = train.loc[train[id_name].isin(val_uids)].index
                new_split.append((train_idx, valid_idx))
            split = new_split
        else:
            # 标准的分层K折交叉验证
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            split = skf.split(train, train[label_name])

        # K折交叉验证训练
        for fold, (trn_index, val_index) in enumerate(split):
            print(f'\n{"="*50}')
            print(f'训练 Fold {fold + 1}/{folds}')
            print(f'{"="*50}')

            evals_result_dic = {}
            train_cids = train.loc[trn_index, id_name].values

            # 数据增强（如果提供）
            if aug:
                train_aug = aug.loc[aug[id_name].isin(train_cids)]
                trn_data = lgb.Dataset(
                    pd.concat([train.loc[trn_index, features], train_aug[features]]),
                    label=pd.concat([train.loc[trn_index, label_name], train_aug[label_name]])
                )
            else:
                trn_data = lgb.Dataset(
                    train.loc[trn_index, features],
                    label=train.loc[trn_index, label_name]
                )

            val_data = lgb.Dataset(
                train.loc[val_index, features],
                label=train.loc[val_index, label_name]
            )

            # 训练模型
            model = lgb.train(
                params,
                train_set=trn_data,
                num_boost_round=rounds,
                valid_sets=[trn_data, val_data],
                valid_names=['training', 'validation'],
                evals_result=evals_result_dic,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose
            )

            # 保存模型
            model.save_model(output_path + f'/fold{fold}.ckpt')

            # 验证集预测
            valid_preds = model.predict(
                train.loc[val_index, features],
                num_iteration=model.best_iteration
            )
            oof.loc[val_index, label_name] = valid_preds

            # 记录训练日志
            for i in range(len(evals_result_dic['validation'][params['metric']]) // verbose):
                Write_log(
                    log,
                    f' - Round {i*verbose} - train_metric: {evals_result_dic["training"][params["metric"]][i*verbose]:.6f} - '
                    f'valid_metric: {evals_result_dic["validation"][params["metric"]][i*verbose]:.6f}'
                )

            # 计算验证集评估指标
            fold_metric = Metric(train.loc[val_index, label_name], valid_preds)
            all_valid_metric.append(fold_metric)
            Write_log(log, f'- Fold {fold} 验证集指标: {fold_metric:.6f}\n')

            # 记录特征重要性
            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
            feature_name = model.feature_name()
            feature_importance.append(pd.DataFrame({
                'feature_name': feature_name,
                'importance_gain': importance_gain,
                'importance_split': importance_split
            }))

        # 保存特征重要性
        feature_importance_df = pd.concat(feature_importance)
        feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
        feature_importance_df = feature_importance_df.sort_values(
            by=['importance_gain'], ascending=False
        )
        feature_importance_df.to_csv(output_path + '/feature_importance.csv', index=False)

        # 计算最终评估指标
        mean_valid_metric = np.mean(all_valid_metric)
        global_valid_metric = Metric(train[label_name].values, oof[label_name].values)
        Write_log(
            log,
            f'平均验证指标: {mean_valid_metric:.6f}, 全局验证指标: {global_valid_metric:.6f}'
        )

        # 保存OOF预测
        oof.to_csv(output_path + '/oof.csv', index=False)

        log.close()
        os.rename(
            output_path + '/train.log',
            output_path + f'/train_{mean_valid_metric:.6f}.log'
        )

        # 记录实验日志
        log_df = pd.DataFrame({
            'run_id': [run_id],
            'mean_metric': [round(mean_valid_metric, 6)],
            'global_metric': [round(global_valid_metric, 6)],
            'remark': [args.remark]
        })
        log_path = output_root + '/experiment_log.csv'
        if not os.path.exists(log_path):
            log_df.to_csv(log_path, index=False)
        else:
            log_df.to_csv(log_path, index=False, header=None, mode='a')

    # ========== 预测阶段 ==========
    if test is not None:
        print('\n开始生成测试集预测...')
        sub = test[[id_name]].copy()
        sub['prediction'] = 0.0

        for fold in range(config['folds']):
            model = lgb.Booster(model_file=output_path + f'/fold{fold}.ckpt')
            test_preds = model.predict(
                test[config['feature_name']],
                num_iteration=model.best_iteration
            )
            sub['prediction'] += test_preds / config['folds']

        # 保存提交文件
        sub[[id_name, 'prediction']].to_csv(
            output_path + '/submission.csv.zip',
            compression='zip',
            index=False
        )
        print(f'预测完成，结果已保存到: {output_path}/submission.csv.zip')

    # 重命名输出目录
    if args.save_dir in output_path:
        os.rename(output_path, output_root + run_id + '/')

    return oof, sub, (mean_valid_metric, global_valid_metric)

class TaskDataset:
    """
    时间序列任务的数据集类

    该数据集类用于处理信用违约预测中的时间序列数据，支持：
    1. 变长序列的处理
    2. 序列特征和统计特征的组合
    3. 动态padding和批处理

    Attributes:
        df_series: 序列数据DataFrame（时间维度）
        df_feature: 聚合特征DataFrame（统计维度）
        df_y: 标签DataFrame
        uidxs: 用户索引列表，每个元素为(start_idx, end_idx, feature_idx)

    技术要点：
        1. 序列长度可变：每个客户的历史记录长度不同（1-13个月）
        2. 特征反转：将特征值转换为1-x+0.001，增强模型对异常值的鲁棒性
        3. Padding策略：使用mask标记有效位置，避免padding值影响模型
    """

    def __init__(
        self,
        df_series: pd.DataFrame,
        df_feature: pd.DataFrame,
        uidxs: List[Tuple[int, int, int]],
        df_y: Optional[pd.DataFrame] = None
    ):
        self.df_series = df_series
        self.df_feature = df_feature
        self.df_y = df_y
        self.uidxs = uidxs

    def __len__(self) -> int:
        return len(self.uidxs)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """
        获取单个样本

        Returns:
            字典，包含以下键：
            - SERIES: 时间序列特征 [seq_len, feature_dim]
            - FEATURE: 聚合统计特征 [feature_dim]
            - LABEL: 标签（训练时） [1]
        """
        i1, i2, idx = self.uidxs[index]

        # 提取时间序列（去除customer_ID列）
        series = self.df_series.iloc[i1:i2+1, 1:].values

        # 处理单样本情况
        if len(series.shape) == 1:
            series = series.reshape((-1,) + series.shape[-1:])

        # 特征反转：将特征值映射到(0,1)区间的补集
        # 这种技巧可以增强模型对异常高值的敏感度
        series_inverted = series.copy()
        series_inverted[series_inverted != 0] = 1.0 - series_inverted[series_inverted != 0] + 0.001

        # 提取聚合特征
        feature = self.df_feature.loc[idx].values[1:]  # 去除customer_ID
        feature_inverted = feature.copy()
        feature_inverted[feature_inverted != 0] = 1.0 - feature_inverted[feature_inverted != 0] + 0.001

        # 返回数据
        if self.df_y is not None:
            label = self.df_y.loc[idx, [label_name]].values
            return {
                'SERIES': series,
                'FEATURE': np.concatenate([feature, feature_inverted]),
                'LABEL': label,
            }
        else:
            return {
                'SERIES': series,
                'FEATURE': np.concatenate([feature, feature_inverted]),
            }

    def collate_fn(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        批处理函数，将多个样本组合成一个batch

        主要功能：
        1. 动态padding：将不同长度的序列padding到统一长度
        2. 生成mask：标记有效位置，避免padding影响计算
        3. 类型转换：numpy -> torch.Tensor

        Args:
            batch: 样本列表

        Returns:
            批次数据字典，包含：
            - batch_series: [batch_size, max_len, series_dim]
            - batch_mask: [batch_size, max_len] (1表示有效，0表示padding)
            - batch_feature: [batch_size, feature_dim]
            - batch_y: [batch_size, 1]

        技术要点：
            使用mask机制处理变长序列，这在RNN/LSTM/GRU中是标准做法
        """
        batch_size = len(batch)
        max_seq_len = 13  # 最大序列长度为13个月

        # 初始化batch tensors
        batch_series = torch.zeros((batch_size, max_seq_len, batch[0]['SERIES'].shape[1]))
        batch_mask = torch.zeros((batch_size, max_seq_len))
        batch_feature = torch.zeros((batch_size, batch[0]['FEATURE'].shape[0]))
        batch_y = torch.zeros((batch_size, 1))

        # 填充数据
        for i, item in enumerate(batch):
            series = item['SERIES']
            seq_len = series.shape[0]

            # 序列数据和mask
            batch_series[i, :seq_len, :] = torch.tensor(series).float()
            batch_mask[i, :seq_len] = 1.0

            # 特征数据
            batch_feature[i] = torch.tensor(item['FEATURE'].astype(np.float32)).float()

            # 标签（如果有）
            if self.df_y is not None:
                batch_y[i] = torch.tensor(item['LABEL'].astype(np.float32)).float()

        return {
            'batch_series': batch_series,
            'batch_mask': batch_mask,
            'batch_feature': batch_feature,
            'batch_y': batch_y
        }


def NN_train_and_predict(train, test, model_class, config, use_series_oof, logit=False, output_root='./output/', run_id=None):
    if not run_id:
        run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_root + f'{args.save_dir}/'
    else:
        output_path = output_root + run_id + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    os.system(f'cp ./*.py {output_path}')
    feature_name = config['feature_name']
    obj_max = config['obj_max']
    epochs = config['epochs']
    smoothing = config['smoothing']
    patience = config['patience']
    lr = config['lr']
    batch_size = config['batch_size']
    folds = config['folds']
    seed = config['seed']
    if train is not None:
        train_series,train_feature,train_y,train_series_idx = train

        oof = train_y[[id_name]]
        oof['fold'] = -1
        oof[label_name] = 0.0
        oof[label_name] = oof[label_name].astype(np.float32)
    else:
        oof = None

    if train is not None:
        log = open(output_path + 'train.log','w',buffering=1)
        log.write(str(config)+'\n')

        all_valid_metric = []

        skf = StratifiedKFold(n_splits = folds, shuffle=True, random_state=seed)

        model_num = 0
        train_folds = []

        for fold, (trn_index, val_index) in enumerate(skf.split(train_y,train_y[label_name])):

            train_dataset = TaskDataset(train_series,train_feature,[train_series_idx[i] for i in trn_index],train_y)
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn,num_workers=args.num_workers)
            valid_dataset = TaskDataset(train_series,train_feature,[train_series_idx[i] for i in val_index],train_y)
            valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False, drop_last=False, collate_fn=valid_dataset.collate_fn,num_workers=args.num_workers)

            model = model_class(223,(6375+13)*2,1,3,128,use_series_oof=use_series_oof)
            scheduler = Adam12()

            model.cuda()
            if args.use_apm:
                scaler = amp.GradScaler()
            optimizer = scheduler.schedule(model, 0, epochs)[0]

            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-8)
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e5,
            #                                                 max_lr=1e-2, epochs=epochs, steps_per_epoch=len(train_dataloader))
            #torch.optim.Adam(model.parameters(),betas=(0.9, 0.99), lr=lr, weight_decay=0.00001,eps=1e-5)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])


            loss_tr = nn.BCELoss()
            loss_tr1 = nn.BCELoss(reduction='none')
            if obj_max == 1:
                best_valid_metric = 0
            else:
                best_valid_metric = 1e9
            not_improve_epochs = 0
            if args.do_train:
                for epoch in range(epochs):
                    # if epoch <= 13:
                    #     continue
                    np.random.seed(666*epoch)
                    train_loss = 0.0
                    train_num = 0
                    scheduler.step(model,epoch,epochs)
                    model.train()
                    bar = tqdm(train_dataloader)
                    for data in bar:
                        optimizer.zero_grad()
                        for k in data:
                            data[k] = data[k].cuda()
                        y = data['batch_y']
                        if args.use_apm:
                            with amp.autocast():
                                outputs = model(data)
                                # loss_series = loss_tr1(series_outputs,y.repeat(1,13))
                                # loss_series = (loss_series * data['batch_mask']).sum() / data['batch_mask'].sum()
                                # if epoch < 30:
                                #     loss = loss_series
                                # else:
                                loss = loss_tr(outputs,y) #+ loss_series # 0.5 * (loss_tr(outputs,y) + loss_feature(feature,y))
                            if str(loss.item()) == 'nan': continue
                            scaler.scale(loss).backward()
                            torch.nn.utils.clip_grad_norm(model.parameters(), clipnorm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(data)
                            loss = loss_tr(outputs,y)
                            loss.backward()
                            optimizer.step()
                        # scheduler.step()
                        train_num += data['batch_feature'].shape[0]
                        train_loss += data['batch_feature'].shape[0] * loss.item()
                        bar.set_description('loss: %.4f' % (loss.item()))

                    train_loss /= train_num

                    # eval
                    model.eval()
                    valid_preds = []
                    for data in tqdm(valid_dataloader):
                        for k in data:
                            data[k] = data[k].cuda()
                        with torch.no_grad():
                            if logit:
                                outputs = model(data).sigmoid()
                                # feature,outputs = model(data)
                                # outputs = outputs.sigmoid()
                            else:
                                outputs = model(data)
                                # feature,outputs = model(data)
                        valid_preds.append(outputs.detach().cpu().numpy())

                    valid_preds = np.concatenate(valid_preds).reshape(-1)
                    valid_Y = train_y.loc[val_index,label_name].values # oof train
                    valid_mean = np.mean(valid_preds)
                    valid_metric = Metric(valid_Y,valid_preds)

                    if obj_max*(valid_metric) > obj_max*best_valid_metric:
                        if len(gpus) > 1:
                            torch.save(model.module.state_dict(),output_path + 'fold%s.ckpt'%fold)
                        else:
                            torch.save(model.state_dict(),output_path + 'fold%s.ckpt'%fold)
                        not_improve_epochs = 0
                        best_valid_metric = valid_metric
                        Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_metric,valid_mean))
                    else:
                        not_improve_epochs += 1
                        Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f, NIE +1 ---> %s'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_metric,valid_mean,not_improve_epochs))
                        if not_improve_epochs >= patience:
                            break

            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, torch.device('cuda' if torch.cuda.is_available() else 'cpu') )

            model = model_class(223,(6375+13)*2,1,3,128,use_series_oof=use_series_oof)
            model.cuda()
            model.load_state_dict(state_dict)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            model.eval()

            valid_preds = []
            valid_Y = []
            for data in tqdm(valid_dataloader):
                for k in data:
                    data[k] = data[k].cuda()
                with torch.no_grad():
                    if logit:
                        outputs = model(data).sigmoid()
                        # feature,outputs = model(data)
                        # outputs = outputs.sigmoid()
                    else:
                        outputs = model(data)
                        # feature,outputs = model(data)
                valid_preds.append(outputs.detach().cpu().numpy())
                valid_Y.append(y.detach().cpu().numpy())

            valid_preds = np.concatenate(valid_preds).reshape(-1)
            valid_Y = train_y.loc[val_index,label_name].values # oof train
            valid_mean = np.mean(valid_preds)
            valid_metric = Metric(valid_Y,valid_preds)
            Write_log(log,'[fold %s] best_valid_metric: %.6f, best_valid_mean: %.6f'%(fold,valid_metric,valid_mean))

            all_valid_metric.append(valid_metric)
            oof.loc[val_index,label_name] = valid_preds
            oof.loc[val_index,'fold'] = fold
            train_folds.append(fold)

        mean_valid_metric = np.mean(all_valid_metric)
        Write_log(log,'all valid mean metric:%.6f'%(mean_valid_metric))
        oof.loc[oof['fold'].isin(train_folds)].to_csv(output_path + 'oof.csv',index=False)

        if test is None:
            log.close()
            os.rename(output_path + 'train.log', output_path + 'train_%.6f.log'%mean_valid_metric)

        log_df = pd.DataFrame({'run_id':[run_id],'folds':folds,'metric':[round(mean_valid_metric,6)],'lb':[np.nan],'remark':[config['remark']]})
        if not os.path.exists(output_root + 'experiment_log.csv'):
            log_df.to_csv(output_root + 'experiment_log.csv',index=False)
        else:
            log_df.to_csv(output_root + 'experiment_log.csv',index=False,mode='a',header=None)

    if test is not None:
        if train is None:
            log = open(output_path + 'test.log','w', buffering=1)
            Write_log(log,str(config)+'\n')
        test_series,test_feature,test_series_idx = test

        sub = test_feature[-len(test_series_idx):][[id_name]].reset_index(drop=True)
        sub['prediction'] = 0

        test_dataset = TaskDataset(test_series,test_feature,test_series_idx)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False, drop_last=False, collate_fn=test_dataset.collate_fn,num_workers=args.num_workers)
        models = []
        for fold in range(folds):
            if not os.path.exists(output_path + 'fold%s.ckpt'%fold):
                continue
            model = model_class(223,(6375+13)*2,1,3,128,use_series_oof=use_series_oof)
            model.cuda()
            state_dict = torch.load(output_path + 'fold%s.ckpt'%fold, torch.device('cuda') )
            model.load_state_dict(state_dict)
            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

            model.eval()
            models.append(model)
        print('model count:',len(models))
        test_preds = []
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                for k in data:
                    data[k] = data[k].cuda()

                if logit:
                    # outputs = model(data).sigmoid()
                    outputs = torch.stack([m(data).sigmoid() for m in models],0).mean(0)
                    # feature,outputs = model(data)
                    # outputs = outputs.sigmoid()
                else:
                    # outputs = model(data)
                    outputs = torch.stack([m(data) for m in models],0).mean(0)
                    # feature,outputs = model(data)
                test_preds.append(outputs.cpu().detach().numpy())
        test_preds = np.concatenate(test_preds).reshape(-1)
        test_mean = np.mean(test_preds)
        Write_log(log,'test_mean: %.6f'%(test_mean))
        sub['prediction'] = test_preds
        sub.to_csv(output_path+'submission.csv.zip',index=False, compression='zip')
    else:
        sub = None

    if args.save_dir in output_path:
        os.rename(output_path,output_root+run_id+'/')
    return oof,sub
