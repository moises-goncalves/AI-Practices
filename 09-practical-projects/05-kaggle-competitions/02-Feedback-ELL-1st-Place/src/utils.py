"""
工具函数模块

本模块提供了训练过程中常用的工具函数和类，包括：
- AverageMeter: 用于跟踪和计算指标的平均值
- 时间格式化函数: 将时间转换为可读格式
- 配置文件加载和保存: 处理YAML格式的配置文件
- 日志记录器: 设置标准的日志输出格式
- 文件路径管理: 动态生成和管理各种文件路径
"""

import sys
import yaml
import os
import time
import math
import json
import argparse
from types import SimpleNamespace
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter


class AverageMeter:
    """
    跟踪和计算数值的移动平均值

    在训练过程中用于计算损失、准确率等指标的平均值。
    支持增量更新和自动计算累积平均值。

    Attributes:
        val: 最近一次更新的值
        avg: 当前的平均值
        sum: 所有值的总和
        count: 更新次数

    Examples:
        >>> meter = AverageMeter()
        >>> meter.update(0.5, n=32)  # 更新值为0.5，批次大小为32
        >>> meter.update(0.4, n=32)
        >>> print(meter.avg)  # 输出平均值
    """

    def __init__(self):
        """初始化AverageMeter实例"""
        self.reset()

    def reset(self):
        """重置所有统计值为0"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        更新统计值

        Args:
            val: 要添加的值
            n: 该值对应的样本数量（默认为1）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def as_minutes(s):
    """
    将秒数转换为分钟和秒的格式

    Args:
        s: 秒数

    Returns:
        格式化的时间字符串，如 "5m 30s"
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    """
    计算已用时间和预计剩余时间

    根据已完成的进度百分比，估算总时间和剩余时间。
    常用于训练循环中显示进度信息。

    Args:
        since: 开始时间戳（time.time()返回值）
        percent: 当前完成的百分比（0-1之间的浮点数）

    Returns:
        格式化的时间字符串，如 "5m 30s (remain 10m 20s)"

    Examples:
        >>> start = time.time()
        >>> # ... 执行一些操作 ...
        >>> time_info = time_since(start, 0.3)  # 完成了30%
    """
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))


def get_evaluation_steps(num_train_steps, n_evaluations):
    """
    计算训练过程中的验证步骤

    将训练步骤均匀分配，确定在哪些步骤进行模型验证。

    Args:
        num_train_steps: 每个epoch的总训练步骤数
        n_evaluations: 每个epoch需要进行的验证次数

    Returns:
        包含验证步骤索引的列表

    Examples:
        >>> steps = get_evaluation_steps(1000, 5)
        >>> print(steps)  # [200, 400, 600, 800, 1000]
    """
    eval_steps = num_train_steps // n_evaluations
    eval_steps = [eval_steps * i for i in range(1, n_evaluations + 1)]
    return eval_steps


def get_config(path):
    """
    从YAML文件加载配置

    Args:
        path: YAML配置文件的路径

    Returns:
        包含配置信息的字典

    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果YAML格式错误
    """
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, path):
    """
    将配置保存到YAML文件

    Args:
        config: 要保存的配置字典
        path: 输出YAML文件的路径
    """
    with open(path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


def load_filepaths():
    """
    从SETTINGS.json加载文件路径配置

    读取项目的配置文件，并将所有相对路径转换为绝对路径。
    这确保了无论从哪个目录运行代码，路径都是正确的。

    Returns:
        包含所有文件路径的字典，所有路径均为绝对路径

    Raises:
        FileNotFoundError: 如果SETTINGS.json文件不存在
        json.JSONDecodeError: 如果JSON格式错误
    """
    with open('../SETTINGS.json') as f:
        filepaths = json.load(f)

    # 将所有相对路径转换为绝对路径
    for key, value in filepaths.items():
        filepaths[key] = os.path.abspath(value)
    return filepaths


def update_filepaths(filepaths, config, run_name, fold):
    """
    根据配置和fold更新文件路径

    基于运行名称和fold编号，动态生成训练过程中需要的所有文件路径，
    包括模型保存路径、日志路径、伪标签路径等。

    Args:
        filepaths: 基础文件路径字典（从load_filepaths获取）
        config: 训练配置字典
        run_name: 本次运行的名称（通常是模型ID）
        fold: 当前的fold编号（用于交叉验证）

    Returns:
        更新后的文件路径字典

    Notes:
        - 模型文件名格式: {backbone_type}_fold{fold}_best.pth
        - 伪标签文件名格式: pseudolabels_fold{fold}.csv
        - 所有路径都会被转换为绝对路径
    """
    # 处理backbone名称中的斜杠（如microsoft/deberta-v3-base -> microsoft-deberta-v3-base）
    backbone_type = config['model']['backbone_type'].replace('/', '-')
    model_fn = f"{backbone_type}_fold{fold}_best.pth"
    pseudolabels_fn = f'pseudolabels_fold{fold}.csv'

    # 生成模型相关路径
    filepaths['run_dir_path'] = os.path.join(filepaths['MODELS_DIR_PATH'], run_name)
    filepaths['model_fn_path'] = os.path.join(filepaths['run_dir_path'], model_fn)
    filepaths['backbone_config_fn_path'] = os.path.join(filepaths['run_dir_path'], 'config.pth')
    filepaths['tokenizer_dir_path'] = os.path.join(filepaths['run_dir_path'], 'tokenizer')
    filepaths['training_config_fn_path'] = os.path.join(filepaths['CONFIGS_DIR_PATH'], f'{run_name}_training_config.yaml')
    filepaths['log_fn_path'] = os.path.join(filepaths['run_dir_path'], 'train.log')
    filepaths['oof_fn_path'] = os.path.join(filepaths['run_dir_path'], f'oof_fold{fold}.csv')

    # 生成伪标签路径
    filepaths['prev_data_pseudo_fn_path'] = os.path.join(filepaths['PREVIOUS_DATA_PSEUDOLABELS_DIR_PATH'],
                                                         config['general']['previous_data_pseudo_version'],
                                                         pseudolabels_fn)

    filepaths['curr_data_pseudo_fn_path'] = os.path.join(filepaths['CURRENT_DATA_PSEUDOLABELS_DIR_PATH'],
                                                         config['general']['current_data_pseudo_version'],
                                                         pseudolabels_fn)

    # 生成checkpoint路径（如果从checkpoint继续训练）
    filepaths['model_checkpoint_fn_path'] = os.path.join(filepaths['MODELS_DIR_PATH'],
                                                         config['model']['checkpoint_id'],
                                                         model_fn) \
        if config['model']['from_checkpoint'] else ''

    # 转换所有路径为绝对路径
    for key, value in filepaths.items():
        filepaths[key] = os.path.abspath(value)
    return filepaths


def dictionary_to_namespace(data):
    """
    递归地将字典转换为SimpleNamespace对象

    SimpleNamespace允许使用点号访问字典值（如config.model.backbone_type），
    使代码更加简洁易读。支持嵌套字典和列表。

    Args:
        data: 要转换的数据（可以是dict、list或其他类型）

    Returns:
        转换后的数据：
        - dict -> SimpleNamespace
        - list -> 元素被递归转换的list
        - 其他类型 -> 原样返回

    Examples:
        >>> config_dict = {'model': {'lr': 0.001, 'epochs': 10}}
        >>> config = dictionary_to_namespace(config_dict)
        >>> print(config.model.lr)  # 0.001
    """
    if type(data) is list:
        return list(map(dictionary_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dictionary_to_namespace(value))
        return sns
    else:
        return data


def get_logger(filename):
    """
    创建并配置日志记录器

    同时输出日志到控制台和文件，方便实时查看和后续分析。

    Args:
        filename: 日志文件的保存路径

    Returns:
        配置好的Logger对象

    Notes:
        - 日志级别设置为INFO
        - 使用简洁的消息格式（只显示消息内容）
        - 同时输出到标准输出和文件
    """
    logger = getLogger(__name__)
    logger.setLevel(INFO)

    # 控制台输出处理器
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))

    # 文件输出处理器
    handler2 = FileHandler(filename=filename)
    handler2.setFormatter(Formatter("%(message)s"))

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def str_to_bool(argument):
    """
    将字符串参数转换为布尔值

    用于argparse，支持多种常见的布尔值表示方式。

    Args:
        argument: 要转换的参数（可以是bool或str）

    Returns:
        对应的布尔值

    Raises:
        argparse.ArgumentTypeError: 如果输入的字符串不是有效的布尔值表示

    Examples:
        >>> str_to_bool('yes')  # True
        >>> str_to_bool('no')   # False
        >>> str_to_bool('1')    # True
    """
    if isinstance(argument, bool):
        return argument
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_dirs_if_not_exists(filepaths):
    """
    创建文件路径字典中所有不存在的目录

    遍历文件路径字典，为所有包含'DIR_PATH'的键创建对应的目录。

    Args:
        filepaths: 文件路径字典

    Notes:
        - 只创建目录路径（键名包含'DIR_PATH'）
        - 如果目录已存在则跳过
        - 使用os.mkdir创建单层目录
    """
    for key, value in filepaths.items():
        if 'DIR_PATH' in key.upper() and not os.path.isdir(value):
            os.mkdir(value)
