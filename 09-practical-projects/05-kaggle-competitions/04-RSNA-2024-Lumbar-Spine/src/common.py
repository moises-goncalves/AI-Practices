"""
通用工具模块
=============

本模块提供了RSNA 2024腰椎退行性分类竞赛项目的通用工具和依赖导入。

主要功能：
    1. 环境配置：设置NumExpr多线程参数
    2. 路径管理：添加第三方库路径
    3. 通用库导入：导入常用的Python和深度学习库
    4. 可视化配置：配置matplotlib后端

依赖库：
    - 数据处理：numpy, pandas, cv2
    - 深度学习：PyTorch (在下方导入)
    - 可视化：matplotlib
    - 工具：tqdm, collections, itertools等

使用方法：
    在项目的其他模块中导入：
    >>> from common import *

注意事项：
    - 本模块使用了import *，在生产代码中应谨慎使用
    - matplotlib后端设置为TkAgg，确保在图形界面环境中运行
    - NumExpr线程数已优化为32/16，可根据硬件调整
"""

import sys
import os

# ===== 环境配置 =====
# 设置NumExpr库的线程数，用于优化numpy数组运算
# MAX_THREADS: 最大允许线程数
# NUM_THREADS: 实际使用线程数
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

# ===== 路径配置 =====
# 添加第三方库路径到Python搜索路径中
# 这允许我们导入自定义的辅助库（my_lib等）
THIRD_PARTY_DIR = os.path.dirname(os.path.realpath(__file__)) + '/third_party'
print('第三方库目录:', THIRD_PARTY_DIR)
sys.path.append(THIRD_PARTY_DIR)

# ===== 自定义工具库导入 =====
# 从my_lib导入各种辅助功能
from my_lib.other import *      # 通用工具函数
from my_lib.draw import *       # 绘图和可视化工具
from my_lib.file import *       # 文件I/O工具

# ===== 标准库导入 =====
import math
import random
import time
import json
import zipfile
import itertools
import collections
from shutil import copyfile
from timeit import default_timer as timer
from collections import OrderedDict
from collections import defaultdict
from glob import glob
from copy import deepcopy
from ast import literal_eval

# ===== 科学计算库 =====
import numpy as np              # 数值计算
import pandas as pd             # 数据分析
import cv2                      # 图像处理

# ===== 进度条和格式化 =====
from tqdm import tqdm           # 进度条显示
from print_dict import format_dict  # 字典格式化输出

# ===== 可视化配置 =====
import matplotlib
import matplotlib.pyplot as plt

# 设置matplotlib后端为TkAgg
# 这是一个跨平台的交互式后端，适合大多数环境
matplotlib.use('TkAgg')
# 其他可选后端（如果TkAgg不可用）：
# matplotlib.use('WXAgg')  # wxPython后端
# matplotlib.use('Qt4Agg') # Qt4后端
# matplotlib.use('Qt5Agg') # Qt5后端
print('matplotlib后端:', matplotlib.get_backend())



# ===== 深度学习框架（PyTorch）=====
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel


def pytorch_version_to_text():
	"""
	生成PyTorch环境信息的格式化文本

	该函数收集并格式化以下信息：
	    - PyTorch版本
	    - CUDA版本
	    - cuDNN版本
	    - GPU数量
	    - GPU属性（名称、计算能力、显存等）

	Returns:
	    str: 格式化的PyTorch环境信息文本

	使用示例：
	    >>> print(pytorch_version_to_text())
	    pytorch
	        torch.__version__              = 2.3.0+cu121
	        torch.version.cuda             = 12.1
	        torch.backends.cudnn.version() = 8902
	        torch.cuda.device_count()      = 2
	        torch.cuda.get_device_properties() = name='NVIDIA RTX 6000 Ada ...'

	参考环境：
	    环境1（测试环境）：
	        - PyTorch 2.0.1+cu117
	        - CUDA 11.7, cuDNN 8500
	        - GPU: NVIDIA TITAN X (Pascal), 12GB显存

	    环境2（训练环境）：
	        - PyTorch 2.3.0+cu121
	        - CUDA 12.1, cuDNN 8902
	        - GPU: 2x NVIDIA RTX 6000 Ada, 48GB显存
	"""
	text = ''
	text += '\tPyTorch环境信息\n'
	text += '\t\ttorch.__version__              = %s\n' % torch.__version__
	text += '\t\ttorch.version.cuda             = %s\n' % torch.version.cuda
	text += '\t\ttorch.backends.cudnn.version() = %s\n' % torch.backends.cudnn.version()
	text += '\t\ttorch.cuda.device_count()      = %d\n' % torch.cuda.device_count()

	if torch.cuda.is_available():
		# 只在CUDA可用时获取GPU属性
		text += '\t\ttorch.cuda.get_device_properties() = %s\n' % str(torch.cuda.get_device_properties(0))[22:-1]
	else:
		text += '\t\ttorch.cuda.get_device_properties() = CUDA不可用\n'

	text += '\n'
	return text


# ===== 医学影像处理库（可选）=====
# 用于处理DICOM格式的医学影像
# 根据需要取消注释以下导入
# import nibabel as nib     # NIfTI格式处理
# import pydicom            # DICOM格式处理
# 参考：https://github.com/tsangel/dicomsdl


# ===== 主程序入口 =====
if __name__ == '__main__':
	"""
	当直接运行本模块时，打印PyTorch环境信息

	运行方法：
	    python common.py
	"""
	print(pytorch_version_to_text())

