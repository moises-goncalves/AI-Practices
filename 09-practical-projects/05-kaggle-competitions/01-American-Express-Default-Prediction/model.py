"""
神经网络模型模块

本模块实现了基于GRU的时间序列违约预测模型。

模型架构：
1. 序列编码分支：使用双向GRU处理时间序列特征
2. 特征编码分支：使用全连接网络处理聚合统计特征
3. 特征融合层：将两个分支的输出进行融合
4. 预测层：输出违约概率

技术亮点：
- 双向GRU：同时捕获前向和后向的时间依赖关系
- 动态序列处理：支持变长序列输入
- 多分支架构：同时利用序列信息和统计信息
- 正则化：使用Dropout和LayerNorm/BatchNorm防止过拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List, Dict
import math


class Amodel(nn.Module):
    """
    时间序列违约预测模型

    该模型结合了序列模型和传统特征工程的优势：
    - 序列分支：使用双向GRU处理客户的历史交易序列
    - 特征分支：处理人工构造的聚合统计特征
    - 融合策略：根据use_series_oof参数决定是否使用特征分支

    Args:
        series_dim: 序列特征维度（每个时间步的特征数）
        feature_dim: 聚合特征维度
        target_num: 目标数量（违约预测为1）
        hidden_num: 隐藏层数量
        hidden_dim: 隐藏层维度
        drop_rate: Dropout比率，默认0.5
        use_series_oof: 是否使用序列OOF特征（聚合特征分支）

    技术说明：
        1. 双向GRU：相比单向GRU，能同时利用过去和未来信息
           公式：h_t = GRU(x_t, h_{t-1})
           双向：h_t = [GRU_forward(x_t), GRU_backward(x_t)]

        2. Pack Padded Sequence：PyTorch处理变长序列的标准方法
           - 避免对padding位置进行无效计算
           - 提高训练效率
           - 保持梯度传播的正确性

        3. LayerNorm vs BatchNorm：
           - LayerNorm：对每个样本的特征维度归一化，适合序列数据
           - BatchNorm：对batch维度归一化，适合固定长度数据
    """

    def __init__(
        self,
        series_dim: int,
        feature_dim: int,
        target_num: int,
        hidden_num: int,
        hidden_dim: int,
        drop_rate: float = 0.5,
        use_series_oof: bool = False
    ):
        super(Amodel, self).__init__()
        self.use_series_oof = use_series_oof

        # ============ 序列编码分支 ============
        # 输入层：将原始序列特征映射到hidden_dim维度
        self.input_series_block = nn.Sequential(
            nn.Linear(series_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)  # LayerNorm适合序列数据
        )

        # GRU层：双向GRU处理时间序列
        # bidirectional=True使输出维度翻倍
        self.gru_series = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,  # 输入格式：[batch, seq, feature]
            bidirectional=True  # 双向GRU
        )

        # ============ 特征编码分支 ============
        # 输入层：处理聚合统计特征
        self.input_feature_block = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm适合固定维度特征
            nn.LeakyReLU()  # LeakyReLU避免死神经元问题
        )

        # 隐藏层：多层MLP
        hidden_feature_layers = []
        for h in range(hidden_num - 1):
            hidden_feature_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(drop_rate),  # Dropout正则化
                nn.LeakyReLU()
            ])
        self.hidden_feature_block = nn.Sequential(*hidden_feature_layers)

        # ============ 输出预测层 ============
        # 根据是否使用特征分支决定输入维度
        # 双向GRU输出维度为2*hidden_dim，特征分支为hidden_dim
        input_dim = 3 * hidden_dim if use_series_oof else 2 * hidden_dim

        self.output_block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),

            nn.Linear(hidden_dim, target_num),
            nn.Sigmoid()  # 输出概率值[0, 1]
        )

    def batch_gru(
        self,
        series: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        批量处理GRU，支持变长序列

        该方法使用PyTorch的pack_padded_sequence处理变长序列，
        只对有效位置（mask=1）进行GRU计算，提高效率。

        Args:
            series: 序列张量 [batch_size, max_len, hidden_dim]
            mask: 有效位置掩码 [batch_size, max_len]

        Returns:
            pooling_feature: 每个序列的最后一个有效位置的隐藏状态 [batch_size, 2*hidden_dim]

        技术说明：
            pack_padded_sequence的优势：
            1. 内存效率：不为padding位置分配计算资源
            2. 计算效率：跳过padding位置的前向传播
            3. 正确性：避免padding影响GRU的隐藏状态更新

            双向GRU的输出：
            - 前向GRU：从第1个时间步到最后一个时间步
            - 后向GRU：从最后一个时间步到第1个时间步
            - 我们取最后一个有效时间步的输出，包含了完整的序列信息
        """
        # 计算每个序列的有效长度
        node_num = mask.sum(dim=-1).detach().cpu()

        # 打包序列：将padding位置去除
        pack = nn.utils.rnn.pack_padded_sequence(
            series,
            node_num,
            batch_first=True,
            enforce_sorted=False  # 允许序列长度无序
        )

        # GRU前向传播
        message, hidden = self.gru_series(pack)

        # 提取每个序列最后一个有效位置的输出
        pooling_feature = []
        for i, n in enumerate(node_num.numpy()):
            n = int(n)
            bi = 0  # batch内的起始索引

            # 找到当前序列在unsorted后的位置
            si = message.unsorted_indices[i]

            # 遍历时间步，找到最后一个有效位置
            for k in range(n):
                if k == n - 1:  # 最后一个有效时间步
                    sample_feature = message.data[bi + si]
                bi = bi + message.batch_sizes[k]

            pooling_feature.append(sample_feature)

        return torch.stack(pooling_feature, 0)

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            data: 输入数据字典，包含：
                - batch_series: [batch_size, max_len, series_dim]
                - batch_mask: [batch_size, max_len]
                - batch_feature: [batch_size, feature_dim]

        Returns:
            预测概率 [batch_size, 1]

        网络流程：
            1. 序列分支：series -> input_block -> GRU -> pooling
            2. 特征分支（可选）：feature -> input_block -> hidden_blocks
            3. 融合：concat(x1, x2) or x1
            4. 预测：output_block -> sigmoid
        """
        # ========== 序列分支 ==========
        # 输入编码
        x1 = self.input_series_block(data['batch_series'])

        # GRU处理并提取最后隐藏状态
        x1 = self.batch_gru(x1, data['batch_mask'])

        # ========== 特征分支（可选） ==========
        if self.use_series_oof:
            # 特征编码
            x2 = self.input_feature_block(data['batch_feature'])
            x2 = self.hidden_feature_block(x2)

            # 特征融合
            x = torch.cat([x1, x2], axis=1)
        else:
            x = x1

        # ========== 输出预测 ==========
        y = self.output_block(x)

        return y
