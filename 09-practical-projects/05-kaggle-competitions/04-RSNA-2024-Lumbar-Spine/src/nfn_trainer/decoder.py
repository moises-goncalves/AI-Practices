"""
解码器网络模块
==============

本模块实现了U-Net风格的解码器网络，支持2D和3D两种解码方式。

解码器作用：
    在编码器（Encoder）提取图像特征后，解码器负责将特征恢复到原始分辨率，
    同时融合编码器的跳跃连接（Skip Connections）以保留细节信息。

网络架构：
    编码器特征 → 解码器块1 → 解码器块2 → ... → 输出特征
                  ↑ skip      ↑ skip

主要组件：
    1. MyDecoderBlock: 2D解码器基本块（用于SCS模型）
    2. MyUnetDecoder: 2D U-Net解码器（用于SCS模型）
    3. MyDecoderBlock3d: 3D解码器基本块（用于NFN模型）
    4. MyUnetDecoder3d: 3D U-Net解码器（用于NFN模型）

设计特点：
    - 采用上采样 + 卷积的方式进行解码
    - 使用BatchNorm + ReLU激活
    - 支持跳跃连接融合编码器特征
    - 提供Attention接口（当前使用Identity）

参考资料：
    - U-Net: https://arxiv.org/abs/1505.04597
    - 3D U-Net: https://arxiv.org/abs/1606.06650
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ===== 2D解码器（用于椎管狭窄SCS模型）=====

class MyDecoderBlock(nn.Module):
    """
    2D解码器基本块

    该模块执行以下操作：
        1. 特征上采样（2倍）
        2. 与跳跃连接特征拼接
        3. 两次3x3卷积 + BatchNorm + ReLU

    参数：
        in_channel (int): 输入特征通道数
        skip_channel (int): 跳跃连接特征通道数
        out_channel (int): 输出特征通道数

    输入形状：
        x: (batch_size, in_channel, H, W)
        skip: (batch_size, skip_channel, 2H, 2W) 或 None

    输出形状：
        (batch_size, out_channel, 2H, 2W)

    使用示例：
        >>> block = MyDecoderBlock(512, 256, 256)
        >>> x = torch.randn(2, 512, 10, 10)
        >>> skip = torch.randn(2, 256, 20, 20)
        >>> out = block(x, skip)
        >>> out.shape
        torch.Size([2, 256, 20, 20])
    """
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()

        # 第一次卷积：融合上采样特征和跳跃连接
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()  # 预留注意力机制接口

        # 第二次卷积：进一步细化特征
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()  # 预留注意力机制接口

    def forward(self, x, skip=None):
        """
        前向传播

        参数：
            x: 输入特征，来自上一层解码器或编码器
            skip: 跳跃连接特征，来自对应编码器层

        返回：
            解码后的特征
        """
        # 上采样到2倍分辨率
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # 如果存在跳跃连接，进行拼接
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        # 两次卷积处理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyUnetDecoder(nn.Module):
    """
    2D U-Net解码器

    完整的解码器网络，包含多个解码器块和跳跃连接。
    主要用于椎管狭窄（Spinal Canal Stenosis, SCS）模型。

    参数：
        in_channel (int): 编码器最后一层的通道数
        skip_channel (list[int]): 各层跳跃连接的通道数（从深到浅）
        out_channel (list[int]): 各层解码器输出通道数

    网络结构：
        输入: 编码器特征 [512 channels, 10x10]
        Block1: 512+320 → 384 [20x20]
        Block2: 384+128 → 192 [40x40]
        Block3: 192+64 → 96 [80x80]
        输出: 96 channels, 80x80

    使用示例：
        >>> decoder = MyUnetDecoder(
        ...     in_channel=512,
        ...     skip_channel=[320, 128, 64],
        ...     out_channel=[384, 192, 96]
        ... )
        >>> feature = torch.randn(2, 512, 10, 10)
        >>> skips = [
        ...     torch.randn(2, 320, 20, 20),
        ...     torch.randn(2, 128, 40, 40),
        ...     torch.randn(2, 64, 80, 80)
        ... ]
        >>> last, decode_list = decoder(feature, skips)
        >>> last.shape
        torch.Size([2, 96, 80, 80])
    """
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()  # 中心层（可扩展）

        # 构建多个解码器块
        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        """
        前向传播

        参数：
            feature: 编码器最后一层特征
            skip: 跳跃连接特征列表（从深到浅）

        返回：
            last: 最后一层解码特征
            decode: 所有层解码特征列表
        """
        d = self.center(feature)
        decode = []

        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)

        last = d
        return last, decode


# ===== 3D解码器（用于神经孔狭窄NFN模型）=====

class MyDecoderBlock3d(nn.Module):
    """
    3D解码器基本块

    与2D版本类似，但处理3D特征（深度/时间维度）。
    用于神经孔狭窄（Neural Foraminal Narrowing, NFN）模型。

    关键区别：
        - 使用3D卷积（Conv3d）代替2D卷积
        - 上采样时只在H和W维度进行，深度维度保持不变
        - 使用BatchNorm3d代替BatchNorm2d

    参数：
        in_channel (int): 输入特征通道数
        skip_channel (int): 跳跃连接特征通道数
        out_channel (int): 输出特征通道数

    输入形状：
        x: (batch_size, in_channel, D, H, W)
        skip: (batch_size, skip_channel, D, 2H, 2W) 或 None

    输出形状：
        (batch_size, out_channel, D, 2H, 2W)

    使用示例：
        >>> block = MyDecoderBlock3d(512, 256, 256)
        >>> x = torch.randn(2, 512, 7, 10, 10)  # D=7层切片
        >>> skip = torch.randn(2, 256, 7, 20, 20)
        >>> out = block(x, skip)
        >>> out.shape
        torch.Size([2, 256, 7, 20, 20])
    """
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()

        # 第一次3D卷积
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()

        # 第二次3D卷积
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        """
        前向传播

        参数：
            x: 输入3D特征
            skip: 跳跃连接3D特征

        返回：
            解码后的3D特征
        """
        # 上采样：深度维度保持不变，只在H和W上2倍放大
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyUnetDecoder3d(nn.Module):
    """
    3D U-Net解码器

    完整的3D解码器网络，用于处理MRI序列的多层切片。
    主要用于神经孔狭窄（NFN）模型。

    应用场景：
        在Sagittal T1 MRI序列中，有多个连续的切片（如7-15层）。
        3D解码器可以建模相邻切片之间的空间关系，提升分类精度。

    参数：
        in_channel (int): 编码器最后一层的通道数
        skip_channel (list[int]): 各层跳跃连接的通道数
        out_channel (list[int]): 各层解码器输出通道数

    网络结构：
        与2D版本类似，但所有操作都在3D空间进行：
        输入: [B, 512, D, 10, 10]
        → Block1: [B, 384, D, 20, 20]
        → Block2: [B, 192, D, 40, 40]
        → Block3: [B, 96, D, 80, 80]

    使用示例：
        >>> decoder = MyUnetDecoder3d(
        ...     in_channel=512,
        ...     skip_channel=[320, 128, 64],
        ...     out_channel=[384, 192, 96]
        ... )
        >>> feature = torch.randn(2, 512, 7, 10, 10)
        >>> skips = [
        ...     torch.randn(2, 320, 7, 20, 20),
        ...     torch.randn(2, 128, 7, 40, 40),
        ...     torch.randn(2, 64, 7, 80, 80)
        ... ]
        >>> last, decode_list = decoder(feature, skips)
        >>> last.shape
        torch.Size([2, 96, 7, 80, 80])
    """
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
    ):
        super().__init__()
        self.center = nn.Identity()

        # 构建多个3D解码器块
        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyDecoderBlock3d(i, s, o)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        """
        前向传播

        参数：
            feature: 编码器最后一层3D特征
            skip: 跳跃连接3D特征列表

        返回：
            last: 最后一层解码特征
            decode: 所有层解码特征列表
        """
        d = self.center(feature)
        decode = []

        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)

        last = d
        return last, decode
