"""
3D Segmentation Model for Organ Mask Generation.

This module implements a 3D segmentation model based on ResNet18d for generating
organ masks from CT scans. The model is trained to segment 5 organs:
- Liver (class 0)
- Spleen (class 1)
- Right Kidney (class 2)
- Left Kidney (class 3)
- Bowel (class 4)

Architecture:
    - Encoder: timm ResNet18d with 3D convolutions
    - Decoder: U-Net style decoder
    - Loss: Dice Loss for multi-label segmentation
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from transformers import get_cosine_schedule_with_warmup

import sys
sys.path.append('./Configs/')
from segmentation_config import CFG

class Model(nn.Module):
    """
    3D Segmentation Model for multi-organ segmentation.

    This model uses a U-Net architecture with a timm encoder (ResNet18d by default)
    converted to 3D convolutions. It performs dense pixel-wise prediction to generate
    segmentation masks for 5 different organs.

    Args:
        backbone (str, optional): Not used, kept for compatibility. Model name is from CFG.
        segtype (str): Decoder type, currently only 'unet' is supported. Default: 'unet'.
        pretrained (bool): Whether to use ImageNet pretrained weights. Default: False.

    Architecture Details:
        - Input: (B, C, H, W) where C is 1 (grayscale CT)
        - The input is replicated to 3 channels for timm compatibility
        - Encoder: ResNet18d with 4 blocks
        - Decoder: U-Net decoder with skip connections
        - Output: (B, 5, H, W) for 5 organ classes

    Forward Pass:
        1. Replicate grayscale input to 3 channels
        2. Extract multi-scale features from encoder
        3. Decode features with skip connections
        4. Generate 5-channel segmentation map
    """

    def __init__(self, backbone=None, segtype='unet', pretrained=False):
        super(Model, self).__init__()

        # Number of encoder blocks to use
        n_blocks = 4
        self.n_blocks = n_blocks

        # Create encoder with feature extraction
        self.encoder = timm.create_model(
            CFG.model_name,
            in_chans=3,  # RGB input required by timm models
            features_only=True,  # Return intermediate features for U-Net skip connections
            drop_rate=0.1,  # Dropout rate for regularization
            drop_path_rate=0.1,  # Stochastic depth rate
            pretrained=pretrained  # Use ImageNet weights if True
        )

        # Determine encoder output channels dynamically
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]

        # Decoder channel configuration
        decoder_channels = [256, 128, 64, 32, 16]

        # Build U-Net decoder
        if segtype == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )

        # Final segmentation head: maps decoder output to 5 organ classes
        self.segmentation_head = nn.Conv2d(
            decoder_channels[n_blocks-1],
            5,  # 5 organ classes
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

    def forward(self, x):
        """
        Forward pass of the segmentation model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W) or (B, 1, H, W)

        Returns:
            torch.Tensor: Segmentation logits of shape (B, 5, H, W)
        """
        # Convert grayscale to RGB by replicating channels
        x = torch.stack([x]*3, 1)

        # Extract multi-scale features
        global_features = [0] + self.encoder(x)[:self.n_blocks]

        # Decode features with skip connections
        seg_features = self.decoder(*global_features)

        # Generate final segmentation map
        seg_features = self.segmentation_head(seg_features)

        return seg_features
    
#from timm.models.layers.conv2d_same import Conv2dSame
class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return timm.models.layers.conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1, 1), value: float = 0):
    ih, iw, iz = x.size()[-3:]
    pad_h = get_same_padding(ih, k[0], s[0], d[0])
    pad_w = get_same_padding(iw, k[1], s[1], d[1])
    pad_z = get_same_padding(iz, k[2], s[2], d[2])
    if pad_h > 0 or pad_w > 0 or pad_z > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_z // 2, pad_z - pad_z // 2], value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def conv3d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0), dilation: Tuple[int, int, int] = (1, 1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)


class Conv3dSame(nn.Conv3d):
    """ Tensorflow like 'SAME' convolution wrapper for 3d convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv3d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv3d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv3dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv3d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def convert_3d(module):

    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
            
    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE)
        
    def forward(self, masks_outputs, masks):
        loss = self.dice(masks_outputs.float(), masks.float())
        return loss

def define_criterion_optimizer_scheduler_scaler(model):
    criterion = CustomLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=CFG.steps_per_epoch * CFG.n_epochs * CFG.upscale_steps, num_warmup_steps=CFG.n_warmup_steps)
    
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast)
    
    return criterion, optimizer, scheduler, scaler