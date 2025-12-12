"""
损失函数模块

本模块实现多种用于回归和分类任务的损失函数，包括：
1. RMSELoss: 均方根误差损失
2. MCRMSELoss: 多目标RMSE损失
3. FocalLoss: Focal损失（用于处理类别不平衡）
4. DenseCrossEntropy: 密集交叉熵损失
5. WeightedDenseCrossEntropy: 加权密集交叉熵损失

根据配置文件动态选择和初始化损失函数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    """
    均方根误差损失(Root Mean Squared Error Loss)

    RMSE是回归任务的常用损失函数，对大误差更敏感。

    计算公式：
        RMSE = sqrt(MSE + eps)

    其中eps用于数值稳定性，防止梯度爆炸。

    Attributes:
        mse: MSE损失函数
        reduction: 损失聚合方式('mean', 'sum', 'none')
        eps: 数值稳定项
    """

    def __init__(self, reduction='mean', eps=1e-9):
        """
        Args:
            reduction: 损失聚合方式
                - 'none': 返回每个样本的损失
                - 'mean': 返回平均损失
                - 'sum': 返回损失总和
            eps: 添加到MSE的小常数，防止sqrt(0)导致的梯度问题
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        计算RMSE损失

        Args:
            y_pred: 预测值
            y_true: 真实值

        Returns:
            RMSE损失（根据reduction方式聚合）
        """
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)

        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class MCRMSELoss(nn.Module):
    """
    多目标均方根误差损失(Mean Columnwise RMSE Loss)

    对多目标回归任务，分别计算每个目标的RMSE，然后加权平均。
    这确保了模型在所有目标上都有良好的性能。

    Attributes:
        rmse: RMSE损失函数
        num_scored: 目标数量
        weights: 每个目标的权重
    """

    def __init__(self, num_scored=6, weights=None):
        """
        Args:
            num_scored: 目标变量的数量
            weights: 每个目标的权重列表
                如果为None，则所有目标权重相等(1/num_scored)
        """
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored
        self.weights = [1/num_scored for _ in range(num_scored)] if weights is None else weights

    def forward(self, yhat, y):
        """
        计算MCRMSE损失

        Args:
            yhat: 预测值，shape为(batch_size, seq_len, num_targets)
            y: 真实值，shape为(batch_size, seq_len, num_targets)

        Returns:
            加权MCRMSE损失

        Notes:
            对于标准的二维输入，seq_len维度应该为1
        """
        score = 0
        for i, w in enumerate(self.weights):
            score += self.rmse(yhat[:, :, i], y[:, :, i]) * w
        return score


class FocalLoss(nn.Module):
    """
    Focal损失

    Focal Loss专门用于处理类别不平衡问题，通过降低易分类样本的权重，
    使模型更关注难分类样本。

    计算公式：
        FL = -(1 - p_t)^gamma * log(p_t)

    其中p_t是正确类别的预测概率，gamma是聚焦参数。

    Attributes:
        weight: 类别权重
        gamma: 聚焦参数，gamma越大，易分类样本的权重降低越多
        reduction: 损失聚合方式
    """

    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        """
        Args:
            weight: 类别权重，用于进一步平衡类别
            gamma: 聚焦参数，通常设置为2.0
            reduction: 损失聚合方式
        """
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        """
        计算Focal损失

        Args:
            input_tensor: 模型输出的logits
            target_tensor: one-hot编码的目标

        Returns:
            Focal损失
        """
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor.argmax(dim=1),
            weight=self.weight,
            reduction=self.reduction,
        )


class DenseCrossEntropy(nn.Module):
    """
    密集交叉熵损失

    用于软标签或标签平滑的交叉熵损失。
    与标准交叉熵不同，它接受连续的目标值而不是离散的类别索引。
    """

    def forward(self, x, target, weights=None):
        """
        计算密集交叉熵损失

        Args:
            x: 模型输出的logits
            target: 软标签，shape与logits相同
            weights: 样本权重（当前未使用）

        Returns:
            交叉熵损失的平均值
        """
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class WeightedDenseCrossEntropy(nn.Module):
    """
    加权密集交叉熵损失

    密集交叉熵的加权版本，支持对不同样本赋予不同权重。
    常用于处理样本不平衡或重要性不同的情况。
    """

    def forward(self, x, target, weights=None):
        """
        计算加权密集交叉熵损失

        Args:
            x: 模型输出的logits
            target: 软标签
            weights: 每个样本的权重
                如果为None，则所有样本权重相等

        Returns:
            加权交叉熵损失
        """
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        loss = -logprobs * target
        loss = loss.sum(-1)

        if weights is not None:
            loss = loss * weights
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.mean()

        return loss


def get_criterion(config):
    """
    根据配置获取损失函数

    从配置文件中读取损失函数类型和参数，返回对应的损失函数实例。

    Args:
        config: 配置对象，包含criterion相关设置

    Returns:
        损失函数实例

    Supported criterion types:
        - 'SmoothL1Loss': 平滑L1损失（对异常值更鲁棒）
        - 'RMSELoss': 均方根误差损失
        - 'MCRMSELoss': 多目标RMSE损失
        - 其他: 默认返回MSELoss

    Examples:
        >>> config.criterion.criterion_type = 'RMSELoss'
        >>> config.criterion.rmse_loss.eps = 1e-6
        >>> criterion = get_criterion(config)
    """
    if config.criterion.criterion_type == 'SmoothL1Loss':
        return torch.nn.SmoothL1Loss(
            reduction=config.criterion.smooth_l1_loss.reduction,
            beta=config.criterion.smooth_l1_loss.beta
        )

    elif config.criterion.criterion_type == 'RMSELoss':
        return RMSELoss(
            eps=config.criterion.rmse_loss.eps,
            reduction=config.criterion.rmse_loss.reduction
        )

    elif config.criterion.criterion_type == 'MCRMSELoss':
        return MCRMSELoss(
            weights=config.criterion.mcrmse_loss.weights,
        )

    # 默认返回MSE损失
    return nn.MSELoss()
