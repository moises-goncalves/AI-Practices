"""
对抗权重扰动(AWP)模块

Adversarial Weight Perturbation (AWP)是一种对抗训练技术，
通过在权重空间添加扰动来提高模型的鲁棒性和泛化能力。

核心思想：
1. 在前向传播前，对模型权重添加对抗性扰动
2. 计算损失并反向传播
3. 恢复原始权重
4. 使用梯度更新模型

参考论文：
Adversarial Weight Perturbation Helps Robust Generalization
"""

import torch


class AWP:
    """
    对抗权重扰动训练器

    在训练过程中对模型权重添加对抗性扰动，提高模型的鲁棒性。
    这种方法在NLP竞赛中被广泛使用，能显著提升模型性能。

    Attributes:
        model: 要训练的模型
        optimizer: 优化器（需要访问梯度信息）
        adv_param: 要扰动的参数名称关键词（默认为'weight'）
        adv_lr: 扰动学习率
        adv_eps: 扰动的最大幅度
        adv_epoch: 开始应用AWP的epoch数
        adv_started: 标记是否已开始AWP
        backup: 存储原始权重的字典

    Examples:
        >>> model = YourModel()
        >>> optimizer = AdamW(model.parameters(), lr=1e-5)
        >>> awp = AWP(model, optimizer, adv_lr=0.001, adv_eps=0.001, adv_epoch=2)
        >>>
        >>> for epoch in range(num_epochs):
        >>>     for batch in dataloader:
        >>>         # 前向传播前应用扰动
        >>>         awp.perturb(epoch)
        >>>         loss = model(batch)
        >>>         loss.backward()
        >>>         # 反向传播后恢复权重
        >>>         awp.restore()
        >>>         optimizer.step()
    """

    def __init__(self,
                 model,
                 optimizer,
                 *,
                 adv_param='weight',
                 adv_lr=0.001,
                 adv_eps=0.001,
                 adv_epoch=2):
        """
        初始化AWP训练器

        Args:
            model: PyTorch模型
            optimizer: 优化器（通常是AdamW）
            adv_param: 要扰动的参数名称关键词
            adv_lr: 扰动学习率，控制扰动的更新速度
            adv_eps: 扰动的最大相对幅度
            adv_epoch: 从第几个epoch开始应用AWP（0-indexed）
        """
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.adv_epoch = adv_epoch
        self.adv_started = False
        self.backup = {}

    def perturb(self, epoch):
        """
        对模型参数施加扰动

        应该在损失计算之前调用。
        如果当前epoch >= adv_epoch，则对权重添加对抗性扰动。

        Args:
            epoch: 当前epoch编号（0-indexed）

        Notes:
            - 扰动基于优化器中的指数移动平均梯度(exp_avg)
            - 扰动幅度受adv_eps限制
            - 第一次调用时会打印提示信息
        """
        if (epoch+1) >= self.adv_epoch:
            if not self.adv_started:
                print('AWP: Start perturbing')
                self.adv_started = True

            self._save()  # 保存原始权重
            self._attack_step()  # 施加扰动

    def _attack_step(self):
        """
        执行对抗性扰动步骤

        基于优化器的exp_avg（动量）计算扰动方向，
        然后在受限范围内更新参数。

        扰动公式：
            w' = w + (adv_lr * |w| / |grad|) * grad
        其中：
            - grad来自optimizer.state中的exp_avg
            - 扰动被限制在 [w - adv_eps*|w|, w + adv_eps*|w|] 范围内
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                # 获取梯度的指数移动平均
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # 计算扰动的上下界（相对于参数值）
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # 沿梯度方向扰动：w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # 限制扰动幅度
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        """
        保存原始参数值

        将需要扰动的参数值保存到backup字典中，
        以便在restore()时恢复。
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        恢复原始参数值

        应该在loss.backward()之后、optimizer.step()之前调用。
        AWP不修改模型权重，只是让梯度基于扰动后的权重计算。

        Notes:
            这是AWP的关键：我们不更新扰动后的权重，
            而是使用扰动产生的梯度来更新原始权重。
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
