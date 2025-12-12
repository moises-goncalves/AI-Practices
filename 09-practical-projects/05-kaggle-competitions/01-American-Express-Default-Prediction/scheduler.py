"""
学习率调度器模块

本模块实现了自定义的学习率调度策略，用于神经网络训练过程中的学习率动态调整。

技术要点：
- 阶段性学习率衰减：在训练的不同阶段使用不同的学习率
- Adam优化器：使用自适应学习率优化算法
- 学习率预热：逐步提高学习率以稳定训练初期
"""

import torch.optim as optim
from typing import Tuple, Any


class SchedulerBase(object):
    """
    学习率调度器基类

    定义了学习率调度器的通用接口和配置参数。
    所有具体的调度器都应该继承这个基类。

    Attributes:
        _is_load_best_weight: 是否加载最佳权重
        _is_load_best_optim: 是否加载最佳优化器状态
        _is_freeze_bn: 是否冻结BatchNorm层
        _is_adjust_lr: 是否调整学习率
        _lr: 当前学习率
        _cur_optimizer: 当前优化器实例
    """

    def __init__(self):
        self._is_load_best_weight = True
        self._is_load_best_optim = True
        _is_freeze_bn = False
        self._is_adjust_lr = True
        self._lr = 0.01
        self._cur_optimizer = None

    def schedule(self, net: Any, epoch: int, epochs: int, **kwargs) -> Tuple[Any, float]:
        """
        学习率调度方法（需要子类实现）

        Args:
            net: 神经网络模型
            epoch: 当前epoch
            epochs: 总epoch数
            **kwargs: 其他参数

        Returns:
            (optimizer, learning_rate): 优化器和学习率

        Raises:
            Exception: 子类必须实现此方法
        """
        raise Exception('调度方法未实现，子类必须重写此方法')

    def step(self, net: Any, epoch: int, epochs: int) -> list:
        """
        执行一步学习率调整

        Args:
            net: 神经网络模型
            epoch: 当前epoch
            epochs: 总epoch数

        Returns:
            各参数组的学习率列表
        """
        optimizer, lr = self.schedule(net, epoch, epochs)

        # 更新所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 收集所有参数组的学习率
        lr_list = [param_group['lr'] for param_group in optimizer.param_groups]
        return lr_list

    def is_load_best_weight(self) -> bool:
        """是否加载最佳权重"""
        return self._is_load_best_weight

    def is_load_best_optim(self) -> bool:
        """是否加载最佳优化器状态"""
        return self._is_load_best_optim

    def is_freeze_bn(self) -> bool:
        """是否冻结BatchNorm层"""
        return self._is_freeze_bn

    def reset(self) -> None:
        """重置调度器状态"""
        self._is_load_best_weight = True
        self._is_load_best_optim = True
        self._is_freeze_bn = False

    def is_adjust_lr(self) -> bool:
        """是否调整学习率"""
        return self._is_adjust_lr


class Adam12(SchedulerBase):
    """
    Adam优化器的阶段性学习率调度器

    该调度器实现了三阶段学习率衰减策略：
    - 阶段1（epoch 0-4）：较高学习率 100e-5，快速收敛
    - 阶段2（epoch 5-8）：中等学习率 10e-5，精细调整
    - 阶段3（epoch 9+）：较低学习率 1e-5，稳定优化

    技术原理：
        学习率调度是深度学习训练的关键技巧之一。阶段性衰减策略：
        1. 初期使用大学习率快速逼近最优解
        2. 中期降低学习率进行精细搜索
        3. 后期使用小学习率稳定收敛

    Args:
        params_list: 参数列表（可选）
    """

    def __init__(self, params_list=None):
        super().__init__()
        self._lr = 3e-4  # 初始学习率
        self._cur_optimizer = None
        self.params_list = params_list

    def schedule(self, net: Any, epoch: int, epochs: int, **kwargs) -> Tuple[Any, float]:
        """
        根据当前epoch返回对应的优化器和学习率

        学习率调度策略：
        - epoch <= 4: lr = 100e-5 = 0.001  (初期快速训练)
        - epoch 5-8:  lr = 10e-5  = 0.0001 (中期精细调整)
        - epoch >= 9: lr = 1e-5   = 0.00001 (后期稳定收敛)

        Args:
            net: 神经网络模型
            epoch: 当前epoch（从0开始）
            epochs: 总epoch数
            **kwargs: 其他参数

        Returns:
            (optimizer, learning_rate): Adam优化器和对应学习率

        技术说明：
            这种手动设置的学习率调度策略通常比余弦退火或指数衰减
            更容易理解和调试，适合快速实验和迭代
        """
        # 根据epoch阶段设置学习率
        lr = 100e-5  # 默认初期学习率: 0.001

        if epoch > 4:
            lr = 10e-5  # 中期学习率: 0.0001

        if epoch > 8:
            lr = 1e-5  # 后期学习率: 0.00001

        self._lr = lr

        # 首次调用时创建优化器
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(
                net.parameters(),
                lr=lr
                # 可选参数（已注释）:
                # eps=1e-5,           # 数值稳定性参数
                # weight_decay=0.001  # L2正则化系数
            )

        return self._cur_optimizer, self._lr
