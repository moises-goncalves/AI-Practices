"""
数据批处理整理器模块

本模块提供动态批处理整理功能，用于优化内存使用和计算效率。
主要功能是将批次中的序列截断到实际需要的最大长度。
"""


def collate(inputs):
    """
    动态截断批次中的序列

    根据批次中实际最长的序列长度，截断所有输入。
    这可以显著减少padding tokens的数量，提高训练效率。

    工作原理：
    1. 计算批次中实际有效token的最大长度（通过attention_mask）
    2. 将所有输入截断到这个长度
    3. 返回截断后的输入字典

    Args:
        inputs: 包含模型输入的字典，通常包含：
            - input_ids: token IDs
            - attention_mask: 注意力掩码
            - 其他可能的输入

    Returns:
        截断后的输入字典

    Examples:
        >>> inputs = {
        ...     'input_ids': torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]),
        ...     'attention_mask': torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
        ... }
        >>> collated = collate(inputs)
        >>> print(collated['input_ids'].shape)  # torch.Size([2, 3])

    Notes:
        - 减少不必要的padding可以节省大约20-30%的显存
        - 对于长度变化较大的数据集特别有效
        - 不影响模型输出，因为padding部分会被attention_mask屏蔽
    """
    # 计算批次中最长的有效序列长度
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())

    # 截断所有输入到这个长度
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]

    return inputs
