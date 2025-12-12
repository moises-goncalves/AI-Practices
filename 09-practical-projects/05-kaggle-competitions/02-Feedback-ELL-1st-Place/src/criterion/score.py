"""
评分指标计算模块

本模块实现MCRMSE（Mean Columnwise Root Mean Squared Error）指标的计算。
MCRMSE是本竞赛的评价指标，用于多目标回归任务。
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def get_score(y_trues, y_preds):
    """
    计算MCRMSE分数

    MCRMSE (Mean Columnwise Root Mean Squared Error) 是多目标回归的常用指标。
    它首先对每个目标分别计算RMSE，然后取所有目标的平均值。

    计算公式：
        MCRMSE = (1/n) * Σ RMSE(y_true_i, y_pred_i)

    其中n是目标数量，RMSE_i是第i个目标的均方根误差。

    Args:
        y_trues: 真实值数组，shape为(n_samples, n_targets)
        y_preds: 预测值数组，shape为(n_samples, n_targets)

    Returns:
        tuple: (mcrmse_score, scores)
            - mcrmse_score: 所有目标的平均RMSE
            - scores: 每个目标的RMSE列表

    Examples:
        >>> y_true = np.array([[3.0, 2.5], [2.5, 3.0]])
        >>> y_pred = np.array([[3.1, 2.4], [2.4, 3.1]])
        >>> mcrmse, individual_scores = get_score(y_true, y_pred)
        >>> print(f"MCRMSE: {mcrmse:.4f}")
        >>> print(f"Individual RMSEs: {individual_scores}")

    Notes:
        - squared=False 使mean_squared_error返回RMSE而不是MSE
        - 对于本竞赛，有6个目标维度：
          cohesion, syntax, vocabulary, phraseology, grammar, conventions
        - 分数越低越好
    """
    scores = []
    idxes = y_trues.shape[1]

    # 对每个目标分别计算RMSE
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)

    # 计算所有目标的平均RMSE
    mcrmse_score = np.mean(scores)

    return mcrmse_score, scores
