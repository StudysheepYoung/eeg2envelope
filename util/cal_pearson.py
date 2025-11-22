# PYTORCH version of the vlaai original code.
import torch
import pdb

def pearson_correlation(y_true, y_pred, axis=1):

    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation.
    numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean),
        dim=axis,
        keepdim=False)

    std_true = torch.sum((y_true - y_true_mean)**2, dim=axis, keepdim=False)
    std_pred = torch.sum((y_pred - y_pred_mean)**2, dim=axis, keepdim=False)
    denominator = torch.sqrt(std_true * std_pred)
    
    pearsonR = torch.div(numerator, denominator + 1e-6)

    assert torch.all(torch.lt(pearsonR, 1)) and torch.all(torch.gt(pearsonR, -1)), "Loss contains values outside the range of -1 to 1"

    return pearsonR


def pearson_loss(y_true, y_pred, axis=1):
    return 1-pearson_correlation(y_true, y_pred, axis=axis)

def pearson_metric(y_true, y_pred, axis=1):
    return pearson_correlation(y_true, y_pred, axis=axis)
    
def l1_loss(y_true, y_pred, axis=1):
    l1_dist = torch.abs(y_true - y_pred)
    l1_loss = torch.mean(l1_dist, axis = axis, keepdim=False)
    return l1_loss

def mse_loss(y_true, y_pred, axis=1):
    """
    计算均方误差损失

    参数:
    y_true: 真实值
    y_pred: 预测值
    axis: 计算均值的维度

    返回:
    mse: 均方误差
    """
    squared_diff = (y_true - y_pred)**2
    mse = torch.mean(squared_diff, dim=axis, keepdim=False)
    return mse


def multi_scale_pearson_loss(y_pred, y_true, scales=[32, 64, 128], axis=1):
    """
    多尺度 Pearson Loss
    在不同时间尺度计算相关性，帮助模型学习多尺度特征

    参数:
    y_pred: 预测值 [B, T, C]
    y_true: 真实值 [B, T, C]
    scales: 下采样尺度列表
    axis: 计算 Pearson 的维度（默认为时间维度 axis=1）

    返回:
    loss: 多尺度 Pearson Loss 的平均值
    """
    import torch.nn.functional as F

    # 原始尺度的 Pearson Loss
    total_loss = pearson_loss(y_true, y_pred, axis=axis)

    # 对于每个尺度
    for scale in scales:
        if y_pred.shape[axis] >= scale:
            # 需要转置以使用 avg_pool1d (需要 [B, C, T] 格式)
            # 输入是 [B, T, 1]，转置为 [B, 1, T]
            pred_transposed = y_pred.transpose(1, 2)  # [B, 1, T]
            true_transposed = y_true.transpose(1, 2)  # [B, 1, T]

            # 平均池化降采样
            pred_pooled = F.avg_pool1d(pred_transposed, kernel_size=scale, stride=scale)
            true_pooled = F.avg_pool1d(true_transposed, kernel_size=scale, stride=scale)

            # 转回 [B, T', 1] 格式
            pred_pooled = pred_pooled.transpose(1, 2)
            true_pooled = true_pooled.transpose(1, 2)

            # 计算这个尺度的 Pearson Loss
            scale_loss = pearson_loss(true_pooled, pred_pooled, axis=axis)
            total_loss = total_loss + scale_loss

    # 返回平均 Loss
    return total_loss / (1 + len(scales))