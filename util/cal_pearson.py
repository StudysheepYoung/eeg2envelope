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
    # total_loss = pearson_loss(y_true, y_pred, axis=axis)
    total_loss = 0

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


def variance_ratio_loss(y_true, y_pred, axis=1):
    """
    方差比损失：约束预测和真实信号的方差比接近 1.0

    这个损失函数解决纯 Pearson Loss 导致的幅度失控问题。
    Pearson 只关注相关性（波形形状），对幅度不敏感。
    通过约束方差比 = var(pred)/var(true) ≈ 1，我们可以：
    - 保持预测信号的波动幅度与真实信号一致
    - 不惩罚均值偏移（符合 EEG 信号特性）
    - 提供平滑的梯度，易于优化

    参数:
    y_true: 真实值 [B, T, C]
    y_pred: 预测值 [B, T, C]
    axis: 计算方差的维度（默认为时间维度 axis=1）

    返回:
    loss: 方差比损失 (variance_ratio - 1)^2

    示例:
    >>> y_true = torch.randn(32, 640, 1)
    >>> y_pred = torch.randn(32, 640, 1) * 2  # 幅度2倍
    >>> loss = variance_ratio_loss(y_true, y_pred)
    >>> # loss ≈ (2^2 - 1)^2 = 9 (因为方差比 = 4)
    """
    # 计算方差：var = E[(x - mean(x))^2]
    var_true = torch.var(y_true, dim=axis, keepdim=False, unbiased=False)
    var_pred = torch.var(y_pred, dim=axis, keepdim=False, unbiased=False)

    # 计算方差比
    variance_ratio = var_pred / (var_true + 1e-6)

    # 目标是方差比为 1.0，使用平方损失
    # (ratio - 1)^2: ratio=1 时损失为0，ratio 偏离1时损失增大
    loss = (variance_ratio - 1.0) ** 2

    return loss


def si_sdr(y_pred, y_true, axis=1, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    尺度不变信号失真比

    SI-SDR 是一个同时优化相关性和幅度的优雅损失函数，特别适合信号重建任务。

    核心思想：
    1. 将预测信号投影到真实信号方向，得到"目标信号"
    2. 计算预测与目标的残差，得到"失真/噪声"
    3. SI-SDR = 10 * log10(目标能量 / 失真能量)

    优势：
    - 尺度不变：自动找到最佳幅度缩放，不受信号绝对幅度影响
    - 同时优化：通过投影最大化相关性，通过残差最小化失真
    - 数学优雅：单一损失函数，无需手动调整多个损失权重
    - 语音领域验证：在语音分离/增强任务中广泛使用并验证有效

    参数:
    y_pred: 预测信号 [B, T, C]
    y_true: 真实信号 [B, T, C]
    axis: 时间维度（默认为 axis=1）
    eps: 数值稳定性常数

    返回:
    si_sdr: SI-SDR 值（dB），越大越好

    数学公式：
    α = <y_pred, y_true> / ||y_true||²
    s_target = α * y_true
    e_noise = y_pred - s_target
    SI-SDR = 10 * log10(||s_target||² / ||e_noise||²)

    示例:
    >>> y_true = torch.randn(32, 640, 1)
    >>> y_pred_good = y_true * 2.0 + 0.01 * torch.randn(32, 640, 1)  # 高质量预测
    >>> y_pred_bad = torch.randn(32, 640, 1)  # 随机预测
    >>> si_sdr(y_pred_good, y_true)  # ~20 dB
    >>> si_sdr(y_pred_bad, y_true)   # ~0 dB
    """
    # 去均值（可选，但建议加上以提高稳定性）
    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)
    y_true_zm = y_true - y_true_mean
    y_pred_zm = y_pred - y_pred_mean

    # 计算投影系数：α = <y_pred, y_true> / ||y_true||²
    dot_product = torch.sum(y_pred_zm * y_true_zm, dim=axis, keepdim=True)
    s_true_power = torch.sum(y_true_zm ** 2, dim=axis, keepdim=True)
    alpha = dot_product / (s_true_power + eps)

    # 投影：s_target = α * y_true（去均值后的版本）
    s_target = alpha * y_true_zm

    # 残差/失真：e_noise = y_pred - s_target
    e_noise = y_pred_zm - s_target

    # 计算能量
    target_power = torch.sum(s_target ** 2, dim=axis, keepdim=False)
    noise_power = torch.sum(e_noise ** 2, dim=axis, keepdim=False)

    # SI-SDR (dB)
    si_sdr_value = 10 * torch.log10(target_power / (noise_power + eps) + eps)

    return si_sdr_value


def si_sdr_loss(y_pred, y_true, axis=1):
    """
    SI-SDR Loss: 用于训练的损失函数版本

    返回负的 SI-SDR，使其变为最小化目标。
    SI-SDR 越大越好，所以 -SI-SDR 越小越好。

    参数:
    y_pred: 预测信号 [B, T, C]
    y_true: 真实信号 [B, T, C]
    axis: 时间维度（默认为 axis=1）

    返回:
    loss: -SI-SDR，用于梯度下降优化

    用法:
    >>> loss = si_sdr_loss(outputs, labels).mean()
    >>> loss.backward()
    """
    return -si_sdr(y_pred, y_true, axis=axis)