"""
Checkpoint 评估脚本
读取 .pt 文件，加载模型，在测试集上进行前向推理并计算性能指标

使用方法:
python eval_checkpoint.py --checkpoint path/to/model.pt
python eval_checkpoint.py --checkpoint test_results/conformer_v2_xxx/model_step_1000.pt --batch_size 1
"""

import argparse
import torch
import os
import glob
import numpy as np
from tqdm import tqdm
from models.FFT_block_conformer_v2 import Decoder
from util.cal_pearson import pearson_loss, pearson_metric, multi_scale_pearson_loss
from util.dataset import RegressionDataset
import torch.nn.functional as F

def multi_scale_pearson_metric(y_pred, y_true, scales=[2, 4, 8, 16, 32], axis=1):
    """
    计算多尺度 Pearson 相关系数（用于评估）
    """
    metrics = {}

    # 原始尺度的 Pearson 系数
    metrics['pearson_scale_1'] = pearson_metric(y_true, y_pred, axis=axis).mean()

    # 对于每个尺度
    for scale in scales:
        if y_pred.shape[axis] >= scale:
            # 需要转置以使用 avg_pool1d (需要 [B, C, T] 格式)
            pred_transposed = y_pred.transpose(1, 2)  # [B, 1, T]
            true_transposed = y_true.transpose(1, 2)  # [B, 1, T]

            # 平均池化降采样
            pred_pooled = F.avg_pool1d(pred_transposed, kernel_size=scale, stride=scale)
            true_pooled = F.avg_pool1d(true_transposed, kernel_size=scale, stride=scale)

            # 转回 [B, T', 1] 格式
            pred_pooled = pred_pooled.transpose(1, 2)
            true_pooled = true_pooled.transpose(1, 2)

            # 计算这个尺度的 Pearson 系数
            scale_metric = pearson_metric(true_pooled, pred_pooled, axis=axis).mean()
            metrics[f'pearson_scale_{scale}'] = scale_metric

    return metrics


def load_checkpoint(checkpoint_path, device='cuda'):
    """加载 checkpoint 并返回模型配置"""
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 打印 checkpoint 信息
    print("\n" + "="*60)
    print("Checkpoint 信息")
    print("="*60)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Step: {checkpoint.get('step', 'N/A')}")
    print(f"Learning Rate: {checkpoint.get('learning_rate', 'N/A')}")

    # 获取模型配置
    args = checkpoint.get('args', {})

    return checkpoint, args


def create_model(args, device='cuda'):
    """根据 checkpoint 的配置创建模型"""
    print("\n" + "="*60)
    print("创建模型")
    print("="*60)

    model = Decoder(
        in_channel=args.get('in_channel', 64),
        d_model=args.get('d_model', 256),
        d_inner=args.get('d_inner', 1024),
        n_head=args.get('n_head', 4),
        n_layers=args.get('n_layers', 8),
        fft_conv1d_kernel=args.get('fft_conv1d_kernel', (9, 1)),
        fft_conv1d_padding=args.get('fft_conv1d_padding', (4, 0)),
        dropout=args.get('dropout', 0.5),
        g_con=args.get('g_con', True),
        within_sub_num=args.get('within_sub_num', 71),
        conv_kernel_size=args.get('conv_kernel_size', 31),
        use_relative_pos=args.get('use_relative_pos', True),
        use_macaron_ffn=args.get('use_macaron_ffn', True),
        use_sinusoidal_pos=args.get('use_sinusoidal_pos', False),
        use_gated_residual=args.get('use_gated_residual', True),
        use_mlp_head=args.get('use_mlp_head', True),
        gradient_scale=args.get('gradient_scale', 1.0)
    ).to(device)

    # 打印模型配置
    print(f"  n_layers: {args.get('n_layers', 8)}")
    print(f"  d_model: {args.get('d_model', 256)}")
    print(f"  d_inner: {args.get('d_inner', 1024)}")
    print(f"  n_head: {args.get('n_head', 4)}")
    print(f"  dropout: {args.get('dropout', 0.5)}")
    print(f"  gradient_scale: {args.get('gradient_scale', 1.0)}")
    print(f"  use_gated_residual: {args.get('use_gated_residual', True)}")
    print(f"  use_mlp_head: {args.get('use_mlp_head', True)}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / (1024 * 1024):.2f} MB")

    return model


def evaluate(model, dataloader, device='cuda', verbose=True):
    """在数据集上评估模型"""
    model.eval()

    test_loss = 0
    test_metric = 0

    # 初始化多尺度指标累加器
    multi_scale_metrics = {
        'pearson_scale_1': 0,
        'pearson_scale_2': 0,
        'pearson_scale_4': 0,
        'pearson_scale_8': 0,
        'pearson_scale_16': 0,
        'pearson_scale_32': 0
    }

    with torch.no_grad():
        # 使用 tqdm 显示进度
        if verbose:
            pbar = tqdm(dataloader, desc="评估进度", ncols=100)
        else:
            pbar = dataloader

        for test_inputs, test_labels, test_sub_id in pbar:
            test_inputs = test_inputs.squeeze(0).to(device)
            test_labels = test_labels.squeeze(0).to(device)
            test_sub_id = test_sub_id.to(device)

            # 前向推理
            test_outputs = model(test_inputs, test_sub_id)

            # 计算损失和指标
            test_loss += pearson_loss(test_outputs, test_labels).mean()
            test_metric += pearson_metric(test_outputs, test_labels).mean()

            # 计算多尺度 Pearson 相关系数
            batch_multi_scale = multi_scale_pearson_metric(test_outputs, test_labels, scales=[2, 4, 8, 16, 32])
            for key in multi_scale_metrics:
                if key in batch_multi_scale:
                    multi_scale_metrics[key] += batch_multi_scale[key].item()

        # 平均化
        test_loss /= len(dataloader)
        test_metric /= len(dataloader)

        for key in multi_scale_metrics:
            multi_scale_metrics[key] /= len(dataloader)

    return test_loss.item(), test_metric.item(), multi_scale_metrics


def main():
    parser = argparse.ArgumentParser(description='评估 Conformer-v2 模型 checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint 文件路径')
    parser.add_argument('--dataset_folder', type=str,
                       default="/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data",
                       help='数据集文件夹路径')
    parser.add_argument('--split_folder', type=str, default="split_data", help='split 文件夹名称')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='评估哪个数据集')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (测试集建议用1)')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')

    args = parser.parse_args()

    # 检查 checkpoint 是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: checkpoint 文件不存在: {args.checkpoint}")
        return

    # 加载 checkpoint
    checkpoint, ckpt_args = load_checkpoint(args.checkpoint, args.device)

    # 创建模型
    model = create_model(ckpt_args, args.device)

    # 加载模型权重
    print("\n加载模型权重...")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ 模型权重加载成功")
    except Exception as e:
        print(f"✗ 加载模型权重失败: {e}")
        print("尝试使用 strict=False 加载...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ 部分权重加载成功")

    # 创建数据集
    print("\n" + "="*60)
    print(f"加载 {args.split} 数据集")
    print("="*60)

    data_folder = os.path.join(args.dataset_folder, args.split_folder)
    features = ["eeg", "envelope"]

    files = [x for x in glob.glob(os.path.join(data_folder, f"{args.split}_-_*"))
             if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]

    dataset = RegressionDataset(
        files,
        input_length=640,  # 默认 64Hz * 10s = 640
        channels=64,
        task=args.split,
        g_con=ckpt_args.get('g_con', True),
        windows_per_sample=1  # 评估时每个样本只用1个窗口
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        drop_last=False
    )

    print(f"数据集大小: {len(dataset)} 样本")
    print(f"Batch 数量: {len(dataloader)}")

    # 评估
    print("\n" + "="*60)
    print("开始评估")
    print("="*60)

    test_loss, test_metric, multi_scale_metrics = evaluate(model, dataloader, args.device)

    # 打印结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"Loss (1 - Pearson): {test_loss:.4f}")
    print(f"Pearson Correlation: {test_metric:.4f}")
    print()

    print("多尺度 Pearson 相关系数:")
    print("-" * 60)
    for scale_name, scale_value in sorted(multi_scale_metrics.items()):
        scale_num = scale_name.split('_')[-1]
        print(f"  Scale {scale_num:>3}: {scale_value:.4f}")

    # 计算高低频差距
    high_freq = (multi_scale_metrics['pearson_scale_1'] + multi_scale_metrics['pearson_scale_2']) / 2
    low_freq = (multi_scale_metrics['pearson_scale_8'] +
                multi_scale_metrics['pearson_scale_16'] +
                multi_scale_metrics['pearson_scale_32']) / 3

    print("-" * 60)
    print(f"  高频平均 (scale 1,2):      {high_freq:.4f}")
    print(f"  低频平均 (scale 8,16,32):  {low_freq:.4f}")
    print(f"  高低频差距:                {low_freq - high_freq:.4f} ({(low_freq-high_freq)/high_freq*100:.1f}%)")
    print("="*60)

    # 保存结果到文件
    result_file = args.checkpoint.replace('.pt', '_eval_result.txt')
    with open(result_file, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'N/A')}\n")
        f.write(f"Step: {checkpoint.get('step', 'N/A')}\n")
        f.write(f"\nPearson Correlation: {test_metric:.4f}\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"\nMulti-scale Pearson:\n")
        for scale_name, scale_value in sorted(multi_scale_metrics.items()):
            scale_num = scale_name.split('_')[-1]
            f.write(f"  Scale {scale_num:>3}: {scale_value:.4f}\n")
        f.write(f"\nHigh-freq avg: {high_freq:.4f}\n")
        f.write(f"Low-freq avg: {low_freq:.4f}\n")
        f.write(f"Gap: {low_freq - high_freq:.4f}\n")

    print(f"\n结果已保存到: {result_file}")


if __name__ == '__main__':
    main()
