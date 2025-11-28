"""
预测测试数据脚本

使用训练好的模型对test_data目录下的测试数据进行预测
测试数据格式: JSON文件，每个受试者60秒数据分成6个10秒片段

Usage:
    python predict_test_data.py --checkpoint <path_to_checkpoint> --test_data_dir <test_data_directory>
"""

import os
import json
import argparse
import numpy as np
import torch
from models.FFT_block_conformer_v2 import Decoder
from util.cal_pearson import pearson_metric
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Predict on test data using trained model')

    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--test_data_dir', type=str,
                        default='/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data/test_data',
                        help='Directory containing test data (preprocessed_eeg and labels folders)')

    # 可选参数
    parser.add_argument('--output_dir', type=str, default='test_predictions',
                        help='Directory to save prediction results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=6,
                        help='Batch size for inference (number of 10s segments to process together)')

    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    """
    加载模型检查点

    Args:
        checkpoint_path: 检查点文件路径
        device: 设备

    Returns:
        model: 加载权重后的模型
        checkpoint: 检查点字典(包含训练参数等信息)
    """
    print(f"\n正在加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从检查点中恢复模型参数
    args = argparse.Namespace(**checkpoint['args'])

    # 从sub_proj层权重推断within_sub_num（如果模型使用了g_con）
    # sub_proj.weight shape: [d_model, within_sub_num]
    if 'model_state_dict' in checkpoint and 'sub_proj.weight' in checkpoint['model_state_dict']:
        within_sub_num = checkpoint['model_state_dict']['sub_proj.weight'].shape[1]
        print(f"  - 从checkpoint推断: within_sub_num={within_sub_num}")
    else:
        # 如果没有sub_proj层（g_con=False），使用默认值
        within_sub_num = 71
        print(f"  - 未找到sub_proj层，使用默认: within_sub_num={within_sub_num}")

    # 创建模型
    model = Decoder(
        in_channel=args.in_channel,
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_head=args.n_head,
        n_layers=args.n_layers,
        fft_conv1d_kernel=args.fft_conv1d_kernel,
        fft_conv1d_padding=args.fft_conv1d_padding,
        dropout=args.dropout,
        g_con=args.g_con,
        within_sub_num=within_sub_num,  # 使用从checkpoint推断的值
        conv_kernel_size=args.conv_kernel_size,
        use_relative_pos=args.use_relative_pos,
        use_macaron_ffn=args.use_macaron_ffn,
        use_sinusoidal_pos=args.use_sinusoidal_pos,
        use_gated_residual=args.use_gated_residual,
        use_mlp_head=args.use_mlp_head,
        gradient_scale=args.gradient_scale,
        skip_cnn=args.skip_cnn
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ 成功加载检查点 (Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']})")
    print(f"  - Model: Conformer-v2")
    print(f"  - Layers: {args.n_layers}, d_model: {args.d_model}, heads: {args.n_head}")
    print(f"  - Within_sub_num: {within_sub_num} (支持受试者索引 0-{within_sub_num-1})")
    print(f"  - Skip CNN: {args.skip_cnn}")

    return model, checkpoint, within_sub_num


def load_test_data(test_data_dir, subject_id, segment_idx, normalize=True):
    """
    加载单个受试者的单个片段测试数据并进行归一化

    Args:
        test_data_dir: 测试数据根目录
        subject_id: 受试者ID (如 'sub-002')
        segment_idx: 片段索引 (0-5)
        normalize: 是否进行Z-score归一化 (默认True)

    Returns:
        eeg_data: 归一化后的EEG数据 [samples, channels], 均值≈0, 标准差≈1
        label_data: envelope标签数据 [samples, 1]
        audio_filename: 音频文件名
    """
    eeg_path = os.path.join(test_data_dir, 'preprocessed_eeg', f'{subject_id}.json')
    label_path = os.path.join(test_data_dir, 'labels', f'{subject_id}.json')

    # 加载JSON数据
    with open(eeg_path, 'r') as f:
        eeg_json = json.load(f)
    with open(label_path, 'r') as f:
        label_json = json.load(f)

    # 加载指定片段
    key = f"{subject_id}_{segment_idx}"

    # 检查key是否存在
    if key not in eeg_json:
        return None, None, None

    # EEG: 保持 [samples, channels] 格式
    eeg_data = np.array(eeg_json[key], dtype=np.float32)  # [samples, channels]

    # Label: [envelope_data, audio_filename]
    label_data = np.array(label_json[key][0], dtype=np.float32)  # [samples, 1]
    audio_filename = label_json[key][1]  # 音频文件名

    if normalize:
        # ============ 归一化处理 (参考官方baseline) ============
        # 按片段、按通道进行Z-score标准化: (x - mean) / std
        data_mean = np.mean(eeg_data, axis=0, keepdims=True)  # [1, channels]
        data_std = np.std(eeg_data, axis=0, keepdims=True)    # [1, channels]
        eeg_data = (eeg_data - data_mean) / (data_std + 1e-8)  # 避免除0
        # =====================================================

    return eeg_data, label_data, audio_filename


def predict_subject(model, eeg_data, subject_idx, device, win_len=10, sample_rate=64):
    """
    对单个受试者的EEG数据进行预测

    Args:
        model: 训练好的模型
        eeg_data: EEG数据 [total_samples, channels]
        subject_idx: 受试者索引 (从0开始)
        device: 设备
        win_len: 窗口长度(秒)
        sample_rate: 采样率

    Returns:
        predictions: 预测的envelope [total_samples, 1]
    """
    input_length = sample_rate * win_len  # 640
    total_samples = eeg_data.shape[0]

    # 计算可以切分的完整窗口数量
    n_windows = total_samples // input_length

    # 只使用完整的窗口
    valid_samples = n_windows * input_length
    eeg_data = eeg_data[:valid_samples, :]

    # 切分窗口: [valid_samples, channels] -> [n_windows, input_length, channels]
    windows = []
    for i in range(n_windows):
        start_idx = i * input_length
        end_idx = start_idx + input_length
        window = eeg_data[start_idx:end_idx, :]  # [input_length, channels]
        windows.append(window)

    windows = np.stack(windows, axis=0)  # [n_windows, input_length, channels]

    # 转换为tensor
    windows_tensor = torch.FloatTensor(windows).to(device)  # [n_windows, input_length, channels]
    sub_id_tensor = torch.LongTensor([subject_idx] * n_windows).to(device)  # [n_windows]

    # 批量推理
    with torch.no_grad():
        predictions = model(windows_tensor, sub_id_tensor)  # [n_windows, input_length, 1]

    # 拼接所有窗口的预测: [n_windows, input_length, 1] -> [valid_samples, 1]
    predictions = predictions.cpu().numpy()
    predictions = predictions.reshape(-1, 1)  # [valid_samples, 1]

    return predictions


def compute_metrics(predictions, labels):
    """
    计算评估指标

    Args:
        predictions: 预测值 [samples, 1]
        labels: 真实值 [samples, 1]

    Returns:
        metrics: 指标字典
    """
    # 转换为tensor
    pred_tensor = torch.FloatTensor(predictions)
    label_tensor = torch.FloatTensor(labels)

    # 计算Pearson相关系数
    pearson = pearson_metric(pred_tensor, label_tensor, axis=0).mean().item()

    # 计算MSE
    mse = np.mean((predictions - labels) ** 2)

    metrics = {
        'pearson': pearson,
        'mse': mse
    }

    return metrics


def visualize_prediction(predictions, labels, subject_id, output_dir, audio_filename=None, n_samples=6400):
    """
    可视化预测结果（归一化后的数据）

    Args:
        predictions: 预测值 [samples, 1]
        labels: 真实值 [samples, 1] (原始未归一化)
        subject_id: 受试者ID或片段ID
        output_dir: 输出目录
        audio_filename: 音频文件名 (可选)
        n_samples: 显示的样本数量 (默认100秒，但10秒片段只有640个样本)
    """
    # 只显示前n_samples个点，避免图像过于密集
    n_samples = min(n_samples, len(predictions))

    # 对label也进行归一化，使其与预测值在同一尺度
    labels_subset = labels[:n_samples]
    label_mean = labels_subset.mean()
    label_std = labels_subset.std()
    labels_normalized = (labels_subset - label_mean) / (label_std + 1e-8)

    pred_plot = predictions[:n_samples, 0]
    label_plot = labels_normalized[:, 0]

    # 计算归一化后的Pearson系数
    pearson = compute_metrics(predictions[:n_samples], labels_normalized)['pearson']

    # 创建单图
    plt.figure(figsize=(16, 6))
    time_axis = np.arange(n_samples) / 64  # 转换为秒

    # 归一化后的对比
    plt.plot(time_axis, label_plot, label='Ground Truth (Normalized)', alpha=0.7, linewidth=1.5, color='blue')
    plt.plot(time_axis, pred_plot, label='Prediction (Normalized)', alpha=0.7, linewidth=1.5, color='orange')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Normalized Envelope Amplitude', fontsize=12)
    title = f'{subject_id} - Normalized Data (Mean=0, Std=1) | Pearson: {pearson:.4f}'
    if audio_filename:
        title += f'\nAudio: {audio_filename}'
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f'Prediction: μ={pred_plot.mean():.3f}, σ={pred_plot.std():.3f}\n'
    stats_text += f'Ground Truth: μ={label_plot.mean():.3f}, σ={label_plot.std():.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(output_dir, 'visualizations', f'{subject_id}_prediction.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 保存可视化: {save_path}")


def main():
    args = parse_args()

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)

    # 加载模型
    model, checkpoint, within_sub_num = load_checkpoint(args.checkpoint, device)

    print(f"\n✓ 数据归一化: 启用 (按受试者、按通道进行Z-score标准化)")
    print(f"  - 参考官方baseline归一化方法")
    print(f"  - 确保测试数据与训练数据分布一致 (均值=0, 方差=1)")

    # 获取所有测试受试者
    eeg_dir = os.path.join(args.test_data_dir, 'preprocessed_eeg')
    subject_files = sorted([f for f in os.listdir(eeg_dir) if f.endswith('.json')])
    subject_ids = [f.replace('.json', '') for f in subject_files]

    print(f"\n找到 {len(subject_ids)} 个测试受试者: {subject_ids}")

    # 存储所有受试者的结果
    all_metrics = {}

    # 对每个受试者的每个片段进行预测
    print("\n开始预测...")
    if within_sub_num < 85:
        print(f"\n⚠️  重要提示：受试者{within_sub_num+1}-85超出训练集范围（1-{within_sub_num}），将使用最大索引{within_sub_num-1}作为替代")
        print(f"  这些受试者的性能可能较低，因为模型未见过这些受试者的数据\n")

    for subject_id in tqdm(subject_ids, desc="处理受试者"):
        # 提取受试者索引 (sub-002 -> 1, sub-003 -> 2, ...)
        subject_idx = int(subject_id.split('-')[1]) - 1

        # ⚠️ 修复：将超出训练范围的受试者索引限制在within_sub_num范围内
        # 例如: 训练时within_sub_num=85，支持索引0-84
        # 对于超出范围的受试者，使用最大有效索引作为替代
        if subject_idx >= within_sub_num:
            subject_idx = within_sub_num - 1  # 使用最大有效索引

        subject_metrics = []

        # 处理6个片段
        for segment_idx in range(6):
            # 加载单个片段数据
            eeg_data, label_data, audio_filename = load_test_data(
                args.test_data_dir, subject_id, segment_idx
            )

            # 跳过不存在的片段
            if eeg_data is None:
                continue

            # 预测 (单个10秒片段)
            predictions = predict_subject(model, eeg_data, subject_idx, device)

            # 裁剪label_data以匹配predictions的长度
            labels = label_data[:len(predictions)]

            # 计算指标
            metrics = compute_metrics(predictions, labels)
            subject_metrics.append(metrics['pearson'])

            # 保存预测结果
            result_path = os.path.join(
                args.output_dir, 'predictions',
                f'{subject_id}_seg{segment_idx}_prediction.npy'
            )
            np.save(result_path, predictions)

            # 可视化
            segment_name = f"{subject_id}_seg{segment_idx}"
            visualize_prediction(
                predictions, labels, segment_name, args.output_dir,
                audio_filename=audio_filename
            )

        # 计算该受试者的平均指标
        if len(subject_metrics) > 0:
            avg_metric = np.mean(subject_metrics)
            all_metrics[subject_id] = {'pearson': avg_metric, 'segments': len(subject_metrics)}
        else:
            all_metrics[subject_id] = {'pearson': 0.0, 'segments': 0}

    # 计算平均指标
    avg_pearson = np.mean([m['pearson'] for m in all_metrics.values()])

    # 打印结果
    print("\n" + "="*80)
    print("测试结果汇总 (按10秒片段)")
    print("="*80)
    print(f"{'受试者':<15} {'片段数':>10} {'平均Pearson':>15}")
    print("-"*80)

    for subject_id, metrics in all_metrics.items():
        print(f"{subject_id:<15} {metrics['segments']:>10} {metrics['pearson']:>15.4f}")

    print("-"*80)
    print(f"{'总平均':<15} {'':<10} {avg_pearson:>15.4f}")
    print("="*80)

    # 保存结果到JSON
    results = {
        'checkpoint': args.checkpoint,
        'epoch': checkpoint['epoch'],
        'step': checkpoint['step'],
        'average_pearson': avg_pearson,
        'per_subject_metrics': all_metrics,
        'note': 'Metrics are averaged across 10-second segments for each subject'
    }

    results_path = os.path.join(args.output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 结果已保存到: {results_path}")
    print(f"✓ 预测结果已保存到: {os.path.join(args.output_dir, 'predictions')}")
    print(f"✓ 可视化已保存到: {os.path.join(args.output_dir, 'visualizations')}")

    # ============ 生成统计分析和箱线图 ============
    print("\n" + "="*80)
    print("生成统计分析和箱线图...")
    print("="*80)

    # 收集所有片段的Pearson系数
    all_pearson_scores = []
    subject_pearson_map = {}  # {subject_id: [pearson_scores]}

    for subject_id, metrics in all_metrics.items():
        if metrics['segments'] > 0:
            # 从保存的预测结果中重新计算每个片段的Pearson
            subject_idx_num = int(subject_id.split('-')[1])
            subject_pearson_map[subject_id] = []

            for seg_idx in range(6):
                pred_path = os.path.join(
                    args.output_dir, 'predictions',
                    f'{subject_id}_seg{seg_idx}_prediction.npy'
                )
                if os.path.exists(pred_path):
                    # 加载预测结果
                    predictions = np.load(pred_path)

                    # 加载对应的label
                    _, label_data, _ = load_test_data(
                        args.test_data_dir, subject_id, seg_idx, normalize=False
                    )
                    if label_data is not None:
                        labels = label_data[:len(predictions)]
                        # 归一化label用于计算Pearson
                        label_mean = labels.mean()
                        label_std = labels.std()
                        labels_norm = (labels - label_mean) / (label_std + 1e-8)

                        # 计算Pearson
                        pearson = compute_metrics(predictions, labels_norm)['pearson']
                        all_pearson_scores.append(pearson)
                        subject_pearson_map[subject_id].append(pearson)

    # 按受试者编号分组：1-71 和 72-85
    group1_scores = []  # sub-001 to sub-071
    group2_scores = []  # sub-072 to sub-085

    for subject_id, scores in subject_pearson_map.items():
        subject_num = int(subject_id.split('-')[1])
        if 1 <= subject_num <= 71:
            group1_scores.extend(scores)
        elif 72 <= subject_num <= 85:
            group2_scores.extend(scores)

    # 打印统计信息
    print("\n" + "="*80)
    print("样本统计分析")
    print("="*80)

    if len(group1_scores) > 0:
        print(f"\n样本 1-71 (训练集受试者):")
        print(f"  片段数: {len(group1_scores)}")
        print(f"  平均Pearson: {np.mean(group1_scores):.4f}")
        print(f"  中位数: {np.median(group1_scores):.4f}")
        print(f"  标准差: {np.std(group1_scores):.4f}")
        print(f"  最小值: {np.min(group1_scores):.4f}")
        print(f"  最大值: {np.max(group1_scores):.4f}")
        print(f"  25%分位: {np.percentile(group1_scores, 25):.4f}")
        print(f"  75%分位: {np.percentile(group1_scores, 75):.4f}")

    if len(group2_scores) > 0:
        print(f"\n样本 72-85 (测试集受试者):")
        print(f"  片段数: {len(group2_scores)}")
        print(f"  平均Pearson: {np.mean(group2_scores):.4f}")
        print(f"  中位数: {np.median(group2_scores):.4f}")
        print(f"  标准差: {np.std(group2_scores):.4f}")
        print(f"  最小值: {np.min(group2_scores):.4f}")
        print(f"  最大值: {np.max(group2_scores):.4f}")
        print(f"  25%分位: {np.percentile(group2_scores, 25):.4f}")
        print(f"  75%分位: {np.percentile(group2_scores, 75):.4f}")

    print("="*80)

    # 生成箱线图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：样本1-71的箱线图
    if len(group1_scores) > 0:
        bp1 = ax1.boxplot([group1_scores], positions=[0], widths=0.6,
                          patch_artist=True, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                          boxprops=dict(facecolor='lightblue'),
                          medianprops=dict(color='darkblue', linewidth=2))

        ax1.set_ylabel('Pearson Correlation', fontsize=12)
        ax1.set_title(f'Samples 1-71 (n={len(group1_scores)} segments)\nMean: {np.mean(group1_scores):.4f}',
                     fontsize=14, fontweight='bold')
        ax1.set_xticks([0])
        ax1.set_xticklabels(['Subjects 1-71'])
        ax1.grid(True, alpha=0.3, axis='y')

        # 添加统计文本
        stats_text1 = f'Mean: {np.mean(group1_scores):.4f}\n'
        stats_text1 += f'Median: {np.median(group1_scores):.4f}\n'
        stats_text1 += f'Std: {np.std(group1_scores):.4f}\n'
        stats_text1 += f'Range: [{np.min(group1_scores):.4f}, {np.max(group1_scores):.4f}]'
        ax1.text(0.5, 0.05, stats_text1, transform=ax1.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 右图：样本72-85的箱线图
    if len(group2_scores) > 0:
        bp2 = ax2.boxplot([group2_scores], positions=[0], widths=0.6,
                          patch_artist=True, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                          boxprops=dict(facecolor='lightgreen'),
                          medianprops=dict(color='darkgreen', linewidth=2))

        ax2.set_ylabel('Pearson Correlation', fontsize=12)
        ax2.set_title(f'Samples 72-85 (n={len(group2_scores)} segments)\nMean: {np.mean(group2_scores):.4f}',
                     fontsize=14, fontweight='bold')
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Subjects 72-85'])
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加统计文本
        stats_text2 = f'Mean: {np.mean(group2_scores):.4f}\n'
        stats_text2 += f'Median: {np.median(group2_scores):.4f}\n'
        stats_text2 += f'Std: {np.std(group2_scores):.4f}\n'
        stats_text2 += f'Range: [{np.min(group2_scores):.4f}, {np.max(group2_scores):.4f}]'
        ax2.text(0.5, 0.05, stats_text2, transform=ax2.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存箱线图
    boxplot_path = os.path.join(args.output_dir, 'pearson_boxplot_comparison.png')
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 箱线图已保存到: {boxplot_path}")

    # 保存统计数据到JSON
    stats_results = {
        'group_1_71': {
            'n_segments': len(group1_scores),
            'mean': float(np.mean(group1_scores)) if len(group1_scores) > 0 else 0,
            'median': float(np.median(group1_scores)) if len(group1_scores) > 0 else 0,
            'std': float(np.std(group1_scores)) if len(group1_scores) > 0 else 0,
            'min': float(np.min(group1_scores)) if len(group1_scores) > 0 else 0,
            'max': float(np.max(group1_scores)) if len(group1_scores) > 0 else 0,
            'q25': float(np.percentile(group1_scores, 25)) if len(group1_scores) > 0 else 0,
            'q75': float(np.percentile(group1_scores, 75)) if len(group1_scores) > 0 else 0,
        },
        'group_72_85': {
            'n_segments': len(group2_scores),
            'mean': float(np.mean(group2_scores)) if len(group2_scores) > 0 else 0,
            'median': float(np.median(group2_scores)) if len(group2_scores) > 0 else 0,
            'std': float(np.std(group2_scores)) if len(group2_scores) > 0 else 0,
            'min': float(np.min(group2_scores)) if len(group2_scores) > 0 else 0,
            'max': float(np.max(group2_scores)) if len(group2_scores) > 0 else 0,
            'q25': float(np.percentile(group2_scores, 25)) if len(group2_scores) > 0 else 0,
            'q75': float(np.percentile(group2_scores, 75)) if len(group2_scores) > 0 else 0,
        }
    }

    stats_path = os.path.join(args.output_dir, 'group_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_results, f, indent=2)

    print(f"✓ 统计数据已保存到: {stats_path}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
