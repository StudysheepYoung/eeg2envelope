"""
Test Script for Conformer Model Checkpoints
测试脚本 - 支持从checkpoint自动读取配置并在test集上评估

Usage:
    # 测试单个checkpoint
    python test_model.py --checkpoint test_results/exp1/best_model.pt

    # 批量测试多个checkpoints
    python test_model.py --checkpoint_dir test_results/ --pattern "*/best_model.pt"
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import glob
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models.FFT_block_conformer_v2 import Decoder
from util.cal_pearson import pearson_loss, pearson_metric
from util.dataset import RegressionDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test Conformer Model on Test Set')

    # Checkpoint相关
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='单个checkpoint文件路径')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='包含多个checkpoint的目录')
    parser.add_argument('--pattern', type=str, default='*/best_model.pt',
                        help='checkpoint文件匹配模式（用于批量测试）')

    # 数据集路径（固定）
    parser.add_argument('--split_data_dir', type=str,
                        default='/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data/split_data',
                        help='split_data目录路径（包含test_-_开头的文件）')
    parser.add_argument('--test_data_dir', type=str,
                        default='/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data/test_data',
                        help='test_data目录路径（包含sub-72到sub-85的文件）')

    # 输出相关
    parser.add_argument('--output_dir', type=str, default='test_results_eval',
                        help='结果保存目录')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备号')
    parser.add_argument('--save_predictions', action='store_true',
                        help='是否保存每个样本的预测结果')

    return parser.parse_args()


def load_checkpoint_and_create_model(checkpoint_path, device):
    """
    从checkpoint加载模型配置和权重

    Returns:
        model: 加载好权重的模型
        args_dict: checkpoint中保存的训练参数
    """
    print(f"\n{'='*80}")
    print(f"加载checkpoint: {checkpoint_path}")
    print(f"{'='*80}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 获取保存的参数
    if 'args' in checkpoint:
        args_dict = checkpoint['args']
    else:
        raise ValueError("Checkpoint中没有找到'args'字段，无法重建模型配置")

    # 打印模型配置
    print("\n【模型配置】")
    print(f"  n_layers: {args_dict.get('n_layers', 'N/A')}")
    print(f"  d_model: {args_dict.get('d_model', 'N/A')}")
    print(f"  d_inner: {args_dict.get('d_inner', 'N/A')}")
    print(f"  n_head: {args_dict.get('n_head', 'N/A')}")
    print(f"  dropout: {args_dict.get('dropout', 'N/A')}")
    print(f"\n【消融实验配置】")
    print(f"  skip_cnn: {args_dict.get('skip_cnn', 'N/A')}")
    print(f"  use_se: {args_dict.get('use_se', 'N/A')}")
    print(f"  use_gated_residual: {args_dict.get('use_gated_residual', 'N/A')}")
    print(f"  use_mlp_head: {args_dict.get('use_mlp_head', 'N/A')}")
    print(f"  gradient_scale: {args_dict.get('gradient_scale', 'N/A')}")

    if 'val_loss' in checkpoint:
        print(f"\n【验证集性能】")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  Val Pearson: {checkpoint.get('val_pearson', 'N/A')}")
        print(f"  Best Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 根据配置创建模型
    model = Decoder(
        in_channel=args_dict.get('in_channel', 64),
        d_model=args_dict.get('d_model', 256),
        d_inner=args_dict.get('d_inner', 1024),
        n_head=args_dict.get('n_head', 4),
        n_layers=args_dict.get('n_layers', 4),
        fft_conv1d_kernel=args_dict.get('fft_conv1d_kernel', (9, 1)),
        fft_conv1d_padding=args_dict.get('fft_conv1d_padding', (4, 0)),
        dropout=args_dict.get('dropout', 0.4),
        g_con=args_dict.get('g_con', True),
        within_sub_num=85,  # 固定为85（覆盖1-85所有受试者）
        conv_kernel_size=args_dict.get('conv_kernel_size', 31),
        use_relative_pos=args_dict.get('use_relative_pos', True),
        use_macaron_ffn=args_dict.get('use_macaron_ffn', True),
        use_sinusoidal_pos=args_dict.get('use_sinusoidal_pos', False),
        use_gated_residual=args_dict.get('use_gated_residual', True),
        use_mlp_head=args_dict.get('use_mlp_head', True),
        gradient_scale=args_dict.get('gradient_scale', 1.0),
        skip_cnn=args_dict.get('skip_cnn', False),
        use_se=args_dict.get('use_se', True),
    ).to(device)

    # 加载权重（兼容DDP和非DDP的checkpoint）
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError("Checkpoint中没有找到'model_state_dict'或'state_dict'")

    # 移除DDP的'module.'前缀（如果有）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    print(f"\n✓ 模型权重加载成功")
    print(f"{'='*80}\n")

    return model, args_dict


def load_test_data(split_data_dir, test_data_dir, sample_rate=64, win_len=10):
    """
    加载测试数据

    包含两部分:
    1. split_data中的test文件（受试者1-71）
    2. test_data中sub-72到sub-85的文件

    Returns:
        test_samples: list of (eeg_data, envelope_data, subject_id)
    """
    print(f"\n{'='*80}")
    print("加载测试数据")
    print(f"{'='*80}")

    input_length = sample_rate * win_len
    test_samples = []

    # ========== 第1部分: split_data中的test文件 ==========
    print("\n[1/2] 从split_data加载test文件...")
    test_files = sorted(glob.glob(os.path.join(split_data_dir, 'test_-_*')))

    # 按受试者分组
    from itertools import groupby
    test_files_grouped = []
    test_files_sorted = sorted(test_files,
                               key=lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))

    for recording_name, feature_paths in groupby(test_files_sorted,
                                                  lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3])):
        feature_list = sorted(list(feature_paths),
                            key=lambda x: "0" if "eeg" in x else x)
        test_files_grouped.append(feature_list)

    print(f"  找到 {len(test_files_grouped)} 组test文件")

    for file_group in test_files_grouped:
        # file_group: [eeg文件, envelope文件]
        eeg_file = [f for f in file_group if 'eeg.npy' in f][0]
        envelope_file = [f for f in file_group if 'envelope.npy' in f][0]

        # 提取受试者ID
        sub_id_str = os.path.basename(eeg_file).split('_-_')[1]  # 'sub-XX'
        sub_id = int(sub_id_str.split('-')[1]) - 1  # 转为0-based索引

        # 加载数据
        eeg_data = np.load(eeg_file)
        envelope_data = np.load(envelope_file)

        test_samples.append((eeg_data, envelope_data, sub_id))

    print(f"  ✓ 加载完成，共 {len(test_samples)} 个样本")

    # ========== 第2部分: test_data中sub-72到sub-85的JSON文件 ==========
    print("\n[2/2] 从test_data加载sub-72到sub-85（JSON格式）...")

    additional_count = 0
    eeg_json_dir = os.path.join(test_data_dir, 'preprocessed_eeg')
    label_json_dir = os.path.join(test_data_dir, 'labels')

    for sub_id in range(72, 86):  # 72-85
        # JSON文件命名: sub-072.json
        eeg_json_file = os.path.join(eeg_json_dir, f'sub-{sub_id:03d}.json')
        label_json_file = os.path.join(label_json_dir, f'sub-{sub_id:03d}.json')

        if not os.path.exists(eeg_json_file):
            # print(f"  警告: 未找到sub-{sub_id:03d}的文件（可能缺失）")
            continue

        if not os.path.exists(label_json_file):
            print(f"  警告: 未找到sub-{sub_id:03d}的label文件")
            continue

        # 加载JSON数据
        with open(eeg_json_file, 'r') as f:
            eeg_json = json.load(f)

        with open(label_json_file, 'r') as f:
            label_json = json.load(f)

        # JSON结构: {"sub-072_0": [[eeg_channels], ...], "sub-072_1": [...], ...}
        # 遍历每个recording
        for key in eeg_json.keys():
            if key not in label_json:
                print(f"  警告: {key} 在label文件中不存在")
                continue

            # eeg_json[key]: [[ch1, ch2, ...], [ch1, ch2, ...], ...]  # [time, channels]
            # label_json[key]: [[env1_samples], [metadata]]，其中每个sample是[value]格式
            eeg_data = np.array(eeg_json[key])  # [time_steps, n_channels]

            # Label第一个元素是attended envelope
            envelope_list = label_json[key][0]  # [[val], [val], ...]
            envelope_data = np.array([val[0] for val in envelope_list]).reshape(-1, 1)  # [time_steps, 1]

            test_samples.append((eeg_data, envelope_data, sub_id - 1))  # 0-based
            additional_count += 1

    print(f"  ✓ 加载完成，新增 {additional_count} 个样本（来自test_data的JSON文件）")

    print(f"\n{'='*80}")
    print(f"测试集总计: {len(test_samples)} 个样本")
    print(f"{'='*80}\n")

    return test_samples


def segment_data(data, input_length):
    """将数据分段为固定长度的窗口"""
    nsegment = data.shape[0] // input_length
    data = data[:int(nsegment * input_length)]
    segments = [data[i:i+input_length] for i in range(0, data.shape[0], input_length)]
    return segments


def evaluate_model(model, test_samples, device, input_length=640, batch_size=64):
    """
    评估模型

    注意：为了与训练时的评估一致，使用按batch计算Pearson然后平均的方法

    Args:
        model: 模型
        test_samples: list of (eeg_data, envelope_data, subject_id)
        device: 设备
        input_length: 窗口长度（默认640 = 10秒 @ 64Hz）
        batch_size: batch大小（默认64，与训练时一致）

    Returns:
        results: dict包含每个样本的预测和指标
    """
    print(f"\n{'='*80}")
    print("开始评估模型")
    print(f"{'='*80}\n")

    results = {
        'samples': [],  # 每个样本的详细结果
        'by_subject': {},  # 按受试者统计
    }

    model.eval()

    with torch.no_grad():
        for idx, (eeg_data, envelope_data, sub_id) in enumerate(tqdm(test_samples, desc="评估进度")):
            # 分段
            eeg_segments = segment_data(eeg_data, input_length)
            envelope_segments = segment_data(envelope_data, input_length)

            if len(eeg_segments) == 0:
                continue

            # 转为numpy数组
            eeg_array = np.array(eeg_segments, dtype=np.float32)  # [n_seg, T, C]
            envelope_array = np.array(envelope_segments, dtype=np.float32)  # [n_seg, T, 1]

            # 按batch评估（与训练时eval一致）
            num_segments = len(eeg_segments)
            num_batches = (num_segments + batch_size - 1) // batch_size

            pearson_sum = 0.0
            all_predictions = []
            all_labels = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_segments)

                # 获取当前batch
                eeg_batch = torch.FloatTensor(eeg_array[start_idx:end_idx]).to(device)
                envelope_batch = torch.FloatTensor(envelope_array[start_idx:end_idx]).to(device)
                sub_id_batch = torch.tensor([sub_id] * (end_idx - start_idx)).to(device)

                # 预测
                pred_batch = model(eeg_batch, sub_id_batch)

                # 计算当前batch的Pearson
                pearson_batch = pearson_metric(envelope_batch, pred_batch).item()
                pearson_sum += pearson_batch

                # 保存预测和标签（用于后续保存）
                all_predictions.append(pred_batch.cpu().numpy())
                all_labels.append(envelope_batch.cpu().numpy())

            # 对所有batch的Pearson取平均（与训练时eval一致）
            avg_pearson = pearson_sum / num_batches

            # 拼接所有预测和标签（仅用于保存）
            pred_full = np.concatenate(all_predictions, axis=0).reshape(-1, 1)
            label_full = np.concatenate(all_labels, axis=0).reshape(-1, 1)

            # 保存结果
            sample_result = {
                'sample_idx': idx,
                'subject_id': sub_id + 1,  # 转回1-based
                'pearson': avg_pearson,
                'prediction': pred_full,
                'label': label_full,
            }

            results['samples'].append(sample_result)

            # 按受试者统计
            if sub_id not in results['by_subject']:
                results['by_subject'][sub_id] = []
            results['by_subject'][sub_id].append(avg_pearson)

    return results


def save_results(results, args_dict, output_dir, checkpoint_name, save_predictions=False):
    """
    保存评估结果
    """
    os.makedirs(output_dir, exist_ok=True)

    # ========== 计算统计指标 ==========
    all_pearsons = [s['pearson'] for s in results['samples']]

    # 按受试者范围分组
    group_1_71 = []  # 受试者1-71
    group_72_85 = []  # 受试者72-85

    for sample in results['samples']:
        sub_id = sample['subject_id']
        pearson = sample['pearson']

        if 1 <= sub_id <= 71:
            group_1_71.append(pearson)
        elif 72 <= sub_id <= 85:
            group_72_85.append(pearson)

    summary = {
        'checkpoint': checkpoint_name,
        'model_config': {
            'n_layers': args_dict.get('n_layers'),
            'skip_cnn': args_dict.get('skip_cnn'),
            'use_se': args_dict.get('use_se'),
            'use_gated_residual': args_dict.get('use_gated_residual'),
            'use_mlp_head': args_dict.get('use_mlp_head'),
            'd_model': args_dict.get('d_model'),
        },
        'overall': {
            'num_samples': len(all_pearsons),
            'mean_pearson': float(np.mean(all_pearsons)),
            'std_pearson': float(np.std(all_pearsons)),
            'median_pearson': float(np.median(all_pearsons)),
            'min_pearson': float(np.min(all_pearsons)),
            'max_pearson': float(np.max(all_pearsons)),
        },
        'group_1_71': {
            'num_samples': len(group_1_71),
            'mean_pearson': float(np.mean(group_1_71)) if group_1_71 else 0,
            'std_pearson': float(np.std(group_1_71)) if group_1_71 else 0,
            'median_pearson': float(np.median(group_1_71)) if group_1_71 else 0,
        },
        'group_72_85': {
            'num_samples': len(group_72_85),
            'mean_pearson': float(np.mean(group_72_85)) if group_72_85 else 0,
            'std_pearson': float(np.std(group_72_85)) if group_72_85 else 0,
            'median_pearson': float(np.median(group_72_85)) if group_72_85 else 0,
        },
        'per_sample': [
            {
                'sample_idx': s['sample_idx'],
                'subject_id': s['subject_id'],
                'pearson': s['pearson'],
            }
            for s in results['samples']
        ]
    }

    # 保存JSON
    json_path = os.path.join(output_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ 结果已保存到: {json_path}")

    # ========== 绘制箱线图（只生成1-71的图）==========
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # 只绘制受试者1-71
    bp = ax.boxplot([group_1_71], labels=['Subjects 1-71'],
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=10),
                     boxprops=dict(facecolor='lightblue'),
                     medianprops=dict(color='darkblue', linewidth=2.5))

    ax.set_ylabel('Pearson Correlation', fontsize=14)
    ax.set_title(f'Test Set Performance (Subjects 1-71)\nn={len(group_1_71)}, μ={np.mean(group_1_71):.4f}',
                  fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    boxplot_path = os.path.join(output_dir, 'pearson_boxplot_1_71.png')
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 箱线图（1-71）已保存到: {boxplot_path}")

    # ========== 保存预测结果（可选）==========
    if save_predictions:
        pred_dir = os.path.join(output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)

        for sample in results['samples']:
            pred_file = os.path.join(pred_dir,
                                    f"sample_{sample['sample_idx']:03d}_sub{sample['subject_id']:03d}.npz")
            np.savez_compressed(pred_file,
                              prediction=sample['prediction'],
                              label=sample['label'],
                              pearson=sample['pearson'])

        print(f"✓ 预测结果已保存到: {pred_dir}")

    # ========== 打印汇总 ==========
    print(f"\n{'='*80}")
    print("评估结果汇总")
    print(f"{'='*80}")
    print(f"\n【全部样本 (1-85)】")
    print(f"  样本数: {summary['overall']['num_samples']}")
    print(f"  平均Pearson: {summary['overall']['mean_pearson']:.4f} ± {summary['overall']['std_pearson']:.4f}")
    print(f"  中位数: {summary['overall']['median_pearson']:.4f}")
    print(f"  范围: [{summary['overall']['min_pearson']:.4f}, {summary['overall']['max_pearson']:.4f}]")

    print(f"\n【受试者1-71】")
    print(f"  样本数: {summary['group_1_71']['num_samples']}")
    print(f"  平均Pearson: {summary['group_1_71']['mean_pearson']:.4f} ± {summary['group_1_71']['std_pearson']:.4f}")

    print(f"\n【受试者72-85】")
    print(f"  样本数: {summary['group_72_85']['num_samples']}")
    print(f"  平均Pearson: {summary['group_72_85']['mean_pearson']:.4f} ± {summary['group_72_85']['std_pearson']:.4f}")
    print(f"{'='*80}\n")

    return summary


def test_single_checkpoint(checkpoint_path, args):
    """测试单个checkpoint"""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model, args_dict = load_checkpoint_and_create_model(checkpoint_path, device)

    # 加载测试数据
    test_samples = load_test_data(args.split_data_dir, args.test_data_dir)

    # 评估
    results = evaluate_model(model, test_samples, device)

    # 保存结果
    checkpoint_name = Path(checkpoint_path).parent.name + '_' + Path(checkpoint_path).stem
    output_dir = os.path.join(args.output_dir, checkpoint_name)

    summary = save_results(results, args_dict, output_dir, checkpoint_name, args.save_predictions)

    return summary


def test_multiple_checkpoints(args):
    """批量测试多个checkpoints"""
    # 查找所有符合条件的checkpoints
    pattern = os.path.join(args.checkpoint_dir, args.pattern)
    checkpoints = sorted(glob.glob(pattern))

    if len(checkpoints) == 0:
        print(f"错误: 未找到符合模式'{args.pattern}'的checkpoint文件")
        return

    print(f"\n找到 {len(checkpoints)} 个checkpoint文件:")
    for cp in checkpoints:
        print(f"  - {cp}")
    print()

    all_summaries = []

    for checkpoint_path in checkpoints:
        try:
            summary = test_single_checkpoint(checkpoint_path, args)
            all_summaries.append(summary)
        except Exception as e:
            print(f"\n错误: 测试 {checkpoint_path} 失败")
            print(f"  {str(e)}\n")
            continue

    # 生成对比表格
    if len(all_summaries) > 1:
        comparison_data = []
        for summary in all_summaries:
            row = {
                'Experiment': summary['checkpoint'],
                'n_layers': summary['model_config']['n_layers'],
                'skip_cnn': summary['model_config']['skip_cnn'],
                'use_se': summary['model_config']['use_se'],
                'use_gated_residual': summary['model_config']['use_gated_residual'],
                'use_mlp_head': summary['model_config']['use_mlp_head'],
                'd_model': summary['model_config']['d_model'],
                'Pearson_All': summary['overall']['mean_pearson'],
                'Pearson_1-71': summary['group_1_71']['mean_pearson'],
                'Pearson_72-85': summary['group_72_85']['mean_pearson'],
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        csv_path = os.path.join(args.output_dir, 'comparison_summary.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')

        print(f"\n{'='*80}")
        print("实验对比表")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        print(f"\n✓ 对比表已保存到: {csv_path}")
        print(f"{'='*80}\n")


def main():
    args = parse_args()

    # 验证参数
    if args.checkpoint is None and args.checkpoint_dir is None:
        print("错误: 必须指定 --checkpoint 或 --checkpoint_dir")
        return

    if args.checkpoint is not None and args.checkpoint_dir is not None:
        print("错误: --checkpoint 和 --checkpoint_dir 不能同时指定")
        return

    # 单个checkpoint测试
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            print(f"错误: checkpoint文件不存在: {args.checkpoint}")
            return

        test_single_checkpoint(args.checkpoint, args)

    # 批量测试
    else:
        if not os.path.exists(args.checkpoint_dir):
            print(f"错误: checkpoint目录不存在: {args.checkpoint_dir}")
            return

        test_multiple_checkpoints(args)


if __name__ == '__main__':
    main()
