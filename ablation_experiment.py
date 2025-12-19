"""
快速测试脚本 - 给定checkpoint文件夹，自动找best_model.pt并生成箱线图

Usage:
    # 使用别名
    python quick_test_boxplot.py --model conformer_v2

    # 使用完整路径
    python quick_test_boxplot.py --folder test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251203_154629

    # 批量测试多个模型（使用别名）
    python quick_test_boxplot.py --models conformer_v2 conformer_v3

    # 添加新的模型别名
    python quick_test_boxplot.py --add-alias my_model /path/to/checkpoint/folder
"""

import torch
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models.FFT_block_conformer_v2 import Decoder
from util.cal_pearson import pearson_metric
from test_model import load_test_data, segment_data


# ========== 模型别名配置文件 ==========
MODEL_ALIASES_FILE = 'model_aliases.json'

# 默认模型别名（可以通过配置文件扩展）
DEFAULT_ALIASES = {
    # 基准模型
    'Exp-00': 'test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251216_000230',

    # 消融实验
    'Exp-01-无CNN': 'test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251215_163055',
    'Exp-02-无SE': 'test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251215_163353',
    'Exp-03-无MLP_Head': 'test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251215_163517',
    'Exp-04-无Gated_Residual': 'test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251215_163722',
    'Exp-05-无LLRD': 'test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251216_000047',

    # 深度实验
    'Exp-07-2层Conformer': 'test_results/conformer_v2_nlayer2_dmodel256_nhead4_gscale1.0_dist_20251203_233114',
    'Exp-08-6层Conformer': 'test_results/conformer_v2_nlayer6_dmodel256_nhead4_gscale1.0_dist_20251208_113351',
    'Exp-09-8层Conformer': 'test_results/conformer_v2_nlayer8_dmodel256_nhead4_gscale1.0_dist_20251208_113500',

    # 损失函数实验
    'Exp-10-只用HuberLoss': 'test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251209_233040',
    'Exp-11-只用多层皮尔逊': 'test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251209_233613',
}


def load_aliases():
    """加载模型别名配置"""
    aliases = DEFAULT_ALIASES.copy()

    if os.path.exists(MODEL_ALIASES_FILE):
        with open(MODEL_ALIASES_FILE, 'r') as f:
            user_aliases = json.load(f)
            aliases.update(user_aliases)

    return aliases


def save_alias(alias_name, folder_path):
    """保存新的模型别名"""
    aliases = {}

    if os.path.exists(MODEL_ALIASES_FILE):
        with open(MODEL_ALIASES_FILE, 'r') as f:
            aliases = json.load(f)

    aliases[alias_name] = folder_path

    with open(MODEL_ALIASES_FILE, 'w') as f:
        json.dump(aliases, f, indent=2)

    print(f"✓ 已保存别名: {alias_name} -> {folder_path}")


def list_aliases():
    """列出所有可用的模型别名"""
    aliases = load_aliases()

    print("\n可用的模型别名:")
    print("="*80)
    for alias, path in sorted(aliases.items()):
        exists = "✓" if find_best_model(path) else "✗"
        print(f"  {exists} {alias:20s} -> {path}")
    print("="*80)


def find_best_model(folder_path):
    """在给定文件夹中查找best_model.pt"""
    checkpoint_path = os.path.join(folder_path, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    # 如果不在根目录，尝试在test_results子目录中查找
    checkpoint_path = os.path.join('test_results', folder_path, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    return None


def load_model_from_checkpoint(checkpoint_path, device):
    """加载模型"""
    print(f"\n加载checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'args' in checkpoint:
        args_dict = checkpoint['args']
    else:
        raise ValueError("Checkpoint中没有找到'args'字段")

    # 创建模型
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
        within_sub_num=85,
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

    # 加载权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError("Checkpoint中没有找到权重")

    # 移除DDP前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 验证权重已加载（打印第一个参数的部分值作为指纹）
    first_param = next(model.parameters())
    weight_fingerprint = first_param.flatten()[:5].detach().cpu().numpy()

    print(f"✓ 模型加载成功")
    print(f"  n_layers: {args_dict.get('n_layers')}, d_model: {args_dict.get('d_model')}, n_head: {args_dict.get('n_head')}")
    print(f"  权重指纹: {weight_fingerprint}")

    return model, args_dict


def resolve_folder_path(folder_or_alias):
    """解析文件夹路径或别名"""
    aliases = load_aliases()

    # 如果是别名，返回对应路径
    if folder_or_alias in aliases:
        return aliases[folder_or_alias], folder_or_alias

    # 否则直接返回路径，别名为路径本身
    return folder_or_alias, os.path.basename(folder_or_alias)


def evaluate_model_only(model, test_samples, device):
    """仅评估模型，不生成图表"""
    print(f"\n开始评估...")

    input_length = 640
    batch_size = 64

    subject_pearsons = {}  # {subject_id: [pearson1, pearson2, ...]}

    model.eval()

    with torch.no_grad():
        for idx, (eeg_data, envelope_data, sub_id) in enumerate(tqdm(test_samples, desc="评估")):
            # 分段
            eeg_segments = segment_data(eeg_data, input_length)
            envelope_segments = segment_data(envelope_data, input_length)

            if len(eeg_segments) == 0:
                continue

            # 转为数组
            eeg_array = np.array(eeg_segments, dtype=np.float32)
            envelope_array = np.array(envelope_segments, dtype=np.float32)

            # 按batch评估
            num_segments = len(eeg_segments)
            num_batches = (num_segments + batch_size - 1) // batch_size

            pearson_sum = 0.0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_segments)

                eeg_batch = torch.FloatTensor(eeg_array[start_idx:end_idx]).to(device)
                envelope_batch = torch.FloatTensor(envelope_array[start_idx:end_idx]).to(device)
                sub_id_batch = torch.tensor([sub_id] * (end_idx - start_idx)).to(device)

                pred_batch = model(eeg_batch, sub_id_batch)
                pearson_batch = pearson_metric(envelope_batch, pred_batch).mean().item()
                pearson_sum += pearson_batch

            # 平均
            avg_pearson = pearson_sum / num_batches

            # 保存结果（转为1-based）
            subject_id = sub_id + 1
            if subject_id not in subject_pearsons:
                subject_pearsons[subject_id] = []
            subject_pearsons[subject_id].append(avg_pearson)

    # 计算每个受试者的平均Pearson
    subject_avg_pearsons = {}
    for sub_id in sorted(subject_pearsons.keys()):
        subject_avg_pearsons[sub_id] = np.mean(subject_pearsons[sub_id])

    # 分组（1-71和72-85）
    group_1_71 = [subject_avg_pearsons[i] for i in range(1, 72) if i in subject_avg_pearsons]
    group_72_85 = [subject_avg_pearsons[i] for i in range(72, 86) if i in subject_avg_pearsons]

    print(f"\n评估完成:")
    print(f"  受试者1-71: {len(group_1_71)}个, 平均Pearson = {np.mean(group_1_71):.4f}")
    print(f"  受试者72-85: {len(group_72_85)}个, 平均Pearson = {np.mean(group_72_85):.4f}")

    # 返回评估结果
    return {
        'group_1_71': group_1_71,
        'group_72_85': group_72_85,
        'subject_avg_pearsons': subject_avg_pearsons,
    }


def plot_all_models(all_results, output_dir='ablation_results'):
    """为所有模型生成统一的箱线图"""
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据
    model_names = []
    data_1_71_list = []
    mean_values = []

    for result in all_results:
        model_names.append(result['display_name'])
        data_1_71_list.append(result['group_1_71'])
        mean_values.append(np.mean(result['group_1_71']))

    # 创建图表
    num_models = len(model_names)
    fig_width = max(10, num_models * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))

    # 绘制箱线图
    positions = range(1, num_models + 1)
    bp = ax.boxplot(data_1_71_list, positions=positions,
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='darkblue', linewidth=2))

    # 设置标签
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Pearson Correlation (per subject avg)', fontsize=14)
    ax.set_title(f'Model Comparison on Test Set (Subjects 1-71)\nn_models={num_models}',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 在每个箱线图上方标注平均值
    for i, (pos, mean_val) in enumerate(zip(positions, mean_values)):
        ax.text(pos, mean_val, f'{mean_val:.3f}',
                ha='center', va='bottom', fontsize=9, color='darkred', fontweight='bold')

    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(output_dir, 'all_models_comparison_boxplot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 统一箱线图已保存: {plot_path}")

    # 保存详细结果JSON
    summary = {
        'num_models': num_models,
        'models': []
    }

    for result in all_results:
        summary['models'].append({
            'display_name': result['display_name'],
            'folder': result['folder'],
            'overall': {
                'num_subjects': len(result['subject_avg_pearsons']),
                'mean_pearson': float(np.mean(list(result['subject_avg_pearsons'].values()))),
                'std_pearson': float(np.std(list(result['subject_avg_pearsons'].values()))),
            },
            'group_1_71': {
                'num_subjects': len(result['group_1_71']),
                'mean_pearson': float(np.mean(result['group_1_71'])) if result['group_1_71'] else 0,
                'std_pearson': float(np.std(result['group_1_71'])) if result['group_1_71'] else 0,
            },
            'group_72_85': {
                'num_subjects': len(result['group_72_85']),
                'mean_pearson': float(np.mean(result['group_72_85'])) if result['group_72_85'] else 0,
                'std_pearson': float(np.std(result['group_72_85'])) if result['group_72_85'] else 0,
            },
        })

    json_path = os.path.join(output_dir, 'all_models_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ 详细结果已保存: {json_path}\n")


def main():
    parser = argparse.ArgumentParser(description='快速测试并生成箱线图')

    # 模型指定方式
    parser.add_argument('--model', type=str, default=None,
                       help='模型别名（如: conformer_v2）')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='多个模型别名')
    parser.add_argument('--folder', type=str, default=None,
                       help='checkpoint文件夹路径')
    parser.add_argument('--folders', type=str, nargs='+', default=None,
                       help='多个checkpoint文件夹路径')

    # 别名管理
    parser.add_argument('--add-alias', type=str, nargs=2, metavar=('ALIAS', 'PATH'),
                       help='添加新的模型别名: --add-alias my_model /path/to/folder')
    parser.add_argument('--list-aliases', action='store_true',
                       help='列出所有可用的模型别名')

    # 数据和输出
    parser.add_argument('--split_data_dir', type=str,
                       default='/RAID5/projects/likeyang/happy/NeuroConformer/data/split_data',
                       help='split_data目录')
    parser.add_argument('--test_data_dir', type=str,
                       default='/RAID5/projects/likeyang/happy/NeuroConformer/data/test_data',
                       help='test_data目录')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='输出目录')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')

    args = parser.parse_args()

    # 处理别名管理命令
    if args.add_alias:
        alias_name, folder_path = args.add_alias
        save_alias(alias_name, folder_path)
        return

    if args.list_aliases:
        list_aliases()
        return

    # 确定要测试的文件夹列表
    folders_to_test = []

    if args.model is not None:
        folders_to_test.append(args.model)
    elif args.models is not None:
        folders_to_test.extend(args.models)
    elif args.folder is not None:
        folders_to_test.append(args.folder)
    elif args.folders is not None:
        folders_to_test.extend(args.folders)
    else:
        print("错误: 必须指定 --model, --models, --folder 或 --folders")
        print("使用 --list-aliases 查看可用的模型别名")
        return

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载测试数据（所有模型共用）
    print("\n" + "="*80)
    print("加载测试数据...")
    print("="*80)
    test_samples = load_test_data(args.split_data_dir, args.test_data_dir)

    # 测试每个文件夹
    all_results = []
    model = None  # 初始化model变量

    for folder_or_alias in folders_to_test:
        print("\n" + "="*80)
        print(f"测试: {folder_or_alias}")
        print("="*80)

        # 解析路径或别名
        folder_path, display_name = resolve_folder_path(folder_or_alias)
        print(f"路径: {folder_path}")
        print(f"显示名称: {display_name}")

        # 查找checkpoint
        checkpoint_path = find_best_model(folder_path)
        if checkpoint_path is None:
            print(f"错误: 未找到 {folder_path}/best_model.pt")
            continue

        try:
            # 清理之前的模型（释放GPU内存）
            if model is not None:
                del model
                torch.cuda.empty_cache()
                print("  已清理之前的模型")

            # 加载模型
            model, model_args = load_model_from_checkpoint(checkpoint_path, device)

            # 强制设置为eval模式
            model.eval()

            # 确认模型已加载到正确设备
            print(f"  模型设备: {next(model.parameters()).device}")

            # 打印模型配置以确认加载了不同的模型
            print(f"  模型配置: skip_cnn={model_args.get('skip_cnn')}, use_se={model_args.get('use_se')}, "
                  f"use_gated_residual={model_args.get('use_gated_residual')}, use_mlp_head={model_args.get('use_mlp_head')}")

            # 评估模型
            eval_results = evaluate_model_only(model, test_samples, device)

            # 保存结果和模型配置
            eval_results['folder'] = folder_path
            eval_results['display_name'] = display_name
            eval_results['model_config'] = {
                'n_layers': model_args.get('n_layers'),
                'skip_cnn': model_args.get('skip_cnn'),
                'use_se': model_args.get('use_se'),
                'use_gated_residual': model_args.get('use_gated_residual'),
                'use_mlp_head': model_args.get('use_mlp_head'),
            }
            all_results.append(eval_results)

        except Exception as e:
            print(f"错误: 测试失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 生成统一的箱线图和对比表格
    if len(all_results) > 0:
        # 绘制统一箱线图
        plot_all_models(all_results, args.output_dir)

        # 生成对比表格
        print("\n" + "="*80)
        print("多模型对比")
        print("="*80)

        comparison = []
        for r in all_results:
            group_1_71_mean = np.mean(r['group_1_71']) if r['group_1_71'] else 0
            group_72_85_mean = np.mean(r['group_72_85']) if r['group_72_85'] else 0
            overall_mean = np.mean(list(r['subject_avg_pearsons'].values()))

            comparison.append({
                'Model': r['display_name'],
                'Subjects 1-71 (Mean)': f"{group_1_71_mean:.4f}",
                'Subjects 72-85 (Mean)': f"{group_72_85_mean:.4f}",
                'Overall (Mean)': f"{overall_mean:.4f}",
            })

        import pandas as pd
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))

        csv_path = os.path.join(args.output_dir, 'comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n✓ 对比结果已保存: {csv_path}")
    else:
        print("\n警告: 没有成功评估任何模型")

    print("\n完成！")


if __name__ == '__main__':
    main()
