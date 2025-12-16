"""
消融实验推理脚本 - 只进行模型推理并保存结果

Usage:
    # 测试指定实验
    python ablation_inference.py --models Exp-00 Exp-01 Exp-02

    # 测试所有预定义实验
    python ablation_inference.py --models Exp-00 Exp-01-无CNN Exp-02-无SE Exp-03-无MLP_Head Exp-04-无Gated_Residual Exp-05-无LLRD
"""

import torch
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models.FFT_block_conformer_v2 import Decoder
from util.cal_pearson import pearson_metric
from test_model import load_test_data, segment_data


# ========== 模型别名配置 ==========
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

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'args' in checkpoint:
        args_dict = checkpoint['args']
    else:
        raise ValueError("Checkpoint中没有找到'args'字段")

    # 处理args可能是dict的情况
    if isinstance(args_dict, dict):
        get_arg = lambda key, default: args_dict.get(key, default)
    else:
        get_arg = lambda key, default: getattr(args_dict, key, default)

    # 创建模型
    model = Decoder(
        in_channel=get_arg('in_channel', 64),
        d_model=get_arg('d_model', 256),
        d_inner=get_arg('d_inner', 1024),
        n_head=get_arg('n_head', 4),
        n_layers=get_arg('n_layers', 4),
        fft_conv1d_kernel=get_arg('fft_conv1d_kernel', (9, 1)),
        fft_conv1d_padding=get_arg('fft_conv1d_padding', (4, 0)),
        dropout=get_arg('dropout', 0.4),
        g_con=get_arg('g_con', True),
        within_sub_num=85,
        conv_kernel_size=get_arg('conv_kernel_size', 31),
        use_relative_pos=get_arg('use_relative_pos', True),
        use_macaron_ffn=get_arg('use_macaron_ffn', True),
        use_sinusoidal_pos=get_arg('use_sinusoidal_pos', False),
        use_gated_residual=get_arg('use_gated_residual', True),
        use_mlp_head=get_arg('use_mlp_head', True),
        gradient_scale=get_arg('gradient_scale', 1.0),
        skip_cnn=get_arg('skip_cnn', True),
        use_se=get_arg('use_se', True),
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

    # 转换args_dict为dict格式
    if not isinstance(args_dict, dict):
        args_dict = vars(args_dict)

    print(f"✓ 模型加载成功")
    print(f"  配置: n_layers={get_arg('n_layers', 4)}, skip_cnn={get_arg('skip_cnn', True)}, "
          f"use_se={get_arg('use_se', True)}, use_gated_residual={get_arg('use_gated_residual', True)}")

    return model, args_dict


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
        subject_avg_pearsons[sub_id] = float(np.mean(subject_pearsons[sub_id]))

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


def main():
    parser = argparse.ArgumentParser(description='消融实验推理脚本')

    # 模型指定方式
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='模型别名列表')
    parser.add_argument('--folders', type=str, nargs='+', default=None,
                       help='checkpoint文件夹路径列表')

    # 数据和输出
    parser.add_argument('--split_data_dir', type=str,
                       default='/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data/split_data',
                       help='split_data目录')
    parser.add_argument('--test_data_dir', type=str,
                       default='/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data/test_data',
                       help='test_data目录')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='输出目录')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备号')

    args = parser.parse_args()

    # 确定要测试的模型
    if args.models is not None:
        models_to_test = args.models
    elif args.folders is not None:
        models_to_test = args.folders
    else:
        print("错误: 必须指定 --models 或 --folders")
        return

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载测试数据（所有模型共用）
    print("\n" + "="*80)
    print("加载测试数据...")
    print("="*80)
    test_samples = load_test_data(args.split_data_dir, args.test_data_dir)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 测试每个模型
    all_results = []
    model = None  # 初始化model变量

    for model_alias in models_to_test:
        print("\n" + "="*80)
        print(f"测试: {model_alias}")
        print("="*80)

        # 解析路径或别名
        if model_alias in DEFAULT_ALIASES:
            folder_path = DEFAULT_ALIASES[model_alias]
            display_name = model_alias
        else:
            folder_path = model_alias
            display_name = os.path.basename(model_alias)

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

            # 评估模型
            eval_results = evaluate_model_only(model, test_samples, device)

            # 构建完整结果
            result = {
                'model_alias': display_name,
                'folder_path': folder_path,
                'checkpoint_path': checkpoint_path,
                'model_config': {
                    'n_layers': model_args.get('n_layers'),
                    'd_model': model_args.get('d_model'),
                    'n_head': model_args.get('n_head'),
                    'skip_cnn': model_args.get('skip_cnn'),
                    'use_se': model_args.get('use_se'),
                    'use_gated_residual': model_args.get('use_gated_residual'),
                    'use_mlp_head': model_args.get('use_mlp_head'),
                    'use_llrd': model_args.get('use_llrd'),
                },
                'results': {
                    'subject_avg_pearsons': eval_results['subject_avg_pearsons'],
                    'group_1_71': {
                        'num_subjects': len(eval_results['group_1_71']),
                        'mean': float(np.mean(eval_results['group_1_71'])),
                        'std': float(np.std(eval_results['group_1_71'])),
                        'median': float(np.median(eval_results['group_1_71'])),
                        'values': eval_results['group_1_71'],
                    },
                    'group_72_85': {
                        'num_subjects': len(eval_results['group_72_85']),
                        'mean': float(np.mean(eval_results['group_72_85'])) if eval_results['group_72_85'] else 0,
                        'std': float(np.std(eval_results['group_72_85'])) if eval_results['group_72_85'] else 0,
                        'median': float(np.median(eval_results['group_72_85'])) if eval_results['group_72_85'] else 0,
                        'values': eval_results['group_72_85'],
                    },
                }
            }

            all_results.append(result)

            # 保存单个模型的结果
            single_result_path = os.path.join(args.output_dir, f'{display_name}_results.json')
            with open(single_result_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ 单个模型结果已保存: {single_result_path}")

        except Exception as e:
            print(f"错误: 测试失败 - {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # 保存所有结果的汇总
    if len(all_results) > 0:
        summary_path = os.path.join(args.output_dir, 'ablation_all_results.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'num_models': len(all_results),
                'models': all_results
            }, f, indent=2)

        print("\n" + "="*80)
        print("推理完成！")
        print("="*80)
        print(f"✓ 共测试 {len(all_results)} 个模型")
        print(f"✓ 汇总结果已保存: {summary_path}")
        print(f"✓ 单个模型结果保存在: {args.output_dir}/")
    else:
        print("\n警告: 没有成功评估任何模型")

    print("\n完成！")


if __name__ == '__main__':
    main()
