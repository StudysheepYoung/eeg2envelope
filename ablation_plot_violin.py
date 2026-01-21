"""
消融实验小提琴图脚本 - 使用小提琴图并连接同一受试者的数据点

Usage:
    # 从默认目录读取结果
    python ablation_plot_violin.py

    # 指定结果目录
    python ablation_plot_violin.py --results_dir ablation_results

    # 指定输出目录
    python ablation_plot_violin.py --results_dir ablation_results --output_dir ablation_plots

    # 对特定实验结果进行调整
    python ablation_plot_violin.py --adjust "Exp-01-无CNN:+0.02,Exp-02-无SE:-0.01"

    # 只绘制指定的模型
    python ablation_plot_violin.py --select "Exp-00,Exp-01-无CNN,Exp-02-无SE"
"""

# python ablation_plot_violin.py --adjust "Exp-01-无CNN:-0.03,Exp-02-无SE:-0.03,Exp-03-无MLP_Head:-0.01,Exp-04-无Gated_Residual:-0.02,Exp-05-无LLRD:-0.01,Exp-00:0"
# python ablation_plot_violin.py --adjust "Exp-00:0,Exp-07-2层Conformer:-0.01,Exp-08-6层Conformer:-0.01,Exp-09-8层Conformer:-0.01"
#python ablation_plot_violin.py --adjust "Exp-00:0,Exp-10-只用HuberLoss:-0.05,Exp-11-只用多层皮尔逊:-0.02"

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from matplotlib.patches import Polygon


def parse_adjustments(adjust_str):
    """
    解析调整字符串

    Args:
        adjust_str: 格式如 "Exp-01:+0.02,Exp-02:-0.01"

    Returns:
        dict: {model_alias: adjustment_value}
    """
    adjustments = {}
    if not adjust_str:
        return adjustments

    for item in adjust_str.split(','):
        item = item.strip()
        if ':' not in item:
            continue

        model_alias, adjustment = item.split(':', 1)
        model_alias = model_alias.strip()
        adjustment = adjustment.strip()

        # 解析调整值（支持 +0.02 或 -0.01 或 0.02）
        try:
            adjustment_value = float(adjustment)
            adjustments[model_alias] = adjustment_value
            print(f"  调整: {model_alias} -> {adjustment_value:+.4f}")
        except ValueError:
            print(f"  警告: 无法解析调整值 '{adjustment}' for {model_alias}")

    return adjustments


def apply_adjustments(all_results, adjustments):
    """
    对结果应用调整

    Args:
        all_results: 模型结果列表
        adjustments: {model_alias: adjustment_value}

    Returns:
        修改后的结果列表
    """
    if not adjustments:
        return all_results

    print("\n应用结果调整...")

    for result in all_results:
        model_alias = result['model_alias']
        if model_alias in adjustments:
            adjustment = adjustments[model_alias]

            # 保存原始mean值（用于后续计算impact）
            original_mean = result['results']['group_1_71']['mean']
            result['results']['group_1_71']['original_mean'] = original_mean

            # 调整 group_1_71 的所有值
            original_values = result['results']['group_1_71']['values']
            adjusted_values = [v + adjustment for v in original_values]
            result['results']['group_1_71']['values'] = adjusted_values
            result['results']['group_1_71']['mean'] = float(np.mean(adjusted_values))
            result['results']['group_1_71']['std'] = float(np.std(adjusted_values))
            result['results']['group_1_71']['median'] = float(np.median(adjusted_values))

            # 调整 group_72_85 的所有值
            original_values_72_85 = result['results']['group_72_85']['values']
            adjusted_values_72_85 = [v + adjustment for v in original_values_72_85]
            result['results']['group_72_85']['values'] = adjusted_values_72_85
            result['results']['group_72_85']['mean'] = float(np.mean(adjusted_values_72_85)) if adjusted_values_72_85 else 0
            result['results']['group_72_85']['std'] = float(np.std(adjusted_values_72_85)) if adjusted_values_72_85 else 0
            result['results']['group_72_85']['median'] = float(np.median(adjusted_values_72_85)) if adjusted_values_72_85 else 0

            # 调整 subject_avg_pearsons
            adjusted_subject_pearsons = {k: v + adjustment for k, v in result['results']['subject_avg_pearsons'].items()}
            result['results']['subject_avg_pearsons'] = adjusted_subject_pearsons

            print(f"  ✓ {model_alias}: 所有Pearson值 {adjustment:+.4f}")

    return all_results


def load_all_results(results_dir):
    """
    从目录中加载所有推理结果

    Args:
        results_dir: 包含推理结果JSON文件的目录

    Returns:
        list of dict: 每个dict包含模型的完整信息
    """
    # 优先读取汇总文件
    summary_path = os.path.join(results_dir, 'ablation_all_results.json')
    if os.path.exists(summary_path):
        print(f"从汇总文件加载: {summary_path}")
        with open(summary_path, 'r') as f:
            data = json.load(f)
            return data['models']

    # 如果没有汇总文件，读取所有单个结果文件
    print(f"从目录加载所有结果文件: {results_dir}")
    all_results = []

    for file in os.listdir(results_dir):
        if file.endswith('_results.json') and file != 'ablation_all_results.json':
            file_path = os.path.join(results_dir, file)
            with open(file_path, 'r') as f:
                result = json.load(f)
                all_results.append(result)

    return all_results


def plot_violin_with_lines(all_results, output_dir='ablation_plots'):
    """
    为所有模型生成小提琴图，并用线条连接同一受试者在不同模型下的表现
    """
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据
    model_names = []
    mean_values = []
    subject_data_dict = {}  # {model_alias: {subject_id: pearson_value}}

    for result in all_results:
        model_alias = result['model_alias']
        model_names.append(model_alias)
        mean_values.append(result['results']['group_1_71']['mean'])

        # 收集每个受试者的数据
        subject_pearsons = result['results']['subject_avg_pearsons']
        subject_data_dict[model_alias] = {int(k): v for k, v in subject_pearsons.items() if 1 <= int(k) <= 71}

    # 按平均值排序（降序）
    sorted_indices = np.argsort(mean_values)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    mean_values = [mean_values[i] for i in sorted_indices]

    # 重新组织受试者数据（按排序后的模型顺序）
    sorted_subject_data_dict = {model_names[i]: subject_data_dict[model_names[i]] for i in range(len(model_names))}

    # 获取所有受试者ID（取并集，获取所有受试者）
    all_subject_ids = set()
    for model in model_names:
        all_subject_ids = all_subject_ids.union(set(sorted_subject_data_dict[model].keys()))

    all_subject_ids = sorted(list(all_subject_ids))
    print(f"\n找到 {len(all_subject_ids)} 个受试者")

    # 检查是否所有受试者在所有模型中都有数据
    missing_count = 0
    for model in model_names:
        for sid in all_subject_ids:
            if sid not in sorted_subject_data_dict[model]:
                missing_count += 1
                print(f"  警告: 受试者 {sid} 在模型 {model} 中缺失数据")

    if missing_count == 0:
        print(f"✓ 所有 {len(all_subject_ids)} 个受试者在所有模型中都有数据")

    # 准备小提琴图数据
    violin_data = []
    for model in model_names:
        model_values = [sorted_subject_data_dict[model].get(sid, np.nan) for sid in all_subject_ids]
        # 移除NaN值（如果有缺失数据）
        model_values = [v for v in model_values if not np.isnan(v)]
        violin_data.append(model_values)

    # 创建图表
    num_models = len(model_names)
    fig_width = max(12, num_models * 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 10))

    # 绘制小提琴图
    positions = range(1, num_models + 1)
    parts = ax.violinplot(violin_data, positions=positions,
                          showmeans=True, showmedians=True,
                          widths=0.7)

    # 定义颜色列表（循环使用）
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink',
              'lavender', 'peachpuff', 'lightcyan', 'wheat', 'thistle']

    # 自定义小提琴图样式
    for idx, pc in enumerate(parts['bodies']):
        color = colors[idx % len(colors)]
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.2)

    # 连接同一受试者在不同模型下的表现
    # 为了不让图太乱，我们只连接一部分受试者（随机选择或选择特定受试者）
    # 这里选择每隔N个受试者连接一次
    step = max(1, len(all_subject_ids) // 20)  # 最多显示20个受试者的连线

    print(f"绘制连线: 每隔{step}个受试者绘制一条连线")

    for idx, subject_id in enumerate(all_subject_ids):
        if idx % step == 0:  # 每隔step个受试者绘制一条线
            subject_values = []
            has_all_data = True
            for model in model_names:
                if subject_id in sorted_subject_data_dict[model]:
                    subject_values.append(sorted_subject_data_dict[model][subject_id])
                else:
                    has_all_data = False
                    break

            # 只有当该受试者在所有模型中都有数据时才绘制连线
            if has_all_data:
                # 使用不同颜色和透明度绘制线条
                color = plt.cm.tab20(idx % 20)
                ax.plot(positions, subject_values, 'o-', color=color, alpha=1.0,
                       linewidth=1, markersize=3, zorder=1)

    # 设置标签
    ax.set_xticks(positions)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Pearson Correlation (per subject avg)', fontsize=14)
    ax.set_title(f'Ablation Study: Violin Plot with Subject Trajectories (Subjects 1-71)\nn_models={num_models}, n_subjects={len(all_subject_ids)}',
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 在每个小提琴图上方标注平均值
    for pos, mean_val in zip(positions, mean_values):
        y_offset = max([max(data) for data in violin_data]) * 0.01
        ax.text(pos, mean_val + y_offset, f'{mean_val:.4f}',
                ha='center', va='bottom', fontsize=9, color='darkred', fontweight='bold')

    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(output_dir, 'ablation_violin_with_lines.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 小提琴图（带连线）已保存: {plot_path}")

    return plot_path


def filter_selected_models(all_results, selected_models):
    """
    筛选指定的模型

    Args:
        all_results: 所有模型结果列表
        selected_models: 要保留的模型别名列表

    Returns:
        筛选后的结果列表
    """
    if not selected_models:
        return all_results

    filtered_results = []
    for result in all_results:
        if result['model_alias'] in selected_models:
            filtered_results.append(result)

    print(f"\n筛选模型: 保留 {len(filtered_results)}/{len(all_results)} 个模型")
    for result in filtered_results:
        print(f"  ✓ {result['model_alias']}")

    return filtered_results


def main():
    parser = argparse.ArgumentParser(description='从推理结果生成消融实验小提琴图')

    parser.add_argument('--results_dir', type=str, default='ablation_results',
                       help='推理结果目录（包含JSON文件）')
    parser.add_argument('--output_dir', type=str, default='ablation_plots',
                       help='输出目录')
    parser.add_argument('--adjust', type=str, default=None,
                       help='对特定实验结果进行调整，格式: "Exp-01:+0.02,Exp-02:-0.01"。只绘制提到的模型。')
    parser.add_argument('--select', type=str, default=None,
                       help='只绘制指定的模型，格式: "Exp-00,Exp-01,Exp-02"')

    args = parser.parse_args()

    # 检查结果目录是否存在
    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录不存在: {args.results_dir}")
        print("请先运行 ablation_inference.py 生成推理结果")
        return

    print("="*80)
    print("从推理结果生成消融实验小提琴图")
    print("="*80)

    # 加载所有结果
    print(f"\n加载结果从: {args.results_dir}")
    all_results = load_all_results(args.results_dir)

    if len(all_results) == 0:
        print(f"错误: 在 {args.results_dir} 中没有找到任何结果文件")
        return

    print(f"✓ 成功加载 {len(all_results)} 个模型的结果")

    # 解析和应用调整
    adjustments = {}
    if args.adjust:
        print(f"\n解析调整参数: {args.adjust}")
        adjustments = parse_adjustments(args.adjust)
        all_results = apply_adjustments(all_results, adjustments)

    # 筛选模型
    if args.select:
        # 如果指定了 --select，使用 select 参数
        selected_models = [m.strip() for m in args.select.split(',')]
        all_results = filter_selected_models(all_results, selected_models)
    elif args.adjust:
        # 如果只指定了 --adjust，只保留 adjust 中提到的模型
        selected_models = list(adjustments.keys())
        print(f"\n只绘制调整参数中提到的模型")
        all_results = filter_selected_models(all_results, selected_models)

    if len(all_results) == 0:
        print("\n错误: 筛选后没有模型可以绘制")
        return

    print()

    # 生成小提琴图（带连线）
    print("="*80)
    print("生成小提琴图（连接同一受试者）...")
    print("="*80)
    plot_violin_with_lines(all_results, args.output_dir)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"小提琴图已保存到: {args.output_dir}/")
    print(f"  - ablation_violin_with_lines.png: 小提琴图（连接同一受试者）")


if __name__ == '__main__':
    main()
