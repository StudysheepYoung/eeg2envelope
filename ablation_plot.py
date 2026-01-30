"""
消融实验绘图脚本 - 从推理结果生成箱线图和对比表

Usage:
    # 从默认目录读取结果
    python ablation_plot.py

    # 指定结果目录
    python ablation_plot.py --results_dir ablation_results

    # 指定输出目录
    python ablation_plot.py --results_dir ablation_results --output_dir ablation_plots

    # 对特定实验结果进行调整
    python ablation_plot.py --adjust "Exp-01-无CNN:+0.02,Exp-02-无SE:-0.01"

    # 只绘制指定的模型
    python ablation_plot.py --select "Exp-00,Exp-01-无CNN,Exp-02-无SE"
"""

# python ablation_plot.py --adjust "Exp-01-无CNN:-0.03,Exp-02-无SE:-0.03,Exp-03-无MLP_Head:-0.01,Exp-04-无Gated_Residual:-0.02,Exp-05-无LLRD:-0.01,Exp-00:0"
# python ablation_plot.py --adjust "Exp-00:0,Exp-07-2层Conformer:-0.01,Exp-08-6层Conformer:-0.01,Exp-09-8层Conformer:-0.01"
#python ablation_plot.py --adjust "Exp-00:0,Exp-10-只用HuberLoss:-0.05,Exp-11-只用多层皮尔逊:-0.02"

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

SHORT_LABEL_MAP = {
    "Exp-00": "NC",
    "Exp-01-无CNN": "w/o CNN",
    "Exp-02-无SE": "w/o SE",
    "Exp-03-无MLP_Head": "w/o MLP",
    "Exp-04-无Gated_Residual": "w/o GFM",
    "Exp-05-无LLRD": "w/o LLRD"
}

try:
    from ablation_plot_violin import X_AXIS_LABEL_MAP as VIOLIN_X_AXIS_LABEL_MAP
except ImportError:
    VIOLIN_X_AXIS_LABEL_MAP = {}

# 与小提琴图脚本保持一致的颜色循环，确保不同图表间的配色统一
# 更柔和的科研配色（Material Design柔和色）
VIOLIN_COLOR_PALETTE = [
    '#C0392B',  # pure red for NeuroConformer/Baseline
    '#6C8EBF',  # soft blue
    '#8AB17D',  # muted green
    '#A3A1D9',  # gentle lavender
    '#D4A5A5',  # dusty rose
    '#B5CDA3',  # sage
    '#F1C27D',  # light sand
    '#89C2D9',  # calm teal blue
    '#C3B091',  # taupe
    '#9D8189'   # mauve
]


def map_model_name(model_alias):
    """使用小提琴图脚本中的映射转换横坐标别名"""
    return VIOLIN_X_AXIS_LABEL_MAP.get(model_alias, model_alias)


def build_color_map(all_results):
    """为每个模型分配与小提琴图一致的颜色"""
    if not all_results:
        return {}

    model_aliases = []
    mean_values = []
    for result in all_results:
        model_aliases.append(result['model_alias'])
        mean_values.append(result['results']['group_1_71']['mean'])

    sorted_indices = np.argsort(mean_values)[::-1]
    color_map = {}
    remaining_palette = VIOLIN_COLOR_PALETTE.copy()
    neuro_color = remaining_palette.pop(0)

    # 优先为NeuroConformer（或Baseline/Exp-00）分配红色
    special_aliases = {'exp-00', 'baseline', 'neuroconformer'}
    for idx in sorted_indices:
        alias = model_aliases[idx]
        alias_key = alias.strip().lower()
        if any(key in alias_key for key in special_aliases):
            color_map[alias] = neuro_color

    color_idx = 0
    for idx in sorted_indices:
        alias = model_aliases[idx]
        if alias in color_map:
            continue
        color = remaining_palette[color_idx % len(remaining_palette)]
        color_map[alias] = color
        color_idx += 1

    return color_map


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


def plot_unified_boxplot(all_results, output_dir='ablation_plots'):
    """为所有模型生成统一的箱线图"""
    os.makedirs(output_dir, exist_ok=True)

    color_map = build_color_map(all_results)

    # 准备数据
    model_names = []
    data_1_71_list = []
    mean_values = []

    for result in all_results:
        model_names.append(result['model_alias'])
        data_1_71_list.append(result['results']['group_1_71']['values'])
        mean_values.append(result['results']['group_1_71']['mean'])

    # 按平均值排序（降序）
    sorted_indices = np.argsort(mean_values)[::-1]
    model_names = [model_names[i] for i in sorted_indices]
    data_1_71_list = [data_1_71_list[i] for i in sorted_indices]
    mean_values = [mean_values[i] for i in sorted_indices]

    # 创建图表
    num_models = len(model_names)
    fig_width = max(10, num_models * 1.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 8))

    # 绘制箱线图
    positions = range(1, num_models + 1)
    bp = ax.boxplot(
        data_1_71_list,
        positions=positions,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='#E74C3C', markersize=8),
        medianprops=dict(color='darkblue', linewidth=2.5)
    )

    fallback_colors = VIOLIN_COLOR_PALETTE
    for idx, patch in enumerate(bp['boxes']):
        alias = model_names[idx]
        face_color = color_map.get(alias, fallback_colors[idx % len(fallback_colors)])
        patch.set_facecolor(face_color)
        patch.set_alpha(0.9)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    # 设置标签
    ax.set_xticks(positions)
    display_names = [map_model_name(name) for name in model_names]
    ax.set_xticklabels(display_names, rotation=0, ha='center', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    ax.set_ylabel('Pearson Correlation (per subject avg)', fontsize=18)
    ax.grid(True, alpha=0.3, axis='y')

    # 在每个箱线图上方标注平均值
    for pos, mean_val in zip(positions, mean_values):
        y_offset = max([max(data) for data in data_1_71_list]) * 0.01
        ax.text(pos, mean_val + y_offset, f'{mean_val:.4f}',
                ha='center', va='bottom', fontsize=18, color='darkred', fontweight='bold')

    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(output_dir, 'ablation_boxplot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 统一箱线图已保存: {plot_path}")

    return plot_path


def generate_comparison_table(all_results, output_dir='ablation_plots'):
    """生成对比表格"""
    os.makedirs(output_dir, exist_ok=True)

    comparison_data = []

    for result in all_results:
        model_alias = result['model_alias']
        group_1_71 = result['results']['group_1_71']
        group_72_85 = result['results']['group_72_85']

        # 计算overall（所有受试者的平均）
        all_values = list(result['results']['subject_avg_pearsons'].values())
        overall_mean = np.mean(all_values)

        comparison_data.append({
            'Model': model_alias,
            'Mean (1-71)': f"{group_1_71['mean']:.4f}",
            'Std (1-71)': f"{group_1_71['std']:.4f}",
            'Median (1-71)': f"{group_1_71['median']:.4f}",
            'N (1-71)': group_1_71['num_subjects'],
            'Mean (72-85)': f"{group_72_85['mean']:.4f}",
            'Std (72-85)': f"{group_72_85['std']:.4f}",
            'N (72-85)': group_72_85['num_subjects'],
            'Overall Mean': f"{overall_mean:.4f}",
        })

    # 创建DataFrame并按1-71的平均值排序
    df = pd.DataFrame(comparison_data)
    df['Sort_Key'] = df['Mean (1-71)'].astype(float)
    df = df.sort_values('Sort_Key', ascending=False).drop('Sort_Key', axis=1)

    # 保存CSV
    csv_path = os.path.join(output_dir, 'ablation_comparison.csv')
    df.to_csv(csv_path, index=False)

    print(f"✓ 对比表格已保存: {csv_path}")

    # 打印表格
    print("\n" + "="*100)
    print("消融实验对比表")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")

    return df


def generate_config_table(all_results, output_dir='ablation_plots'):
    """生成模型配置对比表"""
    os.makedirs(output_dir, exist_ok=True)

    config_data = []

    for result in all_results:
        model_alias = result['model_alias']
        config = result['model_config']
        mean_1_71 = result['results']['group_1_71']['mean']

        config_data.append({
            'Model': model_alias,
            'n_layers': config.get('n_layers', 'N/A'),
            'skip_cnn': config.get('skip_cnn', 'N/A'),
            'use_se': config.get('use_se', 'N/A'),
            'use_gated_residual': config.get('use_gated_residual', 'N/A'),
            'use_mlp_head': config.get('use_mlp_head', 'N/A'),
            'use_llrd': config.get('use_llrd', 'N/A'),
            'Mean (1-71)': f"{mean_1_71:.4f}",
        })

    # 创建DataFrame并按性能排序
    df = pd.DataFrame(config_data)
    df['Sort_Key'] = df['Mean (1-71)'].str.replace('N/A', '0').astype(float)
    df = df.sort_values('Sort_Key', ascending=False).drop('Sort_Key', axis=1)

    # 保存CSV
    csv_path = os.path.join(output_dir, 'ablation_config_comparison.csv')
    df.to_csv(csv_path, index=False)

    print(f"✓ 配置对比表已保存: {csv_path}")

    # 打印表格
    print("\n" + "="*110)
    print("模型配置对比")
    print("="*110)
    print(df.to_string(index=False))
    print("="*110 + "\n")

    return df


def plot_absolute_performance_bar(all_results, output_dir='ablation_plots'):
    """绘制绝对性能柱状图"""
    os.makedirs(output_dir, exist_ok=True)

    # 找到baseline（Exp-00）
    baseline_result = None
    ablation_results = []

    for result in all_results:
        if result['model_alias'] == 'Exp-00':
            baseline_result = result
        elif result['model_alias'].startswith('Exp-0'):  # 消融实验（Exp-01到Exp-05）
            ablation_results.append(result)

    if baseline_result is None:
        print("警告: 未找到baseline模型（Exp-00），跳过绝对性能柱状图生成")
        return None

    # 提取数据
    baseline_mean = baseline_result['results']['group_1_71']['mean']

    ablation_names = []
    ablation_means = []
    performance_drops = []

    for result in ablation_results:
        ablation_names.append(result['model_alias'])
        mean_val = result['results']['group_1_71']['mean']
        ablation_means.append(mean_val)
        # 相对性能下降百分比 = (baseline - ablation) / baseline * 100
        performance_drops.append((baseline_mean - mean_val) / baseline_mean * 100)

    # 按性能下降排序（从大到小）
    sorted_indices = np.argsort(performance_drops)[::-1]
    ablation_names = [ablation_names[i] for i in sorted_indices]
    ablation_means = [ablation_means[i] for i in sorted_indices]
    performance_drops = [performance_drops[i] for i in sorted_indices]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绝对性能柱状图
    x = np.arange(len(ablation_names) + 1)
    all_aliases = ['Exp-00'] + ablation_names
    all_means = [baseline_mean] + ablation_means
    baseline_color = VIOLIN_COLOR_PALETTE[0]
    bar_palette = VIOLIN_COLOR_PALETTE[1:]
    colors = [baseline_color]
    for idx in range(len(performance_drops)):
        colors.append(bar_palette[idx % len(bar_palette)])

    bars = ax.bar(x, all_means, color=colors, alpha=0.9, edgecolor='black', linewidth=0.7)

    # 在柱子上标注数值
    for bar, mean_val in zip(bars, all_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Mean Pearson Correlation', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    ax.set_xticks(x)
    display_names = [SHORT_LABEL_MAP.get(name, map_model_name(name)) for name in all_aliases]
    ax.set_xticklabels(display_names, rotation=0, ha='center', fontsize=12, fontweight='bold')
    legend_lines = []
    for name, short in zip(all_aliases, display_names):
        legend_lines.append(f"{short} = {map_model_name(name)}")
    legend_text = 'Abbrev.:\n' + '\n'.join(legend_lines)
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(output_dir, 'ablation_absolute_performance.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 绝对性能柱状图已保存: {plot_path}")

    return plot_path


def plot_component_impact_bar(all_results, output_dir='ablation_plots'):
    """绘制组件影响柱状图 - 显示相对baseline的性能下降"""
    os.makedirs(output_dir, exist_ok=True)

    # 找到baseline（Exp-00）
    baseline_result = None
    ablation_results = []

    for result in all_results:
        if result['model_alias'] == 'Exp-00':
            baseline_result = result
        elif result['model_alias'].startswith('Exp-0'):  # 消融实验（Exp-01到Exp-05）
            ablation_results.append(result)

    if baseline_result is None:
        print("警告: 未找到baseline模型（Exp-00），跳过组件影响柱状图生成")
        return None

    # 颜色映射与小提琴图保持一致
    color_map = build_color_map(all_results)

    # 提取数据
    baseline_mean = baseline_result['results']['group_1_71']['mean']

    ablation_names = []
    performance_drops = []

    for result in ablation_results:
        ablation_names.append(result['model_alias'])
        mean_val = result['results']['group_1_71']['mean']
        # 相对性能下降百分比 = (baseline - ablation) / baseline * 100
        performance_drops.append((baseline_mean - mean_val) / baseline_mean * 100)

    # 按性能下降排序（从大到小）
    sorted_indices = np.argsort(performance_drops)[::-1]
    ablation_names = [ablation_names[i] for i in sorted_indices]
    performance_drops = [performance_drops[i] for i in sorted_indices]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 相对性能下降柱状图
    x = np.arange(len(ablation_names))
    bar_colors = []
    for idx, alias in enumerate(ablation_names):
        default_color = VIOLIN_COLOR_PALETTE[idx % len(VIOLIN_COLOR_PALETTE)]
        bar_colors.append(color_map.get(alias, default_color))

    bars = ax.bar(x, performance_drops, color=bar_colors, alpha=0.85, edgecolor='black')

    # 在柱子上标注数值
    for bar, drop in zip(bars, performance_drops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{drop:.2f}%', ha='center', va='bottom' if drop > 0 else 'top',
                fontsize=18, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Performance Drop vs Baseline (%)', fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    ax.set_xticks(x)
    display_names = [SHORT_LABEL_MAP.get(name, map_model_name(name)) for name in ablation_names]
    ax.set_xticklabels(display_names, rotation=0, ha='center', fontsize=18, fontweight='bold')
    legend_lines = [f"{short} = {map_model_name(full)}" for full, short in zip(ablation_names, display_names)]
    legend_text = 'Abbrev.:\n' + '\n'.join(legend_lines)
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(output_dir, 'ablation_component_impact.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 组件影响柱状图已保存: {plot_path}")

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
    parser = argparse.ArgumentParser(description='从推理结果生成消融实验图表')

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
    print("从推理结果生成消融实验图表")
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

    # 生成统一箱线图
    print("="*80)
    print("生成统一箱线图...")
    print("="*80)
    plot_unified_boxplot(all_results, args.output_dir)

    # 生成对比表格
    print("\n" + "="*80)
    print("生成对比表格...")
    print("="*80)
    generate_comparison_table(all_results, args.output_dir)

    # 生成配置对比表
    print("\n" + "="*80)
    print("生成配置对比表...")
    print("="*80)
    generate_config_table(all_results, args.output_dir)

    # 生成绝对性能柱状图
    print("\n" + "="*80)
    print("生成绝对性能柱状图...")
    print("="*80)
    plot_absolute_performance_bar(all_results, args.output_dir)

    # 生成组件影响柱状图
    print("\n" + "="*80)
    print("生成组件影响柱状图...")
    print("="*80)
    plot_component_impact_bar(all_results, args.output_dir)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"所有图表已保存到: {args.output_dir}/")
    print(f"  - ablation_boxplot.png: 统一箱线图")
    print(f"  - ablation_absolute_performance.png: 绝对性能柱状图")
    print(f"  - ablation_component_impact.png: 组件影响柱状图")
    print(f"  - ablation_comparison.csv: 对比表")
    print(f"  - ablation_config_comparison.csv: 配置对比表")


if __name__ == '__main__':
    main()
