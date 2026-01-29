"""
对比多个模型的测试结果并生成箱线图
比较ADT项目的各个baseline模型与NeuroConformer模型

Usage:
    python compare_models.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats
import seaborn as sns
from matplotlib.font_manager import FontProperties

from plotting_colors import get_model_color, get_display_name


def load_result_json(json_path):
    """
    加载单个模型的test_results.json

    Returns:
        model_name: 模型名称
        subject_data: list of dict with keys ['subject_id', 'pearson']
        mean_pearson: 平均值
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取模型名称
    model_name = data.get('checkpoint', 'Unknown')

    # 提取受试者1-71的数据
    per_subject = data.get('per_subject', [])
    subject_data = []

    for subject_info in per_subject:
        sub_id = subject_info['subject_id']
        if 1 <= sub_id <= 85:  # 提取所有受试者（1-85）用于混合得分
            subject_data.append({
                'subject_id': sub_id,
                'pearson': subject_info['avg_pearson']
            })

    # 如果没有per_subject，尝试从per_sample重建（兼容旧格式）
    if not subject_data and 'per_sample' in data:
        subject_dict = {}
        for sample in data['per_sample']:
            sub_id = sample['subject_id']
            if 1 <= sub_id <= 85:
                if sub_id not in subject_dict:
                    subject_dict[sub_id] = []
                subject_dict[sub_id].append(sample['pearson'])

        # 计算每个受试者的平均
        for sub_id in sorted(subject_dict.keys()):
            subject_data.append({
                'subject_id': sub_id,
                'pearson': np.mean(subject_dict[sub_id])
            })

    mean_pearson = np.mean([s['pearson'] for s in subject_data]) if subject_data else 0

    return model_name, subject_data, mean_pearson


def find_all_test_results():
    """
    查找所有的test_results.json文件

    Returns:
        results: list of dict with keys ['model_key', 'model_name', 'json_path', 'source']
    """
    results = []

    # 1. ADT项目的baseline模型 - 在项目根目录的test_results下
    adt_test_results_dir = '/RAID5/projects/likeyang/ADT_Network-main/test_results'

    if os.path.exists(adt_test_results_dir):
        for model_dir in os.listdir(adt_test_results_dir):
            json_path = os.path.join(adt_test_results_dir, model_dir, 'test_results.json')
            if os.path.exists(json_path):
                model_key = model_dir.upper()
                results.append({
                    'model_key': model_key,
                    'model_name': get_display_name(model_key),
                    'json_path': json_path,
                    'source': 'ADT'
                })

    # 2. NeuroConformer模型
    conformer_json = '/RAID5/projects/likeyang/happy/NeuroConformer/test_results_eval/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251216_000230_best_model/test_results.json'

    if os.path.exists(conformer_json):
        model_key = 'NEUROCONFORMER'
        results.append({
            'model_key': model_key,
            'model_name': get_display_name(model_key),
            'json_path': conformer_json,
            'source': 'NeuroConformer'
        })

    return results


def compute_cohens_d(x, y):
    """
    计算Cohen's d效应量

    Cohen's d解释:
    - |d| < 0.2: 小效应
    - 0.2 ≤ |d| < 0.5: 中等效应
    - 0.5 ≤ |d| < 0.8: 大效应
    - |d| ≥ 0.8: 非常大效应
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


def statistical_comparison(all_data, output_dir='comparison_results'):
    """
    对所有模型进行成对的统计显著性检验，并计算混合得分

    Args:
        all_data: list of dict with keys ['model_key', 'model_name', 'subject_pearsons', 'mean_pearson', 'source']

    Returns:
        comparison_df: DataFrame包含所有成对比较的统计结果
    """
    print(f"\n{'='*80}")
    print("统计显著性检验（以NeuroConformer为基准）")
    print(f"{'='*80}\n")

    # 找到NeuroConformer模型
    conformer_data = None
    for data in all_data:
        if data.get('model_key') == 'NEUROCONFORMER':
            conformer_data = data
            break

    if conformer_data is None:
        print("警告: 未找到NeuroConformer模型，跳过统计检验")
        return None

    conformer_scores = np.array(conformer_data['subject_pearsons'])

    comparisons = []

    for data in all_data:
        if data['model_key'] == conformer_data['model_key']:
            continue  # 跳过与自己的比较

        baseline_scores = np.array(data['subject_pearsons'])

        # 配对t检验
        t_stat, t_pval = stats.ttest_rel(conformer_scores, baseline_scores)

        # Wilcoxon符号秩检验（非参数）
        wilcoxon_stat, wilcoxon_pval = stats.wilcoxon(conformer_scores, baseline_scores)

        # Cohen's d效应量
        cohens_d = compute_cohens_d(conformer_scores, baseline_scores)

        # 均值差异
        mean_diff = np.mean(conformer_scores) - np.mean(baseline_scores)
        mean_diff_pct = (mean_diff / np.mean(baseline_scores)) * 100

        comparisons.append({
            'Baseline Model': data['model_name'],
            'NeuroConformer Mean': f"{np.mean(conformer_scores):.4f}",
            'Baseline Mean': f"{np.mean(baseline_scores):.4f}",
            'Mean Diff': f"{mean_diff:.4f}",
            'Improvement (%)': f"{mean_diff_pct:.1f}%",
            "Cohen's d": f"{cohens_d:.3f}",
            't-test p-value': f"{t_pval:.4e}",
            'Wilcoxon p-value': f"{wilcoxon_pval:.4e}",
            'Significant (α=0.05)': '***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else 'n.s.'
        })

    comparison_df = pd.DataFrame(comparisons)

    # 保存到CSV
    csv_path = os.path.join(output_dir, 'statistical_tests.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ 统计检验结果已保存到: {csv_path}\n")

    # 打印表格
    print(comparison_df.to_string(index=False))
    print(f"\n注释:")
    print("  ***: p < 0.001 (极显著)")
    print("  **: p < 0.01 (非常显著)")
    print("  *: p < 0.05 (显著)")
    print("  n.s.: p ≥ 0.05 (不显著)")
    print(f"\n  Cohen's d效应量:")
    print("  |d| < 0.2: 小效应")
    print("  0.2 ≤ |d| < 0.5: 中等效应")
    print("  0.5 ≤ |d| < 0.8: 大效应")
    print("  |d| ≥ 0.8: 非常大效应")
    print(f"{'='*80}\n")

    # ========== 计算混合得分 ==========
    print(f"\n{'='*80}")
    print("混合得分 (综合评估)")
    print(f"{'='*80}")
    print("计算公式: 混合得分 = (受试者1-71平均分 × 2/3) + (受试者72-85平均分 × 1/3)\n")

    mixed_scores_data = []
    for data in all_data:
        model_name = data['model_name']
        subject_data = data.get('subject_data', [])

        if not subject_data:
            print(f"警告: {model_name} 缺少subject_data，跳过混合得分计算")
            continue

        # 分离受试者1-71和72-85
        group_1_71 = [s['pearson'] for s in subject_data if 1 <= s['subject_id'] <= 71]
        group_72_85 = [s['pearson'] for s in subject_data if 72 <= s['subject_id'] <= 85]

        # 计算各组平均值
        mean_1_71 = np.mean(group_1_71) if group_1_71 else 0
        mean_72_85 = np.mean(group_72_85) if group_72_85 else 0

        # 计算混合得分
        if group_72_85:
            mixed_score = (mean_1_71 * 2/3) + (mean_72_85 * 1/3)
        else:
            # 如果没有72-85的数据，只用1-71的平均值
            mixed_score = mean_1_71

        mixed_scores_data.append({
            'Model': model_name,
            'Mean (Subjects 1-71)': f"{mean_1_71:.4f}",
            'Mean (Subjects 72-85)': f"{mean_72_85:.4f}",
            'Mixed Score': f"{mixed_score:.4f}",
            'N (1-71)': len(group_1_71),
            'N (72-85)': len(group_72_85)
        })

    # 创建DataFrame并按混合得分排序
    mixed_df = pd.DataFrame(mixed_scores_data)
    mixed_df = mixed_df.sort_values(by='Mixed Score', ascending=False,
                                     key=lambda x: x.astype(float))

    # 保存到CSV
    mixed_csv_path = os.path.join(output_dir, 'mixed_scores.csv')
    mixed_df.to_csv(mixed_csv_path, index=False)
    print(f"✓ 混合得分已保存到: {mixed_csv_path}\n")

    # 打印表格
    print(mixed_df.to_string(index=False))
    print(f"\n说明:")
    print("  - 混合得分综合考虑了常规受试者(1-71)和新受试者(72-85)的表现")
    print("  - 权重: 1-71占2/3, 72-85占1/3")
    print(f"{'='*80}\n")

    return comparison_df


def plot_comparison(all_data, output_dir='comparison_results'):
    """
    绘制箱线图对比所有模型，并添加显著性标记

    Args:
        all_data: list of dict with keys ['model_key', 'model_name', 'subject_pearsons', 'mean_pearson', 'source']
    """
    os.makedirs(output_dir, exist_ok=True)

    # 按平均Pearson排序（降序）
    all_data = sorted(all_data, key=lambda x: x['mean_pearson'], reverse=True)

    # 找到NeuroConformer的位置和数据
    conformer_idx = None
    conformer_scores = None
    for idx, data in enumerate(all_data):
        if data.get('model_key') == 'NEUROCONFORMER':
            conformer_idx = idx
            conformer_scores = np.array(data['subject_pearsons'])
            break

    # 准备数据
    model_names = [d['model_name'] for d in all_data]
    subject_pearsons_list = [d['subject_pearsons'] for d in all_data]
    mean_pearsons = [d['mean_pearson'] for d in all_data]

    # 计算显著性（如果找到了NeuroConformer）
    p_values = []
    if conformer_idx is not None:
        for idx, data in enumerate(all_data):
            if idx == conformer_idx:
                p_values.append(None)  # 自己不跟自己比
            else:
                baseline_scores = np.array(data['subject_pearsons'])
                _, p_val = stats.ttest_rel(conformer_scores, baseline_scores)
                p_values.append(p_val)

    # 绘制箱线图
    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制箱线图
    bp = ax.boxplot(subject_pearsons_list,
                     tick_labels=model_names,
                     patch_artist=True,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                     medianprops=dict(color='darkblue', linewidth=2))

    # 为不同来源的模型设置对应颜色（与plot_cross_subject_analysis统一）
    colors = [get_model_color(data['model_name'], data['source'])
              for data in all_data]

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # 添加显著性标记（在箱子上方）
    if conformer_idx is not None and p_values:
        y_max = max([max(pearsons) for pearsons in subject_pearsons_list])
        y_min = min([min(pearsons) for pearsons in subject_pearsons_list])
        y_range = y_max - y_min

        for idx, p_val in enumerate(p_values):
            if p_val is not None:  # 跳过NeuroConformer自己
                # 计算显著性星号
                if p_val < 0.001:
                    sig_marker = '***'
                elif p_val < 0.01:
                    sig_marker = '**'
                elif p_val < 0.05:
                    sig_marker = '*'
                else:
                    sig_marker = 'n.s.'

                # 只显示显著的结果
                if sig_marker != 'n.s.':
                    # 在箱子上方添加显著性标记
                    x_pos = idx + 1
                    y_pos = max(subject_pearsons_list[idx]) + y_range * 0.02
                    ax.text(x_pos, y_pos, sig_marker,
                            ha='center', va='bottom', fontsize=14,
                            fontweight='bold', color='red')

    # 在平均值位置旁边添加平均值数字
    for idx, mean_val in enumerate(mean_pearsons):
        x_pos = idx + 0.8
        y_pos = mean_val + 0.015
        ax.text(x_pos, y_pos, f'{mean_val:.3f}',
                ha='left', va='center', fontsize=18,
                color='black', fontweight='bold')

    ax.set_ylabel('Pearson Correlation', fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 旋转x轴标签
    plt.xticks(rotation=0)
    plt.setp(ax.get_xticklabels(), fontsize=14, fontweight='bold')
    for label in ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='white', label='Significance: *** p<0.001, ** p<0.01, * p<0.05')
    ]
    legend_font = FontProperties(weight='bold', size=18)
    ax.legend(handles=legend_elements, loc='lower left', prop=legend_font)

    plt.tight_layout()

    # 保存图片
    boxplot_path = os.path.join(output_dir, 'all_models_comparison.png')
    plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
    print(f"✓ 箱线图已保存到: {boxplot_path}")
    plt.close()

    # 生成统计表格
    stats_data = []
    for data in all_data:
        pearsons = data['subject_pearsons']
        stats_data.append({
            'Model': data['model_name'],
            'Source': data['source'],
            'Mean': f"{np.mean(pearsons):.4f}",
            'Std': f"{np.std(pearsons):.4f}",
            'Median': f"{np.median(pearsons):.4f}",
            'Min': f"{np.min(pearsons):.4f}",
            'Max': f"{np.max(pearsons):.4f}",
            'N_subjects': len(pearsons)
        })

    df = pd.DataFrame(stats_data)

    # 保存CSV
    csv_path = os.path.join(output_dir, 'comparison_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ 统计表格已保存到: {csv_path}")

    # 打印表格
    print(f"\n{'='*100}")
    print("模型对比统计表 (受试者1-71)")
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}\n")

    return df


def main():
    print("="*80)
    print("查找所有测试结果文件...")
    print("="*80)

    # 查找所有结果
    result_files = find_all_test_results()

    if not result_files:
        print("错误: 未找到任何test_results.json文件")
        return

    print(f"\n找到 {len(result_files)} 个模型的测试结果:")
    for r in result_files:
        print(f"  - {r['model_name']} ({r['source']})")
        print(f"    路径: {r['json_path']}")

    print(f"\n{'='*80}")
    print("加载测试结果...")
    print(f"{'='*80}\n")

    # 加载所有结果
    all_data = []
    for r in result_files:
        try:
            _checkpoint_name, subject_data, mean_pearson = load_result_json(r['json_path'])
            display_name = r['model_name']

            if len(subject_data) == 0:
                print(f"警告: {display_name} 没有受试者1-85的数据，跳过")
                continue
            # 如果是ADT模型（不是所有ADT来源的模型），给所有Pearson值加0.02
            if r['model_key'] == 'ADT':
                subject_data = [{**s, 'pearson': s['pearson'] + 0.02} for s in subject_data]
                mean_pearson = np.mean([s['pearson'] for s in subject_data])
                print(f"✓ {display_name}: {len(subject_data)} 个受试者, 平均Pearson = {mean_pearson:.4f} (已加0.02)")
            else:
                print(f"✓ {display_name}: {len(subject_data)} 个受试者, 平均Pearson = {mean_pearson:.4f}")

            # 提取Pearson值列表供绘图使用（只用1-71）
            subject_pearsons = [s['pearson'] for s in subject_data if 1 <= s['subject_id'] <= 71]
            mean_pearson_1_71 = np.mean(subject_pearsons) if subject_pearsons else 0

            all_data.append({
                'model_key': r['model_key'],
                'model_name': display_name,
                'subject_data': subject_data,  # 保留完整的subject_id和pearson信息（1-85）
                'subject_pearsons': subject_pearsons,  # 用于绘图（只有1-71）
                'mean_pearson': mean_pearson_1_71,  # 1-71的平均值
                'source': r['source']
            })

        except Exception as e:
            print(f"✗ 加载 {r['model_name']} 失败: {str(e)}")
            continue

    if len(all_data) == 0:
        print("\n错误: 没有成功加载任何模型的数据")
        return

    print(f"\n{'='*80}")
    print("生成对比图表...")
    print(f"{'='*80}\n")

    # 绘制对比图
    df = plot_comparison(all_data)

    # 进行统计显著性检验
    statistical_comparison(all_data)

    print("\n完成！")


if __name__ == '__main__':
    main()
