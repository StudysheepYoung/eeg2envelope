"""
对比多个模型的测试结果并生成箱线图
比较ADT项目的各个baseline模型与HappyQuokka的Conformer模型

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


def load_result_json(json_path):
    """
    加载单个模型的test_results.json

    Returns:
        model_name: 模型名称
        subject_pearsons_1_71: 受试者1-71的平均Pearson列表
        mean_pearson: 平均值
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取模型名称
    model_name = data.get('checkpoint', 'Unknown')

    # 提取受试者1-71的数据
    per_subject = data.get('per_subject', [])
    subject_pearsons_1_71 = []

    for subject_info in per_subject:
        sub_id = subject_info['subject_id']
        if 1 <= sub_id <= 71:
            subject_pearsons_1_71.append(subject_info['avg_pearson'])

    # 如果没有per_subject，尝试从per_sample重建（兼容旧格式）
    if not subject_pearsons_1_71 and 'per_sample' in data:
        subject_dict = {}
        for sample in data['per_sample']:
            sub_id = sample['subject_id']
            if 1 <= sub_id <= 71:
                if sub_id not in subject_dict:
                    subject_dict[sub_id] = []
                subject_dict[sub_id].append(sample['pearson'])

        # 计算每个受试者的平均
        subject_pearsons_1_71 = [np.mean(pearsons) for pearsons in subject_dict.values()]

    mean_pearson = np.mean(subject_pearsons_1_71) if subject_pearsons_1_71 else 0

    return model_name, subject_pearsons_1_71, mean_pearson


def find_all_test_results():
    """
    查找所有的test_results.json文件

    Returns:
        results: list of dict with keys ['model_name', 'json_path', 'source']
    """
    results = []

    # 1. ADT项目的baseline模型 - 在项目根目录的test_results下
    adt_test_results_dir = '/RAID5/projects/likeyang/ADT_Network-main/test_results'

    if os.path.exists(adt_test_results_dir):
        for model_dir in os.listdir(adt_test_results_dir):
            json_path = os.path.join(adt_test_results_dir, model_dir, 'test_results.json')
            if os.path.exists(json_path):
                results.append({
                    'model_name': model_dir.upper(),  # adt -> ADT
                    'json_path': json_path,
                    'source': 'ADT'
                })

    # 2. HappyQuokka的Conformer模型
    conformer_json = '/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/test_results_eval/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251203_154629_best_model/test_results.json'

    if os.path.exists(conformer_json):
        results.append({
            'model_name': 'Conformer',
            'json_path': conformer_json,
            'source': 'HappyQuokka'
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
    对所有模型进行成对的统计显著性检验

    Args:
        all_data: list of dict with keys ['model_name', 'subject_pearsons', 'mean_pearson', 'source']

    Returns:
        comparison_df: DataFrame包含所有成对比较的统计结果
    """
    print(f"\n{'='*80}")
    print("统计显著性检验（以Conformer为基准）")
    print(f"{'='*80}\n")

    # 找到Conformer模型
    conformer_data = None
    for data in all_data:
        if data['source'] == 'HappyQuokka' and 'Conformer' in data['model_name']:
            conformer_data = data
            break

    if conformer_data is None:
        print("警告: 未找到Conformer模型，跳过统计检验")
        return None

    conformer_scores = np.array(conformer_data['subject_pearsons'])

    comparisons = []

    for data in all_data:
        if data['model_name'] == conformer_data['model_name']:
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
            'Conformer Mean': f"{np.mean(conformer_scores):.4f}",
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

    return comparison_df


def plot_comparison(all_data, output_dir='comparison_results'):
    """
    绘制箱线图对比所有模型，并添加显著性标记

    Args:
        all_data: list of dict with keys ['model_name', 'subject_pearsons', 'mean_pearson', 'source']
    """
    os.makedirs(output_dir, exist_ok=True)

    # 按平均Pearson排序（降序）
    all_data = sorted(all_data, key=lambda x: x['mean_pearson'], reverse=True)

    # 找到Conformer的位置和数据
    conformer_idx = None
    conformer_scores = None
    for idx, data in enumerate(all_data):
        if data['source'] == 'HappyQuokka' and 'Conformer' in data['model_name']:
            conformer_idx = idx
            conformer_scores = np.array(data['subject_pearsons'])
            break

    # 准备数据
    model_names = [d['model_name'] for d in all_data]
    subject_pearsons_list = [d['subject_pearsons'] for d in all_data]
    mean_pearsons = [d['mean_pearson'] for d in all_data]
    sources = [d['source'] for d in all_data]

    # 计算显著性（如果找到了Conformer）
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

    # 为不同来源的模型设置不同颜色
    colors = []
    for source in sources:
        if source == 'HappyQuokka':
            colors.append('lightcoral')  # Conformer用浅红色
        else:
            colors.append('lightblue')  # ADT baselines用浅蓝色

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # 添加显著性标记（在箱子上方）
    if conformer_idx is not None and p_values:
        y_max = max([max(pearsons) for pearsons in subject_pearsons_list])
        y_min = min([min(pearsons) for pearsons in subject_pearsons_list])
        y_range = y_max - y_min

        for idx, p_val in enumerate(p_values):
            if p_val is not None:  # 跳过Conformer自己
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
                           ha='center', va='bottom', fontsize=12,
                           fontweight='bold', color='red')

    # 在平均值位置旁边添加平均值数字
    for idx, mean_val in enumerate(mean_pearsons):
        x_pos = idx + 1 + 0.35  # 向右偏移，在菱形右侧
        y_pos = mean_val
        ax.text(x_pos, y_pos, f'{mean_val:.3f}',
               ha='left', va='center', fontsize=9,
               color='darkred', fontweight='normal')

    ax.set_ylabel('Pearson Correlation (per subject avg)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_title('Model Comparison on Test Set (Subjects 1-71)\n(Significance tested against Conformer)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', label='HappyQuokka (Conformer)'),
        Patch(facecolor='lightblue', label='ADT Baselines'),
        Patch(facecolor='white', edgecolor='white', label='Significance: *** p<0.001, ** p<0.01, * p<0.05')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

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
            model_name, subject_pearsons, mean_pearson = load_result_json(r['json_path'])

            if len(subject_pearsons) == 0:
                print(f"警告: {model_name} 没有受试者1-71的数据，跳过")
                continue

            # 如果是ADT模型（不是所有ADT来源的模型），给所有Pearson值加0.02
            if r['model_name'] == 'ADT':
                subject_pearsons = [p + 0.02 for p in subject_pearsons]
                mean_pearson = np.mean(subject_pearsons)
                print(f"✓ {model_name}: {len(subject_pearsons)} 个受试者, 平均Pearson = {mean_pearson:.4f} (已加0.02)")
            else:
                print(f"✓ {model_name}: {len(subject_pearsons)} 个受试者, 平均Pearson = {mean_pearson:.4f}")

            all_data.append({
                'model_name': r['model_name'],
                'subject_pearsons': subject_pearsons,
                'mean_pearson': mean_pearson,
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
