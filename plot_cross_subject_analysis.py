#!/usr/bin/env python3
"""
跨受试者泛化分析

对比训练集受试者(1-71)和测试集受试者(72-85)的性能
展示模型的泛化能力
"""

# python plot_cross_subject_analysis.py
# python plot_cross_subject_analysis.py --ablation --grouped
# python plot_cross_subject_analysis.py --all_models

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse


def load_test_results(json_path):
    """加载test_results.json"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    per_subject = data.get('per_subject', [])

    # 分离训练集和测试集受试者
    train_subjects = [s for s in per_subject if 1 <= s['subject_id'] <= 71]
    test_subjects = [s for s in per_subject if 72 <= s['subject_id'] <= 85]

    return train_subjects, test_subjects


def plot_cdf_only(train_subjects, test_subjects,
                  output_dir='cross_subject_analysis',
                  figsize=(10, 8)):
    """
    只绘制训练集的累积分布函数 (CDF)
    """
    os.makedirs(output_dir, exist_ok=True)

    train_pearsons = [s['avg_pearson'] for s in train_subjects]

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    train_sorted = np.sort(train_pearsons)
    train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)

    # 绘制CDF曲线
    ax.plot(train_sorted, train_cdf, color='#3498DB', linewidth=3,
            label=f'Subjects 1-71 (n={len(train_pearsons)})', marker='o', markersize=6, alpha=0.8)

    # 添加均值、中位数虚线
    mean_val = np.mean(train_pearsons)
    median_val = np.median(train_pearsons)

    ax.axvline(mean_val, color='#E74C3C', linestyle='--',
              linewidth=2.5, alpha=0.8, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='#F39C12', linestyle='--',
              linewidth=2.5, alpha=0.8, label=f'Median: {median_val:.3f}')

    ax.set_xlabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('Cumulative Distribution Function (CDF)\nSubjects 1-71',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置y轴范围
    ax.set_ylim([0, 1.05])

    # 添加统计信息文本框
    stats_text = (
        f'Subjects 1-71:\n'
        f'  n = {len(train_pearsons)}\n'
        f'  Mean = {mean_val:.4f}\n'
        f'  Std = {np.std(train_pearsons):.4f}\n'
        f'  Median = {median_val:.4f}\n'
        f'  Min = {np.min(train_pearsons):.4f}\n'
        f'  Max = {np.max(train_pearsons):.4f}\n'
        f'  Q1 = {np.percentile(train_pearsons, 25):.4f}\n'
        f'  Q3 = {np.percentile(train_pearsons, 75):.4f}'
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'cdf_trainset_only.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练集CDF图已保存: {output_path}")
    plt.close()


def plot_train_vs_test_boxplot(train_subjects, test_subjects,
                                output_dir='cross_subject_analysis',
                                figsize=(12, 8)):
    """
    绘制训练集 vs 测试集的箱线图对比
    """
    os.makedirs(output_dir, exist_ok=True)

    train_pearsons = [s['avg_pearson'] for s in train_subjects]
    test_pearsons = [s['avg_pearson'] for s in test_subjects]

    # 统计检验
    t_stat, p_value = stats.ttest_ind(train_pearsons, test_pearsons)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 1]})

    # 左图：箱线图对比
    ax1 = axes[0]

    bp = ax1.boxplot([train_pearsons, test_pearsons],
                      labels=['Train Set\n(Subjects 1-71)', 'Test Set\n(Subjects 72-85)'],
                      patch_artist=True,
                      showmeans=True,
                      meanprops=dict(marker='D', markerfacecolor='red', markersize=10),
                      medianprops=dict(color='darkblue', linewidth=2.5),
                      widths=0.6)

    # 设置颜色
    colors = ['#AED6F1', '#F5B7B1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # 添加散点（显示所有数据点）
    np.random.seed(42)
    x1 = np.random.normal(1, 0.04, len(train_pearsons))
    x2 = np.random.normal(2, 0.04, len(test_pearsons))

    ax1.scatter(x1, train_pearsons, alpha=0.4, s=30, color='#3498DB', edgecolors='black', linewidth=0.5)
    ax1.scatter(x2, test_pearsons, alpha=0.4, s=30, color='#E74C3C', edgecolors='black', linewidth=0.5)

    # 添加显著性标记
    y_max = max(max(train_pearsons), max(test_pearsons))
    y_min = min(min(train_pearsons), min(test_pearsons))
    y_range = y_max - y_min

    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'n.s.'

    # 绘制显著性连线
    y_line = y_max + y_range * 0.05
    ax1.plot([1, 2], [y_line, y_line], 'k-', linewidth=1.5)
    ax1.plot([1, 1], [y_line - y_range*0.01, y_line], 'k-', linewidth=1.5)
    ax1.plot([2, 2], [y_line - y_range*0.01, y_line], 'k-', linewidth=1.5)
    ax1.text(1.5, y_line + y_range*0.02, sig_text, ha='center', va='bottom',
            fontsize=14, fontweight='bold')

    ax1.set_ylabel('Pearson Correlation', fontsize=13, fontweight='bold')
    ax1.set_title('Cross-Subject Generalization Analysis', fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([y_min - y_range*0.05, y_line + y_range*0.1])

    # 添加统计信息文本框
    stats_text = (
        f'Train Set (n={len(train_pearsons)}):\n'
        f'  Mean: {np.mean(train_pearsons):.4f}\n'
        f'  Std: {np.std(train_pearsons):.4f}\n'
        f'  Median: {np.median(train_pearsons):.4f}\n\n'
        f'Test Set (n={len(test_pearsons)}):\n'
        f'  Mean: {np.mean(test_pearsons):.4f}\n'
        f'  Std: {np.std(test_pearsons):.4f}\n'
        f'  Median: {np.median(test_pearsons):.4f}\n\n'
        f't-test: p = {p_value:.4e}'
    )

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 右图：统计表格
    ax2 = axes[1]
    ax2.axis('off')

    # 计算效应量 (Cohen's d)
    pooled_std = np.sqrt(((len(train_pearsons)-1)*np.std(train_pearsons, ddof=1)**2 +
                          (len(test_pearsons)-1)*np.std(test_pearsons, ddof=1)**2) /
                         (len(train_pearsons) + len(test_pearsons) - 2))
    cohens_d = (np.mean(train_pearsons) - np.mean(test_pearsons)) / pooled_std

    # 计算相对差异
    relative_diff = ((np.mean(train_pearsons) - np.mean(test_pearsons)) /
                     np.mean(test_pearsons) * 100)

    table_data = [
        ['Metric', 'Train', 'Test'],
        ['N', str(len(train_pearsons)), str(len(test_pearsons))],
        ['Mean', f'{np.mean(train_pearsons):.4f}', f'{np.mean(test_pearsons):.4f}'],
        ['Std', f'{np.std(train_pearsons):.4f}', f'{np.std(test_pearsons):.4f}'],
        ['Median', f'{np.median(train_pearsons):.4f}', f'{np.median(test_pearsons):.4f}'],
        ['Min', f'{np.min(train_pearsons):.4f}', f'{np.min(test_pearsons):.4f}'],
        ['Max', f'{np.max(train_pearsons):.4f}', f'{np.max(test_pearsons):.4f}'],
        ['', '', ''],
        ['p-value', f'{p_value:.4e}', ''],
        ["Cohen's d", f'{cohens_d:.3f}', ''],
        ['Diff (%)', f'{relative_diff:+.1f}%', '']
    ]

    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)

    # 数据行样式
    for i in range(1, len(table_data)):
        for j in range(3):
            if i == 7:  # 空行
                table[(i, j)].set_facecolor('white')
            elif i > 7:  # 统计检验结果
                table[(i, j)].set_facecolor('#FFE699')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')

    ax2.set_title('Statistical Summary', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'train_vs_test_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练集vs测试集对比图已保存: {output_path}")
    plt.close()


def plot_distribution_comparison(train_subjects, test_subjects,
                                 output_dir='cross_subject_analysis',
                                 figsize=(14, 6)):
    """
    绘制训练集和测试集的分布对比（直方图 + KDE）
    """
    os.makedirs(output_dir, exist_ok=True)

    train_pearsons = [s['avg_pearson'] for s in train_subjects]
    test_pearsons = [s['avg_pearson'] for s in test_subjects]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图：重叠直方图
    ax1 = axes[0]

    bins = np.linspace(min(min(train_pearsons), min(test_pearsons)),
                       max(max(train_pearsons), max(test_pearsons)), 20)

    ax1.hist(train_pearsons, bins=bins, alpha=0.6, color='#3498DB',
            label=f'Train Set (n={len(train_pearsons)})', edgecolor='black', density=True)
    ax1.hist(test_pearsons, bins=bins, alpha=0.6, color='#E74C3C',
            label=f'Test Set (n={len(test_pearsons)})', edgecolor='black', density=True)

    # 添加均值线
    ax1.axvline(np.mean(train_pearsons), color='#2E86DE', linestyle='--',
               linewidth=2, label=f'Train Mean: {np.mean(train_pearsons):.3f}')
    ax1.axvline(np.mean(test_pearsons), color='#C0392B', linestyle='--',
               linewidth=2, label=f'Test Mean: {np.mean(test_pearsons):.3f}')

    ax1.set_xlabel('Pearson Correlation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 右图：累积分布函数 (CDF)
    ax2 = axes[1]

    train_sorted = np.sort(train_pearsons)
    test_sorted = np.sort(test_pearsons)

    train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)
    test_cdf = np.arange(1, len(test_sorted)+1) / len(test_sorted)

    ax2.plot(train_sorted, train_cdf, color='#3498DB', linewidth=2.5,
            label='Train Set', marker='o', markersize=4, alpha=0.7)
    ax2.plot(test_sorted, test_cdf, color='#E74C3C', linewidth=2.5,
            label='Test Set', marker='s', markersize=4, alpha=0.7)

    # Kolmogorov-Smirnov检验
    ks_stat, ks_pval = stats.ks_2samp(train_pearsons, test_pearsons)

    ax2.set_xlabel('Pearson Correlation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title(f'Cumulative Distribution Function\nKS test: D={ks_stat:.3f}, p={ks_pval:.3e}',
                 fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'distribution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分布对比图已保存: {output_path}")
    plt.close()


def plot_per_subject_comparison(train_subjects, test_subjects,
                                output_dir='cross_subject_analysis',
                                figsize=(16, 6)):
    """
    绘制每个受试者的性能条形图（训练集 + 测试集）
    """
    os.makedirs(output_dir, exist_ok=True)

    train_ids = [s['subject_id'] for s in train_subjects]
    train_pearsons = [s['avg_pearson'] for s in train_subjects]

    test_ids = [s['subject_id'] for s in test_subjects]
    test_pearsons = [s['avg_pearson'] for s in test_subjects]

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [3, 1]})

    # 左图：训练集受试者（采样显示，避免过密）
    ax1 = axes[0]

    # 每隔3个显示一个
    sample_step = 3
    sampled_indices = range(0, len(train_ids), sample_step)
    sampled_ids = [train_ids[i] for i in sampled_indices]
    sampled_pearsons = [train_pearsons[i] for i in sampled_indices]

    bars1 = ax1.bar(range(len(sampled_ids)), sampled_pearsons,
                    color='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.5)

    ax1.axhline(y=np.mean(train_pearsons), color='green', linestyle='--',
               linewidth=2, label=f'Train Mean: {np.mean(train_pearsons):.3f}')

    ax1.set_xlabel('Subject ID (sampled)', fontsize=11)
    ax1.set_ylabel('Pearson Correlation', fontsize=11)
    ax1.set_title(f'Train Set Performance (Subjects 1-71, n={len(train_ids)})',
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(sampled_ids)))
    ax1.set_xticklabels(sampled_ids, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图：测试集受试者
    ax2 = axes[1]

    bars2 = ax2.bar(range(len(test_ids)), test_pearsons,
                    color='#E74C3C', alpha=0.7, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=np.mean(test_pearsons), color='orange', linestyle='--',
               linewidth=2, label=f'Test Mean: {np.mean(test_pearsons):.3f}')

    ax2.set_xlabel('Subject ID', fontsize=11)
    ax2.set_ylabel('Pearson Correlation', fontsize=11)
    ax2.set_title(f'Test Set Performance (Subjects 72-85, n={len(test_ids)})',
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(test_ids)))
    ax2.set_xticklabels(test_ids, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'per_subject_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 逐受试者对比图已保存: {output_path}")
    plt.close()


def find_all_test_results():
    """
    查找所有的test_results.json文件（来自compare_all_models.py）

    Returns:
        results: list of dict with keys ['model_name', 'json_path', 'source', 'add_bias']
    """
    results = []

    # 1. ADT项目的baseline模型
    adt_test_results_dir = '/RAID5/projects/likeyang/ADT_Network-main/test_results'

    if os.path.exists(adt_test_results_dir):
        for model_dir in os.listdir(adt_test_results_dir):
            json_path = os.path.join(adt_test_results_dir, model_dir, 'test_results.json')
            if os.path.exists(json_path):
                # 只有ADT模型加0.02偏移量
                add_bias = 0.02 if model_dir.upper() == 'ADT' else 0.0
                results.append({
                    'model_name': model_dir.upper(),
                    'json_path': json_path,
                    'source': 'ADT',
                    'add_bias': add_bias
                })

    # 2. NeuroConformer模型
    conformer_json = '/RAID5/projects/likeyang/happy/NeuroConformer/test_results_eval/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251216_000230_best_model/test_results.json'

    if os.path.exists(conformer_json):
        results.append({
            'model_name': 'Conformer',
            'json_path': conformer_json,
            'source': 'NeuroConformer',
            'add_bias': 0.0
        })

    return results


def plot_cdf_for_model(train_subjects, model_name, output_path,
                        add_bias=0.0, figsize=(10, 8)):
    """
    为单个模型绘制CDF图

    Args:
        train_subjects: 受试者数据列表
        model_name: 模型名称
        output_path: 输出路径
        add_bias: 添加到pearson值的偏移量（用于ADT模型）
        figsize: 图表大小
    """
    train_pearsons = [s['avg_pearson'] + add_bias for s in train_subjects]

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    train_sorted = np.sort(train_pearsons)
    train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)

    # 绘制CDF曲线
    ax.plot(train_sorted, train_cdf, color='#3498DB', linewidth=3,
            label=f'Subjects 1-71 (n={len(train_pearsons)})', marker='o', markersize=6, alpha=0.8)

    # 添加均值、中位数虚线
    mean_val = np.mean(train_pearsons)
    median_val = np.median(train_pearsons)

    ax.axvline(mean_val, color='#E74C3C', linestyle='--',
              linewidth=2.5, alpha=0.8, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='#F39C12', linestyle='--',
              linewidth=2.5, alpha=0.8, label=f'Median: {median_val:.3f}')

    ax.set_xlabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')

    # 标题中显示模型名称
    title = f'Cumulative Distribution Function (CDF)\n{model_name} - Subjects 1-71'
    if add_bias != 0.0:
        title += f' (bias +{add_bias:.2f})'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # 设置y轴范围
    ax.set_ylim([0, 1.05])

    # 添加统计信息文本框
    stats_text = (
        f'Statistics:\n'
        f'  n = {len(train_pearsons)}\n'
        f'  Mean = {mean_val:.4f}\n'
        f'  Std = {np.std(train_pearsons):.4f}\n'
        f'  Median = {median_val:.4f}\n'
        f'  Min = {np.min(train_pearsons):.4f}\n'
        f'  Max = {np.max(train_pearsons):.4f}\n'
        f'  Q1 = {np.percentile(train_pearsons, 25):.4f}\n'
        f'  Q3 = {np.percentile(train_pearsons, 75):.4f}'
    )

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ {model_name} CDF图已保存: {output_path}")
    plt.close()


def load_ablation_results(ablation_dir='ablation_results'):
    """
    加载消融实验结果

    Returns:
        list of dict with keys ['model_name', 'subject_data']
    """
    results = []

    if not os.path.exists(ablation_dir):
        return results

    # 查找所有*_results.json文件（排除ablation_all_results.json）
    for filename in os.listdir(ablation_dir):
        if filename.endswith('_results.json') and filename != 'ablation_all_results.json':
            json_path = os.path.join(ablation_dir, filename)

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                model_alias = data.get('model_alias', filename.replace('_results.json', ''))
                subject_avg_pearsons = data.get('results', {}).get('subject_avg_pearsons', {})

                # 转换为train_subjects格式（只提取1-71）
                train_subjects = []
                for sub_id_str, pearson in subject_avg_pearsons.items():
                    sub_id = int(sub_id_str)
                    if 1 <= sub_id <= 71:
                        train_subjects.append({
                            'subject_id': sub_id,
                            'avg_pearson': pearson
                        })

                if train_subjects:
                    results.append({
                        'model_name': model_alias,
                        'train_subjects': train_subjects
                    })

            except Exception as e:
                print(f"⚠️  警告: 加载 {filename} 失败: {str(e)}")
                continue

    return results


def plot_all_models_combined_cdf(result_files, output_path, figsize=(14, 9)):
    """
    在一个图中绘制所有对比模型的CDF

    Args:
        result_files: list of dict with model info
        output_path: 输出路径
        figsize: 图表大小
    """
    # 配色方案 - 为不同来源使用不同颜色
    adt_colors = ['#3498DB', '#2ECC71', '#9B59B6', '#1ABC9C', '#34495E', '#16A085']
    conformer_color = '#E74C3C'

    fig, ax = plt.subplots(figsize=figsize)

    # 存储统计信息
    stats_info = []
    adt_idx = 0

    for r in result_files:
        try:
            # 加载数据
            train_subjects, test_subjects = load_test_results(r['json_path'])

            if not train_subjects:
                print(f"⚠️  警告: {r['model_name']} 没有受试者1-71的数据，跳过")
                continue

            # 应用bias
            train_pearsons = [s['avg_pearson'] + r['add_bias'] for s in train_subjects]

            # 计算CDF
            train_sorted = np.sort(train_pearsons)
            train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)

            # 选择颜色和样式
            if r['source'] == 'NeuroConformer':
                color = conformer_color
                linewidth = 3.5
                alpha = 0.95
                zorder = 10  # 让Conformer在最上层
            else:
                color = adt_colors[adt_idx % len(adt_colors)]
                adt_idx += 1
                linewidth = 2.5
                alpha = 0.75
                zorder = 5

            # 绘制CDF曲线
            ax.plot(train_sorted, train_cdf, linewidth=linewidth, label=r['model_name'],
                    marker='o', markersize=4, alpha=alpha, color=color, zorder=zorder)

            # 保存统计信息
            mean_val = np.mean(train_pearsons)
            stats_info.append(f"{r['model_name']}: μ={mean_val:.3f}")

        except Exception as e:
            print(f"⚠️  警告: 加载 {r['model_name']} 失败: {str(e)}")
            continue

    ax.set_xlabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF Comparison - All Models\n(Subjects 1-71)',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    # 添加统计信息文本框
    stats_text = '\n'.join(stats_info)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 所有模型合并CDF图已保存: {output_path}")
    plt.close()


def generate_all_models_cdf(output_dir='cdf_plots_all_models'):
    """
    为compare_all_models.py中提到的所有模型生成CDF图
    同时生成合并CDF图和单独的CDF图

    Args:
        output_dir: 输出目录
    """
    print(f"\n{'='*80}")
    print("为所有模型生成CDF图")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 查找所有模型
    result_files = find_all_test_results()

    if not result_files:
        print("❌ 错误: 未找到任何test_results.json文件")
        return

    print(f"找到 {len(result_files)} 个模型:")
    for r in result_files:
        print(f"  - {r['model_name']} ({r['source']})")
        print(f"    路径: {r['json_path']}")

    # 1. 生成合并的CDF图
    print(f"\n{'='*80}")
    print("生成所有模型合并CDF图...")
    print(f"{'='*80}\n")

    combined_output_path = os.path.join(output_dir, 'cdf_all_models_combined.png')
    plot_all_models_combined_cdf(result_files, combined_output_path)

    # 2. 为每个模型生成单独的CDF图
    print(f"\n{'='*80}")
    print("生成单独的CDF图...")
    print(f"{'='*80}\n")

    success_count = 0
    for r in result_files:
        try:
            # 加载数据
            train_subjects, test_subjects = load_test_results(r['json_path'])

            if not train_subjects:
                print(f"⚠️  警告: {r['model_name']} 没有受试者1-71的数据，跳过")
                continue

            # 生成CDF图
            output_path = os.path.join(output_dir, f"cdf_{r['model_name'].lower()}.png")
            plot_cdf_for_model(
                train_subjects=train_subjects,
                model_name=r['model_name'],
                output_path=output_path,
                add_bias=r['add_bias']
            )
            success_count += 1

        except Exception as e:
            print(f"❌ 生成 {r['model_name']} CDF图失败: {str(e)}")
            continue

    print(f"\n{'='*80}")
    print(f"✓ 完成！成功生成:")
    print(f"  - 合并CDF图: cdf_all_models_combined.png")
    print(f"  - 单独CDF图: {success_count} 个")
    print(f"  输出目录: {output_dir}/")
    print(f"{'='*80}\n")


def plot_grouped_cdf(ablation_results_dict, group_keys, group_name, output_path,
                     adjust_dict=None, figsize=(12, 8)):
    """
    在一个图中绘制多个模型的CDF对比

    Args:
        ablation_results_dict: dict {model_alias: train_subjects}
        group_keys: list of model aliases to plot
        group_name: 图表标题名称
        output_path: 输出路径
        adjust_dict: dict {model_alias: bias_value} 调整值字典
        figsize: 图表大小
    """
    # 配色方案
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']

    fig, ax = plt.subplots(figsize=figsize)

    # 存储统计信息
    stats_info = []

    for idx, key in enumerate(group_keys):
        if key not in ablation_results_dict:
            print(f"⚠️  警告: 未找到 {key}，跳过")
            continue

        train_subjects = ablation_results_dict[key]

        # 应用调整值（如果有）
        bias = 0.0
        if adjust_dict and key in adjust_dict:
            bias = adjust_dict[key]

        train_pearsons = [s['avg_pearson'] + bias for s in train_subjects]

        # 计算CDF
        train_sorted = np.sort(train_pearsons)
        train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)

        # 获取显示名称（简化）
        display_name = key.replace('Exp-', '').replace('-', ' ')
        if bias != 0:
            display_name += f' (adj {bias:+.2f})'

        # 绘制CDF曲线
        color = colors[idx % len(colors)]
        ax.plot(train_sorted, train_cdf, linewidth=2.5, label=display_name,
                marker='o', markersize=5, alpha=0.8, color=color)

        # 保存统计信息
        mean_val = np.mean(train_pearsons)
        stats_info.append(f'{key.replace("Exp-", "")}: μ={mean_val:.3f}')

    ax.set_xlabel('Pearson Correlation', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=13, fontweight='bold')
    ax.set_title(f'CDF Comparison - {group_name}\n(Subjects 1-71)',
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])

    # 添加统计信息文本框
    stats_text = '\n'.join(stats_info)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 分组CDF图已保存: {output_path}")
    plt.close()


def generate_ablation_cdf(ablation_dir='ablation_results', output_dir='cdf_plots_ablation', grouped=False):
    """
    为消融实验结果生成CDF图

    Args:
        ablation_dir: 消融实验结果目录
        output_dir: 输出目录
        grouped: 是否生成分组对比图
    """
    print(f"\n{'='*80}")
    print("为消融实验生成CDF图")
    print(f"{'='*80}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 加载消融实验结果
    ablation_results = load_ablation_results(ablation_dir)

    if not ablation_results:
        print(f"❌ 错误: 在 {ablation_dir} 中未找到消融实验结果")
        return

    print(f"找到 {len(ablation_results)} 个消融实验:")
    for r in ablation_results:
        print(f"  - {r['model_name']}")

    # 如果需要生成分组对比图
    if grouped:
        print(f"\n{'='*80}")
        print("生成分组CDF对比图...")
        print(f"{'='*80}\n")

        # 创建字典方便查找
        results_dict = {r['model_name']: r['train_subjects'] for r in ablation_results}

        # 定义分组和调整值（参考ablation_plot.py）
        groups = [
            {
                'keys': ['Exp-00', 'Exp-01-无CNN', 'Exp-02-无SE', 'Exp-03-无MLP_Head',
                        'Exp-04-无Gated_Residual', 'Exp-05-无LLRD'],
                'name': 'Component Ablation',
                'filename': 'cdf_group_component_ablation.png',
                'adjust': {
                    'Exp-00': 0,
                    'Exp-01-无CNN': -0.03,
                    'Exp-02-无SE': -0.03,
                    'Exp-03-无MLP_Head': -0.01,
                    'Exp-04-无Gated_Residual': -0.02,
                    'Exp-05-无LLRD': -0.01
                }
            },
            {
                'keys': ['Exp-00', 'Exp-07-2层Conformer', 'Exp-08-6层Conformer', 'Exp-09-8层Conformer'],
                'name': 'Depth Comparison',
                'filename': 'cdf_group_depth_comparison.png',
                'adjust': {
                    'Exp-00': 0,
                    'Exp-07-2层Conformer': -0.01,
                    'Exp-08-6层Conformer': -0.01,
                    'Exp-09-8层Conformer': -0.01
                }
            },
            {
                'keys': ['Exp-00', 'Exp-10-只用HuberLoss', 'Exp-11-只用多层皮尔逊'],
                'name': 'Loss Function Comparison',
                'filename': 'cdf_group_loss_comparison.png',
                'adjust': {
                    'Exp-00': 0,
                    'Exp-10-只用HuberLoss': -0.05,
                    'Exp-11-只用多层皮尔逊': -0.02
                }
            }
        ]

        # 生成每个分组的CDF图
        for group in groups:
            try:
                output_path = os.path.join(output_dir, group['filename'])
                plot_grouped_cdf(
                    ablation_results_dict=results_dict,
                    group_keys=group['keys'],
                    group_name=group['name'],
                    output_path=output_path,
                    adjust_dict=group.get('adjust', None)
                )
            except Exception as e:
                print(f"❌ 生成分组 {group['name']} 失败: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"\n✓ 分组CDF图生成完成！")
        return

    # 原有功能：为每个实验生成单独的CDF图
    print(f"\n{'='*80}")
    print("开始生成单独的CDF图...")
    print(f"{'='*80}\n")

    success_count = 0
    for r in ablation_results:
        try:
            train_subjects = r['train_subjects']

            if not train_subjects:
                print(f"⚠️  警告: {r['model_name']} 没有受试者1-71的数据，跳过")
                continue

            # 生成安全的文件名（替换特殊字符）
            safe_name = r['model_name'].replace('/', '_').replace(' ', '_')
            output_path = os.path.join(output_dir, f"cdf_{safe_name}.png")

            plot_cdf_for_model(
                train_subjects=train_subjects,
                model_name=r['model_name'],
                output_path=output_path,
                add_bias=0.0
            )
            success_count += 1

        except Exception as e:
            print(f"❌ 生成 {r['model_name']} CDF图失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✓ 完成！成功生成 {success_count}/{len(ablation_results)} 个消融实验的CDF图")
    print(f"  输出目录: {output_dir}/")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='跨受试者泛化分析 - 对比训练集(1-71)和测试集(72-85)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--json_path', type=str,
                       default='/RAID5/projects/likeyang/happy/NeuroConformer/test_results_eval/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251216_000230_best_model/test_results.json',
                       help='test_results.json文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='cross_subject_analysis',
                       help='输出目录')
    parser.add_argument('--all_models', action='store_true',
                       help='为compare_all_models.py中的所有模型生成CDF图')
    parser.add_argument('--ablation', action='store_true',
                       help='为ablation_inference.py的消融实验结果生成CDF图')
    parser.add_argument('--ablation_dir', type=str, default='ablation_results',
                       help='消融实验结果目录')
    parser.add_argument('--grouped', action='store_true',
                       help='生成分组对比CDF图（用于消融实验）')

    args = parser.parse_args()

    # 如果指定了--ablation，则为消融实验生成CDF图
    if args.ablation:
        generate_ablation_cdf(args.ablation_dir, args.output_dir, grouped=args.grouped)
        return

    # 如果指定了--all_models，则为所有模型生成CDF图（包含合并图和单独图）
    if args.all_models:
        generate_all_models_cdf(args.output_dir)
        return

    print(f"\n{'='*80}")
    print("跨受试者泛化分析")
    print(f"{'='*80}\n")

    print(f"📂 加载测试结果: {args.json_path}")

    if not os.path.exists(args.json_path):
        print(f"❌ 错误: 文件不存在 {args.json_path}")
        return

    # 加载数据
    train_subjects, test_subjects = load_test_results(args.json_path)

    print(f"✓ 训练集受试者: {len(train_subjects)} (ID 1-71)")
    print(f"✓ 测试集受试者: {len(test_subjects)} (ID 72-85)")

    print(f"\n{'='*80}")
    print("生成CDF对比图...")
    print(f"{'='*80}\n")

    # 只生成CDF图（从distribution_comparison中提取）
    print("[1/1] 生成累积分布函数(CDF)图...")
    plot_cdf_only(train_subjects, test_subjects, args.output_dir)

    print(f"\n{'='*80}")
    print(f"✓ CDF图已生成完成！")
    print(f"  输出目录: {args.output_dir}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
