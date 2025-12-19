#!/usr/bin/env python3
"""
预测质量详细分析图

生成多种预测质量可视化：
1. 时序对比图（预测 vs 真值）
2. 误差分布直方图
3. 散点图（预测 vs 真值）
4. 受试者相关性分布
5. 时间窗口质量分析
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import argparse


def load_test_results(json_path):
    """
    加载test_results.json

    Returns:
        dict with keys:
            - model_name: 模型名称
            - per_subject: 每个受试者的统计信息
            - per_sample: 每个样本的详细信息（如果有）
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    model_name = data.get('checkpoint', 'Unknown')
    per_subject = data.get('per_subject', [])
    per_sample = data.get('per_sample', [])

    return {
        'model_name': model_name,
        'per_subject': per_subject,
        'per_sample': per_sample
    }


def plot_time_series_comparison(per_sample, output_dir='prediction_analysis',
                                 n_samples=5, sample_indices=None, figsize=(16, 10)):
    """
    绘制时序对比图（预测 vs 真值）

    Args:
        per_sample: 样本级别的预测结果
        n_samples: 显示多少个样本
        sample_indices: 指定显示哪些样本（None则随机选择）
    """
    os.makedirs(output_dir, exist_ok=True)

    if not per_sample or 'predictions' not in per_sample[0]:
        print("⚠️  警告: 数据中没有predictions字段，跳过时序对比图")
        return

    # 选择样本
    if sample_indices is None:
        # 随机选择n_samples个样本
        if len(per_sample) <= n_samples:
            sample_indices = list(range(len(per_sample)))
        else:
            np.random.seed(42)
            sample_indices = np.random.choice(len(per_sample), n_samples, replace=False)
            sample_indices = sorted(sample_indices)

    n_rows = len(sample_indices)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=False)

    if n_rows == 1:
        axes = [axes]

    for idx, sample_idx in enumerate(sample_indices):
        sample = per_sample[sample_idx]

        predictions = np.array(sample['predictions'])
        targets = np.array(sample['targets'])
        pearson_r = sample['pearson']
        subject_id = sample.get('subject_id', 'Unknown')

        time_steps = np.arange(len(predictions))

        ax = axes[idx]

        # 绘制真值和预测
        ax.plot(time_steps, targets, label='Ground Truth',
                color='#2E86DE', linewidth=1.5, alpha=0.8)
        ax.plot(time_steps, predictions, label='Prediction',
                color='#EE5A6F', linewidth=1.5, alpha=0.8)

        # 标题和标签
        ax.set_title(f'Subject {subject_id} | Sample {sample_idx} | Pearson r = {pearson_r:.3f}',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Speech Envelope', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    axes[-1].set_xlabel('Time Steps', fontsize=11)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'time_series_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 时序对比图已保存: {output_path}")
    plt.close()


def plot_error_distribution(per_sample, output_dir='prediction_analysis', figsize=(14, 6)):
    """
    绘制误差分布直方图和统计信息
    """
    os.makedirs(output_dir, exist_ok=True)

    if not per_sample or 'predictions' not in per_sample[0]:
        print("⚠️  警告: 数据中没有predictions字段，跳过误差分布图")
        return

    # 收集所有误差
    all_errors = []
    for sample in per_sample:
        predictions = np.array(sample['predictions'])
        targets = np.array(sample['targets'])
        errors = predictions - targets
        all_errors.extend(errors)

    all_errors = np.array(all_errors)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图：误差分布直方图
    ax1 = axes[0]
    n, bins, patches = ax1.hist(all_errors, bins=100, density=True,
                                 color='#3498DB', alpha=0.7, edgecolor='black')

    # 拟合正态分布
    mu, sigma = all_errors.mean(), all_errors.std()
    x = np.linspace(all_errors.min(), all_errors.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label=f'Normal fit\nμ={mu:.4f}, σ={sigma:.4f}')

    ax1.set_xlabel('Prediction Error (Pred - True)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 右图：Q-Q图（检验正态性）
    ax2 = axes[1]
    stats.probplot(all_errors, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 统计信息
    textstr = '\n'.join([
        f'Mean: {mu:.4f}',
        f'Std: {sigma:.4f}',
        f'Median: {np.median(all_errors):.4f}',
        f'MAE: {np.abs(all_errors).mean():.4f}',
        f'RMSE: {np.sqrt((all_errors**2).mean()):.4f}'
    ])

    ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'error_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 误差分布图已保存: {output_path}")
    plt.close()


def plot_prediction_scatter(per_sample, output_dir='prediction_analysis',
                            n_samples_max=500, figsize=(10, 10)):
    """
    绘制预测 vs 真值散点图
    """
    os.makedirs(output_dir, exist_ok=True)

    if not per_sample or 'predictions' not in per_sample[0]:
        print("⚠️  警告: 数据中没有predictions字段，跳过散点图")
        return

    # 收集数据（限制样本数避免过密）
    all_predictions = []
    all_targets = []

    for sample in per_sample[:n_samples_max]:
        predictions = np.array(sample['predictions'])
        targets = np.array(sample['targets'])
        all_predictions.extend(predictions)
        all_targets.extend(targets)

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 绘制散点图
    fig, ax = plt.subplots(figsize=figsize)

    # 使用hexbin处理密集点
    hexbin = ax.hexbin(all_targets, all_predictions, gridsize=50, cmap='YlOrRd',
                       mincnt=1, alpha=0.8)

    # 添加对角线（完美预测线）
    min_val = min(all_targets.min(), all_predictions.min())
    max_val = max(all_targets.max(), all_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2,
           label='Perfect Prediction', alpha=0.7)

    # 计算整体Pearson相关系数
    r, p_val = stats.pearsonr(all_targets, all_predictions)

    # 线性拟合
    z = np.polyfit(all_targets, all_predictions, 1)
    p = np.poly1d(z)
    ax.plot([min_val, max_val], [p(min_val), p(max_val)], 'r-', linewidth=2,
           label=f'Linear Fit: y={z[0]:.3f}x+{z[1]:.3f}', alpha=0.7)

    ax.set_xlabel('Ground Truth', fontsize=13, fontweight='bold')
    ax.set_ylabel('Prediction', fontsize=13, fontweight='bold')
    ax.set_title(f'Prediction vs Ground Truth\nPearson r = {r:.4f} (p < {p_val:.2e})',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 添加colorbar
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Count', fontsize=11)

    # 保持正方形比例
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'prediction_scatter.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 预测散点图已保存: {output_path}")
    plt.close()


def plot_subject_correlation_distribution(per_subject, output_dir='prediction_analysis',
                                          figsize=(14, 8)):
    """
    绘制受试者相关性分布（直方图 + 箱线图）- 只分析受试者1-71
    """
    os.makedirs(output_dir, exist_ok=True)

    if not per_subject:
        print("⚠️  警告: 没有per_subject数据，跳过受试者分析")
        return

    # 提取数据（只保留受试者1-71）
    subject_ids = [s['subject_id'] for s in per_subject if 1 <= s['subject_id'] <= 71]
    pearsons = [s['avg_pearson'] for s in per_subject if 1 <= s['subject_id'] <= 71]

    # 创建组合图
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1],
                          hspace=0.3, wspace=0.3)

    # 左上：每个受试者的Pearson条形图
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#2E86DE' if p >= np.median(pearsons) else '#E74C3C' for p in pearsons]
    bars = ax1.bar(range(len(subject_ids)), pearsons, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=0.5)
    ax1.axhline(y=np.mean(pearsons), color='green', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(pearsons):.4f}')
    ax1.axhline(y=np.median(pearsons), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(pearsons):.4f}')
    ax1.set_xlabel('Subject ID', fontsize=11)
    ax1.set_ylabel('Pearson Correlation', fontsize=11)
    ax1.set_title('Per-Subject Performance', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # 设置x轴标签（每隔5个显示）
    tick_positions = range(0, len(subject_ids), max(1, len(subject_ids) // 20))
    tick_labels = [subject_ids[i] for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')

    # 右上：Pearson分布直方图
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(pearsons, bins=20, orientation='horizontal', color='#3498DB',
            alpha=0.7, edgecolor='black')
    ax2.axhline(y=np.mean(pearsons), color='green', linestyle='--', linewidth=2)
    ax2.axhline(y=np.median(pearsons), color='orange', linestyle='--', linewidth=2)
    ax2.set_ylabel('Pearson Correlation', fontsize=11)
    ax2.set_xlabel('Count', fontsize=11)
    ax2.set_title('Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 下方：箱线图
    ax3 = fig.add_subplot(gs[1, :])
    bp = ax3.boxplot([pearsons], vert=False, patch_artist=True,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=10),
                     medianprops=dict(color='darkblue', linewidth=2))
    bp['boxes'][0].set_facecolor('#AED6F1')
    ax3.set_xlabel('Pearson Correlation', fontsize=11)
    ax3.set_title('Summary Statistics', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    # 添加统计信息
    textstr = '\n'.join([
        f'N subjects: {len(pearsons)}',
        f'Mean: {np.mean(pearsons):.4f}',
        f'Std: {np.std(pearsons):.4f}',
        f'Median: {np.median(pearsons):.4f}',
        f'Min: {np.min(pearsons):.4f}',
        f'Max: {np.max(pearsons):.4f}',
        f'Q1: {np.percentile(pearsons, 25):.4f}',
        f'Q3: {np.percentile(pearsons, 75):.4f}'
    ])

    ax3.text(0.98, 0.98, textstr, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'subject_correlation_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 受试者相关性分布图已保存: {output_path}")
    plt.close()


def plot_correlation_by_performance_group(per_subject, output_dir='prediction_analysis',
                                          figsize=(12, 6)):
    """
    按性能分组分析（高/中/低性能受试者）- 只分析受试者1-71
    """
    os.makedirs(output_dir, exist_ok=True)

    if not per_subject:
        print("⚠️  警告: 没有per_subject数据，跳过分组分析")
        return

    # 只保留受试者1-71
    pearsons = np.array([s['avg_pearson'] for s in per_subject if 1 <= s['subject_id'] <= 71])

    # 按三分位数分组
    q33 = np.percentile(pearsons, 33.33)
    q67 = np.percentile(pearsons, 66.67)

    low_group = pearsons[pearsons <= q33]
    mid_group = pearsons[(pearsons > q33) & (pearsons <= q67)]
    high_group = pearsons[pearsons > q67]

    # 绘制分组对比
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 左图：小提琴图
    ax1 = axes[0]
    data_to_plot = [low_group, mid_group, high_group]
    parts = ax1.violinplot(data_to_plot, positions=[1, 2, 3], showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#3498DB')
        pc.set_alpha(0.7)

    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Low\n(Bottom 33%)', 'Mid\n(Middle 33%)', 'High\n(Top 33%)'])
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title('Performance Distribution by Group', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图：统计表格
    ax2 = axes[1]
    ax2.axis('off')

    table_data = [
        ['Group', 'N', 'Mean', 'Std', 'Range'],
        ['Low', len(low_group), f'{low_group.mean():.4f}', f'{low_group.std():.4f}',
         f'{low_group.min():.3f}-{low_group.max():.3f}'],
        ['Mid', len(mid_group), f'{mid_group.mean():.4f}', f'{mid_group.std():.4f}',
         f'{mid_group.min():.3f}-{mid_group.max():.3f}'],
        ['High', len(high_group), f'{high_group.mean():.4f}', f'{high_group.std():.4f}',
         f'{high_group.min():.3f}-{high_group.max():.3f}']
    ]

    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.15, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 表头样式
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 数据行样式
    for i in range(1, 4):
        for j in range(5):
            table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')

    ax2.set_title('Group Statistics', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'performance_groups.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 性能分组分析图已保存: {output_path}")
    plt.close()


def generate_all_plots(json_path, output_dir='prediction_analysis'):
    """
    生成所有预测质量分析图
    """
    print(f"\n{'='*80}")
    print(f"预测质量分析")
    print(f"{'='*80}\n")

    print(f"📂 加载测试结果: {json_path}")

    if not os.path.exists(json_path):
        print(f"❌ 错误: 文件不存在 {json_path}")
        return

    # 加载数据
    results = load_test_results(json_path)
    model_name = results['model_name']
    per_subject = results['per_subject']
    per_sample = results['per_sample']

    print(f"✓ 模型: {model_name}")
    print(f"✓ 受试者数: {len(per_subject)}")
    print(f"✓ 样本数: {len(per_sample)}")

    print(f"\n{'='*80}")
    print("生成可视化图表...")
    print(f"{'='*80}\n")

    # 1. 时序对比图
    if per_sample:
        print("[1/5] 生成时序对比图...")
        plot_time_series_comparison(per_sample, output_dir, n_samples=5)

    # 2. 误差分布图
    if per_sample:
        print("[2/5] 生成误差分布图...")
        plot_error_distribution(per_sample, output_dir)

    # 3. 预测散点图
    if per_sample:
        print("[3/5] 生成预测散点图...")
        plot_prediction_scatter(per_sample, output_dir)

    # 4. 受试者相关性分布
    if per_subject:
        print("[4/4] 生成受试者相关性分布图...")
        plot_subject_correlation_distribution(per_subject, output_dir)

    print(f"\n{'='*80}")
    print(f"✓ 所有图表已生成完成！")
    print(f"  输出目录: {output_dir}/")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='预测质量详细分析 - 生成多种可视化图表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认路径
  python plot_prediction_quality.py

  # 指定test_results.json路径
  python plot_prediction_quality.py --json_path path/to/test_results.json

  # 指定输出目录
  python plot_prediction_quality.py --output_dir my_analysis
"""
    )

    parser.add_argument('--json_path', type=str,
                       default='/RAID5/projects/likeyang/happy/NeuroConformer/test_results_eval/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251216_000230_best_model/test_results.json',
                       help='test_results.json文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='prediction_analysis',
                       help='输出目录')

    args = parser.parse_args()

    generate_all_plots(args.json_path, args.output_dir)


if __name__ == '__main__':
    main()
