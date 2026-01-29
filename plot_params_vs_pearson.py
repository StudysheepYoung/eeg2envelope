"""
绘制模型参数量与重建相关性的关系图

根据论文中的表格数据，绘制参数量-性能曲线图
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties

from plotting_colors import get_model_color

# 模型数据
models_data = {
    'Linear': {'params': 2_049, 'pearson': 0.113, 'marker': 'o', 'color_key': 'LINEAR'},
    'EEGNet': {'params': 802_158, 'pearson': 0.159, 'marker': 's', 'color_key': 'EEGNET'},
    'FCNN': {'params': 23_528_435, 'pearson': 0.108, 'marker': '^', 'color_key': 'FCNN'},
    'VLAAI': {'params': 12_881_409, 'pearson': 0.138, 'marker': 'v', 'color_key': 'VLAAI'},
    'ADT Network': {'params': 527_088, 'pearson': 0.166, 'marker': 'D', 'color_key': 'ADT'},
    'HappyQuokka': {'params': 1_275_073, 'pearson': 0.180, 'marker': 'p', 'color_key': 'HAPPYQUOKKA'},
    'Ours (NeuroConformer)': {'params': 7_544_389, 'pearson': 0.239, 'marker': '*', 'color_key': 'NEUROCONFORMER'},
}


def get_color(model_name):
    """Return consistent color for the given model name."""
    color_key = models_data[model_name].get('color_key', model_name)
    return get_model_color(color_key)

def plot_params_vs_pearson(output_dir='comparison_results'):
    """
    绘制参数量与Pearson相关性的散点图
    """
    os.makedirs(output_dir, exist_ok=True)

    # 提取数据
    model_names = list(models_data.keys())
    params = [models_data[m]['params'] for m in model_names]
    pearsons = [models_data[m]['pearson'] for m in model_names]
    colors = [get_color(m) for m in model_names]
    markers = [models_data[m]['marker'] for m in model_names]

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制散点
    for i, model in enumerate(model_names):
        # 特殊标记Ours模型
        if 'Ours' in model:
            ax.scatter(params[i], pearsons[i],
                      c=colors[i], marker=markers[i], s=650,
                      edgecolors='black', linewidths=2.8,
                      label=model, zorder=5, alpha=0.92)
        else:
            ax.scatter(params[i], pearsons[i],
                      c=colors[i], marker=markers[i], s=280,
                      edgecolors='black', linewidths=1.8,
                      label=model, zorder=3, alpha=0.85)

    # 设置对数刻度（参数量范围很大）
    ax.set_xscale('log')

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # 添加标签和标题
    ax.set_xlabel('Number of Parameters (log scale)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

    # 设置y轴范围，留出空间
    ax.set_ylim([0.09, 0.26])

    # 添加图例（放在左上角）
    legend_font = FontProperties(weight='bold', size=20)
    ax.legend(loc='upper left', framealpha=0.95,
              ncol=1, columnspacing=1.0, handletextpad=0.5,
              prop=legend_font)

    # 在每个点旁边添加数值标注
    for i, model in enumerate(model_names):
        # 计算标注位置偏移
        if 'Ours' in model:
            offset_x = 1.15
            offset_y = 0.005
            fontsize = 22
            fontweight = 'bold'
        else:
            offset_x = 1.15
            offset_y = 0.003
            fontsize = 22
            fontweight = 'bold'

        # 添加Pearson值标注
        ax.annotate(f'{pearsons[i]:.3f}',
                   xy=(params[i], pearsons[i]),
                   xytext=(params[i] * offset_x, pearsons[i] + offset_y),
                   fontsize=fontsize,
                   fontweight=fontweight,
                   color=colors[i],
                   ha='left',
                   va='bottom')

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'params_vs_pearson.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 参数量-性能曲线图已保存到: {output_path}")
    plt.close()

    # 同时保存为PDF（矢量格式，适合论文）
    output_pdf = os.path.join(output_dir, 'params_vs_pearson.pdf')
    fig, ax = plt.subplots(figsize=(12, 8))

    # 重新绘制（为PDF）
    for i, model in enumerate(model_names):
        if 'Ours' in model:
            ax.scatter(params[i], pearsons[i],
                      c=colors[i], marker=markers[i], s=650,
                      edgecolors='black', linewidths=2.8,
                      label=model, zorder=5, alpha=0.92)
        else:
            ax.scatter(params[i], pearsons[i],
                      c=colors[i], marker=markers[i], s=280,
                      edgecolors='black', linewidths=1.8,
                      label=model, zorder=3, alpha=0.85)

    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel('Number of Parameters (log scale)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')
    ax.set_ylim([0.09, 0.26])
    ax.legend(loc='upper left', framealpha=0.95,
              ncol=1, columnspacing=1.0, handletextpad=0.5,
              prop=legend_font)

    for i, model in enumerate(model_names):
        if 'Ours' in model:
            offset_x = 1.15
            offset_y = 0.005
            fontsize = 18
            fontweight = 'bold'
        else:
            offset_x = 1.15
            offset_y = 0.003
            fontsize = 16
            fontweight = 'bold'

        ax.annotate(f'{pearsons[i]:.3f}',
                   xy=(params[i], pearsons[i]),
                   xytext=(params[i] * offset_x, pearsons[i] + offset_y),
                   fontsize=fontsize,
                   fontweight=fontweight,
                   color=colors[i],
                   ha='left',
                   va='bottom')

    plt.tight_layout()
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"✓ 参数量-性能曲线图（PDF）已保存到: {output_pdf}")
    plt.close()


def plot_params_vs_pearson_with_table(output_dir='comparison_results'):
    """
    绘制参数量与Pearson相关性的组合图（曲线图+表格）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 提取数据
    model_names = list(models_data.keys())
    params = [models_data[m]['params'] for m in model_names]
    pearsons = [models_data[m]['pearson'] for m in model_names]
    colors = [get_color(m) for m in model_names]
    markers = [models_data[m]['marker'] for m in model_names]

    # 按Pearson排序
    sorted_indices = np.argsort(pearsons)[::-1]

    # 创建组合图（上方散点图，下方表格）
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.3)

    # ========== 上方：散点图 ==========
    ax1 = fig.add_subplot(gs[0])

    for i, model in enumerate(model_names):
        if 'Ours' in model:
            ax1.scatter(params[i], pearsons[i],
                       c=colors[i], marker=markers[i], s=500,
                       edgecolors='black', linewidths=2.5,
                       label=model, zorder=5, alpha=0.9)
        else:
            ax1.scatter(params[i], pearsons[i],
                       c=colors[i], marker=markers[i], s=200,
                       edgecolors='black', linewidths=1.5,
                       label=model, zorder=3, alpha=0.8)

    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_xlabel('Number of Parameters (log scale)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Pearson Correlation', fontsize=13, fontweight='bold')
    ax1.set_ylim([0.09, 0.26])
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=2)

    # ========== 下方：表格 ==========
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    # 准备表格数据（按性能排序）
    table_data = []
    for idx in sorted_indices:
        model = model_names[idx]
        param_str = f"{params[idx]:,}"
        pearson_str = f"{pearsons[idx]:.3f}"
        table_data.append([model, param_str, pearson_str])

    # 创建表格
    table = ax2.table(cellText=table_data,
                     colLabels=['Model', 'Parameters', 'Pearson Correlation'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.3, 0.3])

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    # 表头样式
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)

    # 数据行样式
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('white')

            # 高亮Ours行
            if 'Ours' in table_data[i-1][0]:
                cell.set_facecolor('#FFE699')
                cell.set_text_props(weight='bold', fontsize=12)

    plt.tight_layout()

    # 保存
    output_path = os.path.join(output_dir, 'params_vs_pearson_with_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 参数量-性能组合图已保存到: {output_path}")
    plt.close()


def print_model_summary():
    """
    打印模型汇总表格
    """
    print("\n" + "="*80)
    print("模型参数量与性能对比")
    print("="*80)
    print(f"{'模型':<30} {'参数量':>15} {'Pearson相关系数':>18}")
    print("-"*80)

    # 按Pearson排序
    sorted_models = sorted(models_data.items(),
                          key=lambda x: x[1]['pearson'],
                          reverse=True)

    for model, data in sorted_models:
        params_str = f"{data['params']:,}"
        pearson_str = f"{data['pearson']:.3f}"

        # 高亮显示Ours
        if 'Ours' in model:
            print(f"★ {model:<28} {params_str:>15} {pearson_str:>18} ★")
        else:
            print(f"  {model:<28} {params_str:>15} {pearson_str:>18}")

    print("="*80)

    # 计算性能提升
    ours_pearson = models_data['Ours (NeuroConformer)']['pearson']
    best_baseline_pearson = max([data['pearson'] for name, data in models_data.items()
                                  if 'Ours' not in name])
    improvement = (ours_pearson - best_baseline_pearson) / best_baseline_pearson * 100

    print(f"\n性能提升:")
    print(f"  最佳baseline (HappyQuokka): {best_baseline_pearson:.3f}")
    print(f"  Ours (NeuroConformer): {ours_pearson:.3f}")
    print(f"  相对提升: +{improvement:.1f}%")
    print("="*80 + "\n")


def main():
    """
    主函数
    """
    print("\n" + "="*80)
    print("生成模型参数量与性能对比图")
    print("="*80 + "\n")

    # 打印模型汇总
    print_model_summary()

    # 生成图表
    output_dir = 'comparison_results'
    os.makedirs(output_dir, exist_ok=True)

    print("\n生成图表中...")
    plot_params_vs_pearson(output_dir)
    plot_params_vs_pearson_with_table(output_dir)

    print("\n✓ 所有图表生成完成！")
    print(f"  输出目录: {output_dir}/")
    print(f"    - params_vs_pearson.png (散点图, PNG)")
    print(f"    - params_vs_pearson.pdf (散点图, PDF)")
    print(f"    - params_vs_pearson_with_table.png (组合图)")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
