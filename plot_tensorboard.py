#!/usr/bin/env python3
"""
简洁版TensorBoard绘图工具 - 只生成原始曲线，支持自定义范围
"""

import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_scalar_from_events(log_dir, scalar_name):
    """从TensorBoard events文件加载标量数据"""
    # 查找events文件
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))

    if not event_files:
        print(f"❌ 错误: 在 {log_dir} 中未找到events文件")
        return None, None

    # 使用最新的events文件
    event_file = max(event_files, key=os.path.getctime)
    print(f"📂 读取: {os.path.basename(event_file)}")

    # 加载数据
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # 列出所有可用标签
    available_tags = ea.Tags()['scalars']

    if scalar_name not in available_tags:
        print(f"❌ 错误: 未找到标签 '{scalar_name}'")
        print(f"📋 可用标签: {available_tags}")
        return None, None

    # 提取数据
    scalar_data = ea.Scalars(scalar_name)
    steps = np.array([x.step for x in scalar_data])
    values = np.array([x.value for x in scalar_data])

    print(f"✓ 成功加载 {len(steps)} 个数据点")

    return steps, values


def plot_simple_curve(steps, values, scalar_name,
                      step_min=None, step_max=None,
                      output_file='tensorboard_curve.png',
                      figsize=(14, 6), linewidth=2, color='#2E86DE',
                      show_grid=True, show_best=True, dpi=300):
    """
    绘制简洁的原始曲线

    Args:
        steps: 步数数组
        values: 值数组
        scalar_name: 标量名称
        step_min: 最小步数（None表示从头）
        step_max: 最大步数（None表示到尾）
        output_file: 输出文件名
        figsize: 图像大小
        linewidth: 线宽
        color: 线条颜色
        show_grid: 是否显示网格
        show_best: 是否标注最佳点
        dpi: 图像分辨率
    """
    # 筛选范围
    mask = np.ones(len(steps), dtype=bool)
    if step_min is not None:
        mask &= (steps >= step_min)
    if step_max is not None:
        mask &= (steps <= step_max)

    steps_filtered = steps[mask]
    values_filtered = values[mask]

    if len(steps_filtered) == 0:
        print(f"❌ 警告: 在范围 [{step_min}, {step_max}] 内没有数据点")
        return

    # 统计信息
    print(f"\n{'='*70}")
    print(f"数据统计:")
    print(f"{'='*70}")
    print(f"  数据点数: {len(steps_filtered)}")
    print(f"  Step 范围: {steps_filtered.min()} - {steps_filtered.max()}")
    print(f"  Value 范围: {values_filtered.min():.4f} - {values_filtered.max():.4f}")

    # 判断是loss还是metric：loss越小越好，metric越大越好
    is_loss = 'loss' in scalar_name.lower()

    if is_loss:
        best_idx = values_filtered.argmin()  # loss: 最小值最好
        print(f"  最佳值（最小）: {values_filtered[best_idx]:.4f} @ step {steps_filtered[best_idx]}")
    else:
        best_idx = values_filtered.argmax()  # metric: 最大值最好
        print(f"  最佳值（最大）: {values_filtered[best_idx]:.4f} @ step {steps_filtered[best_idx]}")

    print(f"  起始值: {values_filtered[0]:.4f}")
    print(f"  结束值: {values_filtered[-1]:.4f}")

    # 绘图
    plt.figure(figsize=figsize)
    plt.plot(steps_filtered, values_filtered, linewidth=linewidth,
             color=color, alpha=0.9)

    # 标注最佳点
    if show_best:
        best_step = steps_filtered[best_idx]
        best_value = values_filtered[best_idx]
        if is_loss:
            label_text = f'Best (Min): {best_value:.4f} @ step {best_step}'
        else:
            label_text = f'Best (Max): {best_value:.4f} @ step {best_step}'
        plt.scatter([best_step], [best_value],
                   color='red', s=150, zorder=5, marker='*',
                   label=label_text)
        plt.legend(fontsize=12, loc='best')

    # 设置标签
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel(scalar_name.replace('_', ' ').replace('/', ' ').title(), fontsize=14)

    # 标题
    title = scalar_name.replace('_', ' ').replace('/', ' ').title()
    if step_min or step_max:
        title += f' (Steps {step_min or "start"}-{step_max or "end"})'
    plt.title(title, fontsize=16, fontweight='bold')

    if show_grid:
        plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 图表已保存: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='简洁版TensorBoard绘图 - 只生成原始曲线，支持范围控制',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 绘制完整曲线
  python plot_tb_simple.py --log_dir runs/exp1 --scalar "Validation/pearson"

  # 只绘制前6000步
  python plot_tb_simple.py --log_dir runs/exp1 --scalar "Validation/pearson" --step_max 6000

  # 绘制5000-15000步
  python plot_tb_simple.py --log_dir runs/exp1 --scalar "Validation/pearson" --step_min 5000 --step_max 15000

  # 自定义颜色和线宽
  python plot_tb_simple.py --log_dir runs/exp1 --scalar "Train/loss_total" --color red --linewidth 3
        """
    )

    # 必需参数
    parser.add_argument('--log_dir', type=str, default="/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251203_154629",
                        help='TensorBoard日志目录')
    parser.add_argument('--scalar', type=str, default="Validation/pearson",
                        help='标量名称（如 "Validation/pearson"）')

    # 范围控制
    parser.add_argument('--step_min', type=int, default=None,
                        help='最小步数（默认: 从头开始）')
    parser.add_argument('--step_max', type=int, default=None,
                        help='最大步数（默认: 到末尾）')

    # 输出设置
    parser.add_argument('--output', type=str, default='tensorboard_curve.png',
                        help='输出文件名（默认: tensorboard_curve.png）')

    # 样式设置
    parser.add_argument('--figsize', type=str, default='14,6',
                        help='图像大小，格式: 宽,高（默认: 14,6）')
    parser.add_argument('--linewidth', type=float, default=2,
                        help='线条宽度（默认: 2）')
    parser.add_argument('--color', type=str, default='#2E86DE',
                        help='线条颜色（默认: #2E86DE 蓝色）')
    parser.add_argument('--dpi', type=int, default=300,
                        help='图像分辨率（默认: 300）')
    parser.add_argument('--no_grid', action='store_true',
                        help='不显示网格')
    parser.add_argument('--no_best', action='store_true',
                        help='不标注最佳点')

    args = parser.parse_args()

    # 检查目录
    if not os.path.exists(args.log_dir):
        print(f"❌ 错误: 目录不存在: {args.log_dir}")
        return

    # 解析figsize
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        print(f"❌ 错误: figsize格式不正确，使用默认值 (14, 6)")
        figsize = (14, 6)

    # 加载数据
    print(f"\n{'='*70}")
    print(f"从 TensorBoard 加载: {args.scalar}")
    print(f"{'='*70}\n")

    steps, values = load_scalar_from_events(args.log_dir, args.scalar)

    if steps is None:
        return

    # 绘图
    plot_simple_curve(
        steps, values, args.scalar,
        step_min=args.step_min,
        step_max=args.step_max,
        output_file=args.output,
        figsize=figsize,
        linewidth=args.linewidth,
        color=args.color,
        show_grid=not args.no_grid,
        show_best=not args.no_best,
        dpi=args.dpi
    )


if __name__ == '__main__':
    main()
