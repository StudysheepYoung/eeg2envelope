"""
分析 TensorBoard 日志中的直方图数据
检查梯度和权重分布是否健康
"""
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# 日志路径
log_path = "/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/test_results/conformer_nlayer8_dmodel256_nhead4_conv31_dist_20251121_172042"

# 加载 TensorBoard 数据
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

print("=" * 80)
print("TensorBoard 日志分析")
print("=" * 80)

# 1. 查看所有标量
print("\n【标量指标】")
scalars = ea.Tags()['scalars']
for tag in scalars:
    events = ea.Scalars(tag)
    if len(events) > 0:
        latest = events[-1]
        print(f"  {tag}: {latest.value:.6f} (step {latest.step})")

# 2. 查看所有直方图
print("\n" + "=" * 80)
print("【直方图分析】")
print("=" * 80)

histograms = ea.Tags().get('histograms', [])
if not histograms:
    print("  ⚠️  未找到直方图数据")
else:
    print(f"\n共有 {len(histograms)} 个直方图\n")

    # 分析梯度直方图
    gradient_hists = [h for h in histograms if 'Gradient' in h or '梯度' in h]
    weight_hists = [h for h in histograms if 'Weight' in h or '权重' in h]

    print(f"梯度直方图: {len(gradient_hists)} 个")
    print(f"权重直方图: {len(weight_hists)} 个")

    # 详细分析最近的梯度分布
    print("\n" + "-" * 80)
    print("【梯度分布健康检查】")
    print("-" * 80)

    for tag in gradient_hists[:10]:  # 只显示前10个
        events = ea.Histograms(tag)
        if len(events) > 0:
            latest = events[-1]
            hist_values = latest.histogram_value

            # 提取统计信息
            min_val = hist_values.min
            max_val = hist_values.max

            # 健康检查
            issues = []
            if abs(min_val) < 1e-10 and abs(max_val) < 1e-10:
                issues.append("⚠️ 梯度消失")
            elif abs(max_val) > 100 or abs(min_val) > 100:
                issues.append("⚠️ 梯度爆炸")

            status = "✓" if not issues else " ".join(issues)

            print(f"\n  {tag}")
            print(f"    范围: [{min_val:.6e}, {max_val:.6e}]")
            print(f"    状态: {status}")

    # 详细分析权重分布
    print("\n" + "-" * 80)
    print("【权重分布健康检查】")
    print("-" * 80)

    for tag in weight_hists[:10]:  # 只显示前10个
        events = ea.Histograms(tag)
        if len(events) > 0:
            latest = events[-1]
            hist_values = latest.histogram_value

            min_val = hist_values.min
            max_val = hist_values.max

            # 健康检查
            issues = []
            if abs(min_val) < 1e-10 and abs(max_val) < 1e-10:
                issues.append("⚠️ 权重全为0")
            elif abs(max_val) > 10:
                issues.append("⚠️ 权重过大")

            status = "✓" if not issues else " ".join(issues)

            print(f"\n  {tag}")
            print(f"    范围: [{min_val:.6f}, {max_val:.6f}]")
            print(f"    状态: {status}")

# 3. 查看图像
print("\n" + "=" * 80)
images = ea.Tags().get('images', [])
print(f"【图像】: {len(images)} 个")
for tag in images:
    print(f"  - {tag}")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
