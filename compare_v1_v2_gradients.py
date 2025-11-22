"""
对比原始版本和 v2 版本的梯度改善效果
"""
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# 两个模型的日志路径
v1_log_path = "/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/test_results/conformer_nlayer8_dmodel256_nhead4_conv31_dist_20251121_172042"
v2_log_path = "/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/test_results/conformer_v2_nlayer8_dmodel256_nhead4_gscale2.0_dist_20251122_114434"

print("=" * 80)
print("Conformer v1 vs v2 梯度对比分析")
print("=" * 80)

# 加载两个模型的数据
ea_v1 = event_accumulator.EventAccumulator(v1_log_path)
ea_v1.Reload()

ea_v2 = event_accumulator.EventAccumulator(v2_log_path)
ea_v2.Reload()

print("\n【标量指标对比】")
print("=" * 80)

# 对比标量指标
scalars_v1 = ea_v1.Tags()['scalars']
scalars_v2 = ea_v2.Tags()['scalars']

print("\n训练损失和性能:")
print(f"{'指标':<30} {'v1 原始版':<15} {'v2 改进版':<15} {'变化':<15}")
print("-" * 80)

metrics_to_compare = [
    'Train/loss_total',
    'Train/loss_mse',
    'Train/loss_pearson',
    'Gradient/norm',
    'Validation/loss',
    'Validation/pearson',
    'Test/loss',
    'Test/pearson'
]

for metric in metrics_to_compare:
    if metric in scalars_v1 and metric in scalars_v2:
        v1_events = ea_v1.Scalars(metric)
        v2_events = ea_v2.Scalars(metric)

        if len(v1_events) > 0 and len(v2_events) > 0:
            v1_val = v1_events[-1].value
            v2_val = v2_events[-1].value

            # 计算变化
            if 'loss' in metric.lower():
                change = (v1_val - v2_val) / v1_val * 100  # 损失降低是好的
                change_str = f"{change:+.1f}% {'↓' if change > 0 else '↑'}"
            else:
                change = (v2_val - v1_val) / v1_val * 100  # 指标提升是好的
                change_str = f"{change:+.1f}% {'↑' if change > 0 else '↓'}"

            print(f"{metric:<30} {v1_val:<15.6f} {v2_val:<15.6f} {change_str:<15}")

# 梯度直方图分析
print("\n" + "=" * 80)
print("【梯度分布对比】")
print("=" * 80)

histograms_v1 = ea_v1.Tags().get('histograms', [])
histograms_v2 = ea_v2.Tags().get('histograms', [])

gradient_hists_v1 = [h for h in histograms_v1 if 'Gradient' in h or '梯度' in h]
gradient_hists_v2 = [h for h in histograms_v2 if 'Gradient' in h or '梯度' in h]

# 分层分析
def get_layer_gradients(gradient_hists, ea):
    """提取各层梯度幅值"""
    early_layers = []  # 前层
    middle_layers = []  # 中间层
    final_layers = []  # 输出层

    for tag in gradient_hists:
        events = ea.Histograms(tag)
        if len(events) > 0:
            latest = events[-1]
            hist_values = latest.histogram_value
            max_abs_grad = max(abs(hist_values.min), abs(hist_values.max))

            if 'layer_stack.0' in tag or 'layer_stack.1' in tag:
                early_layers.append((tag, max_abs_grad))
            elif 'layer_stack.6' in tag or 'layer_stack.7' in tag:
                middle_layers.append((tag, max_abs_grad))
            elif 'fc.' in tag and 'se.fc' not in tag:
                final_layers.append((tag, max_abs_grad))

    return early_layers, middle_layers, final_layers

early_v1, middle_v1, final_v1 = get_layer_gradients(gradient_hists_v1, ea_v1)
early_v2, middle_v2, final_v2 = get_layer_gradients(gradient_hists_v2, ea_v2)

def avg_grad(layers):
    return np.mean([g for _, g in layers]) if layers else 0

print("\n梯度幅值对比:")
print(f"{'层位置':<20} {'v1 原始版':<15} {'v2 改进版':<15} {'提升倍数':<15}")
print("-" * 80)

v1_early = avg_grad(early_v1)
v2_early = avg_grad(early_v2)
v1_middle = avg_grad(middle_v1)
v2_middle = avg_grad(middle_v2)
v1_final = avg_grad(final_v1)
v2_final = avg_grad(final_v2)

print(f"{'前层 (layer 0-1)':<20} {v1_early:<15.6f} {v2_early:<15.6f} {v2_early/v1_early:>14.1f}x")
print(f"{'后层 (layer 6-7)':<20} {v1_middle:<15.6f} {v2_middle:<15.6f} {v2_middle/v1_middle:>14.1f}x")
print(f"{'输出层 (fc)':<20} {v1_final:<15.6f} {v2_final:<15.6f} {v2_final/v1_final:>14.1f}x")

print("\n梯度比例对比 (前层/输出层):")
print("-" * 80)
v1_ratio = v1_early / v1_final if v1_final > 0 else 0
v2_ratio = v2_early / v2_final if v2_final > 0 else 0

print(f"v1 原始版: {v1_ratio:.4f} ({v1_ratio*100:.2f}%)")
print(f"v2 改进版: {v2_ratio:.4f} ({v2_ratio*100:.2f}%)")
print(f"改进倍数: {v2_ratio/v1_ratio:.1f}x")

# 详细的前层梯度对比
print("\n" + "=" * 80)
print("【前层详细梯度对比】")
print("=" * 80)

print(f"\n{'层名称':<50} {'v1':<12} {'v2':<12} {'提升':<10}")
print("-" * 80)

# 找到共同的前层
v1_early_dict = {tag.split('/')[-1]: grad for tag, grad in early_v1}
v2_early_dict = {tag.split('/')[-1]: grad for tag, grad in early_v2}

common_layers = set(v1_early_dict.keys()) & set(v2_early_dict.keys())
for layer_name in sorted(common_layers)[:10]:  # 只显示前10个
    v1_g = v1_early_dict[layer_name]
    v2_g = v2_early_dict[layer_name]
    improvement = v2_g / v1_g if v1_g > 0 else 0
    print(f"{layer_name:<50} {v1_g:<12.6f} {v2_g:<12.6f} {improvement:>9.1f}x")

# 总结
print("\n" + "=" * 80)
print("【改进效果总结】")
print("=" * 80)

print("\n✅ 梯度改善:")
if v2_ratio > v1_ratio * 5:
    print(f"   🎉 优秀！前层梯度比例从 {v1_ratio*100:.2f}% 提升到 {v2_ratio*100:.2f}%，提升 {v2_ratio/v1_ratio:.1f} 倍")
elif v2_ratio > v1_ratio * 2:
    print(f"   ✓ 良好！前层梯度比例从 {v1_ratio*100:.2f}% 提升到 {v2_ratio*100:.2f}%，提升 {v2_ratio/v1_ratio:.1f} 倍")
else:
    print(f"   ⚠️ 改善有限。前层梯度比例从 {v1_ratio*100:.2f}% 提升到 {v2_ratio*100:.2f}%，提升 {v2_ratio/v1_ratio:.1f} 倍")

# 性能改善
if 'Validation/pearson' in scalars_v1 and 'Validation/pearson' in scalars_v2:
    v1_val_pearson = ea_v1.Scalars('Validation/pearson')[-1].value
    v2_val_pearson = ea_v2.Scalars('Validation/pearson')[-1].value
    pearson_improvement = (v2_val_pearson - v1_val_pearson) / v1_val_pearson * 100

    print(f"\n✅ 性能改善:")
    if pearson_improvement > 20:
        print(f"   🎉 显著提升！Validation Pearson 从 {v1_val_pearson:.3f} 提升到 {v2_val_pearson:.3f} ({pearson_improvement:+.1f}%)")
    elif pearson_improvement > 5:
        print(f"   ✓ 有提升。Validation Pearson 从 {v1_val_pearson:.3f} 提升到 {v2_val_pearson:.3f} ({pearson_improvement:+.1f}%)")
    elif pearson_improvement > 0:
        print(f"   → 略有提升。Validation Pearson 从 {v1_val_pearson:.3f} 提升到 {v2_val_pearson:.3f} ({pearson_improvement:+.1f}%)")
    else:
        print(f"   ⚠️ 性能下降。Validation Pearson 从 {v1_val_pearson:.3f} 下降到 {v2_val_pearson:.3f} ({pearson_improvement:.1f}%)")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)
