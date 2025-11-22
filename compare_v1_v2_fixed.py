"""
修正版：对比原始版本和 v2 版本的梯度改善效果
"""
import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# 两个模型的日志路径
v1_log_path = "/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/test_results/conformer_nlayer8_dmodel256_nhead4_conv31_dist_20251121_172042"
v2_log_path = "/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/test_results/conformer_v2_nlayer8_dmodel256_nhead4_gscale2.0_dist_20251122_114434"

print("=" * 80)
print("Conformer v1 vs v2 梯度对比分析 (修正版)")
print("=" * 80)

# 加载两个模型的数据
ea_v1 = event_accumulator.EventAccumulator(v1_log_path)
ea_v1.Reload()

ea_v2 = event_accumulator.EventAccumulator(v2_log_path)
ea_v2.Reload()

print("\n【标量指标对比】")
print("=" * 80)

print(f"\n{'指标':<30} {'v1 原始版':<15} {'v2 改进版':<15} {'变化':<15}")
print("-" * 80)

metrics = {
    'Train/loss_total': '训练总损失',
    'Train/loss_mse': '训练MSE损失',
    'Train/loss_pearson': '训练Pearson损失',
    'Gradient/norm': '梯度范数',
    'Validation/loss': '验证损失',
    'Validation/pearson': '验证Pearson',
    'Test/loss': '测试损失',
    'Test/pearson': '测试Pearson'
}

results = {}
for metric, name in metrics.items():
    v1_events = ea_v1.Scalars(metric)
    v2_events = ea_v2.Scalars(metric)

    if len(v1_events) > 0 and len(v2_events) > 0:
        v1_val = v1_events[-1].value
        v2_val = v2_events[-1].value
        results[metric] = (v1_val, v2_val)

        # 计算变化
        if 'loss' in metric.lower():
            change = (v1_val - v2_val) / abs(v1_val) * 100
            symbol = '↓' if change > 0 else '↑'
        else:
            change = (v2_val - v1_val) / abs(v1_val) * 100
            symbol = '↑' if change > 0 else '↓'

        change_str = f"{change:+.1f}% {symbol}"
        print(f"{metric:<30} {v1_val:<15.6f} {v2_val:<15.6f} {change_str:<15}")

# 梯度直方图分析
print("\n" + "=" * 80)
print("【梯度分布对比 - 详细分析】")
print("=" * 80)

histograms_v1 = ea_v1.Tags().get('histograms', [])
histograms_v2 = ea_v2.Tags().get('histograms', [])

gradient_hists_v1 = [h for h in histograms_v1 if 'Gradient' in h]
gradient_hists_v2 = [h for h in histograms_v2 if 'Gradient' in h]

def get_gradient_magnitude(tag, ea):
    """获取指定层的梯度幅值"""
    events = ea.Histograms(tag)
    if len(events) > 0:
        latest = events[-1]
        hist_values = latest.histogram_value
        return max(abs(hist_values.min), abs(hist_values.max))
    return 0

# 统计各层梯度
print("\n1. 前层 Conformer (layer_stack.0) 梯度对比:")
print("-" * 80)
print(f"{'模块':<50} {'v1':<12} {'v2':<12} {'提升':<10}")
print("-" * 80)

layer0_tags_v1 = [t for t in gradient_hists_v1 if 'layer_stack.0' in t]
layer0_tags_v2 = [t for t in gradient_hists_v2 if 'layer_stack.0' in t]

v1_layer0_grads = []
v2_layer0_grads = []

for tag_v1 in layer0_tags_v1:
    module_name = tag_v1.split('layer_stack.0.')[-1]
    tag_v2 = f"Gradient/layer_stack.0.{module_name}"

    if tag_v2 in gradient_hists_v2:
        grad_v1 = get_gradient_magnitude(tag_v1, ea_v1)
        grad_v2 = get_gradient_magnitude(tag_v2, ea_v2)

        v1_layer0_grads.append(grad_v1)
        v2_layer0_grads.append(grad_v2)

        improvement = grad_v2 / grad_v1 if grad_v1 > 0 else 0
        print(f"{module_name:<50} {grad_v1:<12.6f} {grad_v2:<12.6f} {improvement:>9.1f}x")

avg_v1_layer0 = np.mean(v1_layer0_grads) if v1_layer0_grads else 0
avg_v2_layer0 = np.mean(v2_layer0_grads) if v2_layer0_grads else 0

print("-" * 80)
print(f"{'平均值':<50} {avg_v1_layer0:<12.6f} {avg_v2_layer0:<12.6f} {avg_v2_layer0/avg_v1_layer0:>9.1f}x")

# 后层对比
print("\n2. 后层 Conformer (layer_stack.7) 梯度对比:")
print("-" * 80)

layer7_tags_v1 = [t for t in gradient_hists_v1 if 'layer_stack.7' in t]
layer7_tags_v2 = [t for t in gradient_hists_v2 if 'layer_stack.7' in t]

v1_layer7_grads = []
v2_layer7_grads = []

for tag_v1 in layer7_tags_v1[:5]:  # 只显示前5个
    module_name = tag_v1.split('layer_stack.7.')[-1]
    tag_v2 = f"Gradient/layer_stack.7.{module_name}"

    if tag_v2 in gradient_hists_v2:
        grad_v1 = get_gradient_magnitude(tag_v1, ea_v1)
        grad_v2 = get_gradient_magnitude(tag_v2, ea_v2)

        v1_layer7_grads.append(grad_v1)
        v2_layer7_grads.append(grad_v2)

avg_v1_layer7 = np.mean([get_gradient_magnitude(t, ea_v1) for t in layer7_tags_v1])
avg_v2_layer7 = np.mean([get_gradient_magnitude(t, ea_v2) for t in layer7_tags_v2])

print(f"平均梯度: v1={avg_v1_layer7:.6f}, v2={avg_v2_layer7:.6f}, 提升={avg_v2_layer7/avg_v1_layer7:.1f}x")

# CNN 层对比
print("\n3. CNN 特征提取层 (conv1) 梯度对比:")
print("-" * 80)

conv1_tags = ['Gradient/conv1.weight', 'Gradient/conv1.bias']
for tag in conv1_tags:
    grad_v1 = get_gradient_magnitude(tag, ea_v1)
    grad_v2 = get_gradient_magnitude(tag, ea_v2)
    improvement = grad_v2 / grad_v1 if grad_v1 > 0 else 0
    print(f"{tag:<50} {grad_v1:<12.6f} {grad_v2:<12.6f} {improvement:>9.1f}x")

# 梯度范数对比
print("\n" + "=" * 80)
print("【关键发现】")
print("=" * 80)

grad_norm_v1 = results.get('Gradient/norm', (0, 0))[0]
grad_norm_v2 = results.get('Gradient/norm', (0, 0))[1]
grad_norm_increase = (grad_norm_v2 - grad_norm_v1) / grad_norm_v1 * 100

print(f"\n1. 全局梯度范数:")
print(f"   v1: {grad_norm_v1:.4f}")
print(f"   v2: {grad_norm_v2:.4f}")
print(f"   提升: {grad_norm_increase:+.1f}%")

if grad_norm_increase > 50:
    print(f"   ✅ 梯度范数显著增大，说明模型整体学习能力增强")
else:
    print(f"   → 梯度范数略有增大")

print(f"\n2. 前层梯度改善:")
print(f"   layer_stack.0 平均梯度提升: {avg_v2_layer0/avg_v1_layer0:.1f}x")

if avg_v2_layer0/avg_v1_layer0 > 2:
    print(f"   ✅ 前层梯度显著增强 (>{avg_v2_layer0/avg_v1_layer0:.0f}倍)，特征提取能力提升")
elif avg_v2_layer0/avg_v1_layer0 > 1.5:
    print(f"   ✓ 前层梯度有所增强")
else:
    print(f"   → 前层梯度改善有限")

val_pearson_v1 = results.get('Validation/pearson', (0, 0))[0]
val_pearson_v2 = results.get('Validation/pearson', (0, 0))[1]
pearson_change = (val_pearson_v2 - val_pearson_v1) / val_pearson_v1 * 100

print(f"\n3. 性能指标:")
print(f"   Validation Pearson: {val_pearson_v1:.4f} → {val_pearson_v2:.4f} ({pearson_change:+.1f}%)")

test_pearson_v1 = results.get('Test/pearson', (0, 0))[0]
test_pearson_v2 = results.get('Test/pearson', (0, 0))[1]
test_pearson_change = (test_pearson_v2 - test_pearson_v1) / test_pearson_v1 * 100

print(f"   Test Pearson: {test_pearson_v1:.4f} → {test_pearson_v2:.4f} ({test_pearson_change:+.1f}%)")

# 总结
print("\n" + "=" * 80)
print("【总体评估】")
print("=" * 80)

print("\n架构改进效果:")
score = 0

if avg_v2_layer0/avg_v1_layer0 > 2:
    print("  ✅ 前层梯度提升显著")
    score += 3
elif avg_v2_layer0/avg_v1_layer0 > 1.5:
    print("  ✓ 前层梯度有所提升")
    score += 2
else:
    print("  → 前层梯度提升有限")
    score += 1

if grad_norm_increase > 50:
    print("  ✅ 全局梯度范数显著增强")
    score += 2
elif grad_norm_increase > 20:
    print("  ✓ 全局梯度范数有所增强")
    score += 1

if test_pearson_change > 2:
    print("  ✅ 测试性能提升")
    score += 2
elif test_pearson_change > 0:
    print("  ✓ 测试性能略有提升")
    score += 1
else:
    print("  → 测试性能无明显提升")

print(f"\n总分: {score}/7")

if score >= 6:
    print("🎉 改进效果优秀！v2 架构明显优于原始版本")
elif score >= 4:
    print("✓ 改进有效，v2 架构在梯度和性能上都有提升")
else:
    print("→ 改进效果有限，可能需要调整超参数")

print("\n建议:")
if avg_v2_layer0/avg_v1_layer0 < 3:
    print("  - 可以尝试增大 gradient_scale (如 3.0 或 4.0)")
if test_pearson_change < 5:
    print("  - 训练更多 epochs，观察长期效果")
    print("  - 检查损失函数权重 lambda")

print("\n" + "=" * 80)
