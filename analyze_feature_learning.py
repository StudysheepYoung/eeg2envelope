"""
分析特征提取网络的有效性
检查前面的网络（Conformer）是否真正学到了有用特征，还是只依赖最后的线性层
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
print("特征提取有效性分析")
print("=" * 80)

histograms = ea.Tags().get('histograms', [])
gradient_hists = [h for h in histograms if 'Gradient' in h or '梯度' in h]

# ============================================================================
# 分析1: 梯度幅值对比 - 前层 vs 后层
# ============================================================================
print("\n【分析1：梯度幅值分层对比】")
print("如果前面网络没有起作用，前层梯度会非常小（梯度消失）")
print("-" * 80)

# 分类梯度
early_layers = []  # 前层 (Conformer 层)
middle_layers = []  # 中间层
final_layers = []  # 输出层 (fc)

for tag in gradient_hists:
    events = ea.Histograms(tag)
    if len(events) > 0:
        latest = events[-1]
        hist_values = latest.histogram_value
        max_abs_grad = max(abs(hist_values.min), abs(hist_values.max))

        # 分类
        if 'layer_stack.0' in tag or 'layer_stack.1' in tag:
            early_layers.append((tag, max_abs_grad))
        elif 'layer_stack.6' in tag or 'layer_stack.7' in tag:
            middle_layers.append((tag, max_abs_grad))
        elif 'fc.' in tag and 'se.fc' not in tag:
            final_layers.append((tag, max_abs_grad))

# 计算平均梯度幅值
def avg_grad(layers):
    return np.mean([g for _, g in layers]) if layers else 0

early_avg = avg_grad(early_layers)
middle_avg = avg_grad(middle_layers)
final_avg = avg_grad(final_layers)

print(f"\n前层梯度 (layer_stack.0-1):  平均幅值 = {early_avg:.6f}")
print(f"后层梯度 (layer_stack.6-7):  平均幅值 = {middle_avg:.6f}")
print(f"输出层梯度 (fc):            平均幅值 = {final_avg:.6f}")

print(f"\n梯度比例:")
print(f"  前层/输出层 = {early_avg/final_avg:.4f}")
print(f"  后层/输出层 = {middle_avg/final_avg:.4f}")

# 诊断
print("\n【诊断】")
if early_avg / final_avg < 0.01:
    print("  ⚠️  前层梯度过小 (<1%)，可能存在梯度消失，特征提取网络学习不足")
elif early_avg / final_avg < 0.1:
    print("  ⚠️  前层梯度较小 (<10%)，特征提取可能不够充分")
elif early_avg / final_avg > 0.5:
    print("  ✅ 前层梯度充足 (>50%)，特征提取网络正在有效学习")
else:
    print("  ✓  前层梯度适中 (10-50%)，特征提取在学习但可能还有提升空间")

# ============================================================================
# 分析2: 权重变化幅度 - 衡量网络是否在更新
# ============================================================================
print("\n" + "=" * 80)
print("【分析2：权重更新幅度分析】")
print("如果前面网络权重几乎不变，说明没有学到东西")
print("-" * 80)

weight_hists = [h for h in histograms if 'Weight' in h or '权重' in h]

# 对比训练初期和后期的权重变化
early_weight_changes = []
final_weight_changes = []

for tag in weight_hists:
    events = ea.Histograms(tag)
    if len(events) >= 2:
        # 取第一个和最后一个
        first = events[0].histogram_value
        last = events[-1].histogram_value

        # 计算标准差变化（粗略估计分布变化）
        first_std = (first.max - first.min) / 2
        last_std = (last.max - last.min) / 2
        relative_change = abs(last_std - first_std) / (first_std + 1e-10)

        if 'layer_stack.0' in tag or 'layer_stack.1' in tag:
            early_weight_changes.append(relative_change)
        elif 'fc.' in tag and 'se.fc' not in tag:
            final_weight_changes.append(relative_change)

early_change_avg = np.mean(early_weight_changes) if early_weight_changes else 0
final_change_avg = np.mean(final_weight_changes) if final_weight_changes else 0

print(f"\n前层权重变化: {early_change_avg:.4f}")
print(f"输出层权重变化: {final_change_avg:.4f}")

if early_change_avg < 0.01:
    print("\n  ⚠️  前层权重几乎不变，可能没有学到有用特征")
else:
    print(f"\n  ✓  前层权重有明显变化，正在学习")

# ============================================================================
# 分析3: 逐层梯度衰减率
# ============================================================================
print("\n" + "=" * 80)
print("【分析3：逐层梯度流分析】")
print("理想情况下，梯度应该从后向前平滑衰减，而不是突然消失")
print("-" * 80)

# 收集每层的平均梯度
layer_grads = {}
for tag in gradient_hists:
    # 只分析 layer_stack 的层
    if 'layer_stack' in tag:
        # 提取层编号
        for i in range(8):
            if f'layer_stack.{i}.' in tag:
                events = ea.Histograms(tag)
                if len(events) > 0:
                    latest = events[-1].histogram_value
                    max_abs_grad = max(abs(hist_values.min), abs(hist_values.max))

                    if i not in layer_grads:
                        layer_grads[i] = []
                    layer_grads[i].append(max_abs_grad)
                break

# 计算每层平均梯度
layer_avg_grads = {i: np.mean(grads) for i, grads in sorted(layer_grads.items())}

print("\n从输出到输入的梯度流:")
for i in sorted(layer_avg_grads.keys(), reverse=True):
    print(f"  Layer {i}: {layer_avg_grads[i]:.6f}")

# 计算梯度衰减比例
if len(layer_avg_grads) >= 2:
    first_layer = min(layer_avg_grads.keys())
    last_layer = max(layer_avg_grads.keys())
    decay_ratio = layer_avg_grads[first_layer] / layer_avg_grads[last_layer]

    print(f"\n梯度衰减率 (第0层/第7层): {decay_ratio:.4f}")

    if decay_ratio < 0.01:
        print("  ⚠️  严重梯度消失！前层几乎学不到东西")
    elif decay_ratio < 0.1:
        print("  ⚠️  存在梯度消失倾向")
    else:
        print("  ✓  梯度流正常")

# ============================================================================
# 分析4: 特定层的详细梯度分布
# ============================================================================
print("\n" + "=" * 80)
print("【分析4：关键层梯度详情】")
print("-" * 80)

key_layers = [
    'Gradient/layer_stack.0.slf_attn',  # 第一层注意力
    'Gradient/layer_stack.7.slf_attn',  # 最后一层注意力
    'Gradient/fc.weight',                # 输出层
]

for key in key_layers:
    matches = [h for h in gradient_hists if key in h]
    if matches:
        tag = matches[0]
        events = ea.Histograms(tag)
        if len(events) > 0:
            latest = events[-1].histogram_value
            print(f"\n{tag.split('/')[-1]}:")
            print(f"  范围: [{latest.min:.6e}, {latest.max:.6e}]")

# ============================================================================
# 总结建议
# ============================================================================
print("\n" + "=" * 80)
print("【总结与建议】")
print("=" * 80)

print("\n如果前面的网络没有起作用，会看到：")
print("  1. 前层梯度 << 输出层梯度 (比例 < 0.01)")
print("  2. 前层权重基本不变")
print("  3. 逐层梯度急剧衰减")

print("\n如何改进（如果确认特征提取不足）：")
print("  1. 使用残差连接 (ResNet-style)")
print("  2. 调整学习率 - 对前层使用更大的学习率")
print("  3. 使用梯度裁剪")
print("  4. 减少网络深度或增加 skip connections")
print("  5. 使用 Layer-wise Learning Rate Decay (LLRD)")

print("\n" + "=" * 80)
