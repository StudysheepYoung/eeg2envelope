"""
调试 v2 模型的梯度记录
"""
import os
from tensorboard.backend.event_processing import event_accumulator

v2_log_path = "/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/test_results/conformer_v2_nlayer8_dmodel256_nhead4_gscale2.0_dist_20251122_114434"

ea_v2 = event_accumulator.EventAccumulator(v2_log_path)
ea_v2.Reload()

print("=" * 80)
print("v2 模型梯度记录调试")
print("=" * 80)

# 查看所有梯度标签
histograms = ea_v2.Tags().get('histograms', [])
gradient_hists = [h for h in histograms if 'Gradient' in h or '梯度' in h]

print(f"\n共有 {len(gradient_hists)} 个梯度直方图\n")

# 查找输出层相关的梯度
print("输出层相关梯度:")
print("-" * 80)
output_related = [h for h in gradient_hists if 'fc' in h.lower() or 'output' in h.lower() or 'head' in h.lower()]

for tag in sorted(output_related):
    events = ea_v2.Histograms(tag)
    if len(events) > 0:
        latest = events[-1]
        hist_values = latest.histogram_value
        max_abs_grad = max(abs(hist_values.min), abs(hist_values.max))
        print(f"  {tag}: {max_abs_grad:.6f}")

print("\n所有梯度标签 (前20个):")
print("-" * 80)
for tag in sorted(gradient_hists)[:20]:
    events = ea_v2.Histograms(tag)
    if len(events) > 0:
        latest = events[-1]
        hist_values = latest.histogram_value
        max_abs_grad = max(abs(hist_values.min), abs(hist_values.max))
        print(f"  {tag}: {max_abs_grad:.6f}")

# 查看标量
print("\n" + "=" * 80)
print("标量指标:")
print("=" * 80)
scalars = ea_v2.Tags()['scalars']
for tag in sorted(scalars):
    events = ea_v2.Scalars(tag)
    if len(events) > 0:
        print(f"  {tag}: {events[-1].value:.6f} (step {events[-1].step})")
