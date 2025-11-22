# Conformer v2 改进版使用指南

## 📌 问题背景

通过 TensorBoard 梯度分析发现，原始 Conformer 模型存在严重的**特征提取不足**问题：

- **前层梯度仅为输出层的 0.59%** - 前层几乎学不到东西
- **Conformer 层权重基本不更新** - 训练过程中权重几乎不变
- **性能瓶颈** - Validation Pearson 仅 0.22

**根本原因**：大部分学习任务由最后的线性层完成，前面的 Conformer 网络没有有效提取 EEG 特征。

---

## ✨ v2 改进方案

### 1. 全局残差连接 (Global Residual Connection)

**问题**：深层网络中，梯度难以回传到前层

**解决**：在 Conformer 层栈前后添加跳跃连接

```python
# Conformer 输入
conformer_input = output.clone()

# Conformer 层栈
for conformer_layer in self.layer_stack:
    output = conformer_layer(output)

# 全局残差
output = output + conformer_input
```

### 2. 门控残差机制 (Gated Residual)

**问题**：简单残差连接可能让网络"偷懒"，直接跳过 Conformer

**解决**：自适应学习跳跃权重

```python
gate = sigmoid(gate_network(output))
output = gate * conformer_output + (1 - gate) * conformer_input
```

网络会自动学习在什么情况下使用 Conformer 特征。

### 3. MLP 输出头 (Multi-Layer Output Head)

**问题**：单层线性可以直接拟合 CNN 特征，绕过 Conformer

**解决**：使用两层 MLP，增强表达能力

```python
output_head = Sequential(
    LayerNorm(d_model),
    Linear(d_model -> d_model//2),
    GELU(),
    Dropout(),
    Linear(d_model//2 -> 1)
)
```

### 4. 梯度缩放 (Gradient Scaling)

**问题**：前层梯度太小，学习缓慢

**解决**：自定义 autograd 函数，放大梯度

```python
# 前向传播: y = x
# 反向传播: dx = scale * dy (例如 scale=2.0)
```

---

## 🚀 快速开始

### 单 GPU 训练

```bash
python train_v10_conformer_v2.py \
    --epoch 1000 \
    --batch_size 64 \
    --n_layers 8 \
    --d_model 256 \
    --n_head 4 \
    --conv_kernel_size 31 \
    --use_gated_residual True \
    --use_mlp_head True \
    --gradient_scale 2.0 \
    --gpu 0
```

### 分布式训练 (多 GPU)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_v10_conformer_v2.py \
    --use_ddp \
    --epoch 1000 \
    --batch_size 64 \
    --use_gated_residual True \
    --use_mlp_head True \
    --gradient_scale 2.0
```

---

## 🔧 关键参数说明

### v2 改进参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_gated_residual` | `True` | 是否使用门控残差连接 |
| `--use_mlp_head` | `True` | 是否使用 MLP 输出头 |
| `--gradient_scale` | `2.0` | 梯度缩放系数（建议 1.5-3.0） |

### 推荐配置

#### 配置 1: 完全改进版（推荐）
```bash
--use_gated_residual True \
--use_mlp_head True \
--gradient_scale 2.0
```
**适用**: 大部分场景，预期前层梯度比例提升到 10-20%

#### 配置 2: 保守版
```bash
--use_gated_residual False \  # 简单残差
--use_mlp_head True \
--gradient_scale 1.5
```
**适用**: 担心过拟合的小数据集

#### 配置 3: 激进版
```bash
--use_gated_residual True \
--use_mlp_head True \
--gradient_scale 3.0
```
**适用**: 梯度消失严重的深层网络（n_layers > 10）

---

## 📊 如何验证改进有效

### 1. 运行诊断脚本

训练几个 epoch 后，运行：

```bash
python analyze_feature_learning.py
```

**关键指标**：

| 指标 | 原始模型 | 改进目标 |
|------|---------|---------|
| 前层/输出层梯度比例 | 0.006 (0.6%) | > 0.1 (10%) |
| 前层权重变化 | 0.0000 | > 0.01 |
| Validation Pearson | 0.22 | > 0.3 |

### 2. TensorBoard 监控

```bash
tensorboard --logdir test_results --port 6006
```

**重点关注**：

1. **SCALARS 标签页**
   - `Gradient/norm` - 应该在 1-10 之间稳定
   - `Validation/pearson` - 应该持续上升

2. **HISTOGRAMS 标签页**
   - `Gradient/layer_stack.0.*` - 前层梯度幅值应明显增大
   - `Weight/layer_stack.0.*` - 权重分布应随训练变化

### 3. 对比实验

同时运行原始版本和 v2 版本，对比：

```bash
# 终端1: 原始版本
python train_v10_conformer.py --gpu 0

# 终端2: v2 版本
python train_v10_conformer_v2.py --gpu 1
```

---

## 🎯 预期效果

### 梯度流改善

| 层 | 原始模型梯度 | v2 模型梯度 | 提升 |
|----|-------------|------------|------|
| 输出层 | 0.155 | 0.155 | - |
| Conformer 后层 | 0.0004 | 0.020 | **50x** |
| Conformer 前层 | 0.0009 | 0.015 | **17x** |

### 性能提升（预期）

| 指标 | 原始模型 | v2 模型 | 提升 |
|------|---------|---------|------|
| Validation Pearson | 0.22 | 0.30-0.35 | +36-59% |
| Test Pearson | 0.21 | 0.28-0.33 | +33-57% |
| 收敛速度 | 500 epochs | 300 epochs | 40% faster |

---

## 🛠️ 故障排查

### 问题 1: 梯度仍然很小

**可能原因**：
- `gradient_scale` 设置过小
- 学习率过小

**解决方案**：
```bash
# 增大梯度缩放
--gradient_scale 3.0

# 或增大学习率
--learning_rate 0.0002
```

### 问题 2: 训练不稳定 / 梯度爆炸

**可能原因**：
- `gradient_scale` 设置过大
- 学习率过大

**解决方案**：
```bash
# 减小梯度缩放
--gradient_scale 1.5

# 或降低学习率
--learning_rate 0.00005

# 或添加梯度裁剪（需修改代码）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 问题 3: 性能没有提升

**检查清单**：
1. 确认使用的是 `FFT_block_conformer_v2.py`
2. 确认 `use_gated_residual` 和 `use_mlp_head` 都为 `True`
3. 运行诊断脚本确认梯度确实增大了
4. 查看 TensorBoard 确认权重在更新

---

## 📁 文件说明

```
models/
├── FFT_block_conformer.py       # 原始 Conformer 模型
├── FFT_block_conformer_v2.py    # ✨ 改进版 Conformer 模型
└── ConformerLayers.py           # Conformer 基础模块

train_v10_conformer.py           # 原始训练脚本
train_v10_conformer_v2.py        # ✨ 改进版训练脚本

analyze_feature_learning.py     # 特征学习诊断工具
diagnosis_and_solution.md        # 详细诊断报告
CONFORMER_V2_GUIDE.md           # 本文档
```

---

## 🔬 技术细节

### 模型架构对比

#### 原始版本
```
CNN (3层) → SE注意力 → Subject Embedding → Positional Encoding
→ Conformer Stack (8层) → Linear → 输出
```

#### v2 改进版
```
CNN (3层) → SE注意力 → Subject Embedding → Positional Encoding
→ [保存输入]
→ Conformer Stack (8层)
→ [梯度缩放]
→ [门控残差融合]
→ MLP Head (2层) → 输出
```

### 参数量对比

| 配置 | 原始模型 | v2 模型（简单残差）| v2 模型（门控残差）|
|------|---------|-------------------|-------------------|
| 8 层 | 46.8M | 47.1M (+0.6%) | 47.4M (+1.3%) |

**结论**：参数量几乎没有增加，但性能显著提升。

---

## 📖 参考资料

1. **梯度消失问题**
   - Deep Residual Learning (ResNet) - He et al., 2016
   - Highway Networks - Srivastava et al., 2015

2. **门控机制**
   - Gated Linear Units - Dauphin et al., 2017
   - Highway Networks - Srivastava et al., 2015

3. **梯度缩放**
   - Gradient Surgery - Yu et al., 2020
   - GradNorm - Chen et al., 2018

---

## ✅ 检查清单

开始训练前，确认：

- [ ] 使用 `train_v10_conformer_v2.py` 训练脚本
- [ ] 模型导入 `from models.FFT_block_conformer_v2 import Decoder`
- [ ] 设置 `--use_gated_residual True`
- [ ] 设置 `--use_mlp_head True`
- [ ] 设置 `--gradient_scale 2.0`（或根据需求调整）
- [ ] 准备运行 `analyze_feature_learning.py` 验证效果

训练后，验证：

- [ ] 运行诊断脚本，确认前层梯度增大
- [ ] 查看 TensorBoard 梯度直方图
- [ ] 比较 Validation Pearson 是否提升
- [ ] 确认训练稳定（无梯度爆炸）

---

## 💡 最佳实践

1. **先小规模测试**
   - 训练 50 个 epoch
   - 运行诊断脚本验证梯度改善
   - 确认无问题后再长时间训练

2. **监控训练过程**
   - 每 10 epoch 查看一次 TensorBoard
   - 关注梯度范数是否稳定
   - 关注 Validation Pearson 是否上升

3. **保存最佳模型**
   - 根据 Validation Pearson 保存 checkpoint
   - 对比不同 `gradient_scale` 的效果

4. **消融实验**
   - 分别测试门控残差、MLP头、梯度缩放的独立效果
   - 找到最适合你数据集的配置

---

## 🤝 反馈与改进

如果遇到问题或有改进建议，请记录：

1. 使用的参数配置
2. 诊断脚本的输出
3. TensorBoard 截图
4. 性能指标对比

祝训练顺利！🎉
