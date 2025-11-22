# EEG Conformer 模型诊断报告

## 🚨 核心问题

**前面的 Conformer 网络没有有效学习 EEG 特征，大部分工作由最后的线性层完成**

### 证据

1. **梯度幅值差距**
   - 前层梯度: 0.000919
   - 输出层梯度: 0.154806
   - **比例仅 0.59%** ← 前层几乎学不到东西

2. **权重更新停滞**
   - Conformer 层权重在训练过程中基本不变

3. **性能瓶颈**
   - Validation Pearson: 0.22 (很低)
   - Test Pearson: 0.21

---

## 🔍 根本原因

### 1. 损失函数设计问题

**当前代码**:
```python
loss = l_mse + args.lamda * (l_p ** 2)
```

**问题**:
- Pearson 损失被平方后，如果 lambda 较小，贡献微弱
- MSE 损失可能主要依赖线性层的简单拟合

**建议**:
```python
# 方案1: 去掉平方
loss = l_mse + args.lamda * l_p

# 方案2: 调整权重比例
loss = args.alpha * l_mse + args.beta * l_p

# 方案3: 动态调整 (Uncertainty Weighting)
loss = (1 / (2 * sigma_mse**2)) * l_mse + (1 / (2 * sigma_p**2)) * l_p
```

---

### 2. 缺少残差连接 (Skip Connections)

**当前代码**:
```python
for conformer_layer in self.layer_stack:
    output = conformer_layer(output)  # 串行堆叠，无跳跃连接
```

**问题**:
- 深层网络梯度难以回传到前层
- 没有 shortcut 帮助梯度流动

**建议** (修改 `models/FFT_block_conformer.py:200-202`):
```python
# 添加残差连接
residual = output
for conformer_layer in self.layer_stack:
    output = conformer_layer(output)
output = output + residual  # Global residual

# 或者更好的方案：Pre-LN + Residual (Conformer 内部已有)
# 检查 ConformerBlock 内部是否有残差，如果没有需要添加
```

---

### 3. 输出层过于简单

**当前代码**:
```python
self.fc = nn.Linear(d_model, 1)
...
output = self.fc(output)  # 直接映射
```

**问题**:
- 线性层可以直接拟合 CNN 提取的特征，绕过 Conformer
- 没有迫使网络学习更抽象的表征

**建议**:
```python
# 方案1: 添加 MLP 头
self.output_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.LayerNorm(d_model // 2),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(d_model // 2, 1)
)

# 方案2: 添加注意力池化
self.attention_pool = nn.MultiheadAttention(d_model, num_heads=4)
self.fc = nn.Linear(d_model, 1)
```

---

### 4. 学习率策略问题

**可能的问题**:
- 统一的学习率导致前层学习过慢

**建议**:
```python
# Layer-wise Learning Rate Decay (LLRD)
optimizer = torch.optim.AdamW([
    {'params': model.conv1.parameters(), 'lr': base_lr * 0.1},  # CNN 层较小
    {'params': model.layer_stack[:4].parameters(), 'lr': base_lr * 0.5},  # 前半 Conformer
    {'params': model.layer_stack[4:].parameters(), 'lr': base_lr * 0.8},  # 后半 Conformer
    {'params': model.fc.parameters(), 'lr': base_lr},  # 输出层最大
])
```

---

### 5. CNN 特征过强

**观察**:
- 三层 CNN (7x7, 5x5, 3x3) + SE 通道注意力已经提取了很强的局部特征
- Conformer 可能变成了"装饰品"

**建议**:
```python
# 方案1: 减弱 CNN
# - 减少 CNN 层数 (3层 → 2层 或 1层)
# - 减小卷积核 (7x7 → 3x3)

# 方案2: 在 CNN 和 Conformer 之间加瓶颈层
self.bottleneck = nn.Linear(d_model, d_model // 2)
# 强迫 Conformer 从压缩特征中学习
```

---

## 🎯 推荐实施方案 (优先级排序)

### 方案 A: 快速验证 (1小时内)

**只改训练代码，不改模型**

1. **修改损失函数**
   ```python
   # train_v*.py 中
   loss = l_mse + args.lamda * l_p  # 去掉平方
   ```

2. **增大 lambda**
   ```bash
   # 从 0.1 改成 1.0
   --lamda 1.0
   ```

3. **使用差异化学习率**
   ```python
   optimizer = torch.optim.AdamW([
       {'params': model.layer_stack.parameters(), 'lr': args.lr * 2},  # Conformer 提高2倍
       {'params': model.fc.parameters(), 'lr': args.lr},
   ])
   ```

**预期效果**: 前层梯度应该增大到输出层的 5-10%

---

### 方案 B: 架构改进 (需要重新训练)

**修改模型结构**

1. **添加全局残差** (`models/FFT_block_conformer.py:196-204`)
   ```python
   # 保存 Conformer 输入
   conformer_input = output.clone()

   # Conformer 层
   for conformer_layer in self.layer_stack:
       output = conformer_layer(output)

   # 全局残差 + 门控机制
   gate = torch.sigmoid(self.gate_proj(output))  # 学习跳跃权重
   output = gate * output + (1 - gate) * conformer_input
   ```

2. **改进输出头**
   ```python
   self.output_head = nn.Sequential(
       nn.LayerNorm(d_model),
       nn.Linear(d_model, d_model // 2),
       nn.GELU(),
       nn.Dropout(0.1),
       nn.Linear(d_model // 2, 1)
   )
   ```

---

### 方案 C: 全面优化 (最佳效果)

结合 A + B，并额外添加：

1. **梯度监控和自适应调整**
2. **对比学习损失** (让 Conformer 学习判别性特征)
3. **渐进式训练** (先训练 CNN+fc，再解冻 Conformer)

---

## 📊 如何验证改进有效

重新训练后，检查以下指标：

### TensorBoard 中应该看到:

1. **梯度比例改善**
   - 前层/输出层梯度 > 0.1 (从 0.006 提升)

2. **权重更新明显**
   - 前层权重直方图随训练变化

3. **性能提升**
   - Validation Pearson > 0.3 (从 0.22 提升)

### 运行诊断脚本:
```bash
python analyze_feature_learning.py
```

---

## 🛠️ 立即行动

我可以帮你：

1. ✅ **生成改进版训练脚本** (方案 A)
2. ✅ **修改模型架构** (方案 B)
3. ✅ **创建对比实验配置**
4. ✅ **编写自动诊断工具**

需要我实施哪个方案？
