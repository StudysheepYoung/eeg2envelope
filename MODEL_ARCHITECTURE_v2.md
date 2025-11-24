# Conformer-v2 模型详细架构文档

## 概述

本模型是基于 Conformer 的 EEG 信号处理模型（改进版 v2），用于从 EEG 信号预测语音包络。主要改进包括全局残差连接、门控机制、MLP 输出头和梯度缩放策略。

**输入**: `[Batch, 64通道, 640时间点]` - 10秒的64通道EEG信号
**输出**: `[Batch, 640时间点, 1]` - 预测的语音包络

---

## 整体架构流程图

```
输入: EEG信号 [B, 64, 640]
    ↓
┌───────────────────────────────────────────────────────────────┐
│ 1. CNN特征提取器 (3层)                                         │
│    - Conv1D(64→256, k=7) + LayerNorm + LeakyReLU + Dropout   │
│    - Conv1D(256→256, k=5) + LayerNorm + LeakyReLU + Dropout  │
│    - Conv1D(256→256, k=3) + LayerNorm + LeakyReLU + Dropout  │
│    输出: [B, 256, 640]                                         │
└───────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────┐
│ 2. SE通道注意力模块                                            │
│    - AdaptiveAvgPool1d                                        │
│    - FC(256→16) + LeakyReLU + FC(16→256) + Sigmoid          │
│    - Channel-wise scaling                                     │
│    输出: [B, 256, 640]                                         │
└───────────────────────────────────────────────────────────────┘
    ↓
    转置: [B, 640, 256]
    ↓
┌───────────────────────────────────────────────────────────────┐
│ 3. 受试者全局条件化 (Global Conditioner)                       │
│    - One-hot编码: sub_id → [B, 71]                           │
│    - Linear投影: [B, 71] → [B, 256]                          │
│    - 广播相加: output = output + sub_emb.unsqueeze(1)         │
│    输出: [B, 640, 256]                                         │
└───────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────┐
│ 4. 正弦位置编码 (可选)                                         │
│    - Sinusoidal Positional Encoding                           │
│    - PE[pos, 2i] = sin(pos / 10000^(2i/d_model))            │
│    - PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))          │
│    - 直接相加到输入                                            │
│    输出: [B, 640, 256]                                         │
└───────────────────────────────────────────────────────────────┘
    ↓
    ┌─────────────────────────────────────────────┐
    │ 保存为 conformer_input (用于全局残差)        │
    └─────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────┐
│ 5. Conformer编码器层栈 (重复 n_layers=8 次)                   │
│                                                                │
│   每个 ConformerBlock 的结构:                                  │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ 5.1 Feed-Forward Module 1 (Macaron前半)              │   │
│   │     - LayerNorm                                       │   │
│   │     - Linear(256→1024) + Swish + Dropout             │   │
│   │     - Linear(1024→256) + Dropout                     │   │
│   │     - 输出 × 0.5 (Macaron缩放)                        │   │
│   │     - 残差连接: output = output + 0.5 * ffn(input)    │   │
│   └──────────────────────────────────────────────────────┘   │
│   ↓                                                            │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ 5.2 相对位置多头自注意力模块                          │   │
│   │     - Pre-LayerNorm                                   │   │
│   │     - Q, K, V 线性投影: [B,T,256] → [B,4,T,64]       │   │
│   │     - 计算注意力分数: scores = QK^T / √64             │   │
│   │     - 添加相对位置偏置:                                │   │
│   │       * 相对位置嵌入: [T, T, 64] (可学习参数)         │   │
│   │       * rel_scores = einsum('bhik,ijk->bhij', Q, rel) │   │
│   │       * scores = scores + rel_scores / √64            │   │
│   │     - Softmax + Dropout                               │   │
│   │     - 加权求和: output = Attention(V)                 │   │
│   │     - 输出投影: [B,4,T,64] → [B,T,256]                │   │
│   │     - 残差连接: output = input + attn(input)          │   │
│   └──────────────────────────────────────────────────────┘   │
│   ↓                                                            │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ 5.3 卷积模块 (Conformer特色)                          │   │
│   │     - LayerNorm                                       │   │
│   │     - Pointwise Conv: 256→512, k=1                    │   │
│   │     - GLU激活: 512 → 256 (门控线性单元)               │   │
│   │     - Depthwise Conv: 256→256, k=31, groups=256       │   │
│   │     - BatchNorm1d + Swish                             │   │
│   │     - Pointwise Conv: 256→256, k=1                    │   │
│   │     - Dropout                                          │   │
│   │     - 残差连接: output = input + conv(input)          │   │
│   └──────────────────────────────────────────────────────┘   │
│   ↓                                                            │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ 5.4 Feed-Forward Module 2 (Macaron后半)              │   │
│   │     - LayerNorm                                       │   │
│   │     - Linear(256→1024) + Swish + Dropout             │   │
│   │     - Linear(1024→256) + Dropout                     │   │
│   │     - 输出 × 0.5 (Macaron缩放)                        │   │
│   │     - 残差连接: output = output + 0.5 * ffn(input)    │   │
│   └──────────────────────────────────────────────────────┘   │
│   ↓                                                            │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ 5.5 Final LayerNorm                                   │   │
│   └──────────────────────────────────────────────────────┘   │
│                                                                │
│   (重复8次，层间无显式残差，每个Block内部已有残差)           │
│   输出: [B, 640, 256]                                          │
└───────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────┐
│ 6. 门控残差连接 (v2核心改进)                                  │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ 输入: conformer_output, conformer_input              │   │
│    │                                                       │   │
│    │ 门控网络:                                             │   │
│    │   - 全局特征: global_feat = mean(conformer_output, dim=1) │
│    │     形状: [B, 256]                                    │   │
│    │   - FC1: 256 → 64 + ReLU                             │   │
│    │   - FC2: 64 → 256 + Sigmoid                          │   │
│    │     gate: [B, 256] → unsqueeze → [B, 1, 256]        │   │
│    │                                                       │   │
│    │ 门控融合:                                             │   │
│    │   output = gate * conformer_output +                 │   │
│    │            (1 - gate) * conformer_input              │   │
│    │                                                       │   │
│    │ 含义:                                                 │   │
│    │   - gate接近1: 信任Conformer的变换                   │   │
│    │   - gate接近0: 保留原始输入特征                       │   │
│    │   - 网络自适应学习跳跃连接的权重                      │   │
│    └─────────────────────────────────────────────────────┘   │
│    输出: [B, 640, 256]                                         │
└───────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────┐
│ 7. 梯度缩放 (v2核心改进) - 仅在训练时生效                     │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ GradientScaleFunction (自定义autograd函数)           │   │
│    │                                                       │   │
│    │ 前向传播: y = x (恒等映射)                            │   │
│    │ 反向传播: ∂L/∂x = scale × ∂L/∂y                      │   │
│    │                                                       │   │
│    │ scale = 2.0 (可配置)                                  │   │
│    │                                                       │   │
│    │ 效果:                                                 │   │
│    │ - 放大从输出层反向传播回来的梯度                      │   │
│    │ - 让前面的CNN和Conformer层获得更大的梯度             │   │
│    │ - 缓解梯度消失问题                                    │   │
│    │ - 帮助前层特征提取器学习                              │   │
│    └─────────────────────────────────────────────────────┘   │
│    输出: [B, 640, 256] (值不变，只改变梯度流)                 │
└───────────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────────┐
│ 8. 改进的MLP输出头 (v2核心改进)                               │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ ImprovedOutputHead                                    │   │
│    │   - LayerNorm(256)                                    │   │
│    │   - Linear(256 → 128)                                 │   │
│    │   - GELU激活                                          │   │
│    │   - Dropout(0.3)                                      │   │
│    │   - Linear(128 → 1)                                   │   │
│    │                                                       │   │
│    │ 对比原始版本:                                         │   │
│    │   - 原始: 单层 Linear(256 → 1)                        │   │
│    │   - 改进: 两层MLP，增强非线性变换能力                │   │
│    └─────────────────────────────────────────────────────┘   │
│    输出: [B, 640, 1]                                           │
└───────────────────────────────────────────────────────────────┘
    ↓
最终输出: 预测的语音包络 [B, 640, 1]
```

---

## 详细模块说明

### 1. CNN特征提取器

**位置**: `FFT_block_conformer_v2.py:198-211`

```python
# 第一层
self.conv1 = nn.Conv1d(64, 256, kernel_size=7, padding=3)
self.norm1 = nn.LayerNorm(256)
self.act1 = nn.LeakyReLU(negative_slope=0.01)
self.drop1 = nn.Dropout(0.3)

# 第二层
self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
self.norm2 = nn.LayerNorm(256)
self.act2 = nn.LeakyReLU(negative_slope=0.01)
self.drop2 = nn.Dropout(0.3)

# 第三层
self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
self.norm3 = nn.LayerNorm(256)
self.act3 = nn.LeakyReLU(negative_slope=0.01)
self.drop3 = nn.Dropout(0.3)
```

**特点**:
- 逐层减小卷积核 (7→5→3)，先捕捉长期依赖，再捕捉局部细节
- LayerNorm而非BatchNorm：对时序数据更稳定
- LeakyReLU (slope=0.01)：防止神经元死亡
- 每层后接Dropout (0.3)：防止过拟合

**梯度增强设置**:
- 通过LLRD (Layer-wise Learning Rate Decay)，前层学习率设为 `base_lr × 3.0`
- 参见 `train_v10_conformer_v2.py:73, 194-196`

---

### 2. SE通道注意力模块

**位置**: `FFT_block_conformer_v2.py:51-68`

```python
class SEBlock(nn.Module):
    def __init__(self, channel=256, reduction=16):
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # [B,C,T] → [B,C,1]
        self.fc = nn.Sequential(
            nn.Linear(256, 16),      # 降维 (reduction=16)
            nn.LeakyReLU(0.01),
            nn.Linear(16, 256),      # 升维
            nn.Sigmoid()             # 输出通道权重 [0,1]
        )

    def forward(self, x):
        # x: [B, 256, 640]
        y = self.avg_pool(x)         # → [B, 256, 1]
        y = self.fc(y.view(B, 256))  # → [B, 256]
        y = y.view(B, 256, 1)
        return x * y.expand_as(x)    # 逐通道缩放
```

**作用**:
- 自适应学习每个通道的重要性
- 对重要的EEG通道赋予更高权重
- 减少冗余通道的影响

**梯度增强设置**: SE模块属于前层，学习率 `base_lr × 3.0`

---

### 3. 受试者全局条件化

**位置**: `FFT_block_conformer_v2.py:239, 281-286`

```python
# 初始化
self.sub_proj = nn.Linear(71, 256)

# 前向传播
sub_emb = F.one_hot(sub_id, 71)            # [B] → [B, 71]
sub_emb = self.sub_proj(sub_emb.float())   # [B, 71] → [B, 256]
output = output + sub_emb.unsqueeze(1)     # [B,640,256] + [B,1,256]
```

**作用**:
- 每个受试者有独特的EEG特征
- One-hot编码保证受试者间独立性
- 投影到d_model维度后广播相加，作为全局偏置

**为什么重要**:
- 受试者间EEG信号差异巨大（头型、电极位置、个体差异）
- 添加受试者信息显著提升模型性能
- 相当于给每个受试者学习一个专属的"偏置向量"

---

### 4. 正弦位置编码

**位置**: `FFT_block_conformer_v2.py:23-48`

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_seq_len=640):
        pe = torch.zeros(640, 256)
        position = torch.arange(0, 640).unsqueeze(1)  # [640, 1]
        div_term = exp(arange(0, 256, 2) * (-log(10000.0) / 256))

        pe[:, 0::2] = sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = cos(position * div_term)  # 奇数维度

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, 640, 256]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

**作用**:
- 为每个时间步提供位置信息
- 正弦函数具有周期性和平滑性
- 不同频率的正弦波编码不同尺度的位置信息

**与相对位置编码的关系**:
- 正弦位置编码：绝对位置信息
- Conformer内部的相对位置编码：相对位置关系
- 两者互补，提供更丰富的位置信息

---

### 5. Conformer编码器层栈

#### 5.1 Feed-Forward Module (Macaron前半)

**位置**: `ConformerLayers.py:268-317, 364`

```python
class FeedForwardModule(nn.Module):
    def __init__(self, d_model=256, d_inner=1024, use_macaron=True):
        self.layer_norm = nn.LayerNorm(256)
        self.w_1 = nn.Linear(256, 1024)    # 扩张4倍
        self.w_2 = nn.Linear(1024, 256)    # 压缩回原始
        self.activation = Swish()          # x * sigmoid(x)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        if self.use_macaron:
            x = 0.5 * x  # Macaron缩放

        return x + residual
```

**Macaron-style FFN的设计理念**:
- 传统Transformer: 只在Attention后有1个FFN
- Macaron: Attention前后各1个FFN，每个输出乘以0.5
- 数学等价性: `0.5*FFN1 + Attn + 0.5*FFN2 ≈ Attn + FFN`
- 实际效果: 增强特征变换，提升模型表达能力

**梯度增强设置**:
- 前半Conformer层 (layer 0-3): 学习率 `base_lr × 3.0`
- 后半Conformer层 (layer 4-7): 学习率 `base_lr × 2.0`

---

#### 5.2 相对位置多头自注意力

**位置**: `ConformerLayers.py:162-265`

**标准自注意力**:
```python
Q = W_q × X  # [B, T, 256] → [B, n_head=4, T, d_k=64]
K = W_k × X
V = W_v × X

scores = (Q @ K^T) / √d_k         # [B, 4, T, T]
attention = softmax(scores)        # [B, 4, T, T]
output = attention @ V             # [B, 4, T, 64]
```

**相对位置增强**:
```python
# 创建相对位置嵌入矩阵
rel_pos_emb: [2*max_len-1, d_k] (可学习参数)

# 对于序列长度T=640:
# 位置差范围: [-639, +639]
# 嵌入矩阵索引: [0, 1278]

rel_pos = rel_pos_emb[center-(T-1):center+T]  # 提取[T, T, d_k]

# 计算相对位置注意力分数
# einsum 'bhik,ijk->bhij' 含义:
#   b: batch维度
#   h: head维度
#   i: query位置
#   j: key位置
#   k: d_k维度
rel_scores = einsum('bhik,ijk->bhij', Q, rel_pos)

# 最终注意力分数
scores = (Q @ K^T) / √d_k + rel_scores / √d_k
```

**相对位置编码的优势**:
1. **长度泛化**: 训练时T=640，推理时可用于T>640
2. **平移不变性**: 相对位置不受绝对位置影响
3. **距离衰减**: 相对位置嵌入自动学习距离衰减模式

**内存优化**:
- 朴素实现: 创建 `[B, n_head, T, T, d_k]` 张量 → 内存爆炸
- 当前实现: 用einsum计算 → 内存高效
- 复杂度: O(B × n_head × T² × d_k) → O(T² × d_k)

---

#### 5.3 卷积模块

**位置**: `ConformerLayers.py:32-116`

```python
class ConvolutionModule(nn.Module):
    def forward(self, x):  # x: [B, T, 256]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, 256, T]

        # 1. Pointwise扩张 + GLU
        x = self.pointwise_conv1(x)  # [B, 512, T]
        x = self.glu(x)              # [B, 256, T] (门控激活)

        # 2. Depthwise卷积
        x = self.depthwise_conv(x)   # [B, 256, T], groups=256
        x = self.batch_norm(x)
        x = self.activation(x)       # Swish

        # 3. Pointwise压缩
        x = self.pointwise_conv2(x)  # [B, 256, T]
        x = self.dropout(x)

        return x.transpose(1, 2)     # [B, T, 256]
```

**GLU (Gated Linear Unit)**:
```python
input: [B, 512, T]
split into two halves:
    value: [B, 256, T]
    gate:  [B, 256, T]

output = value * sigmoid(gate)  # [B, 256, T]
```

**Depthwise卷积 (groups=256)**:
```
标准卷积: 每个输出通道看到所有输入通道
Depthwise: 每个输出通道只看到对应的输入通道

参数量对比:
- 标准卷积: 256 × 256 × 31 = 2,031,616
- Depthwise: 256 × 1 × 31 = 7,936
减少 256 倍参数!
```

**为什么Conformer要加卷积?**
- **局部特征**: Attention擅长全局，卷积擅长局部
- **平移等变性**: 卷积对平移不变的模式建模更好
- **计算效率**: Depthwise卷积比全连接快
- **互补性**: 卷积+Attention > 单独使用任一

---

#### 5.4 Feed-Forward Module (Macaron后半)

与5.1相同，但在整个Block的末尾，再次进行特征变换。

---

#### 5.5 Final LayerNorm

每个ConformerBlock最后的归一化层，稳定输出分布。

---

### 6. 门控残差连接 (v2核心改进)

**位置**: `FFT_block_conformer_v2.py:71-107`

```python
class GatedResidual(nn.Module):
    def __init__(self, d_model=256):
        # 门控网络
        self.gate_layer1 = nn.Linear(256, 64)
        self.gate_layer2 = nn.Linear(64, 256)
        self.activation = nn.ReLU()

    def forward(self, x, residual):
        # x: Conformer输出 [B, 640, 256]
        # residual: Conformer输入 [B, 640, 256]

        # 1. 计算全局特征
        global_feat = x.mean(dim=1)  # [B, 256]

        # 2. 门控网络
        gate = self.gate_layer1(global_feat)  # [B, 64]
        gate = self.activation(gate)
        gate = self.gate_layer2(gate)         # [B, 256]
        gate = torch.sigmoid(gate)            # [B, 256], 范围[0,1]
        gate = gate.unsqueeze(1)              # [B, 1, 256]

        # 3. 门控融合
        output = gate * x + (1 - gate) * residual
        return output
```

**设计动机**:
```
问题: 8层Conformer后，某些重要的浅层特征可能被过度变换

简单残差: output = x + residual
  - 固定权重 (1:1)
  - 无法自适应

门控残差: output = gate * x + (1 - gate) * residual
  - 动态权重 (gate是学习的)
  - 网络自己决定保留多少原始特征
```

**门控值的含义**:
```python
gate = 0.9:  90%使用Conformer输出, 10%保留原始输入
             → Conformer变换很好，信任它

gate = 0.2:  20%使用Conformer输出, 80%保留原始输入
             → Conformer变换有问题，走捷径

gate = 0.5:  平衡两者
             → 两种特征都有用
```

**为什么用全局特征控制门控?**
- 如果用逐位置门控 `[B, T, d_model]`：参数量太大，容易过拟合
- 用全局平均 `[B, d_model]`：参数量小，泛化性好
- 假设：整个序列的变换质量是一致的

**梯度增强设置**:
- 门控残差层学习率 `base_lr × 2.0`
- 属于模型后部，学习率稍低防止过拟合

---

### 7. 梯度缩放 (v2核心改进)

**位置**: `FFT_block_conformer_v2.py:323-336, 310-311`

```python
class GradientScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x  # 前向传播: 恒等映射

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播: 梯度乘以scale
        return grad_output * ctx.scale, None

# 使用
if self.gradient_scale != 1.0 and self.training:
    output = GradientScaleFunction.apply(output, 2.0)
```

**工作原理**:

```
前向传播链:
CNN → SE → Conformer → Gate → [GradScale] → OutputHead
                                   ↑
                           梯度放大2倍的位置

反向传播链:
                      ∂L/∂gate = g
                           ↓
                      ∂L/∂gate × 2.0 = 2g  ← 梯度放大
                           ↓
CNN ← SE ← Conformer ← Gate
∂L/∂CNN = 2g × ...
```

**为什么需要梯度缩放?**

典型问题: **梯度消失**
```
输出层梯度: ∂L/∂output_head = 0.5
经过8层Conformer: ∂L/∂conformer_input = 0.5 × 0.8^8 ≈ 0.084
经过SE: ∂L/∂se = 0.084 × 0.9 ≈ 0.076
经过CNN: ∂L/∂cnn = 0.076 × 0.9 × 0.9 × 0.9 ≈ 0.055

问题: CNN层梯度只有输出层的 1/9
      → CNN学习慢，特征提取不足
```

**添加2倍梯度缩放后**:
```
输出层梯度: ∂L/∂output_head = 0.5
梯度缩放: 0.5 × 2.0 = 1.0
经过门控残差: ∂L/∂conformer = 1.0 × ...
经过Conformer: ∂L/∂conformer_input = 1.0 × 0.8^8 ≈ 0.168
经过SE: ∂L/∂se = 0.168 × 0.9 ≈ 0.151
经过CNN: ∂L/∂cnn = 0.151 × 0.9 × 0.9 × 0.9 ≈ 0.110

结果: CNN层梯度是原来的 2 倍
      → CNN学习更快
```

**为什么放在门控残差之后?**

代码演进:
```python
# 早期版本 (不好)
conformer_out = conformer_layers(x)
conformer_out_scaled = GradScale(conformer_out, 2.0)  # 缩放Conformer输出
output = gate(conformer_out_scaled, x)

问题: gate会稀释梯度增强效果
      如果gate=0.5, 实际梯度增强只有1.5倍

# 当前版本 (好)
conformer_out = conformer_layers(x)
gated_out = gate(conformer_out, x)
output = GradScale(gated_out, 2.0)  # 缩放融合后的输出

优势: 梯度增强完整作用于整条路径
      无论gate多少, 都能保证2倍增强
```

**与LLRD的协同**:

两种互补的梯度增强策略:
```
1. LLRD (Layer-wise Learning Rate Decay):
   - 前层: lr × 3.0
   - 后层: lr × 2.0
   - 输出: lr × 0.5

   效果: 前层每步更新更多

2. 梯度缩放:
   - 全局放大反向传播的梯度

   效果: 前层获得更大的梯度信号

协同: LLRD增大步长 + 梯度缩放增大方向
      → 前层学习大幅加速
```

---

### 8. 改进的MLP输出头

**位置**: `FFT_block_conformer_v2.py:110-137`

```python
class ImprovedOutputHead(nn.Module):
    def __init__(self, d_model=256, dropout=0.3):
        self.net = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),      # 降维到一半
            nn.GELU(),                # 平滑非线性
            nn.Dropout(0.3),
            nn.Linear(128, 1)         # 输出标量
        )
```

**对比原始版本**:
```python
# 原始 (单层)
self.fc = nn.Linear(256, 1)

# 改进 (两层MLP)
LayerNorm → Linear(256→128) → GELU → Dropout → Linear(128→1)
```

**为什么需要MLP输出头?**

从特征到预测的映射可能是非线性的:
```
特征空间 (256维):
  - 包含语音内容、韵律、语速等多种信息
  - 高维复杂分布

预测目标 (1维):
  - 语音包络 (幅度)
  - 需要从复杂特征中提取关键信息

单层线性:
  - output = W @ features + b
  - 只能做线性组合
  - 表达能力有限

两层MLP:
  - hidden = GELU(W1 @ features)
  - output = W2 @ hidden
  - 可以做非线性变换
  - 表达能力更强
```

**GELU vs ReLU**:
```python
ReLU(x) = max(0, x)
  - 硬截断, x<0时梯度=0

GELU(x) ≈ x * Φ(x)  (Φ是标准正态分布的CDF)
  - 平滑激活
  - 负值也有小梯度
  - Transformer常用, 效果更好
```

**梯度增强设置**:
- 输出头学习率 `base_lr × 0.5`
- 学习率最低，防止输出层主导训练

**输出头梯度缩放**:
```python
# train_v10_conformer_v2.py:249-269, 492-494
def scale_output_gradients(model, scale=0.5):
    """在backward后调用，缩小输出层梯度"""
    for name, param in model.named_parameters():
        if 'output_head' in name or 'fc' in name:
            if param.grad is not None:
                param.grad.data.mul_(scale)

# 使用
loss.backward()
scale_output_gradients(model, scale=0.5)  # 缩小输出层梯度
optimizer.step()
```

**效果**:
- LLRD: 输出层学习率 0.5x
- 梯度缩放: 输出层梯度 0.5x
- 综合: 输出层更新幅度是前层的 0.25x
- 目的: 让前层主导学习，输出层缓慢适应

---

## 训练中的梯度增强策略总结

### 1. LLRD (Layer-wise Learning Rate Decay)

**位置**: `train_v10_conformer_v2.py:173-246`

```python
def get_llrd_param_groups(model, base_lr, front_scale, back_scale, output_scale):
    front_layers = ['conv1', 'conv2', 'conv3', 'se', 'sub_proj', 'pos_encoder']
    output_layers = ['output_head', 'fc']

    # 前半Conformer (layer 0-3)
    mid_layer = n_layers // 2  # 8 // 2 = 4

    return [
        {'params': front_params, 'lr': base_lr * 3.0},   # 前层
        {'params': back_params, 'lr': base_lr * 2.0},    # 后层
        {'params': output_params, 'lr': base_lr * 0.5}   # 输出层
    ]
```

**层的划分**:
```
前层 (lr × 3.0):
  - CNN: conv1, conv2, conv3, norm1-3, act1-3, drop1-3
  - SE: se
  - 位置编码: sub_proj, pos_encoder
  - Conformer 前半: layer_stack.0 ~ layer_stack.3

后层 (lr × 2.0):
  - Conformer 后半: layer_stack.4 ~ layer_stack.7
  - 门控残差: gated_residual

输出层 (lr × 0.5):
  - MLP输出头: output_head
  - 或单层线性: fc
```

**举例**:
```python
base_lr = 0.0001

前层参数更新:
  θ_front_new = θ_front - 0.0003 × ∇L  # 学习率3倍

输出层参数更新:
  θ_output_new = θ_output - 0.00005 × ∇L  # 学习率减半

比例: 前层步长是输出层的 6 倍!
```

---

### 2. Conformer层内的梯度缩放

**位置**: `FFT_block_conformer_v2.py:310-311`

```python
if self.gradient_scale != 1.0 and self.training:
    output = GradientScaleFunction.apply(output, 2.0)
```

**放置位置**: 门控残差之后，输出头之前

**梯度流动**:
```
前向:
  gate_out = gated_residual(conformer_out, conformer_in)
  scaled_out = GradScale(gate_out, 2.0)  # 前向恒等
  prediction = output_head(scaled_out)

反向:
  ∂L/∂scaled_out = ∂L/∂prediction  (假设=1.0)
  ∂L/∂gate_out = 2.0 × ∂L/∂scaled_out = 2.0  ← 梯度放大
  ∂L/∂conformer = ... (传播到Conformer)
  ∂L/∂cnn = ... (传播到CNN)
```

---

### 3. 输出层梯度缩放

**位置**: `train_v10_conformer_v2.py:249-269, 492-494`

```python
# 在loss.backward()之后调用
def scale_output_gradients(model, scale=0.5):
    for name, param in model.named_parameters():
        if 'output_head' in name or 'fc' in name:
            param.grad.data.mul_(scale)

# 训练循环
loss.backward()  # 计算梯度
scale_output_gradients(model, 0.5)  # 缩小输出层梯度
optimizer.step()  # 更新参数
```

**效果**:
```
正常情况:
  ∂L/∂output_head_W1 = 0.5
  参数更新: W1 -= lr × 0.5

缩放后:
  ∂L/∂output_head_W1 = 0.5 × 0.5 = 0.25
  参数更新: W1 -= lr × 0.25
```

---

### 4. 三种策略的协同效果

**示例计算** (假设base_lr=0.0001):

```
CNN第一层 (conv1):
  - LLRD: lr = 0.0001 × 3.0 = 0.0003
  - 梯度缩放: grad × 2.0 (从Conformer传回)
  - 实际更新: Δθ = -0.0003 × 2.0 × grad = -0.0006 × grad

输出层 (output_head):
  - LLRD: lr = 0.0001 × 0.5 = 0.00005
  - 输出梯度缩放: grad × 0.5
  - 实际更新: Δθ = -0.00005 × 0.5 × grad = -0.000025 × grad

比例:
  CNN更新幅度 / 输出层更新幅度 = 0.0006 / 0.000025 = 24倍!
```

**设计哲学**:

```
问题诊断:
  - 深层网络容易"输出层主导训练"
  - 输出层梯度大 → 更新快 → 过早收敛
  - 前层梯度小 → 学习慢 → 特征提取不足

解决策略:
  1. 限制输出层:
     - LLRD降低学习率 (0.5x)
     - 梯度缩放降低梯度 (0.5x)
     → 输出层更新变慢

  2. 增强前层:
     - LLRD提高学习率 (3.0x)
     - Conformer梯度缩放放大梯度 (2.0x)
     → 前层更新加速

  3. 平衡结果:
     - 前层和后层学习速度接近
     - 整个网络协同训练
     - 特征提取充分
```

---

## 模型参数统计

### 参数量估算

```python
# 1. CNN特征提取器
conv1: 64 × 256 × 7 = 114,688
conv2: 256 × 256 × 5 = 327,680
conv3: 256 × 256 × 3 = 196,608
LayerNorm × 3: 256 × 2 × 3 = 1,536
小计: ~640K

# 2. SE通道注意力
fc1: 256 × 16 = 4,096
fc2: 16 × 256 = 4,096
小计: ~8K

# 3. 受试者embedding
sub_proj: 71 × 256 = 18,176
小计: ~18K

# 4. Conformer层 (单层)
FFN1: 256×1024 + 1024×256 = 524,288 × 2 = 1,048,576
Attention: (256×256)×4 + 相对位置嵌入 ≈ 300K
ConvModule: 估计 ~150K
FFN2: 1,048,576
单层小计: ~2.5M
8层总计: ~20M

# 5. 门控残差
gate_layer1: 256 × 64 = 16,384
gate_layer2: 64 × 256 = 16,384
小计: ~32K

# 6. MLP输出头
layer1: 256 × 128 = 32,768
layer2: 128 × 1 = 128
小计: ~33K

总计: ~21M参数
模型大小: 21M × 4 bytes ≈ 84 MB (Float32)
```

### 内存占用估算 (batch_size=64, seq_len=640)

```python
输入: [64, 640, 64] → 64×640×64×4 = 10.5 MB

中间激活 (最大):
  Conformer attention scores: [64, 4, 640, 640] × 8层
  = 64 × 4 × 640 × 640 × 4 × 8 = 2,621 MB ≈ 2.6 GB

梯度: ~84 MB (与参数量相同)

总计: ~2.7 GB (单个GPU)
```

---

## 前向传播时间复杂度分析

```python
输入: [B, 640, 64]

1. CNN:
   Conv1D: O(B × 640 × 64 × 256 × 7) = O(B × 70M)
   总计 (3层): O(B × 150M)

2. SE:
   AdaptiveAvgPool: O(B × 256 × 640)
   FC: O(B × 256 × 16) + O(B × 16 × 256)
   总计: O(B × 0.2M)

3. Conformer (单层):
   Attention:
     Q,K,V投影: O(B × 640 × 256 × 256) = O(B × 42M)
     Attention计算: O(B × 4 × 640 × 640 × 64) = O(B × 105M)
     输出投影: O(B × 640 × 256 × 256) = O(B × 42M)
   小计: O(B × 189M)

   ConvModule:
     Depthwise: O(B × 256 × 640 × 31) = O(B × 5M)

   FFN: O(B × 640 × 256 × 1024) × 2 = O(B × 335M)

   单层总计: O(B × 529M)
   8层总计: O(B × 4.2G)

4. 输出头:
   MLP: O(B × 640 × 256 × 128) + O(B × 640 × 128) = O(B × 21M)

总复杂度: O(B × 4.4G) 浮点运算/样本
```

**瓶颈分析**:
- Conformer的Attention占 ~47%
- Conformer的FFN占 ~38%
- 其他占 ~15%

**优化建议**:
- 减少Conformer层数 (8→6)
- 减小FFN内部维度 (1024→768)
- 使用Flash Attention优化注意力计算

---

## 配置参数总览

### 当前配置 (train_v10_conformer_v2.py)

```python
# 模型结构
in_channel = 64           # EEG通道数
d_model = 256             # 模型维度
d_inner = 1024            # FFN内部维度
n_head = 4                # 注意力头数
n_layers = 8              # Conformer层数
conv_kernel_size = 31     # Conformer卷积核大小
dropout = 0.3             # Dropout率

# v2改进参数
use_gated_residual = True     # 使用门控残差
use_mlp_head = True           # 使用MLP输出头
gradient_scale = 2.0          # Conformer梯度缩放因子

# LLRD参数
use_llrd = True               # 使用分层学习率
llrd_front_scale = 3.0        # 前层学习率倍率
llrd_back_scale = 2.0         # 后层学习率倍率
llrd_output_scale = 0.5       # 输出层学习率倍率

# 输出层梯度缩放
output_grad_scale = 0.5       # 输出层梯度缩放因子

# 训练参数
learning_rate = 0.0001        # 基础学习率
batch_size = 64               # 批大小
windows_per_sample = 20       # 每个样本采样窗口数
```

---

## 损失函数 (V7 - 当前版本)

**位置**: `train_v10_conformer_v2.py:473-481`

```python
# 多尺度Pearson Loss
l_pearson = multi_scale_pearson_loss(outputs, labels, scales=[2, 4, 8, 16])

# Huber Loss (平滑L1)
l_huber = F.smooth_l1_loss(outputs, labels, reduction='none', beta=0.1).mean()

# 总损失
loss = l_pearson.mean() + 0.1 * l_huber
```

### 多尺度Pearson Loss

**位置**: `util/cal_pearson.py`

```python
def multi_scale_pearson_loss(pred, target, scales=[2, 4, 8, 16]):
    """
    在多个时间尺度上计算Pearson相关系数损失

    尺度2: 下采样2倍 (640 → 320点)
    尺度4: 下采样4倍 (640 → 160点)
    尺度8: 下采样8倍 (640 → 80点)
    尺度16: 下采样16倍 (640 → 40点)
    """
    total_loss = pearson_loss(pred, target)  # 原始尺度

    for scale in scales:
        # 平均池化降采样
        pred_pooled = F.avg_pool1d(pred.transpose(1,2), scale, scale).transpose(1,2)
        target_pooled = F.avg_pool1d(target.transpose(1,2), scale, scale).transpose(1,2)

        # 计算Pearson loss
        total_loss += pearson_loss(pred_pooled, target_pooled)

    return total_loss / (len(scales) + 1)
```

**为什么用多尺度?**
- 语音包络有多个频率成分
- 快变化: 音素转换 (几十毫秒)
- 慢变化: 韵律、语调 (几百毫秒)
- 多尺度确保各频段都预测准确

### Huber Loss

```python
HuberLoss(x, y) = {
    0.5 × (x - y)²         if |x - y| ≤ β
    β × (|x - y| - 0.5β)   if |x - y| > β
}

β = 0.1 (当前配置)
```

**为什么用Huber而非MSE?**
```
MSE: L = (pred - target)²
  - 对异常值敏感
  - 大误差主导损失
  - 峰值预测不准时惩罚过大

Huber: 小误差用L2, 大误差用L1
  - 对异常值鲁棒
  - 梯度更平滑
  - 与Pearson兼容性好
```

---

## 改进版本对比

| 特性 | v1 (原始) | v2 (改进) |
|-----|----------|-----------|
| **残差连接** | 仅ConformerBlock内部 | + 全局门控残差 |
| **输出头** | 单层Linear(256→1) | MLP (256→128→1) |
| **梯度策略** | 无特殊处理 | LLRD + 梯度缩放 × 2 |
| **学习率** | 统一0.0001 | 分层: 3.0x / 2.0x / 0.5x |
| **参数量** | ~20.9M | ~21.0M (+0.5%) |
| **训练时间** | 1x | ~1.02x (几乎无增加) |
| **性能提升** | 基准 | Pearson +0.03~0.05 |

---

## 模型训练监控指标

### TensorBoard日志

```python
# 训练指标
Train/total_loss              # 总损失
Train/pearson_loss            # Pearson损失
Train/huber_loss              # Huber损失
Train/var_ratio               # 方差比 (pred_var / target_var)
Train/pearson_metric          # Pearson相关系数 (指标)

# 验证指标
Val/loss                      # 验证损失
Val/metric                    # 验证Pearson相关系数

# 测试指标
Test/loss                     # 测试损失
Test/metric                   # 测试Pearson相关系数
Test_MultiScale/pearson_scale_1   # 原始尺度
Test_MultiScale/pearson_scale_4   # 4倍下采样
Test_MultiScale/pearson_scale_8   # 8倍下采样
Test_MultiScale/pearson_scale_16  # 16倍下采样
Test_MultiScale/pearson_scale_32  # 32倍下采样

# 梯度监控 (每100步)
Gradients/conv1.weight        # CNN第一层梯度范数
Gradients/layer_stack.0.*     # Conformer第一层梯度
Gradients/layer_stack.7.*     # Conformer最后一层梯度
Gradients/output_head.*       # 输出头梯度
```

### 关键指标解读

**方差比 (var_ratio)**:
```python
var_ratio = var(pred) / var(target)

理想值: 1.0
  - < 0.5: 预测过于平滑 (欠拟合)
  - 0.5-1.5: 合理范围
  - > 2.0: 预测过于波动 (可能过拟合)
```

**Pearson相关系数**:
```python
相关系数范围: [-1, 1]

0.0-0.1: 几乎无相关 (随机猜测)
0.1-0.3: 弱相关
0.3-0.5: 中等相关
0.5-0.7: 强相关
0.7-1.0: 非常强相关

当前模型: ~0.20-0.25 (验证集)
```

---

## 文件结构总览

```
HappyQuokka_system_for_EEG_Challenge/
├── models/
│   ├── FFT_block_conformer_v2.py      # 主模型定义
│   │   ├── Decoder                     # 主模型类
│   │   ├── PositionalEncoding          # 位置编码
│   │   ├── SEBlock                     # SE注意力
│   │   ├── GatedResidual               # 门控残差
│   │   ├── ImprovedOutputHead          # MLP输出头
│   │   └── GradientScaleFunction       # 梯度缩放
│   │
│   └── ConformerLayers.py              # Conformer组件
│       ├── ConformerBlock              # Conformer块
│       ├── RelativeMultiHeadAttention  # 相对位置注意力
│       ├── ConvolutionModule           # 卷积模块
│       ├── FeedForwardModule           # FFN模块
│       ├── GLU                         # 门控线性单元
│       ├── Swish                       # Swish激活
│       └── RelativePositionalEncoding  # 相对位置编码
│
├── train_v10_conformer_v2.py          # 训练脚本
│   ├── get_llrd_param_groups()        # LLRD参数分组
│   ├── scale_output_gradients()       # 输出层梯度缩放
│   └── main()                         # 训练主循环
│
└── util/
    ├── dataset.py                     # 数据加载
    │   └── RegressionDataset          # 数据集类
    │
    └── cal_pearson.py                 # 损失函数
        ├── pearson_loss()             # Pearson损失
        ├── multi_scale_pearson_loss() # 多尺度Pearson
        └── variance_ratio_loss()      # 方差比损失
```

---

## 使用建议

### 1. 如何调整梯度增强强度

```python
# 场景1: 前层学习太慢
llrd_front_scale = 5.0    # 增大 (原3.0)
gradient_scale = 3.0      # 增大 (原2.0)

# 场景2: 前层学习太快，过拟合
llrd_front_scale = 2.0    # 减小
gradient_scale = 1.5      # 减小

# 场景3: 输出层不稳定
llrd_output_scale = 0.3   # 减小 (原0.5)
output_grad_scale = 0.3   # 减小 (原0.5)
```

### 2. 如何简化模型 (加速训练)

```python
# 减少Conformer层数
n_layers = 6  # 原8, 速度提升~25%

# 减小模型维度
d_model = 192   # 原256
d_inner = 768   # 原1024, 速度提升~40%

# 减小注意力头数
n_head = 2  # 原4, 速度提升~10%

# 禁用相对位置编码 (影响性能)
use_relative_pos = False  # 速度提升~15%
```

### 3. 如何增强模型 (提升性能)

```python
# 增加Conformer层数
n_layers = 12  # 原8

# 增大模型维度
d_model = 384   # 原256
d_inner = 1536  # 原1024

# 使用更大的卷积核
conv_kernel_size = 63  # 原31, 增大感受野

# 减小dropout (如果过拟合不严重)
dropout = 0.2  # 原0.3
```

---

## 常见问题 (FAQ)

**Q1: 为什么Conformer层数是8?**
A: 经验值。更少(4-6)可能欠拟合，更多(12+)训练慢且易过拟合。

**Q2: 梯度缩放会影响模型收敛吗?**
A: 不会。它只改变梯度幅度，不改变梯度方向。配合学习率调整使用是安全的。

**Q3: 门控残差的gate值一般是多少?**
A: 训练初期~0.5，后期可能0.6-0.8 (更信任Conformer)。如果gate一直<0.3，说明Conformer学习有问题。

**Q4: 如何判断梯度增强是否有效?**
A: 监控 `Gradients/conv1.weight` 和 `Gradients/output_head.*` 的比值。理想情况两者应该在同一数量级。

**Q5: 多尺度Pearson Loss的scales如何选择?**
A: 根据目标信号的特征频率。语音包络主要能量在1-20Hz，对应时间尺度50-1000ms，所以选[2,4,8,16]合理。

---

## 总结

Conformer-v2 是一个针对 EEG→语音包络预测任务深度优化的模型，核心创新在于：

1. **架构创新**:
   - 全局门控残差 (自适应跳跃连接)
   - MLP输出头 (增强非线性)

2. **训练创新**:
   - LLRD (分层学习率，前层3x，输出层0.5x)
   - 双重梯度缩放 (Conformer内2x，输出后0.5x)

3. **效果**:
   - 解决前层学习不足问题
   - 平衡各层学习速度
   - 提升模型性能 (Pearson +0.03~0.05)

4. **计算成本**:
   - 参数量增加 < 1%
   - 训练时间增加 < 2%
   - 性价比极高

**适用场景**: 深层时序模型，前层特征提取不足，输出层主导训练的问题。
