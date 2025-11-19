# HappyQuokka Model Architecture (train_v10_sota)

## 📊 模型概览

**任务**: EEG-to-Speech Envelope Reconstruction (回归任务)
- **输入**: 64通道 EEG信号，10秒 @ 64Hz = [Batch, 64, 640]
- **输出**: 语音包络信号 = [Batch, 640, 1]
- **架构**: CNN Feature Extractor + SE Attention + Transformer Encoder

---

## 🏗️ 完整架构流程

```
输入 EEG [B, 64, 640]
    ↓
┌─────────────────────────────────────────┐
│  1. CNN特征提取器 (3层卷积)             │
│  ┌─────────────────────────────────┐   │
│  │ Conv1D(64→256, k=7, p=3)        │   │
│  │ → LayerNorm → LeakyReLU → Drop  │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ Conv1D(256→256, k=5, p=2)       │   │
│  │ → LayerNorm → LeakyReLU → Drop  │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ Conv1D(256→256, k=3, p=1)       │   │
│  │ → LayerNorm → LeakyReLU → Drop  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    ↓ [B, 256, 640]
┌─────────────────────────────────────────┐
│  2. SE通道注意力模块                    │
│  ┌─────────────────────────────────┐   │
│  │ AdaptiveAvgPool1d → [B, 256, 1] │   │
│  │ FC(256→16) → LeakyReLU           │   │
│  │ FC(16→256) → Sigmoid             │   │
│  │ 加权: X * weights                │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    ↓ [B, 640, 256]
┌─────────────────────────────────────────┐
│  3. 全局条件器 (受试者嵌入)             │
│  ┌─────────────────────────────────┐   │
│  │ One-Hot(sub_id) [B, 71]         │   │
│  │ → Linear(71→256)                │   │
│  │ 广播相加: X + sub_emb           │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    ↓ [B, 640, 256]
┌─────────────────────────────────────────┐
│  4. 位置编码                            │
│  ┌─────────────────────────────────┐   │
│  │ Sinusoidal PE [640, 256]        │   │
│  │ X = X + PE                       │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    ↓ [B, 640, 256]
┌─────────────────────────────────────────┐
│  5. Transformer编码器 (8层)             │
│  ┌─────────────────────────────────┐   │
│  │ PreLNFFTBlock × 8               │   │
│  │  ├─ MultiHeadAttention (4 heads)│   │
│  │  │  ├─ Pre-LayerNorm            │   │
│  │  │  ├─ Q,K,V线性投影             │   │
│  │  │  ├─ Scaled Dot-Product Attn  │   │
│  │  │  ├─ Dropout + Residual        │   │
│  │  │                               │   │
│  │  └─ PositionwiseFeedForward     │   │
│  │     ├─ Pre-LayerNorm             │   │
│  │     ├─ Conv1D(256→1024, k=9)     │   │
│  │     ├─ LeakyReLU                 │   │
│  │     ├─ Conv1D(1024→256, k=1)     │   │
│  │     └─ Dropout + Residual        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    ↓ [B, 640, 256]
┌─────────────────────────────────────────┐
│  6. 输出层                              │
│  ┌─────────────────────────────────┐   │
│  │ Linear(256 → 1)                  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    ↓
输出 Envelope [B, 640, 1]
```

---

## 🔍 核心模块详解

### 1. CNN特征提取器

**目的**: 从原始EEG信号中提取空间-时序特征

**架构**:
```python
# 3层卷积，逐渐减小kernel size
Conv1D(in=64, out=256, kernel=7, padding=3)
→ LayerNorm(256) → LeakyReLU(0.01) → Dropout(0.3)
Conv1D(in=256, out=256, kernel=5, padding=2)
→ LayerNorm(256) → LeakyReLU(0.01) → Dropout(0.3)
Conv1D(in=256, out=256, kernel=3, padding=1)
→ LayerNorm(256) → LeakyReLU(0.01) → Dropout(0.3)
```

**设计理由**:
- **逐渐减小的卷积核** (7→5→3): 先捕获大范围时序依赖，再聚焦局部细节
- **LayerNorm**: 稳定训练，处理不同受试者间的幅值差异
- **LeakyReLU(0.01)**: 避免神经元死亡，保持梯度流动
- **Dropout(0.3)**: 防止过拟合

**数据流**:
```
[B, 64, 640] → transpose → [B, 64, 640]
→ Conv1 → [B, 256, 640] → transpose → [B, 640, 256]
→ Norm+Act+Drop → transpose → [B, 256, 640]
→ Conv2 → [B, 256, 640] → ...
→ Conv3 → [B, 256, 640]
```

---

### 2. SE通道注意力模块

**目的**: 自适应地对不同EEG通道特征赋予权重

**架构**:
```python
class SEBlock:
    AdaptiveAvgPool1d(1)  # [B, 256, 640] → [B, 256, 1]
    ↓
    FC(256 → 16)          # Squeeze
    ↓
    LeakyReLU(0.01)
    ↓
    FC(16 → 256)          # Excitation
    ↓
    Sigmoid               # [B, 256, 1]
    ↓
    X * weights           # 广播乘法
```

**设计理由**:
- **Reduction Ratio=16**: 压缩到16维，减少参数同时保持表达能力
- **自适应权重**: 不同样本可能在不同脑区有不同的信息量
- **轻量级**: 仅增加少量参数，但效果显著

---

### 3. 全局条件器 (Subject Embedding)

**目的**: 建模受试者个体差异

**架构**:
```python
sub_id [B] → One-Hot [B, 71]
→ Linear(71 → 256) → sub_emb [B, 256]
→ unsqueeze(1) → [B, 1, 256]
→ 广播相加到 X [B, 640, 256]
```

**设计理由**:
- **One-Hot编码**: 71个受试者，每个独立嵌入
- **加性融合**: `X + sub_emb`，保持特征完整性
- **全局条件**: 同一受试者的所有时间步共享相同嵌入

**效果**: 模型能学习到每个受试者的脑电特征偏好

---

### 4. 位置编码

**目的**: 为Transformer注入时序位置信息

**公式**:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**特性**:
- **固定编码**: 不需要训练，直接加到输入
- **正弦余弦**: 奇数偶数维度使用不同函数
- **相对位置**: 模型可学习到相对距离关系

---

### 5. Transformer编码器

#### 5.1 MultiHeadAttention

**架构**:
```python
Pre-LayerNorm(X) → Q, K, V
    ↓
Q = Linear(256 → 4×64) → [B, 640, 4, 64]
K = Linear(256 → 4×64) → [B, 640, 4, 64]
V = Linear(256 → 4×64) → [B, 640, 4, 64]
    ↓
Attention(Q, K, V) = softmax(QK^T / √d_k) V
    ↓
Concat → Linear(256 → 256)
    ↓
Dropout + Residual(X)
```

**参数**:
- **n_head = 4**: 4个注意力头
- **d_k = d_v = 64**: 每个头的维度 (256/4)
- **temperature = √64 = 8**: 缩放因子

**Pre-LN设计**:
```
传统 Post-LN:  X → Attn → Add → Norm
本模型 Pre-LN:  X → Norm → Attn → Add
```
优势: 训练更稳定，梯度流动更顺畅

#### 5.2 PositionwiseFeedForward

**架构**:
```python
Pre-LayerNorm(X)
    ↓
Conv1D(256 → 1024, kernel=9, padding=4)
    ↓
LeakyReLU(0.01)
    ↓
Conv1D(1024 → 256, kernel=1, padding=0)
    ↓
Dropout + Residual(X)
```

**设计理由**:
- **扩张因子=4**: 1024 = 256 × 4，增强非线性表达
- **Conv1D替代FC**: 保持时序局部性
- **kernel=9**: 捕获局部时序上下文 (约140ms)

---

### 6. 输出层

```python
Linear(256 → 1)  # 简单线性投影
```

**输出**: [B, 640, 1] - 每个时间步的语音包络幅值

---

## 💡 损失函数设计

### 组合损失
```python
Loss = MSE_loss + λ × (Pearson_loss)²
```

**参数**: λ = 1.0 (默认)

### 损失函数1: MSE (均方误差)
```python
MSE = mean((y_true - y_pred)²)
```
**作用**: 确保预测幅值的准确性

### 损失函数2: Pearson相关系数
```python
r = Σ[(y_true - ȳ_true)(y_pred - ȳ_pred)] /
    √[Σ(y_true - ȳ_true)² × Σ(y_pred - ȳ_pred)²]

Pearson_loss = 1 - r
```
**作用**: 确保预测波形与真实波形的相关性

**为什么平方**: `(Pearson_loss)²`
- Pearson loss ∈ [0, 2]，平方后放大惩罚
- 更强调波形相关性的重要性

---

## 📊 数据流详解

### 训练数据处理

```python
# 原始数据
train_-_sub-001_-_eeg.npy     [T, 64]   # 变长时序
train_-_sub-001_-_envelope.npy [T, 1]    # 对应包络

# 数据加载 (RegressionDataset)
1. 预加载到内存 (减少IO)
2. 随机采样10秒窗口 (640个时间点)
3. windows_per_sample=10 (每个样本采样10个不同窗口)
4. 受试者ID提取: "sub-001" → 0 (0-indexed)

# DataLoader输出
EEG:      [Batch, 64, 640]
Envelope: [Batch, 640, 1]
Sub_ID:   [Batch]
```

### 测试数据处理
```python
# 测试时不随机采样
1. 将完整录音切分为多个10秒段
2. batch_size=1，逐段推理
3. 拼接所有段的预测结果
```

---

## ⚙️ 训练配置 (train_v10_sota)

```python
# 模型参数
d_model = 256           # 主特征维度
d_inner = 1024          # FFN内部维度
n_head = 4              # 注意力头数
n_layers = 8            # Transformer层数
dropout = 0.3           # Dropout率
in_channel = 64         # EEG通道数
win_len = 10            # 窗口长度(秒)
sample_rate = 64        # 采样率

# 训练参数
epoch = 1000            # 训练轮数
batch_size = 64         # 批次大小
learning_rate = 0.0001  # 学习率
lamda = 1.0             # Pearson损失权重
windows_per_sample = 10 # 每样本窗口数

# 优化器
Adam(lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# 学习率调度
StepLR(step_size=50, gamma=0.9)
# 每50个epoch，lr *= 0.9

# 评估间隔
eval_interval = 10      # 每10个epoch评估一次
saving_interval = 50    # 每50个epoch保存模型
```

---

## 📈 模型容量分析

### 参数量估算

```
1. CNN特征提取器
   Conv1: 64×256×7 ≈ 115K
   Conv2: 256×256×5 ≈ 327K
   Conv3: 256×256×3 ≈ 196K
   Norms: 256×3 ≈ 1K
   小计: ~640K

2. SE通道注意力
   FC1: 256×16 = 4K
   FC2: 16×256 = 4K
   小计: ~8K

3. 受试者嵌入
   Linear: 71×256 ≈ 18K

4. Transformer (×8层)
   每层:
     - MHA: 256×256×3(QKV) + 256×256(out) ≈ 262K
     - FFN: 256×1024×9 + 1024×256 ≈ 2.6M
   8层小计: ~23M

5. 输出层
   Linear: 256×1 ≈ 0.3K

总计: ~24M参数
```

---

## 🎯 设计亮点

### 1. 混合架构
- **CNN**: 提取空间-时序局部特征
- **Transformer**: 建模长距离时序依赖
- 结合两者优势

### 2. 多级正则化
- Dropout (0.3)
- LayerNorm
- 数据增强 (多窗口采样)

### 3. 个体化建模
- 受试者嵌入 (71个独立表示)
- 适应不同人的脑电特征差异

### 4. 注意力机制
- **SE注意力**: 通道级别自适应加权
- **Self-Attention**: 时间步之间的全局依赖

### 5. Pre-LN架构
- 比传统Post-LN更稳定
- 更深的网络仍能高效训练

### 6. 数据效率
- 预加载机制 (减少IO)
- 多窗口采样 (扩充训练样本)
- windows_per_sample=10 → 10倍数据量

---

## 🔬 设计选择的理论依据

### 为什么用CNN+Transformer?
- **CNN**: EEG信号有空间拓扑结构 (64通道分布在头皮不同位置)
- **Transformer**: 语音包络跟踪需要长距离时序建模 (韵律、节奏)

### 为什么用Pre-LN?
- **训练稳定性**: 梯度可以直接流经残差连接
- **深度扩展性**: 更容易堆叠更多层

### 为什么用Pearson损失?
- **波形相关**: MSE只关心幅值，Pearson关心波形形状
- **鲁棒性**: 对幅度缩放不敏感
- **平方**: 强化相关性约束

### 为什么用受试者嵌入?
- **个体差异**: 不同人的脑电信号模式差异大
- **泛化能力**: 在测试受试者上也能应用 (通过ID索引)

---

## 📝 模型文件位置

```
HappyQuokka_system_for_EEG_Challenge/
├── models/
│   ├── FFT_block.py          # 主模型: Decoder
│   │   ├── PositionalEncoding
│   │   ├── PreLNFFTBlock
│   │   ├── SEBlock
│   │   └── Decoder
│   └── SubLayers.py          # 子模块
│       ├── ScaledDotProductAttention
│       ├── MultiHeadAttention
│       └── PositionwiseFeedForward
├── util/
│   ├── cal_pearson.py        # 损失函数
│   ├── dataset.py            # 数据加载器
│   └── utils.py              # 工具函数
└── train_v10_sota.py         # 训练脚本
```

---

## 🚀 推理流程

```python
# 1. 加载模型
model = Decoder(**config).cuda()
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 准备输入
eeg_data = torch.FloatTensor(eeg).cuda()  # [640, 64]
eeg_data = eeg_data.unsqueeze(0)          # [1, 64, 640]
eeg_data = eeg_data.transpose(1, 2)       # [1, 640, 64]
sub_id = torch.LongTensor([subject_idx]).cuda()

# 3. 推理
with torch.no_grad():
    envelope_pred = model(eeg_data, sub_id)  # [1, 640, 1]

# 4. 后处理
envelope_pred = envelope_pred.squeeze().cpu().numpy()  # [640]
```

---

## 📊 性能评估

### 评估指标
1. **Pearson相关系数** (主要指标)
   - 衡量预测波形与真实波形的相关性
   - 范围: [-1, 1]，越接近1越好

2. **均方误差 (MSE)**
   - 衡量预测幅值的准确性
   - 越小越好

### 监控内容
- 训练/验证/测试的Loss和Metric
- 梯度范数 (检测梯度爆炸/消失)
- 可视化: 真实 vs 重建包络对比图
- TensorBoard日志

---

**版本**: train_v10_sota.py
**最后更新**: 2025-10-18
**作者**: HappyQuokka Team
