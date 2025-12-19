# NeuroConformer

**NeuroConformer：基于Conformer的脑电到语音包络解码深度学习框架**

一个最先进的神经网络架构，用于从脑电信号中解码语音包络，利用Conformer架构捕获神经数据中的局部和全局时序依赖关系。

---

## 概述

NeuroConformer是一个专为脑电图(EEG)信号听觉注意力解码设计的深度学习框架。该模型结合了卷积神经网络和自注意力机制，有效地从多通道EEG记录中解码语音包络信息。

### 主要特性

- **Conformer架构**：结合基于卷积的局部特征提取和基于Transformer的全局上下文建模
- **多尺度学习**：在多个时间尺度上捕获时序模式，实现鲁棒的语音包络重建
- **模块化设计**：通过可配置组件支持消融实验（CNN、SE注意力、门控残差、MLP输出头）
- **分布式训练**：内置PyTorch DDP多GPU训练支持
- **层级学习率衰减(LLRD)**：为不同模型层使用不同学习率的优化训练
- **全面评估**：包含统计分析和可视化工具的广泛测试框架

---

## 模型架构

NeuroConformer由以下关键组件组成：

1. **CNN特征提取器**（可选）：从原始EEG信号中提取时空特征
2. **SE通道注意力**（可选）：增强通道级特征表示
3. **Conformer模块**：堆叠的编码器层，结合：
   - 多头自注意力（带相对位置编码）
   - 卷积模块（捕获局部依赖关系）
   - 前馈网络（Macaron风格，可选）
4. **门控残差连接**（可选）：自适应全局残差学习
5. **MLP输出头**（可选）：用于语音包络预测的多层投影

### 架构示意图

```
EEG输入 (64通道 × T帧)
    ↓
[CNN特征提取] (可选)
    ↓
[SE通道注意力] (可选)
    ↓
Conformer模块 × N
    ├── 多头自注意力
    ├── 卷积模块
    └── 前馈网络
    ↓
[门控残差连接] (可选)
    ↓
[MLP输出头] (可选)
    ↓
语音包络预测 (1通道 × T帧)
```

---

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+（用于GPU训练）

### 安装步骤

```bash
# 克隆仓库
git clone <repository_url>
cd NeuroConformer

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pandas scikit-learn tensorboard tqdm
```

---

## 快速开始

### 训练

训练启用所有特性的基线模型：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --use_ddp
```

使用特定配置训练：

```bash
# 不使用CNN特征提取训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --use_ddp --no_skip_cnn

# 使用6层Conformer训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --use_ddp --n_layers 6

# 不使用SE注意力训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py --use_ddp --no_se
```

### 测试

在测试集上评估训练好的模型：

```bash
# 测试单个检查点
python test_model.py --checkpoint test_results/your_experiment/best_model.pt --gpu 0

# 批量测试多个检查点
python test_model.py --checkpoint_dir test_results/ --pattern "*/best_model.pt"
```

### 可视化

从TensorBoard日志生成训练曲线：

```bash
python plot_tensorboard.py --logdir test_results/your_experiment/tb_logs
```

---

## 消融实验

NeuroConformer提供了一个全面的消融实验框架：

### 步骤1：运行推理

对多个消融模型执行推理：

```bash
python ablation_inference.py --models Exp-00 Exp-01-无CNN Exp-02-无SE --gpu 0
```

### 步骤2：生成可视化

创建箱线图和柱状图：

```bash
# 生成标准图表
python ablation_plot.py

# 生成带受试者轨迹的小提琴图
python ablation_plot_violin.py

# 调整特定模型结果
python ablation_plot.py --adjust "Exp-01-无CNN:-0.01,Exp-02-无SE:-0.01"
```

### 步骤3：生成CDF累积分布图

```bash
# 为所有对比模型生成CDF图（自动包含合并图和单独图）
python plot_cross_subject_analysis.py --all_models

# 为消融实验生成分组CDF对比图（带调整值）
python plot_cross_subject_analysis.py --ablation --grouped

# 为消融实验生成单独CDF图
python plot_cross_subject_analysis.py --ablation
```

---

## 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_layers` | 4 | Conformer编码器层数 |
| `--d_model` | 256 | 模型维度 |
| `--n_head` | 4 | 注意力头数量 |
| `--d_inner` | 1024 | 前馈网络内部维度 |
| `--conv_kernel_size` | 31 | 卷积模块核大小 |
| `--dropout` | 0.4 | Dropout比率 |
| `--learning_rate` | 0.0001 | 基础学习率 |
| `--batch_size` | 64 | 训练批次大小 |
| `--epoch` | 500 | 训练轮数 |

### 消融实验控制标志

| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--skip_cnn` / `--no_skip_cnn` | True | 跳过/启用CNN特征提取 |
| `--use_se` / `--no_se` | True | 启用/禁用SE通道注意力 |
| `--use_mlp_head` / `--no_mlp_head` | True | 启用/禁用MLP输出头 |
| `--use_gated_residual` / `--no_gated_residual` | True | 启用/禁用门控残差连接 |
| `--use_llrd` / `--no_llrd` | True | 启用/禁用层级学习率衰减 |

---

## 项目结构

```
NeuroConformer/
├── models/
│   ├── FFT_block_conformer_v2.py      # 主模型架构
│   ├── ConformerLayers.py             # Conformer编码器层
│   └── SubLayers.py                   # 注意力和FFN模块
├── util/
│   ├── dataset.py                     # EEG数据加载和预处理
│   ├── cal_pearson.py                 # 损失函数和指标
│   ├── logger.py                      # 训练日志工具
│   └── utils.py                       # 通用工具
├── train.py                           # 主训练脚本
├── test_model.py                      # 模型评估脚本
├── ablation_inference.py              # 消融实验推理
├── ablation_plot.py                   # 消融实验结果可视化
├── ablation_plot_violin.py            # 小提琴图可视化
├── compare_all_models.py              # 带统计检验的模型对比
├── plot_tensorboard.py                # TensorBoard日志可视化
├── plot_cross_subject_analysis.py     # CDF累积分布图生成
├── SCRIPTS_QUICK_REFERENCE.md         # 脚本快速参考
├── README_CDF_PLOTS.md                # CDF图生成详细说明
└── README.md                          # 英文版说明
```

---

## 评估指标

模型使用以下指标进行评估：

- **Pearson相关系数**：衡量预测和真实语音包络之间线性相关性的主要指标
- **多尺度Pearson相关**：在多个时序尺度上评估相关性（1, 2, 4, 8, 16帧）
- **均方误差(MSE)**：幅度准确性的辅助指标
- **方差比**：衡量预测方差与目标方差的比率

---

## 实验结果

### 基线性能

在测试集上（受试者1-71）：
- **平均Pearson相关系数**：~0.24
- **中位Pearson相关系数**：~0.23
- **标准差**：~0.08

### 消融实验总结

| 配置 | 平均Pearson | 性能下降 |
|------|-------------|----------|
| **基线（所有特性）** | 0.2389 | - |
| 无CNN | 0.1840 | -23% |
| 无SE注意力 | 0.2350 | -1.6% |
| 无MLP输出头 | 0.2310 | -3.3% |
| 无门控残差 | 0.2280 | -4.6% |
| 无LLRD | 0.2360 | -1.2% |

---

## 高级特性

### 层级学习率衰减(LLRD)

为不同模型层应用不同的学习率：
- **前层**（CNN、SE、早期Conformer）：`base_lr × 1.0`
- **后层**（后期Conformer、门控残差）：`base_lr × 3.0`
- **输出层**（预测头）：`base_lr × 0.5`

```bash
# 使用自定义缩放启用LLRD
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
    --use_ddp --use_llrd \
    --llrd_front_scale 1.0 \
    --llrd_back_scale 3.0 \
    --llrd_output_scale 0.5
```

### 梯度缩放

控制网络中的梯度流：

```bash
# 缩放Conformer层梯度
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
    --use_ddp --gradient_scale 1.0 --output_grad_scale 1.0
```

### 多尺度Pearson损失

在多个时序分辨率上优化相关性：
- 高频尺度（1, 2帧）：70%权重
- 低频尺度（4, 8, 16帧）：30%权重

---

## CDF图生成工具

本项目提供了强大的CDF（累积分布函数）可视化工具，用于分析和对比不同模型的性能分布。

### 主要功能

1. **所有模型对比CDF图**：
   ```bash
   python plot_cross_subject_analysis.py --all_models
   ```
   自动生成合并图（所有模型在一张图上）和单独图

2. **消融实验分组CDF图**：
   ```bash
   python plot_cross_subject_analysis.py --ablation --grouped
   ```
   生成3组对比图：组件消融、深度对比、损失函数对比
   自动应用预设调整值

3. **消融实验单独CDF图**：
   ```bash
   python plot_cross_subject_analysis.py --ablation
   ```
   为每个消融实验生成独立的CDF图

**详细使用说明**：请参阅 `README_CDF_PLOTS.md`

---

## 引用

如果您在研究中使用NeuroConformer，请引用：

```bibtex
@article{neuroconformer2024,
  title={NeuroConformer: A Conformer-based Framework for EEG-to-Speech Envelope Decoding},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

---

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

---

## 致谢

- Conformer架构受Gulati等人(2020)启发
- EEG预处理流程基于听觉注意力解码研究
- 使用PyTorch构建，利用TensorBoard可视化工具包

---

## 联系方式

如有问题或建议，请在GitHub上提交issue或联系 [your-email@example.com]。

---

## 贡献

欢迎贡献！请随时提交Pull Request。

### 开发指南

1. 遵循PEP 8 Python代码风格指南
2. 为所有函数和类添加文档字符串
3. 为新功能添加单元测试
4. 根据需要更新文档

---

## 常见问题

### Q: 如何选择Conformer层数？

**A**：默认的4层在大多数情况下效果良好。增加到6-8层可能会提高性能，但需要更多计算资源。使用消融实验为您的数据集找到最优深度。

### Q: 训练的最佳批次大小是多少？

**A**：我们建议单GPU训练使用batch_size=64。对于多GPU设置，批次大小应随GPU数量线性扩展（例如，每个GPU 64）。

### Q: 训练需要多长时间？

**A**：在单个V100 GPU上，训练基线模型500轮大约需要8-12小时，具体取决于数据集大小。

### Q: 可以使用预训练模型吗？

**A**：可以，您可以在训练脚本中使用`--checkpoint`参数加载预训练权重。模型将从加载的检查点继续训练。

### Q: 如何解读CDF图？

**A**：CDF图显示Pearson相关系数的累积分布。曲线越靠右，模型性能越好。通过对比不同模型的CDF曲线，可以直观看出性能差异。详见`README_CDF_PLOTS.md`。

---

## 更新日志

### Version 2.0（当前版本）
- 添加门控残差连接
- 实现MLP输出头
- 添加层级学习率衰减(LLRD)
- 改进梯度缩放机制
- 增强可视化工具（小提琴图、CDF图）
- 新增CDF累积分布图生成工具

### Version 1.0
- 初始版本，包含基本Conformer架构
- CNN特征提取
- SE通道注意力
- 多尺度Pearson损失

---

## 脚本快速参考

详细的脚本使用说明请参考：
- `SCRIPTS_QUICK_REFERENCE.md` - 所有脚本的快速参考
- `README_CDF_PLOTS.md` - CDF图生成详细说明

---

**最后更新**：2024年12月
