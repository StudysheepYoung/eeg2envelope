# HappyQuokka: EEG-to-Speech 解码系统

<p align="center">
  <img src="HappyQuokka.png" width="60%" alt="HappyQuokka Logo">
</p>

<p align="center">
  <strong>ICASSP 2023 听觉脑电图挑战赛任务2（回归）的官方PyTorch实现</strong>
</p>

---

## 📖 项目概述

HappyQuokka 是一个基于深度学习的 EEG-to-Speech 解码系统，能够从多通道脑电图（EEG）信号中重建语音包络。该项目参加了 ICASSP 2023 听觉脑电图挑战赛，专注于从神经信号中解码语音信息的回归任务。

### 🎯 核心任务
- **输入**: 64通道EEG信号 (10秒@64Hz采样率)
- **输出**: 重建的语音包络信号
- **目标**: 实现高精度的神经信号到语音信号的解码

### 🏗️ 模型架构

我们的系统采用 **CNN + Transformer 混合架构**，结合了空间特征提取和时序建模：

```
EEG [64×640] → CNN特征提取 → 通道注意力 → 受试者条件化 →
Transformer编码 → 线性输出 → 语音包络 [1×640]
```

<p align="center">
  <img src="model_architecture_with_data.png" width="90%" alt="模型架构图">
</p>

## 🚀 快速开始

### 环境要求

```bash
# Python 环境
Python >= 3.8
PyTorch >= 1.8.0
CUDA >= 11.0 (推荐GPU训练)

# 主要依赖
torch
numpy
matplotlib
scipy
```

### 安装

```bash
git clone https://github.com/your-username/HappyQuokka_system_for_EEG_Challenge.git
cd HappyQuokka_system_for_EEG_Challenge

# 安装依赖
pip install torch numpy matplotlib scipy
```

### 数据准备

1. 下载 [EEG 数据集](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND)
2. 解压 `split_data.zip` 到 `data/` 目录：

```
data/
├── split_data/
│   ├── train_-_*
│   ├── val_-_*
│   └── test_-_*
```

### 训练模型

**单GPU训练：**
```bash
python train_v10_sota.py --experiment_folder my_experiment
```

**多GPU分布式训练：**
```bash
# 使用 DDP
python -m torch.distributed.launch --nproc_per_node=4 train_v10_sota.py --use_ddp --experiment_folder my_experiment_ddp

# 或使用我们的分布式脚本
python run_ddp.py
```

### 关键参数

```bash
python train_v10_sota.py \
    --epoch 1000 \               # 训练轮数
    --batch_size 64 \            # 批次大小
    --learning_rate 0.0001 \     # 学习率
    --win_len 10 \               # 窗口长度(秒)
    --n_layers 8 \               # Transformer层数
    --d_model 256 \              # 模型维度
    --n_head 4 \                 # 注意力头数
    --dropout 0.3 \              # Dropout率
    --g_con True \               # 是否使用全局条件器(受试者ID)
    --dataset_folder /path/to/data
```

## 🔧 模型详解

### 核心组件

1. **CNN特征提取器**
   - 三层1D卷积 (kernel: 7→5→3)
   - LayerNorm + LeakyReLU + Dropout
   - 64通道 → 256维特征

2. **SE通道注意力**
   - Squeeze-and-Excitation机制
   - 自适应通道权重

3. **全局条件器**
   - 受试者ID嵌入 (One-hot[71] → Linear[256])
   - 支持跨被试泛化

4. **Transformer编码器**
   - 8层 PreLNFFTBlock
   - 多头自注意力 (4头×64维)
   - 位置编码

5. **输出层**
   - 线性映射 (256→1维)
   - 语音包络重建

### 损失函数

```python
loss = MSE_loss + λ × (Pearson_loss)²
```

- **MSE**: 确保幅值准确性
- **Pearson**: 确保波形相关性
- **λ=1.0**: 平衡两个损失项

## 📊 实验结果

### 训练监控

模型训练过程中会自动生成：
- TensorBoard日志 (`test_results/experiment_name/`)
- 可视化图表 (真实vs重建包络对比)
- 模型检查点 (每50个epoch保存)

### 评估指标

- **Pearson相关系数**: 衡量波形相关性
- **均方误差(MSE)**: 衡量幅值准确性
- **实时可视化**: 固定测试样本的重建效果

## 🏃‍♂️ 运行不同版本

项目包含多个训练脚本版本：

```bash
# 基础版本
python train_v1.py

# 分布式版本
python train_v2_ddp.py

# 带日志版本
python train_v3_ddp_log.py

# SOTA版本 (推荐)
python train_v10_sota.py

# 最新改进版本
python train_v16.py
```

## 📁 项目结构

```
HappyQuokka_system_for_EEG_Challenge/
├── README.md                    # 项目说明
├── train_v10_sota.py           # 主训练脚本 (推荐)
├── models/
│   ├── FFT_block.py            # 核心模型定义
│   └── SubLayers.py            # Transformer子层
├── util/
│   ├── dataset.py              # 数据加载器
│   ├── cal_pearson.py          # 损失函数
│   └── utils.py                # 工具函数
├── data/                       # 数据目录
├── test_results/               # 训练结果
└── *.py                        # 其他版本的训练脚本
```

## 🎨 可视化功能

### 模型架构图
```bash
python model_architecture_diagram.py      # 生成基础架构图
python model_architecture_with_data.py    # 生成带数据维度的架构图
```

### 语音包络解释
```bash
python explain_envelope.py                # 生成包络概念解释图
```

### 训练过程可视化
```bash
tensorboard --logdir test_results/experiment_name/
```

## ⚙️ 高级功能

### 分布式训练

支持多GPU并行训练以加速收敛：

```bash
# 4 GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 train_v10_sota.py --use_ddp
```

### 混合精度训练

减少显存占用并加速训练：

```bash
python train_v10_sota.py --use_amp
```

### 数据增强

- **多窗口采样**: 每个样本在一个epoch中采样多个窗口
- **预加载机制**: 将数据预加载到内存减少IO开销

## 🔍 调试和监控

### 梯度监控
模型会自动记录：
- 每层梯度直方图
- 梯度范数变化
- 权重分布演化

### 性能分析
```bash
# 查看训练速度
grep "速度" log.txt

# 查看损失变化
grep "loss" log.txt
```

## 🌟 项目特色

1. **🧠 神经科学驱动**: 基于大脑听觉处理机制设计
2. **🏗️ 混合架构**: CNN空间特征 + Transformer时序建模
3. **👥 个体化建模**: 受试者特定的全局条件器
4. **⚡ 高效训练**: 支持分布式和混合精度训练
5. **📈 实时监控**: 丰富的可视化和日志记录
6. **🔧 易于扩展**: 模块化设计，便于改进和定制

## 📚 相关论文和引用

```bibtex
@inproceedings{HappyQuokka2023,
  title={HappyQuokka: EEG-to-Speech Decoding System for ICASSP 2023 Challenge},
  author={Your Name},
  booktitle={ICASSP 2023 - IEEE International Conference on Acoustics, Speech and Signal Processing},
  year={2023}
}

@article{fastspeech,
  title={Fastspeech: Fast, robust and controllable text to speech},
  author={Ren, Yi and Ruan, Yangjun and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}

@data{eegdata_K3VSND_2023,
  author = {Bollens, Lies and Accou, Bernd and Van hamme, Hugo and Francart, Tom},
  publisher = {KU Leuven RDR},
  title = {{A Large Auditory EEG decoding dataset}},
  year = {2023},
  version = {V1},
  doi = {10.48804/K3VSND},
  url = {https://doi.org/10.48804/K3VSND}
}
```

## 🤝 贡献指南

欢迎提交问题和改进建议！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🔗 相关链接

- **挑战赛官网**: [ICASSP 2023 Auditory EEG Challenge](https://github.com/exporl/auditory-eeg-challenge-2023-code)
- **数据集**: [Large Auditory EEG Dataset](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/K3VSND)
- **FastSpeech**: [Original FastSpeech Implementation](https://github.com/xcmyz/FastSpeech)

## 📞 联系方式

如有问题请提交 Issue 或联系项目维护者。

---

<p align="center">
  Made with ❤️ for neuroscience and speech technology
</p>