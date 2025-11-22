#!/bin/bash
# Conformer v2 改进版训练脚本
# 使用说明: chmod +x run_conformer_v2.sh && ./run_conformer_v2.sh

echo "=========================================="
echo "Conformer v2 - 改进版 EEG 模型训练"
echo "=========================================="
echo ""

# ============ 配置参数 ============
EPOCH=1000
BATCH_SIZE=64
N_LAYERS=8
D_MODEL=256
N_HEAD=4
CONV_KERNEL=31
LEARNING_RATE=0.0001

# v2 改进参数
USE_GATED_RESIDUAL=True
USE_MLP_HEAD=True
GRADIENT_SCALE=2.0

GPU=0  # 单GPU训练时使用的GPU编号
# ================================

echo "训练配置:"
echo "  - Epochs: $EPOCH"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Layers: $N_LAYERS"
echo "  - Model Dim: $D_MODEL"
echo "  - Heads: $N_HEAD"
echo ""
echo "v2 改进:"
echo "  - Gated Residual: $USE_GATED_RESIDUAL"
echo "  - MLP Head: $USE_MLP_HEAD"
echo "  - Gradient Scale: ${GRADIENT_SCALE}x"
echo ""
echo "=========================================="
echo ""

# 单 GPU 训练
python train_v10_conformer_v2.py \
    --epoch $EPOCH \
    --batch_size $BATCH_SIZE \
    --n_layers $N_LAYERS \
    --d_model $D_MODEL \
    --n_head $N_HEAD \
    --conv_kernel_size $CONV_KERNEL \
    --learning_rate $LEARNING_RATE \
    --use_gated_residual $USE_GATED_RESIDUAL \
    --use_mlp_head $USE_MLP_HEAD \
    --gradient_scale $GRADIENT_SCALE \
    --gpu $GPU \
    --eval_interval 10 \
    --saving_interval 50 \
    --grad_log_interval 100

echo ""
echo "=========================================="
echo "训练完成！"
echo "查看结果: tensorboard --logdir test_results --port 6006"
echo "=========================================="
