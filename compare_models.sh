#!/bin/bash
# 对比原始版本和 v2 版本的效果
# 使用两个 GPU 同时训练，方便对比

echo "=========================================="
echo "Conformer 模型对比训练"
echo "=========================================="
echo ""
echo "GPU 0: 原始版本 (train_v10_conformer.py)"
echo "GPU 1: v2 改进版 (train_v10_conformer_v2.py)"
echo ""
echo "训练100个epoch后，运行诊断脚本对比梯度"
echo "=========================================="
echo ""

# 确认有两个可用GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到 nvidia-smi"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "警告: 只检测到 $GPU_COUNT 个GPU"
    echo "将在同一个GPU上串行运行"
fi

# 启动原始版本 (GPU 0)
echo "启动原始版本..."
python train_v10_conformer.py \
    --epoch 100 \
    --batch_size 64 \
    --n_layers 8 \
    --d_model 256 \
    --n_head 4 \
    --conv_kernel_size 31 \
    --gpu 0 \
    --experiment_folder "compare_original" &

PID1=$!

# 等待1秒，避免同时初始化冲突
sleep 1

# 启动 v2 版本 (GPU 1)
echo "启动 v2 改进版..."
python train_v10_conformer_v2.py \
    --epoch 100 \
    --batch_size 64 \
    --n_layers 8 \
    --d_model 256 \
    --n_head 4 \
    --conv_kernel_size 31 \
    --use_gated_residual True \
    --use_mlp_head True \
    --gradient_scale 2.0 \
    --gpu 1 \
    --experiment_folder "compare_v2" &

PID2=$!

echo ""
echo "=========================================="
echo "两个版本正在后台训练..."
echo "原始版本 PID: $PID1"
echo "v2版本 PID: $PID2"
echo ""
echo "监控命令:"
echo "  tensorboard --logdir test_results --port 6006"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "停止训练: kill $PID1 $PID2"
echo "=========================================="

# 等待两个进程
wait $PID1 $PID2

echo ""
echo "=========================================="
echo "训练完成！运行诊断对比..."
echo "=========================================="

# 这里可以添加自动诊断和对比的代码
