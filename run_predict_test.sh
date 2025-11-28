#!/bin/bash
# 使用训练好的模型预测测试数据的示例脚本

# 示例1: 使用最新的检查点
CHECKPOINT="/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/test_results/conformer_v2_nlayer4_dmodel256_nhead4_gscale1.0_dist_20251128_162532/conformer_v2_model_step_2925.pt"

# 测试数据目录
TEST_DATA_DIR="/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data/test_data"

# 输出目录
OUTPUT_DIR="test_predictions"

# 运行预测
python predict_test_data.py \
    --checkpoint $CHECKPOINT \
    --test_data_dir $TEST_DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --gpu 0 \
    --batch_size 6

echo "预测完成！结果保存在: $OUTPUT_DIR"
