#!/usr/bin/env python3
"""
查看 PyTorch checkpoint (.pt) 文件的工具脚本
用法: python inspect_checkpoint.py <checkpoint_path>
"""

import torch
import sys
import os

def inspect_checkpoint(ckpt_path):
    """检查 checkpoint 文件内容"""

    if not os.path.exists(ckpt_path):
        print(f"❌ 文件不存在: {ckpt_path}")
        return

    # 读取checkpoint
    print(f"📂 读取: {ckpt_path}\n")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    print("=" * 80)
    print("Checkpoint 内容概览")
    print("=" * 80)

    # 查看checkpoint包含的key
    print("\n【Checkpoint Keys】")
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  - {key} (模型参数)")
        elif key == 'optimizer_state_dict':
            print(f"  - {key} (优化器状态)")
        elif key == 'args':
            print(f"  - {key} (训练参数配置)")
        else:
            print(f"  - {key}")

    # 查看训练参数 (如果保存了)
    if 'args' in checkpoint:
        print("\n" + "=" * 80)
        print("训练参数配置 (args)")
        print("=" * 80)
        args = checkpoint['args']

        # 按类别分组显示
        print("\n【模型架构】")
        for key in ['n_layers', 'd_model', 'd_inner', 'n_head', 'conv_kernel_size', 'in_channel']:
            if key in args:
                print(f"  {key}: {args[key]}")

        print("\n【正则化】")
        for key in ['dropout']:
            if key in args:
                print(f"  {key}: {args[key]}")
        # weight_decay 可能不在args中
        if 'weight_decay' in args:
            print(f"  weight_decay: {args['weight_decay']}")
        else:
            print(f"  weight_decay: N/A (未保存或为默认值0)")

        print("\n【训练配置】")
        for key in ['batch_size', 'learning_rate', 'windows_per_sample', 'epoch']:
            if key in args:
                print(f"  {key}: {args[key]}")

        print("\n【数据配置】")
        for key in ['win_len', 'sample_rate']:
            if key in args:
                print(f"  {key}: {args[key]}")

        print("\n【Conformer-v2 改进特性】")
        for key in ['gradient_scale', 'use_llrd', 'llrd_front_scale', 'llrd_back_scale',
                    'llrd_output_scale', 'output_grad_scale', 'use_gated_residual', 'use_mlp_head']:
            if key in args:
                print(f"  {key}: {args[key]}")

        print("\n【Conformer 特性】")
        for key in ['use_relative_pos', 'use_macaron_ffn', 'use_sinusoidal_pos']:
            if key in args:
                print(f"  {key}: {args[key]}")

        print("\n【分布式训练】")
        for key in ['use_ddp', 'workers']:
            if key in args:
                print(f"  {key}: {args[key]}")

    # 查看训练状态
    print("\n" + "=" * 80)
    print("训练状态")
    print("=" * 80)
    if 'epoch' in checkpoint:
        print(f"  当前 Epoch: {checkpoint['epoch'] + 1}")  # +1因为保存时是从0开始
    if 'step' in checkpoint:
        print(f"  当前 Step: {checkpoint['step']}")
        # 计算实际epoch
        if 'args' in checkpoint and 'batch_size' in checkpoint['args']:
            # 假设每个epoch有158个batch (508*20/64)
            iter_per_epoch = 158
            actual_epoch = checkpoint['step'] // iter_per_epoch
            print(f"  实际 Epoch: ~{actual_epoch}")
    if 'learning_rate' in checkpoint:
        print(f"  Learning Rate: {checkpoint['learning_rate']:.6f}")

    # 查看模型参数统计
    if 'model_state_dict' in checkpoint:
        print("\n" + "=" * 80)
        print("模型参数统计")
        print("=" * 80)
        state_dict = checkpoint['model_state_dict']

        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  模型大小: {total_params * 4 / (1024*1024):.2f} MB (float32)")

        print(f"\n  参数层总数: {len(state_dict)}")

        print("\n  【前10个参数层】")
        for i, (name, param) in enumerate(list(state_dict.items())[:10]):
            print(f"    {name:50s} {str(tuple(param.shape)):30s} {param.numel():>10,} 参数")

        print("\n  【后10个参数层】")
        for name, param in list(state_dict.items())[-10:]:
            print(f"    {name:50s} {str(tuple(param.shape)):30s} {param.numel():>10,} 参数")

        # 统计各模块参数量
        print("\n  【模块参数分布】")
        module_params = {}
        for name, param in state_dict.items():
            module_name = name.split('.')[0]
            if module_name not in module_params:
                module_params[module_name] = 0
            module_params[module_name] += param.numel()

        for module, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
            print(f"    {module:30s} {params:>12,} 参数 ({params/total_params*100:>5.1f}%)")

    # 查看优化器状态
    if 'optimizer_state_dict' in checkpoint:
        print("\n" + "=" * 80)
        print("优化器状态")
        print("=" * 80)
        opt_dict = checkpoint['optimizer_state_dict']
        print(f"  参数组数量: {len(opt_dict.get('param_groups', []))}")
        if 'param_groups' in opt_dict:
            for i, group in enumerate(opt_dict['param_groups']):
                print(f"\n  参数组 {i}:")
                if 'name' in group:
                    print(f"    名称: {group['name']}")
                print(f"    学习率: {group.get('lr', 'N/A')}")
                print(f"    参数数量: {len(group.get('params', []))}")

    print("\n" + "=" * 80)
    print("✓ 检查完成")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python inspect_checkpoint.py <checkpoint_path>")
        print("\n示例:")
        print("  python inspect_checkpoint.py test_results/experiment/model_step_1000.pt")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    inspect_checkpoint(ckpt_path)
