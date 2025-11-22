"""
Conformer-based Training Script for EEG Signal Processing (v2 - Improved)
使用改进版 Conformer 模型，解决特征提取不足问题

主要改进：
1. 使用 FFT_block_conformer_v2.py 中的改进模型
2. 添加全局残差连接和门控机制
3. 使用 MLP 输出头替代单层线性
4. 支持梯度缩放以增强前层学习

Based on train_v10_conformer.py
"""

import glob
import os
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch.backends.cudnn as cudnn
from util.utils import get_writer, save_checkpoint
from util.logger import TrainingLogger
from torch.optim.lr_scheduler import StepLR
from models.FFT_block_conformer_v2 import Decoder  # 使用改进版模型
from util.cal_pearson import l1_loss, mse_loss, pearson_loss, pearson_metric
from util.dataset import RegressionDataset
import time
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--win_len', type=int, default=10)
parser.add_argument('--sample_rate', type=int, default=64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--g_con', default=True, help="experiment for within subject")

parser.add_argument('--in_channel', type=int, default=64, help="channel of the input eeg signal")
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--d_inner', type=int, default=1024)
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--n_layers', type=int, default=8)
parser.add_argument('--fft_conv1d_kernel', type=tuple, default=(9, 1))
parser.add_argument('--fft_conv1d_padding', type=tuple, default=(4, 0))
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lamda', type=float, default=1)
parser.add_argument('--writing_interval', type=int, default=10)
parser.add_argument('--saving_interval', type=int, default=50)
parser.add_argument('--eval_interval', type=int, default=10, help='evaluation interval (epochs)')
parser.add_argument('--viz_sample_idx', type=int, default=0, help='fixed test sample index for visualization')
parser.add_argument('--grad_log_interval', type=int, default=100, help='gradient logging interval (steps)')

# Conformer-specific parameters
parser.add_argument('--conv_kernel_size', type=int, default=31, help='kernel size for Conformer convolution module')
parser.add_argument('--use_relative_pos', type=bool, default=True, help='use relative positional encoding in attention')
parser.add_argument('--use_macaron_ffn', type=bool, default=True, help='use Macaron-style FFN in Conformer')
parser.add_argument('--use_sinusoidal_pos', type=bool, default=True, help='use additional sinusoidal positional encoding')

# ============ v2 改进参数 ============
parser.add_argument('--use_gated_residual', type=bool, default=True, help='use gated residual connection')
parser.add_argument('--use_mlp_head', type=bool, default=True, help='use MLP output head instead of single linear')
parser.add_argument('--gradient_scale', type=float, default=2.0, help='gradient scaling factor for Conformer layers')
# LLRD (Layer-wise Learning Rate Decay) 参数
parser.add_argument('--use_llrd', type=bool, default=True, help='use layer-wise learning rate decay')
parser.add_argument('--llrd_front_scale', type=float, default=3.0, help='LR scale for front layers (CNN, SE, early Conformer)')
parser.add_argument('--llrd_back_scale', type=float, default=2.0, help='LR scale for back layers (late Conformer, gated_residual)')
parser.add_argument('--llrd_output_scale', type=float, default=0.5, help='LR scale for output head')
# 输出层梯度缩放
parser.add_argument('--output_grad_scale', type=float, default=0.5, help='scale factor for output head gradients after backward')
# ===================================

parser.add_argument('--dataset_folder', type=str, default="/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data", help='write down your absolute path of dataset folder')
parser.add_argument('--split_folder', type=str, default="split_data")
parser.add_argument('--experiment_folder', default=None, help='write down experiment name')

# Distributed training parameters
parser.add_argument('--use_ddp', action='store_true', help='use distributed training')
parser.add_argument('--local-rank', default=-1, type=int, help='local process rank for distributed training')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training')

# Data augmentation
parser.add_argument('--windows_per_sample', type=int, default=10, help='number of windows sampled per sample in one epoch')

args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
cudnn.benchmark = True

# Determine whether to enable DDP based on command line arguments
use_ddp = args.use_ddp
local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", -1))

# Initialize distributed environment (if enabled)
if use_ddp and local_rank != -1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
else:
    world_size = 1
    rank = 0
    local_rank = -1

# Set input length = sample_rate * window_length (seconds)
input_length = args.sample_rate * args.win_len
device = torch.device(f"cuda:{local_rank}" if use_ddp and local_rank != -1 else f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Provide the path of the dataset
data_folder = os.path.join(args.dataset_folder, args.split_folder)
features = ["eeg"] + ["envelope"]

# Create a directory to store results
result_folder = 'test_results'
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.experiment_folder is None:
    if use_ddp:
        experiment_folder = "conformer_v2_nlayer{}_dmodel{}_nhead{}_gscale{}_dist_{}".format(
            args.n_layers, args.d_model, args.n_head, args.gradient_scale, current_time)
    else:
        experiment_folder = "conformer_v2_nlayer{}_dmodel{}_nhead{}_gscale{}_single_{}".format(
            args.n_layers, args.d_model, args.n_head, args.gradient_scale, current_time)
else:
    experiment_folder = args.experiment_folder

save_path = os.path.join(result_folder, experiment_folder)
# Only create writer and logger in main process
is_main_process = (not use_ddp) or rank == 0
writer = get_writer(result_folder, experiment_folder) if is_main_process else None
logger = TrainingLogger(writer, save_path, is_main_process, enable_grad_histogram=True)


def create_dataloader(split_name, data_folder, features, input_length, args, use_ddp, local_rank):
    is_train = split_name == 'train'
    files = [x for x in glob.glob(os.path.join(data_folder, f"{split_name}_-_*"))
             if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]

    # Add windows_per_sample parameter for training set
    windows_per_sample = args.windows_per_sample if hasattr(args, 'windows_per_sample') else 10

    dataset = RegressionDataset(
        files,
        input_length,
        args.in_channel,
        split_name,
        args.g_con,
        windows_per_sample=windows_per_sample if is_train else 1  # Only use multi-window for training
    )

    # Only use DistributedSampler for training set in distributed mode
    sampler = DistributedSampler(dataset) if is_train and use_ddp and local_rank != -1 else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else 1,
        num_workers=args.workers,
        sampler=sampler,
        drop_last=True,
        shuffle=(sampler is None and is_train)  # Only shuffle for training without sampler
    )


def get_llrd_param_groups(model, base_lr, front_scale, back_scale, output_scale, n_layers):
    """
    获取Layer-wise Learning Rate Decay参数组

    分层策略：
    - 前层（CNN, SE, 位置编码, Conformer前半）：base_lr * front_scale
    - 后层（Conformer后半, gated_residual）：base_lr * back_scale
    - 输出层（output_head/fc）：base_lr * output_scale

    Args:
        model: 模型
        base_lr: 基础学习率
        front_scale: 前层学习率倍率
        back_scale: 后层学习率倍率
        output_scale: 输出层学习率倍率
        n_layers: Conformer层数

    Returns:
        param_groups: 参数组列表
    """
    # 定义层的划分
    front_layers = ['conv1', 'conv2', 'conv3', 'norm1', 'norm2', 'norm3',
                    'act1', 'act2', 'act3', 'drop1', 'drop2', 'drop3',
                    'se', 'sub_proj', 'pos_encoder']
    output_layers = ['output_head', 'fc']

    # Conformer层的划分：前半为front，后半为back
    mid_layer = n_layers // 2  # 例如8层时，0-3为前半，4-7为后半

    front_params = []
    back_params = []
    output_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 判断参数属于哪一层
        is_front = any(layer in name for layer in front_layers)
        is_output = any(layer in name for layer in output_layers)

        # 检查是否是Conformer层
        is_conformer_layer = 'layer_stack' in name
        if is_conformer_layer:
            # 提取层号：layer_stack.0.xxx -> 0
            try:
                layer_idx = int(name.split('layer_stack.')[1].split('.')[0])
                if layer_idx < mid_layer:
                    is_front = True
                else:
                    is_front = False
            except (ValueError, IndexError):
                is_front = False

        # 检查是否是gated_residual
        is_gated_residual = 'gated_residual' in name

        if is_output:
            output_params.append(param)
        elif is_front:
            front_params.append(param)
        elif is_gated_residual or (is_conformer_layer and not is_front):
            back_params.append(param)
        else:
            # 默认归为back层
            back_params.append(param)

    param_groups = [
        {'params': front_params, 'lr': base_lr * front_scale, 'name': 'front_layers'},
        {'params': back_params, 'lr': base_lr * back_scale, 'name': 'back_layers'},
        {'params': output_params, 'lr': base_lr * output_scale, 'name': 'output_layers'}
    ]

    return param_groups


def scale_output_gradients(model, scale, use_ddp, local_rank):
    """
    缩放输出层的梯度

    在 loss.backward() 之后调用，用于降低输出层梯度的幅度，
    避免输出层梯度过大导致前层学习不足。

    Args:
        model: 模型
        scale: 缩放因子 (小于1表示缩小梯度)
        use_ddp: 是否使用分布式训练
        local_rank: 本地rank
    """
    base_model = model.module if use_ddp and local_rank != -1 else model

    # 定义输出层
    output_layers = ['output_head', 'fc']

    for name, param in base_model.named_parameters():
        if param.grad is not None and any(layer in name for layer in output_layers):
            param.grad.data.mul_(scale)


def main():
    # ============ 使用改进版 Conformer 模型 ============
    model = Decoder(
        in_channel=args.in_channel,
        d_model=args.d_model,
        d_inner=args.d_inner,
        n_head=args.n_head,
        n_layers=args.n_layers,
        fft_conv1d_kernel=args.fft_conv1d_kernel,
        fft_conv1d_padding=args.fft_conv1d_padding,
        dropout=args.dropout,
        g_con=args.g_con,
        within_sub_num=71,
        # Conformer-specific parameters
        conv_kernel_size=args.conv_kernel_size,
        use_relative_pos=args.use_relative_pos,
        use_macaron_ffn=args.use_macaron_ffn,
        use_sinusoidal_pos=args.use_sinusoidal_pos,
        # v2 改进参数
        use_gated_residual=args.use_gated_residual,
        use_mlp_head=args.use_mlp_head,
        gradient_scale=args.gradient_scale
    ).to(device)
    # ==============================================

    # Print model info in main process
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'=' * 60}")
        print(f"Conformer Model Configuration (v2 - Improved)")
        print(f"{'=' * 60}")
        print(f"基础配置:")
        print(f"  - Model dimension: {args.d_model}")
        print(f"  - FFN inner dimension: {args.d_inner}")
        print(f"  - Number of heads: {args.n_head}")
        print(f"  - Number of layers: {args.n_layers}")
        print(f"  - Conv kernel size: {args.conv_kernel_size}")
        print(f"  - Dropout: {args.dropout}")
        print(f"\nConformer特性:")
        print(f"  - Use relative pos: {args.use_relative_pos}")
        print(f"  - Use Macaron FFN: {args.use_macaron_ffn}")
        print(f"  - Use sinusoidal pos: {args.use_sinusoidal_pos}")
        print(f"\n【v2 改进】:")
        print(f"  - Use gated residual: {args.use_gated_residual}")
        print(f"  - Use MLP head: {args.use_mlp_head}")
        print(f"  - Gradient scale: {args.gradient_scale}x")
        print(f"\n参数统计:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Model size: {total_params * 4 / (1024 * 1024):.2f} MB")
        print(f"{'=' * 60}\n")

    # Wrap model with DDP if using distributed training
    if use_ddp and local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ============ 优化器设置 (支持LLRD) ============
    if args.use_llrd:
        # 使用Layer-wise Learning Rate Decay
        # 注意：DDP模型需要使用model.module来获取参数
        base_model = model.module if use_ddp and local_rank != -1 else model
        param_groups = get_llrd_param_groups(
            base_model,
            args.learning_rate,
            args.llrd_front_scale,
            args.llrd_back_scale,
            args.llrd_output_scale,
            args.n_layers
        )
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.98), eps=1e-09)

        if is_main_process:
            print(f"\n【LLRD 配置】:")
            for group in param_groups:
                print(f"  - {group['name']}: lr={group['lr']:.6f} ({len(group['params'])} params)")
    else:
        # 使用统一学习率
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     betas=(0.9, 0.98),
                                     eps=1e-09)
    # ==============================================

    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    # Create data loaders
    train_dataloader = create_dataloader('train', data_folder, features, input_length, args, use_ddp, local_rank)
    val_dataloader = create_dataloader('val', data_folder, features, input_length, args, use_ddp, local_rank)
    test_dataloader = create_dataloader('test', data_folder, features, input_length, args, use_ddp, local_rank)

    # Get fixed test sample for visualization
    viz_sample = None
    if is_main_process:
        for i, (test_inputs, test_labels, test_sub_id) in enumerate(test_dataloader):
            if i == args.viz_sample_idx:
                viz_sample = (test_inputs.squeeze(0).to(device),
                              test_labels.squeeze(0).to(device),
                              test_sub_id.to(device))
                break

    # Calculate total steps per epoch
    iter_per_epoch = len(train_dataloader)
    global_step = 0

    # Train the model
    for epoch in range(args.epoch):
        model.train()
        start_time = time.time()

        # Set epoch for distributed sampler
        if use_ddp and train_dataloader.sampler is not None:
            train_dataloader.sampler.set_epoch(epoch)

        for step, (inputs, labels, sub_id) in enumerate(train_dataloader):
            global_step += 1
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            sub_id = sub_id.to(device)
            outputs = model(inputs, sub_id)

            l_p = pearson_loss(outputs, labels)
            l_mse = mse_loss(outputs, labels)
            loss = l_mse + args.lamda * (l_p ** 2)
            loss = loss.mean()
            loss.backward()

            # 策略4：缩放输出层梯度
            if args.output_grad_scale != 1.0:
                scale_output_gradients(model, args.output_grad_scale, use_ddp, local_rank)

            # Log gradient information
            if global_step % args.grad_log_interval == 0:
                logger.log_gradients(model, global_step, key_layers_only=True)

            optimizer.step()

            # Log training progress
            if step % args.writing_interval == 0:
                spend_time = time.time() - start_time
                learning_rate = optimizer.param_groups[0]["lr"]

                # Calculate time per step and speed
                time_per_step = spend_time / (step + 1)
                speed = 1.0 / time_per_step if time_per_step > 0 else 0

                # Calculate remaining time
                steps_per_epoch = iter_per_epoch
                total_steps = args.epoch * steps_per_epoch
                current_total_steps = epoch * steps_per_epoch + step
                remaining_steps = total_steps - current_total_steps
                total_remaining = time_per_step * remaining_steps

                # Use unified logging interface
                logger.log_training(
                    epoch=epoch + 1,
                    total_epochs=args.epoch,
                    step=step,
                    total_steps=iter_per_epoch,
                    loss_dict={
                        'total': loss.item(),
                        'mse': l_mse.mean().item(),
                        'pearson': l_p.mean().item()
                    },
                    lr=learning_rate,
                    speed=speed,
                    time_remaining=total_remaining,
                    global_step=global_step
                )

        # Epoch-based evaluation
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            val_loss = 0
            val_metric = 0

            with torch.no_grad():
                for val_inputs, val_labels, val_sub_id in val_dataloader:
                    val_inputs = val_inputs.squeeze(0).to(device)
                    val_labels = val_labels.squeeze(0).to(device)
                    val_sub_id = val_sub_id.to(device)

                    val_outputs = model(val_inputs, val_sub_id)
                    val_loss += pearson_loss(val_outputs, val_labels).mean()
                    val_metric += pearson_metric(val_outputs, val_labels).mean()

                val_loss /= len(val_dataloader)
                val_metric /= len(val_dataloader)
                val_metric = val_metric.mean()

                # Log validation results
                logger.log_validation(global_step, val_loss.mean().item(), val_metric.item())

                # Test the model
                test_loss = 0
                test_metric = 0

                for test_inputs, test_labels, test_sub_id in test_dataloader:
                    test_inputs = test_inputs.squeeze(0).to(device)
                    test_labels = test_labels.squeeze(0).to(device)
                    test_sub_id = test_sub_id.to(device)

                    test_outputs = model(test_inputs, test_sub_id)
                    test_loss += pearson_loss(test_outputs, test_labels).mean()
                    test_metric += pearson_metric(test_outputs, test_labels).mean()

                test_loss /= len(test_dataloader)
                test_metric /= len(test_dataloader)
                test_metric = test_metric.mean()

                # Log test results
                logger.log_test(global_step, test_loss.mean().item(), test_metric.item())

                # Visualization
                if is_main_process and viz_sample is not None:
                    viz_inputs, viz_labels, viz_sub_id = viz_sample
                    viz_outputs = model(viz_inputs, viz_sub_id)

                    # Convert to numpy for plotting
                    viz_outputs_np = viz_outputs[0].cpu().numpy()
                    viz_labels_np = viz_labels[0].cpu().numpy()

                    # Log visualization
                    logger.log_visualization(
                        predictions=viz_outputs_np,
                        targets=viz_labels_np,
                        epoch=epoch + 1,
                        global_step=global_step,
                        save_png=True
                    )

            model.train()

        # Save model (only in main process)
        if (epoch + 1) % args.saving_interval == 0 and is_main_process:
            learning_rate = optimizer.param_groups[0]["lr"]
            # Special handling for DDP model state_dict
            if use_ddp and local_rank != -1:
                checkpoint = {
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'learning_rate': learning_rate,
                    'args': vars(args)  # Save all args including v2 improvements
                }
                torch.save(checkpoint, os.path.join(save_path, f'conformer_v2_model_step_{global_step}.pt'))
            else:
                save_checkpoint(model, optimizer, learning_rate, epoch, save_path, step=global_step)

        scheduler.step()


if __name__ == '__main__':
    # Ensure directory exists (only in main process)
    if is_main_process:
        os.makedirs(save_path, exist_ok=True)
        print(f"\n💾 Results will be saved to: {save_path}\n")

    # Synchronize all processes if using distributed training
    if use_ddp and local_rank != -1:
        dist.barrier()

    main()
