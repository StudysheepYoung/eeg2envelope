"""
Conformer-based Training Script for EEG Signal Processing
Uses ConformerBlock instead of Transformer for better local-global feature modeling

Based on train_v10_sota.py with Conformer-specific modifications
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
from models.FFT_block_conformer import Decoder  # Import Conformer-based model
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
        experiment_folder = "conformer_nlayer{}_dmodel{}_nhead{}_conv{}_dist_{}".format(
            args.n_layers, args.d_model, args.n_head, args.conv_kernel_size, current_time)
    else:
        experiment_folder = "conformer_nlayer{}_dmodel{}_nhead{}_conv{}_single_{}".format(
            args.n_layers, args.d_model, args.n_head, args.conv_kernel_size, current_time)
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


def main():
    # Set up Conformer model with all parameters
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
        use_sinusoidal_pos=args.use_sinusoidal_pos
    ).to(device)

    # Print model info in main process
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{'=' * 50}")
        print(f"Conformer Model Configuration")
        print(f"{'=' * 50}")
        print(f"  - Model dimension: {args.d_model}")
        print(f"  - FFN inner dimension: {args.d_inner}")
        print(f"  - Number of heads: {args.n_head}")
        print(f"  - Number of layers: {args.n_layers}")
        print(f"  - Conv kernel size: {args.conv_kernel_size}")
        print(f"  - Use relative pos: {args.use_relative_pos}")
        print(f"  - Use Macaron FFN: {args.use_macaron_ffn}")
        print(f"  - Use sinusoidal pos: {args.use_sinusoidal_pos}")
        print(f"  - Dropout: {args.dropout}")
        print(f"\nParameter count:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"{'=' * 50}\n")

    # Wrap model with DDP if using distributed training
    if use_ddp and local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.98),
                                 eps=1e-09)
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
                    'args': vars(args)  # Save Conformer-specific args
                }
                torch.save(checkpoint, os.path.join(save_path, f'conformer_model_step_{global_step}.pt'))
            else:
                save_checkpoint(model, optimizer, learning_rate, epoch, save_path, step=global_step)

        scheduler.step()


if __name__ == '__main__':
    # Ensure directory exists (only in main process)
    if is_main_process:
        os.makedirs(save_path, exist_ok=True)

    # Synchronize all processes if using distributed training
    if use_ddp and local_rank != -1:
        dist.barrier()

    main()
