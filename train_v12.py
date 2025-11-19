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
from torch.optim.lr_scheduler import StepLR
from models.FFT_block import Decoder
from util.cal_pearson import l1_loss, mse_loss, pearson_loss, pearson_metric
from util.dataset import RegressionDataset
import time
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--epoch',type=int, default=1000)
parser.add_argument('--batch_size',type=int, default=64)
parser.add_argument('--win_len',type=int, default =10)
parser.add_argument('--sample_rate',type=int, default = 64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--g_con', default=False, help="experiment for within subject")

parser.add_argument('--in_channel', type=int, default=64, help="channel of the input eeg signal")
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--d_inner', type=int, default=1024) 
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--n_layers',type=int, default=12)
parser.add_argument('--fft_conv1d_kernel', type=tuple,default=(9, 1))
parser.add_argument('--fft_conv1d_padding',type=tuple, default= (4, 0))
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--dropout',type=float,default=0.3)
parser.add_argument('--lamda',type=float,default=1)
parser.add_argument('--writing_interval', type=int, default=10)
parser.add_argument('--saving_interval', type=int, default=50)
parser.add_argument('--eval_interval', type=int, default=10, help='评估间隔（epoch数）')
parser.add_argument('--viz_sample_idx', type=int, default=0, help='用于可视化的固定测试样本索引')
parser.add_argument('--grad_log_interval', type=int, default=100, help='记录梯度信息的间隔步数')

parser.add_argument('--dataset_folder',type= str, default="/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data", help='write down your absolute path of dataset folder')
parser.add_argument('--split_folder',type= str, default="split_data")
parser.add_argument('--experiment_folder',default=None, help='write down experiment name')

# 添加分布式训练相关参数
parser.add_argument('--use_ddp', action='store_true', help='是否使用分布式训练')
parser.add_argument('--local-rank', default=-1, type=int, help='分布式训练的本地进程序号')
parser.add_argument('--seed', default=42, type=int, help='随机种子')
parser.add_argument('--workers', default=4, type=int, help='数据加载的工作线程数')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training')

# 添加在其他参数之后
parser.add_argument('--windows_per_sample', type=int, default=10, help='每个样本在一个epoch中采样的窗口数')

args = parser.parse_args()

# 设置随机种子以确保实验可重复性
torch.manual_seed(args.seed)
cudnn.benchmark = True

# 根据命令行参数决定是否启用DDP
use_ddp = args.use_ddp
local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", -1))

# 初始化分布式环境（如果启用）
if use_ddp and local_rank != -1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
else:
    world_size = 1
    rank = 0
    local_rank = -1

# Set the parameters and device
# 设置输入长度=采样率*窗口长度（秒）
input_length = args.sample_rate * args.win_len 
device = torch.device(f"cuda:{local_rank}" if use_ddp and local_rank != -1 else f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Provide the path of the dataset.
# which is split already to train, val, test (1:1:1).
data_folder = os.path.join(args.dataset_folder, args.split_folder)
features = ["eeg"] + ["envelope"]

# Create a directory to store (intermediate) results.
result_folder = 'test_results'
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.experiment_folder is None:
    if use_ddp:
        experiment_folder = "fft_nlayer{}_dmodel{}_nhead{}_win{}_dist_{}".format(args.n_layers, args.d_model, args.n_head, args.win_len, current_time)
    else:
        experiment_folder = "fft_nlayer{}_dmodel{}_nhead{}_win{}_single_{}".format(args.n_layers, args.d_model, args.n_head, args.win_len, current_time)
else: experiment_folder = args.experiment_folder

save_path = os.path.join(result_folder, experiment_folder)
# 只在主进程创建writer
is_main_process = (not use_ddp) or rank == 0
writer = get_writer(result_folder, experiment_folder) if is_main_process else None

def log_gradients_simple(model, writer, step):
    """
    记录简化版的梯度信息到TensorBoard
    """
    # 获取实际模型
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 按照层类型分组收集梯度
    layer_gradients = {}
    
    for name, param in actual_model.named_parameters():
        if param.grad is not None:
            # 提取层类型（如conv、fc、bn等）
            layer_type = name.split('.')[0] if '.' in name else 'other'
            
            if layer_type not in layer_gradients:
                layer_gradients[layer_type] = []
            
            # 存储梯度范数
            layer_gradients[layer_type].append(param.grad.norm(2).item())
    
    # 记录每种层类型的平均梯度范数
    for layer_type, grads in layer_gradients.items():
        avg_norm = sum(grads) / len(grads)
        writer.add_scalar(f"梯度/{layer_type}", avg_norm, step)
    
    # 计算全局梯度范数
    total_norm = 0
    for p in actual_model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    writer.add_scalar("梯度/全局范数", total_norm, step)

def Logger(content):
    if is_main_process:
        print(content)

def create_dataloader(split_name, data_folder, features, input_length, args, use_ddp, local_rank):
    is_train = split_name == 'train'
    files = [x for x in glob.glob(os.path.join(data_folder, f"{split_name}_-_*")) 
             if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    
    # 为训练集添加windows_per_sample参数（可以通过命令行参数控制）
    windows_per_sample = args.windows_per_sample if hasattr(args, 'windows_per_sample') else 10
    
    dataset = RegressionDataset(
        files, 
        input_length, 
        args.in_channel, 
        split_name, 
        args.g_con,
        windows_per_sample=windows_per_sample if is_train else 1  # 只在训练时使用多窗口
    )
    
    # 只在训练集和分布式模式下使用DistributedSampler
    sampler = DistributedSampler(dataset) if is_train and use_ddp and local_rank != -1 else None
    
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size if is_train else 1,
        num_workers=args.workers,
        sampler=sampler,
        drop_last=True,
        shuffle=(sampler is None and is_train)  # 只在训练集且无采样器时打乱
    )

def main():
    # Set the model and optimizer, scheduler.
    model = Decoder(**vars(args)).to(device)
    
    # 根据是否使用DDP决定是否包装模型
    if use_ddp and local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate,
                                betas=(0.9, 0.98),
                                eps=1e-09)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    # 创建数据加载器
    train_dataloader = create_dataloader('train', data_folder, features, input_length, args, use_ddp, local_rank)
    val_dataloader = create_dataloader('val', data_folder, features, input_length, args, use_ddp, local_rank)
    test_dataloader = create_dataloader('test', data_folder, features, input_length, args, use_ddp, local_rank)

    # 为可视化获取固定的测试样本
    viz_sample = None
    if is_main_process:
        for i, (test_inputs, test_labels, test_sub_id) in enumerate(test_dataloader):
            if i == args.viz_sample_idx:
                viz_sample = (test_inputs.squeeze(0).to(device), 
                              test_labels.squeeze(0).to(device), 
                              test_sub_id.to(device))
                break

    # 计算每个epoch的总步数
    iter_per_epoch = len(train_dataloader)
    global_step = 0
    
    # Train the model.
    for epoch in range(args.epoch):
        model.train()
        start_time = time.time()
        
        # 设置epoch，确保分布式采样器在每个epoch中提供不同的数据顺序
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
            
            # 在更新参数前记录梯度信息
            if writer and is_main_process and global_step % args.grad_log_interval == 0:
                log_gradients_simple(model, writer, global_step)
            
            optimizer.step()

            # 只显示训练进度
            if step % args.writing_interval == 0:
                spend_time = time.time() - start_time
                learning_rate = optimizer.param_groups[0]["lr"]
                
                # 计算当前步骤的平均时间和速度（batch/秒）
                time_per_step = spend_time / (step + 1)
                speed = 1.0 / time_per_step if time_per_step > 0 else 0
                
                # 计算所有剩余步骤的总时间
                steps_per_epoch = iter_per_epoch
                total_steps = args.epoch * steps_per_epoch
                current_total_steps = epoch * steps_per_epoch + step
                remaining_steps = total_steps - current_total_steps
                
                # 计算总剩余时间（秒）
                total_remaining = time_per_step * remaining_steps
                
                Logger(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.5f} 速度:{:.2f}批/秒 总剩余:{:.2f}秒:'.format(
                        epoch + 1,
                        args.epoch,
                        step,
                        iter_per_epoch,
                        loss.item(),
                        learning_rate,
                        speed,
                        total_remaining))
                
                if writer and is_main_process:
                    writer.add_losses("Loss", "train",  loss.item(), global_step)
                    # 记录损失函数组成部分
                    writer.add_scalar("损失/MSE", l_mse.mean().item(), global_step)
                    writer.add_scalar("损失/Pearson", l_p.mean().item(), global_step)
            
        # 修改为基于epoch的评估
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
                    val_loss   += pearson_loss(val_outputs, val_labels).mean()
                    val_metric += pearson_metric(val_outputs, val_labels).mean()

                val_loss /= len(val_dataloader)
                val_metric /= len(val_dataloader)
                val_metric = val_metric.mean()

                # 只在主进程上打印和记录日志
                if is_main_process:
                    Logger(f'|-Validation-|Step:{global_step}: loss:{val_loss.mean().item():.3f} metric:{val_metric.item():.3f}')
                    if writer:
                        writer.add_losses("Loss", "Validation",  val_loss.mean().item(), global_step)
                        writer.add_losses("Pearson", "Validation",  val_metric.item(), global_step)

                # Test the model.
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
                
                # 在主进程中进行测试样本的可视化
                if is_main_process and viz_sample is not None:
                    viz_inputs, viz_labels, viz_sub_id = viz_sample
                    viz_outputs = model(viz_inputs, viz_sub_id)
                    
                    # 转换为numpy以便绘图
                    viz_outputs_np = viz_outputs[0].cpu().numpy()
                    viz_labels_np = viz_labels[0].cpu().numpy()
                    
                    # 创建可视化图
                    plt.figure(figsize=(12, 6))
                    plt.plot(viz_labels_np, 'b-', label='real envelop')
                    plt.plot(viz_outputs_np, 'r-', label='rebuild envelop')
                    plt.title(f'Epoch {epoch+1}')
                    plt.xlabel('time')
                    plt.ylabel('amplitude')
                    plt.legend()
                    plt.grid(True)
                    
                    # 保存图像到结果目录
                    viz_save_path = os.path.join(save_path, 'visualizations')
                    os.makedirs(viz_save_path, exist_ok=True)
                    plt.savefig(os.path.join(viz_save_path, f'envelope_viz_epoch_{epoch+1}.png'))
                    plt.close()
                    
                    # 如果有writer，也添加到tensorboard
                    if writer:
                        # 创建一个网格图，用于tensorboard
                        fig = plt.figure(figsize=(12, 6))
                        plt.figure(figsize=(12, 6))
                        plt.plot(viz_labels_np, 'b-', label='real envelop')
                        plt.plot(viz_outputs_np, 'r-', label='rebuild envelop')
                        plt.title(f'Epoch {epoch+1}')
                        plt.xlabel('time')
                        plt.ylabel('amplitude')
                        plt.legend()
                        plt.grid(True)
                        writer.add_figure('语音包络可视化', fig, global_step)
                
                # 只在主进程上打印和记录日志
                if is_main_process:    
                    Logger(f'|-Test-|Step:{global_step}: loss:{test_loss.mean().item():.3f} metric:{test_metric.item():.3f}')
                    if writer:
                        writer.add_losses("Loss", "Test",  test_loss.mean().item(), global_step)
                        writer.add_losses("Pearson", "Test",  test_metric.item(), global_step)
                
            model.train()

        # 只在主进程上保存模型
        if (epoch + 1) % args.saving_interval == 0 and is_main_process:
            learning_rate = optimizer.param_groups[0]["lr"]
            # 分布式训练时保存模型的state_dict需要特殊处理
            if use_ddp and local_rank != -1:
                checkpoint = {
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'learning_rate': learning_rate
                }
                torch.save(checkpoint, os.path.join(save_path, f'model_step_{global_step}.pt'))
            else:
                save_checkpoint(model, optimizer, learning_rate, epoch, save_path, step=global_step)    

        scheduler.step()


if __name__ == '__main__':
    # 确保目录存在，仅在主进程中创建
    if is_main_process:
        os.makedirs(save_path, exist_ok=True)
    
    # 如果使用分布式训练，需要同步所有进程
    if use_ddp and local_rank != -1:
        dist.barrier()
        
    main() 