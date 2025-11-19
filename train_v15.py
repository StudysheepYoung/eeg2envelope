import glob
import os
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import torch.backends.cudnn as cudnn
from util.utils import get_writer, save_checkpoint, get_parser
from torch.optim.lr_scheduler import StepLR
from models.FFT_block import Decoder
from util.cal_pearson import l1_loss, mse_loss, pearson_loss, pearson_metric
from util.dataset import RegressionDataset
import time
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np

def init_all():
    parser = get_parser()
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

    # 设置输入长度=采样率*窗口长度（秒）
    input_length = args.sample_rate * args.win_len 
    device = torch.device(f"cuda:{local_rank}" if use_ddp and local_rank != -1 else f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 数据集路径
    data_folder = os.path.join(args.dataset_folder, args.split_folder)
    features = ["eeg"] + ["envelope"]

    # 结果保存路径
    result_folder = 'test_results'
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_folder is None:
        if use_ddp:
            experiment_folder = "fft_nlayer{}_dmodel{}_nhead{}_win{}_dist_{}".format(args.n_layers, args.d_model, args.n_head, args.win_len, current_time)
        else:
            experiment_folder = "fft_nlayer{}_dmodel{}_nhead{}_win{}_single_{}".format(args.n_layers, args.d_model, args.n_head, args.win_len, current_time)
    else:
        experiment_folder = args.experiment_folder
    save_path = os.path.join(result_folder, experiment_folder)
    is_main_process = (not use_ddp) or rank == 0
    writer = get_writer(result_folder, experiment_folder) if is_main_process else None

    # 只在主进程创建目录
    if is_main_process:
        os.makedirs(save_path, exist_ok=True)
    # 分布式同步
    if use_ddp and local_rank != -1:
        dist.barrier()

    return args, input_length, device, data_folder, features, save_path, is_main_process, writer, use_ddp, local_rank, rank

# 定义记录梯度信息的函数
def log_gradients(model, writer, step):
    """
    记录模型梯度信息到TensorBoard
    
    参数:
    model: 训练模型
    writer: TensorBoard SummaryWriter实例
    step: 全局步数
    """
    # 获取实际模型（如果是DDP模型）
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 记录每一层的梯度直方图和模型参数
    for name, param in actual_model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"梯度/{name}", param.grad.data, step)
            writer.add_histogram(f"权重/{name}", param.data, step)
    
    # 计算并记录梯度的全局统计信息
    total_norm = 0
    param_norm = 0
    for p in actual_model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    # 记录梯度范数
    writer.add_scalar("梯度/范数", total_norm, step)

def Logger(content):
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

def show_progress(mode, step, loss, metric=None, epoch=None, total_epoch=None, iter_per_epoch=None, lr=None, start_time=None, total_steps=None):
    if mode == 'train':
        learning_rate = lr
        if epoch is not None and total_epoch is not None and iter_per_epoch is not None:
            Logger(
                f'Epoch:[{epoch}/{total_epoch}]({step}/{iter_per_epoch}) loss:{loss:.3f} lr:{learning_rate:.5f}'
            )
    elif mode == 'val':
        Logger(
            f'|-Validation-|Step:{step}: loss:{loss:.3f} metric:{metric:.3f}'
        )
    elif mode == 'test':
        Logger(
            f'|-Test-|Step:{step}: loss:{loss:.3f} metric:{metric:.3f}'
        )

def save_model(use_ddp, local_rank, model, optimizer, learning_rate, epoch, save_path, global_step):
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

def visualize_sample(model, viz_sample, epoch, save_path, device):
    viz_inputs, viz_labels, viz_sub_id = viz_sample
    with torch.no_grad():
        viz_inputs = viz_inputs.to(device)
        viz_labels = viz_labels.to(device)
        viz_sub_id = viz_sub_id.to(device)
        viz_outputs = model(viz_inputs, viz_sub_id)
    viz_outputs_np = viz_outputs[0].cpu().numpy()
    viz_labels_np = viz_labels[0].cpu().numpy()
    plt.figure(figsize=(12, 6))
    plt.plot(viz_labels_np, 'b-', label='real envelop')
    plt.plot(viz_outputs_np, 'r-', label='rebuild envelop')
    plt.title(f'Epoch {epoch+1}')
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.legend()
    plt.grid(True)
    viz_save_path = os.path.join(save_path, 'visualizations')
    os.makedirs(viz_save_path, exist_ok=True)
    plt.savefig(os.path.join(viz_save_path, f'envelope_viz_epoch_{epoch+1}.png'))
    plt.close()

def evaluate_and_visualize(model, val_dataloader, test_dataloader, writer, global_step, is_main_process, viz_sample, epoch, save_path, device):
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
        if is_main_process:
            show_progress('val', global_step, val_loss.mean().item(), metric=val_metric.item())
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
        # 可视化
        if is_main_process and viz_sample is not None:
            visualize_sample(model, viz_sample, epoch, save_path, device)
        if is_main_process:
            show_progress('test', global_step, test_loss.mean().item(), metric=test_metric.item())
        if writer:
            writer.add_losses("Loss", "Test",  test_loss.mean().item(), global_step)
            writer.add_losses("Pearson", "Test",  test_metric.item(), global_step)
    model.train()

def main():
    # 用init_all初始化所有变量
    args, input_length, device, data_folder, features, save_path, is_main_process, writer, use_ddp, local_rank, rank = init_all()

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
                log_gradients(model, writer, global_step)
            
            optimizer.step()

            # 只显示训练进度
            if step % args.writing_interval == 0:
                learning_rate = optimizer.param_groups[0]["lr"]
                
                if is_main_process:
                    show_progress('train', step, loss.item(), epoch=epoch+1, total_epoch=args.epoch, iter_per_epoch=iter_per_epoch, lr=learning_rate, start_time=start_time)
                
                if writer and is_main_process:
                    writer.add_losses("Loss", "train",  loss.item(), global_step)
                    # 记录损失函数组成部分
                    writer.add_scalar("损失/MSE", l_mse.mean().item(), global_step)
                    writer.add_scalar("损失/Pearson", l_p.mean().item(), global_step)
            
        # 修改为基于epoch的评估
        if (epoch + 1) % args.eval_interval == 0:
            evaluate_and_visualize(model, val_dataloader, test_dataloader, writer, global_step, is_main_process, viz_sample, epoch, save_path, device)

        # 只在主进程上保存模型
        if (epoch + 1) % args.saving_interval == 0 and is_main_process:
            learning_rate = optimizer.param_groups[0]["lr"]
            save_model(use_ddp, local_rank, model, optimizer, learning_rate, epoch, save_path, global_step)    

        scheduler.step()


if __name__ == '__main__':
    # 只需直接调用main()
    main() 