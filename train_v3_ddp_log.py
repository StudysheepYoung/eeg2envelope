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
from util.cal_pearson import l1_loss, pearson_loss, pearson_metric
from util.dataset import RegressionDataset
import time
import math

parser = argparse.ArgumentParser()

parser.add_argument('--epoch',type=int, default=1000)
parser.add_argument('--batch_size',type=int, default=64)
parser.add_argument('--win_len',type=int, default = 10)
parser.add_argument('--sample_rate',type=int, default = 64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--g_con', default=True, help="experiment for within subject")

parser.add_argument('--in_channel', type=int, default=64, help="channel of the input eeg signal")
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_inner', type=int, default=1024) 
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_layers',type=int, default=8)
parser.add_argument('--fft_conv1d_kernel', type=tuple,default=(9, 1))
parser.add_argument('--fft_conv1d_padding',type=tuple, default= (4, 0))
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--dropout',type=float,default=0.3)
parser.add_argument('--lamda',type=float,default=0.2)
parser.add_argument('--writing_interval', type=int, default=10)
parser.add_argument('--saving_interval', type=int, default=50)
parser.add_argument('--eval_interval', type=int, default=50)

parser.add_argument('--dataset_folder',type= str, default="/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data", help='write down your absolute path of dataset folder')
parser.add_argument('--split_folder',type= str, default="split_data")
parser.add_argument('--experiment_folder',default=None, help='write down experiment name')

# 添加分布式训练相关参数
parser.add_argument('--local-rank', default=-1, type=int, help='分布式训练的本地进程序号')
parser.add_argument('--seed', default=42, type=int, help='随机种子')
parser.add_argument('--workers', default=4, type=int, help='数据加载的工作线程数')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training')

args = parser.parse_args()

# 设置随机种子以确保实验可重复性
torch.manual_seed(args.seed)
cudnn.benchmark = True

# 初始化分布式环境
local_rank = args.local_rank if args.local_rank != -1 else int(os.environ.get("LOCAL_RANK", -1))
if local_rank != -1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
else:
    world_size = 1
    rank = 0

# Set the parameters and device
input_length = args.sample_rate * args.win_len 
device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Provide the path of the dataset.
# which is split already to train, val, test (1:1:1).
data_folder = os.path.join(args.dataset_folder, args.split_folder)
features = ["eeg"] + ["envelope"]

# Create a directory to store (intermediate) results.
result_folder = 'test_results'
if args.experiment_folder is None:
    experiment_folder = "fft_nlayer{}_dmodel{}_nhead{}_win{}_dist".format(args.n_layers, args.d_model, args.n_head, args.win_len)
else: experiment_folder = args.experiment_folder

save_path = os.path.join(result_folder, experiment_folder)
# 只在主进程创建writer
writer = get_writer(result_folder, experiment_folder) if (rank == 0 or local_rank == -1) else None

def Logger(content):
    if rank == 0 or local_rank == -1:
        print(content)

def main():
    # Set the model and optimizer, scheduler.
    model = Decoder(**vars(args)).to(device)
    
    # 分布式训练包装
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate,
                                betas=(0.9, 0.98),
                                eps=1e-09)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    # Define train set and loader.
    train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    train_set= RegressionDataset(train_files, input_length, args.in_channel, 'train', args.g_con)
    
    # 使用DistributedSampler进行分布式训练
    if local_rank != -1:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = None
        
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size = args.batch_size,
            num_workers = args.workers,
            sampler = train_sampler,
            drop_last=True,
            shuffle=(train_sampler is None))

    # Define validation set and loader.
    val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    val_set = RegressionDataset(val_files, input_length, args.in_channel, 'val', args.g_con)
    val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size = 1,
            num_workers = args.workers,
            sampler = None,
            drop_last=True,
            shuffle=False)

    # Define test set and loader.
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    test_set = RegressionDataset(test_files, input_length, args.in_channel, 'test', args.g_con)
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size = 1,
        num_workers = args.workers,
        sampler = None,
        drop_last=True,
        shuffle=False)

    # 计算每个epoch的总步数
    iter_per_epoch = len(train_dataloader)
    global_step = 0
    
    # Train the model.
    for epoch in range(args.epoch):
        model.train()
        start_time = time.time()
        
        # 设置epoch，确保分布式采样器在每个epoch中提供不同的数据顺序
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, (inputs, labels, sub_id) in enumerate(train_dataloader):
            global_step += 1
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)
            sub_id = sub_id.to(device)
            outputs = model(inputs, sub_id)

            l_p = pearson_loss(outputs, labels) 
            l_1 = l1_loss(outputs, labels)
            loss = l_p + args.lamda * l_1
            loss = loss.mean()
            loss.backward()
            optimizer.step()

            # 只显示训练进度
            if step % args.writing_interval == 0:
                spend_time = time.time() - start_time
                learning_rate = optimizer.param_groups[0]["lr"]
                Logger(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.5f} epoch_Time:{}min:'.format(
                        epoch + 1,
                        args.epoch,
                        step,
                        iter_per_epoch,
                        loss.item(),
                        learning_rate,
                        spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
                
                if writer and (rank == 0 or local_rank == -1):
                    writer.add_losses("Loss", "train",  loss.item(), global_step)
            
            # 每eval_interval步评估一次验证集和测试集
            if global_step % args.eval_interval == 0:
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
                    if rank == 0 or local_rank == -1:
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
                    
                    # 只在主进程上打印和记录日志
                    if rank == 0 or local_rank == -1:    
                        Logger(f'|-Test-|Step:{global_step}: loss:{test_loss.mean().item():.3f} metric:{test_metric.item():.3f}')
                        if writer:
                            writer.add_losses("Loss", "Test",  test_loss.mean().item(), global_step)
                            writer.add_losses("Pearson", "Test",  test_metric.item(), global_step)
                
                model.train()

            # 只在主进程上保存模型
            if global_step % args.saving_interval == 0 and (rank == 0 or local_rank == -1):
                learning_rate = optimizer.param_groups[0]["lr"]
                # 分布式训练时保存模型的state_dict需要特殊处理
                if local_rank != -1:
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
    if rank == 0 or local_rank == -1:
        os.makedirs(save_path, exist_ok=True)
    
    # 如果使用分布式训练，需要同步所有进程
    if local_rank != -1:
        dist.barrier()
        
    main()