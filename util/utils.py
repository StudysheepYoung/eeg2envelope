import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import math
import argparse

def get_writer(output_directory, log_directory):

    logging_path=f'{output_directory}/{log_directory}'
    if os.path.exists(logging_path) == False:
        os.makedirs(logging_path)
    writer = CustomWriter(logging_path)
            
    return writer

class CustomWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(CustomWriter, self).__init__(log_dir)
        
    def add_losses(self, name, phase, loss, global_step):
        self.add_scalar(f'{name}/{phase}', loss, global_step)
        

def save_checkpoint(model, optimizer, learning_rate, epoch, filepath):
    print(f"Saving model and optimizer state at iteration {epoch} to {filepath}")
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{epoch}')
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',type=int, default=1000)
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--win_len',type=int, default =10)
    parser.add_argument('--sample_rate',type=int, default = 64)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--g_con', default=True, help="experiment for within subject")
    parser.add_argument('--in_channel', type=int, default=64, help="channel of the input eeg signal")
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_inner', type=int, default=1024) 
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_layers',type=int, default=8)
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
    return parser
