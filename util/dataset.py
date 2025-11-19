import torch
import itertools
import os
import numpy as np
from torch.utils.data import Dataset
import pdb
from random import randint

class RegressionDataset(Dataset):
    """Generate data for the regression task."""

    def __init__(
        self,
        files,
        input_length,
        channels,
        task,
        g_con = True,
        windows_per_sample = 1  # 新增参数，控制每个样本采样的窗口数
    ):

        self.input_length = input_length
        self.files = self.group_recordings(files)
        self.channels = channels
        self.task = task
        self.g_con = g_con
        self.windows_per_sample = windows_per_sample  # 每个样本采样的窗口数
        
        # 预加载数据到内存，减少IO开销
        self.preloaded_data = {}
        self.preload_data()
        
    def preload_data(self):
        """预加载所有数据到内存，减少训练过程中的IO操作"""
        if self.task == "train":    
            print(f"开始预加载{len(self.files)}组训练用eeg和对应envelope到内存...")
        elif self.task == "test":
            print(f"开始预加载{len(self.files)}组验证用eeg和对应envelope到内存...")
        else:
            print(f"开始预加载{len(self.files)}组测试用eeg和对应envelope到内存...")
        for i, file_group in enumerate(self.files):
            self.preloaded_data[i] = []
            for feature in file_group:
                data = np.load(feature)
                # 获取受试者ID
                if self.g_con == True:
                    sub_idx = feature.split('/')[-1].split('_-_')[1].split('-')[-1]
                    sub_idx = int(sub_idx) - 1
                else:
                    sub_idx = 0
                self.preloaded_data[i].append((data, sub_idx))
        if self.task == "train":    
            print("训练数据预加载完成！")
        elif self.task == "test":
            print("验证数据预加载完成！")
        else:
            print("测试数据预加载完成！")
        

    def group_recordings(self, files):
 
        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]

        return new_files

    def __len__(self):
        # 新的长度为原始样本数 × 每个样本的窗口数
        return len(self.files) * self.windows_per_sample if self.task == "train" else len(self.files)

    def __getitem__(self, index):
        # 对于训练数据，计算真实的记录索引和窗口索引
        if self.task == "train":
            recording_index = index // self.windows_per_sample
            window_index = index % self.windows_per_sample
            x, y, sub_id = self.__train_data__(recording_index, window_index)
        else:
            x, y, sub_id = self.__test_data__(index)

        return x, y, sub_id


    def __train_data__(self, recording_index, window_index=0):
        """从预加载的数据中获取训练数据，支持多窗口采样"""
        framed_data = []
        
        # 使用预加载的数据
        for idx, (data, sub_idx) in enumerate(self.preloaded_data[recording_index]):
            # 计算可采样的最大起始位置
            max_start = len(data) - self.input_length
            
            if idx == 0:
                # 根据窗口索引确定采样位置，确保不同窗口采样不同区域
                if self.windows_per_sample > 1:
                    # 将可用范围平均分配给每个窗口
                    segment_size = max_start // self.windows_per_sample
                    # 在分配的段内随机选择起点，增加随机性
                    segment_start = window_index * segment_size
                    segment_end = segment_start + segment_size if window_index < self.windows_per_sample-1 else max_start
                    start_idx = randint(segment_start, segment_end)
                else:
                    # 保持原有的完全随机采样
                    start_idx = randint(0, max_start)
            
            framed_data += [data[start_idx:start_idx + self.input_length]]
        
        # 获取预加载数据中的受试者ID
        sub_idx = self.preloaded_data[recording_index][0][1]
            
        return torch.FloatTensor(framed_data[0]), torch.FloatTensor(framed_data[1]), sub_idx

    def __test_data__(self, recording_index):
        """从预加载的数据中获取测试数据"""
        framed_data = []
        
        # 使用预加载的数据
        for idx, (data, sub_idx) in enumerate(self.preloaded_data[recording_index]):
            nsegment = data.shape[0] // self.input_length
            data = data[:int(nsegment * self.input_length)]
            segment_data = [torch.FloatTensor(data[i:i+self.input_length]).unsqueeze(0) for i in range(0, data.shape[0], self.input_length)]
            segment_data = torch.cat(segment_data)
            framed_data += [segment_data]
        
        # 获取预加载数据中的受试者ID
        sub_idx = self.preloaded_data[recording_index][0][1]

        return framed_data[0], framed_data[1], sub_idx
