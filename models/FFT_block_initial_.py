import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
import pdb
import math
from models.SubLayers import MultiHeadAttention, PositionwiseFeedForward

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_seq_len=640):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 使用正弦和余弦函数计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        
        # 注册为缓冲区，使其成为模型状态的一部分但不是参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, d_model]
        """
        # 添加位置编码到输入
        x = x + self.pe[:, :x.size(1), :]
        return x

class PreLNFFTBlock(torch.nn.Module):

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 **kwargs):

        super(PreLNFFTBlock, self).__init__()

        d_k = d_model // n_head
        d_v = d_model // n_head

        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel,fft_conv1d_padding, dropout=dropout)

    def forward(self, fft_input):

        # dec_input size: [B,T,C]
        fft_output, _= self.slf_attn(
            fft_input, fft_input, fft_input)

        fft_output = self.pos_ffn(fft_output)

        return fft_output

class SEBlock(nn.Module):
    """Squeeze-and-Excitation通道注意力模块"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Decoder(nn.Module):

    def __init__(self,
                 in_channel,
                 d_model,
                 d_inner,
                 n_head,
                 n_layers,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,             
                 dropout,
                 g_con=True,
                 within_sub_num = 71,
                 **kwargs):

        super(Decoder, self).__init__()
        self.g_con = g_con
        self.within_sub_num = within_sub_num
        self.slf_attn = MultiHeadAttention
        self.fc = nn.Linear(d_model, 1)   
        self.conv = nn.Conv1d(in_channel, d_model, kernel_size = 7, padding=3)
        self.se = SEBlock(d_model, reduction=16)  # 新增SEBlock
        
        # 添加位置编码层
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layer_stack = nn.ModuleList([PreLNFFTBlock(
            d_model, d_inner, n_head, fft_conv1d_kernel,fft_conv1d_padding, dropout) for _ in range(n_layers)])
        self.sub_proj = nn.Linear(self.within_sub_num, d_model)

    def forward(self, dec_input, sub_id):

        dec_output = self.conv(dec_input.transpose(1,2))  # [B, d_model, T]
        dec_output = self.se(dec_output)  # 加入通道注意力
        dec_output = dec_output.transpose(1,2)  # [B, T, d_model]

        # Global conditioner.
        if self.g_con == True:
            sub_emb    = F.one_hot(sub_id, self.within_sub_num)
            sub_emb    = self.sub_proj(sub_emb.float())
            output = dec_output + sub_emb.unsqueeze(1)
        else:
            output = dec_output
        
        # 应用位置编码
        output = self.pos_encoder(output)
        
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output)

        output = self.fc(output)

        return output
