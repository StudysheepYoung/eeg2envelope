"""
Conformer-based Model for EEG Signal Processing
Replaces Transformer blocks with Conformer blocks for better local-global feature fusion

Based on "Conformer: Convolution-augmented Transformer for Speech Recognition"
https://arxiv.org/abs/2005.08100
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
import pdb
import math
from models.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from models.ConformerLayers import ConformerBlock


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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation通道注意力模块"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
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
    """
    Conformer-based Decoder for EEG Signal Processing

    This version replaces the original Transformer (PreLNFFTBlock) with ConformerBlock
    for better local-global feature modeling.

    Architecture:
        CNN Feature Extractor (3 layers)
        → SE Channel Attention
        → Subject Embedding (Global Conditioner)
        → Positional Encoding (optional, since Conformer has relative pos encoding)
        → Conformer Encoder Stack (n_layers)
        → Output Linear Layer
    """

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
                 within_sub_num=71,
                 # Conformer-specific parameters
                 conv_kernel_size=31,
                 use_relative_pos=True,
                 use_macaron_ffn=True,
                 use_sinusoidal_pos=True,  # Whether to use additional sinusoidal PE
                 **kwargs):

        super(Decoder, self).__init__()
        self.g_con = g_con
        self.within_sub_num = within_sub_num
        self.use_sinusoidal_pos = use_sinusoidal_pos

        self.fc = nn.Linear(d_model, 1)

        # 三层卷积：每层后依次 LayerNorm -> LeakyReLU -> Dropout
        self.conv1 = nn.Conv1d(in_channel, d_model, kernel_size=7, padding=3)
        self.norm1 = nn.LayerNorm(d_model)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.norm2 = nn.LayerNorm(d_model)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm3 = nn.LayerNorm(d_model)
        self.act3 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.drop3 = nn.Dropout(dropout)

        self.se = SEBlock(d_model, reduction=16)  # SE通道注意力

        # 位置编码层 (可选，因为Conformer已有相对位置编码)
        if use_sinusoidal_pos:
            self.pos_encoder = PositionalEncoding(d_model)
        else:
            self.pos_encoder = None

        # Conformer编码器层栈 (替换原有的Transformer层)
        self.layer_stack = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                d_inner=d_inner,
                n_head=n_head,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                use_relative_pos=use_relative_pos,
                use_macaron_ffn=use_macaron_ffn
            ) for _ in range(n_layers)
        ])

        # Subject embedding projection
        self.sub_proj = nn.Linear(self.within_sub_num, d_model)

    def forward(self, dec_input, sub_id):
        """
        Forward pass

        Args:
            dec_input: [batch_size, 64, 640] - EEG input (64 channels, 640 time steps)
            sub_id: [batch_size] - Subject IDs for global conditioning

        Returns:
            output: [batch_size, 640, 1] - Predicted values
        """

        # 三层卷积 + (LayerNorm -> LeakyReLU -> Dropout)
        x = dec_input.transpose(1, 2)  # [B, C, T]

        x = self.conv1(x)
        x = x.transpose(1, 2)          # [B, T, d_model]
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = x.transpose(1, 2)          # [B, d_model, T]

        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = x.transpose(1, 2)

        x = self.conv3(x)
        x = x.transpose(1, 2)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.drop3(x)
        dec_output = x.transpose(1, 2)  # [B, d_model, T]

        dec_output = self.se(dec_output)  # 加入通道注意力
        dec_output = dec_output.transpose(1, 2)  # [B, T, d_model]

        # Global conditioner.
        if self.g_con == True:
            sub_emb = F.one_hot(sub_id, self.within_sub_num)
            sub_emb = self.sub_proj(sub_emb.float())
            output = dec_output + sub_emb.unsqueeze(1)
        else:
            output = dec_output

        # 应用位置编码 (如果启用)
        if self.pos_encoder is not None:
            output = self.pos_encoder(output)

        # Conformer编码器层
        for conformer_layer in self.layer_stack:
            output = conformer_layer(output)

        output = self.fc(output)

        return output


if __name__ == "__main__":
    # Test the Conformer-based model
    batch_size = 4
    in_channel = 64
    seq_len = 640
    d_model = 256
    d_inner = 1024
    n_head = 4
    n_layers = 8

    # Create test input - Note: Input should be [B, T, C] format
    # EEG input: 640 time steps, 64 channels
    x = torch.randn(batch_size, seq_len, in_channel)  # [B, T, C] = [4, 640, 64]
    sub_id = torch.randint(0, 71, (batch_size,))

    # Create Conformer model
    model = Decoder(
        in_channel=in_channel,
        d_model=d_model,
        d_inner=d_inner,
        n_head=n_head,
        n_layers=n_layers,
        fft_conv1d_kernel=[3, 3],
        fft_conv1d_padding=[1, 1],
        dropout=0.3,
        g_con=True,
        within_sub_num=71,
        conv_kernel_size=31,
        use_relative_pos=True,
        use_macaron_ffn=True,
        use_sinusoidal_pos=True
    )

    # Forward pass
    output = model(x, sub_id)

    print(f"Input shape: {x.shape}")
    print(f"Subject ID shape: {sub_id.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nModel Architecture:")
    print(f"  - Input channels: {in_channel}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - FFN inner dimension: {d_inner}")
    print(f"  - Number of heads: {n_head}")
    print(f"  - Number of layers: {n_layers}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter count:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Memory estimation
    param_memory = total_params * 4 / (1024 * 1024)  # Float32 = 4 bytes
    print(f"  - Estimated model size: {param_memory:.2f} MB")

    print("\n✓ Conformer model test passed!")
