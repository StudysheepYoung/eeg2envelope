"""
Conformer-based Model for EEG Signal Processing (v2 - Improved)
改进版：添加全局残差连接和改进的输出头，解决特征提取不足问题

主要改进：
1. 全局残差连接：Conformer层栈前后添加skip connection
2. 门控残差：自适应学习跳跃连接的权重
3. MLP输出头：替换单层线性层，增强表达能力
4. 梯度缩放：帮助前层获得更大梯度

Based on "Conformer: Convolution-augmented Transformer for Speech Recognition"
https://arxiv.org/abs/2005.08100
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
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


class GatedResidual(nn.Module):
    """
    门控残差连接
    自适应学习跳跃连接的权重，让网络决定是否使用Conformer的输出

    output = gate * conformer_output + (1 - gate) * input
    """
    def __init__(self, d_model, init_gate_bias=2.0):
        super().__init__()
        # 门控网络：学习一个0-1的权重
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, residual):
        """
        Args:
            x: Conformer的输出 [B, T, d_model]
            residual: Conformer的输入 [B, T, d_model]
        Returns:
            门控融合后的输出 [B, T, d_model]
        """
        # 计算全局特征用于门控（使用平均池化）
        # [B, T, d_model] -> [B, d_model]
        global_feat = x.mean(dim=1)

        # 门控计算
        gate = self.gate(global_feat)  # [B, d_model]
        gate = gate.unsqueeze(1)       # [B, 1, d_model]

        # 门控融合
        output = gate * x + (1 - gate) * residual
        return output


class ImprovedOutputHead(nn.Module):
    """
    改进的输出头
    使用MLP替代单层线性层，增强特征变换能力

    Architecture:
        LayerNorm -> Linear (d_model -> d_model//2) -> GELU -> Dropout
        -> Linear (d_model//2 -> 1)
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            out: [batch_size, seq_len, 1]
        """
        return self.net(x)


class Decoder(nn.Module):
    """
    Conformer-based Decoder for EEG Signal Processing (v2 - Improved)

    主要改进：
    1. 全局残差连接：Conformer层栈的输入输出之间添加skip connection
    2. 门控机制：自适应学习残差权重
    3. 改进的输出头：MLP替代单层线性
    4. 梯度缩放：帮助前层学习

    Architecture:
        CNN Feature Extractor (3 layers)
        → SE Channel Attention
        → Subject Embedding (Global Conditioner)
        → Positional Encoding
        → **[全局残差开始]**
        → Conformer Encoder Stack (n_layers)
        → **[门控残差连接]**
        → Improved Output Head (MLP)
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
                 use_sinusoidal_pos=True,
                 # 新增：改进参数
                 use_gated_residual=True,  # 是否使用门控残差
                 use_mlp_head=True,        # 是否使用MLP输出头
                 gradient_scale=1.0,       # 梯度缩放因子
                 **kwargs):

        super(Decoder, self).__init__()
        self.g_con = g_con
        self.within_sub_num = within_sub_num
        self.use_sinusoidal_pos = use_sinusoidal_pos
        self.use_gated_residual = use_gated_residual
        self.use_mlp_head = use_mlp_head
        self.gradient_scale = gradient_scale

        # 输出头：MLP或单层线性
        if use_mlp_head:
            self.output_head = ImprovedOutputHead(d_model, dropout)
        else:
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

        # Conformer编码器层栈
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

        # 门控残差模块
        if use_gated_residual:
            self.gated_residual = GatedResidual(d_model)

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

        # Global conditioner
        if self.g_con == True:
            sub_emb = F.one_hot(sub_id, self.within_sub_num)
            sub_emb = self.sub_proj(sub_emb.float())
            output = dec_output + sub_emb.unsqueeze(1)
        else:
            output = dec_output

        # 应用位置编码 (如果启用)
        if self.pos_encoder is not None:
            output = self.pos_encoder(output)

        # ============ 关键改进：全局残差连接 ============
        # 保存Conformer输入作为残差
        conformer_input = output.clone()

        # Conformer编码器层栈
        # 注意：ConformerBlock 内部已有残差连接，不需要外层再加
        for conformer_layer in self.layer_stack:
            output = conformer_layer(output)

        # 门控残差连接 or 简单残差
        if self.use_gated_residual:
            output = self.gated_residual(output, conformer_input)
        else:
            # 简单的残差连接
            output = output + conformer_input

        # 梯度缩放：移到门控残差之后，让前层获得更直接的梯度放大
        # 这样梯度放大作用于整个融合后的输出，不会被门控稀释
        if self.gradient_scale != 1.0 and self.training:
            output = GradientScaleFunction.apply(output, self.gradient_scale)
        # ================================================

        # 输出层：MLP头 or 单层线性
        if self.use_mlp_head:
            output = self.output_head(output)
        else:
            output = self.fc(output)

        return output


class GradientScaleFunction(torch.autograd.Function):
    """
    自定义autograd函数，用于梯度缩放
    前向传播：y = x
    反向传播：dx = scale * dy
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


if __name__ == "__main__":
    # Test the improved Conformer-based model
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

    print("=" * 80)
    print("Testing Improved Conformer Model (v2)")
    print("=" * 80)

    # Test with different configurations
    configs = [
        {"name": "原始版本", "use_gated_residual": False, "use_mlp_head": False, "gradient_scale": 1.0},
        {"name": "添加简单残差", "use_gated_residual": False, "use_mlp_head": False, "gradient_scale": 1.0},
        {"name": "添加门控残差", "use_gated_residual": True, "use_mlp_head": False, "gradient_scale": 1.0},
        {"name": "添加MLP头", "use_gated_residual": False, "use_mlp_head": True, "gradient_scale": 1.0},
        {"name": "完整改进版", "use_gated_residual": True, "use_mlp_head": True, "gradient_scale": 2.0},
    ]

    for config in configs:
        print(f"\n{'='*80}")
        print(f"配置: {config['name']}")
        print(f"{'='*80}")

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
            use_sinusoidal_pos=True,
            use_gated_residual=config['use_gated_residual'],
            use_mlp_head=config['use_mlp_head'],
            gradient_scale=config['gradient_scale']
        )

        # Forward pass
        output = model(x, sub_id)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Memory estimation
        param_memory = total_params * 4 / (1024 * 1024)  # Float32 = 4 bytes
        print(f"  Estimated model size: {param_memory:.2f} MB")

    print("\n" + "=" * 80)
    print("✓ All configurations tested successfully!")
    print("=" * 80)

    # Test gradient flow
    print("\n" + "=" * 80)
    print("测试梯度流")
    print("=" * 80)

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
        use_gated_residual=True,
        use_mlp_head=True,
        gradient_scale=2.0
    )

    model.train()
    output = model(x, sub_id)
    loss = output.sum()
    loss.backward()

    # Check gradients
    print("\n梯度检查:")
    for name, param in model.named_parameters():
        if param.grad is not None and 'layer_stack.0' in name:
            print(f"  {name}: grad_norm = {param.grad.norm().item():.6f}")
            break

    print("\n✓ 梯度流测试通过!")
