"""
Conformer Layers for EEG Signal Processing
Based on "Conformer: Convolution-augmented Transformer for Speech Recognition"
https://arxiv.org/abs/2005.08100

Adapted for EEG time-series modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit: split input in half along channel dim, apply sigmoid gate"""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


class ConvolutionModule(nn.Module):
    """
    Convolution Module in Conformer

    Architecture:
        LayerNorm -> Pointwise Conv (expansion) -> GLU ->
        Depthwise Conv -> BatchNorm -> Swish ->
        Pointwise Conv (projection) -> Dropout

    Args:
        d_model: input/output dimension
        kernel_size: depthwise conv kernel size (default: 31)
        expansion_factor: expansion factor for pointwise conv (default: 2)
        dropout: dropout rate
    """
    def __init__(self, d_model, kernel_size=31, expansion_factor=2, dropout=0.1):
        super().__init__()

        inner_dim = d_model * expansion_factor
        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise convolution (expansion)
        # Input: [B, T, d_model] -> [B, d_model, T]
        # Output channels = inner_dim * 2 (for GLU)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, inner_dim * 2, kernel_size=1, stride=1, padding=0
        )

        # GLU activation
        self.glu = GLU(dim=1)  # Apply along channel dimension

        # Depthwise convolution
        # Ensure kernel_size is odd for 'same' padding
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Conv1d(
            inner_dim, inner_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=inner_dim  # Depthwise: each channel convolved separately
        )

        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.activation = Swish()

        # Pointwise convolution (projection)
        self.pointwise_conv2 = nn.Conv1d(
            inner_dim, d_model, kernel_size=1, stride=1, padding=0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            out: [batch_size, seq_len, d_model]
        """
        # Layer normalization
        x = self.layer_norm(x)  # [B, T, d_model]

        # Transpose for conv1d: [B, T, d_model] -> [B, d_model, T]
        x = x.transpose(1, 2)

        # Pointwise expansion + GLU
        x = self.pointwise_conv1(x)  # [B, inner_dim*2, T]
        x = self.glu(x)  # [B, inner_dim, T]

        # Depthwise convolution
        x = self.depthwise_conv(x)  # [B, inner_dim, T]
        x = self.batch_norm(x)
        x = self.activation(x)

        # Pointwise projection
        x = self.pointwise_conv2(x)  # [B, d_model, T]
        x = self.dropout(x)

        # Transpose back: [B, d_model, T] -> [B, T, d_model]
        x = x.transpose(1, 2)

        return x


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding for Self-Attention
    Based on "Self-Attention with Relative Position Representations"
    https://arxiv.org/abs/1803.02155
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create relative position embeddings
        # Shape: [2*max_len-1, d_model]
        self.rel_pos_emb = nn.Parameter(
            torch.randn(2 * max_len - 1, d_model) / np.sqrt(d_model)
        )

    def forward(self, seq_len):
        """
        Generate relative position embeddings for sequence length

        Args:
            seq_len: sequence length
        Returns:
            rel_pos: [seq_len, seq_len, d_model]
        """
        # Create relative position matrix
        # pos_i - pos_j ranges from -(seq_len-1) to +(seq_len-1)
        center = self.max_len - 1
        start = center - (seq_len - 1)
        end = center + seq_len

        # Extract relevant slice
        rel_pos = self.rel_pos_emb[start:end]  # [2*seq_len-1, d_model]

        # Create position difference matrix
        # rel_pos[i, j] = embedding for position difference (i - j)
        positions = torch.arange(seq_len, device=rel_pos.device)
        rel_indices = positions.unsqueeze(1) - positions.unsqueeze(0) + (seq_len - 1)

        return rel_pos[rel_indices]  # [seq_len, seq_len, d_model]


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with Relative Position Encoding

    Args:
        d_model: model dimension
        n_head: number of attention heads
        dropout: dropout rate
        use_relative_pos: whether to use relative positional encoding
    """
    def __init__(self, d_model, n_head, dropout=0.1, use_relative_pos=True):
        super().__init__()

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.use_relative_pos = use_relative_pos

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Relative position encoding
        if use_relative_pos:
            self.rel_pos_enc = RelativePositionalEncoding(self.d_k, max_len=1000)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # Xavier initialization
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q, k, v: [batch_size, seq_len, d_model]
            mask: optional attention mask
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = q.size(0), q.size(1)
        residual = q

        # Pre-LayerNorm
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        # Linear projections and split into heads
        # [B, T, d_model] -> [B, T, n_head, d_k] -> [B, n_head, T, d_k]
        q = self.w_q(q).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # [B, n_head, T, d_k] x [B, n_head, d_k, T] -> [B, n_head, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Add relative position bias if enabled
        if self.use_relative_pos:
            # More memory-efficient relative position computation
            # Instead of creating [B, n_head, T, T, d_k] tensor, use einsum

            # Get relative position embeddings: [T, T, d_k]
            rel_pos = self.rel_pos_enc(seq_len)

            # q: [B, n_head, T, d_k]
            # rel_pos: [T, T, d_k]
            # We want: rel_pos_scores[b, h, i, j] = sum_k(q[b, h, i, k] * rel_pos[i, j, k])

            # Use einsum for memory-efficient computation
            # 'bhik,ijk->bhij' means: batch, head, query_pos, key_dim x query_pos, key_pos, key_dim -> batch, head, query_pos, key_pos
            rel_pos_scores = torch.einsum('bhik,ijk->bhij', q, rel_pos)

            # Add to attention scores
            scores = scores + rel_pos_scores / np.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        # [B, n_head, T, T] x [B, n_head, T, d_k] -> [B, n_head, T, d_k]
        output = torch.matmul(attn, v)

        # Concatenate heads and project
        # [B, n_head, T, d_k] -> [B, T, n_head, d_k] -> [B, T, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        output = self.dropout(output)

        # Residual connection
        return output + residual


class FeedForwardModule(nn.Module):
    """
    Feed-Forward Module in Conformer
    Can be used in Macaron-style (half the output dimension)

    Architecture:
        LayerNorm -> Linear (expansion) -> Swish -> Dropout ->
        Linear (projection) -> Dropout

    Args:
        d_model: input dimension
        d_inner: inner dimension (expansion)
        dropout: dropout rate
        use_macaron: if True, apply 0.5 scaling for Macaron-style FFN
    """
    def __init__(self, d_model, d_inner, dropout=0.1, use_macaron=False):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_inner)
        self.w_2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = Swish()
        self.use_macaron = use_macaron

        # Xavier initialization
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            out: [batch_size, seq_len, d_model]
        """
        residual = x

        x = self.layer_norm(x)
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        # Macaron-style: scale by 0.5
        if self.use_macaron:
            x = 0.5 * x

        return x + residual


class ConformerBlock(nn.Module):
    """
    Conformer Block

    Architecture:
        Input
        ↓
        Feed-Forward Module (Macaron-style, 1/2)
        ↓
        Multi-Head Self-Attention Module (with relative pos encoding)
        ↓
        Convolution Module
        ↓
        Feed-Forward Module (Macaron-style, 1/2)
        ↓
        LayerNorm
        ↓
        Output

    Args:
        d_model: model dimension
        d_inner: FFN inner dimension
        n_head: number of attention heads
        conv_kernel_size: convolution kernel size
        dropout: dropout rate
        use_relative_pos: whether to use relative positional encoding in attention
        use_macaron_ffn: whether to use Macaron-style FFN
    """
    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        conv_kernel_size=31,
        dropout=0.1,
        use_relative_pos=True,
        use_macaron_ffn=True
    ):
        super().__init__()

        self.use_macaron_ffn = use_macaron_ffn

        # First FFN (Macaron-style)
        if use_macaron_ffn:
            self.ffn1 = FeedForwardModule(d_model, d_inner, dropout, use_macaron=True)

        # Multi-Head Self-Attention
        self.self_attn = RelativeMultiHeadAttention(
            d_model, n_head, dropout, use_relative_pos=use_relative_pos
        )

        # Convolution Module
        self.conv_module = ConvolutionModule(
            d_model, kernel_size=conv_kernel_size, expansion_factor=2, dropout=dropout
        )

        # Second FFN (Macaron-style)
        if use_macaron_ffn:
            self.ffn2 = FeedForwardModule(d_model, d_inner, dropout, use_macaron=True)
        else:
            # Standard FFN without Macaron
            self.ffn2 = FeedForwardModule(d_model, d_inner, dropout, use_macaron=False)

        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: optional attention mask
        Returns:
            out: [batch_size, seq_len, d_model]
        """
        # First FFN (if Macaron-style)
        if self.use_macaron_ffn:
            x = self.ffn1(x)

        # Multi-Head Self-Attention
        x = self.self_attn(x, x, x, mask)

        # Convolution Module
        x = x + self.conv_module(x)  # Residual connection

        # Second FFN
        x = self.ffn2(x)

        # Final layer norm
        x = self.layer_norm(x)

        return x


if __name__ == "__main__":
    # Test ConformerBlock
    batch_size = 4
    seq_len = 640
    d_model = 256
    d_inner = 1024
    n_head = 4

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)

    # Create Conformer block
    conformer_block = ConformerBlock(
        d_model=d_model,
        d_inner=d_inner,
        n_head=n_head,
        conv_kernel_size=31,
        dropout=0.1,
        use_relative_pos=True,
        use_macaron_ffn=True
    )

    # Forward pass
    output = conformer_block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in conformer_block.parameters()):,}")

    # Test individual modules
    print("\n=== Testing Individual Modules ===")

    # Test ConvolutionModule
    conv_module = ConvolutionModule(d_model=256, kernel_size=31, dropout=0.1)
    conv_out = conv_module(x)
    print(f"ConvolutionModule output shape: {conv_out.shape}")

    # Test RelativeMultiHeadAttention
    attn_module = RelativeMultiHeadAttention(d_model=256, n_head=4, use_relative_pos=True)
    attn_out = attn_module(x, x, x)
    print(f"RelativeMultiHeadAttention output shape: {attn_out.shape}")

    # Test FeedForwardModule
    ffn_module = FeedForwardModule(d_model=256, d_inner=1024, use_macaron=True)
    ffn_out = ffn_module(x)
    print(f"FeedForwardModule output shape: {ffn_out.shape}")

    print("\n✓ All tests passed!")
