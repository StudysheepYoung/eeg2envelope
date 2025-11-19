#!/usr/bin/env python3
"""
模型架构可视化脚本
生成 HappyQuokka (train_v10_sota) 的详细架构图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_architecture_diagram():
    """创建模型架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 24))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    ax.axis('off')

    # 颜色方案
    colors = {
        'input': '#E8F4F8',      # 浅蓝 - 输入输出
        'cnn': '#FFE5B4',        # 浅橙 - CNN
        'attention': '#FFE4E1',  # 浅粉 - 注意力
        'embedding': '#E0FFE0',  # 浅绿 - 嵌入
        'transformer': '#E6E6FA', # 浅紫 - Transformer
        'output': '#FAFAD2',     # 浅黄 - 输出
    }

    y_pos = 28  # 起始Y坐标
    box_width = 8
    box_height = 0.8
    x_center = 5

    def add_box(y, text, color, height=box_height, width=box_width, fontsize=11, bold=False):
        """添加方框"""
        box = FancyBboxPatch(
            (x_center - width/2, y), width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2 if bold else 1
        )
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x_center, y + height/2, text,
                ha='center', va='center', fontsize=fontsize, weight=weight)
        return y - height - 0.3

    def add_arrow(y_from, y_to, text='', offset=0):
        """添加箭头"""
        arrow = FancyArrowPatch(
            (x_center + offset, y_from), (x_center + offset, y_to),
            arrowstyle='->', mutation_scale=20, linewidth=2,
            color='black'
        )
        ax.add_patch(arrow)
        if text:
            ax.text(x_center + offset + 0.5, (y_from + y_to)/2, text,
                   fontsize=9, style='italic', color='#555')

    def add_subbox(y, texts, color, indent=1):
        """添加子模块"""
        sub_width = box_width - indent
        sub_height = 0.6
        current_y = y
        for text in texts:
            box = FancyBboxPatch(
                (x_center - sub_width/2 + indent/2, current_y),
                sub_width, sub_height,
                boxstyle="round,pad=0.05",
                edgecolor='gray',
                facecolor=color,
                linewidth=0.8,
                linestyle='--'
            )
            ax.add_patch(box)
            ax.text(x_center + indent/2, current_y + sub_height/2, text,
                   ha='center', va='center', fontsize=9)
            current_y -= sub_height + 0.15
        return current_y + 0.15

    # 标题
    ax.text(x_center, 29.5, 'HappyQuokka Model Architecture',
           ha='center', fontsize=18, weight='bold')
    ax.text(x_center, 29, 'EEG-to-Speech Envelope Reconstruction (train_v10_sota)',
           ha='center', fontsize=12, style='italic', color='#555')

    # ========== 输入层 ==========
    y_pos = add_box(y_pos, 'Input: EEG Signal\n[Batch, 64, 640]', colors['input'], bold=True)
    add_arrow(y_pos + 0.3, y_pos - 0.2)

    # ========== 1. CNN特征提取器 ==========
    y_pos = add_box(y_pos - 0.3, '① CNN Feature Extractor', colors['cnn'],
                   height=1, fontsize=12, bold=True)

    # Conv1
    y_pos = add_subbox(y_pos - 0.2, [
        'Conv1D(64→256, kernel=7, padding=3)',
        '→ LayerNorm(256)',
        '→ LeakyReLU(0.01)',
        '→ Dropout(0.3)'
    ], colors['cnn'])

    # Conv2
    y_pos = add_subbox(y_pos - 0.1, [
        'Conv1D(256→256, kernel=5, padding=2)',
        '→ LayerNorm(256)',
        '→ LeakyReLU(0.01)',
        '→ Dropout(0.3)'
    ], colors['cnn'])

    # Conv3
    y_pos = add_subbox(y_pos - 0.1, [
        'Conv1D(256→256, kernel=3, padding=1)',
        '→ LayerNorm(256)',
        '→ LeakyReLU(0.01)',
        '→ Dropout(0.3)'
    ], colors['cnn'])

    ax.text(9.5, y_pos + 2, 'Output:\n[B,256,640]', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    add_arrow(y_pos + 0.2, y_pos - 0.3)

    # ========== 2. SE通道注意力 ==========
    y_pos = add_box(y_pos - 0.4, '② SE Channel Attention', colors['attention'],
                   height=1, fontsize=12, bold=True)

    y_pos = add_subbox(y_pos - 0.2, [
        'AdaptiveAvgPool1d(1) → [B, 256, 1]',
        'FC(256→16) → LeakyReLU',
        'FC(16→256) → Sigmoid',
        'X * channel_weights'
    ], colors['attention'])

    ax.text(9.5, y_pos + 1.5, 'SE Block\nReduction\n=16', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='pink', alpha=0.3))

    add_arrow(y_pos + 0.2, y_pos - 0.3)
    ax.text(x_center, y_pos - 0.15, '[B, 640, 256]', ha='center',
           fontsize=9, style='italic', color='blue')

    # ========== 3. 全局条件器 ==========
    y_pos = add_box(y_pos - 0.5, '③ Global Conditioner (Subject Embedding)',
                   colors['embedding'], height=1, fontsize=12, bold=True)

    y_pos = add_subbox(y_pos - 0.2, [
        'One-Hot(sub_id) → [B, 71]',
        'Linear(71→256) → sub_emb',
        'X = X + sub_emb.unsqueeze(1)'
    ], colors['embedding'])

    ax.text(9.5, y_pos + 1, '71个\n受试者', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    add_arrow(y_pos + 0.2, y_pos - 0.3)

    # ========== 4. 位置编码 ==========
    y_pos = add_box(y_pos - 0.4, '④ Positional Encoding', colors['embedding'],
                   height=0.9, fontsize=12, bold=True)

    y_pos = add_subbox(y_pos - 0.2, [
        'Sinusoidal PE(pos, dim)',
        'X = X + PE[:seq_len]'
    ], colors['embedding'])

    ax.text(9.5, y_pos + 0.8, 'Fixed\nEncoding', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    add_arrow(y_pos + 0.2, y_pos - 0.3)

    # ========== 5. Transformer编码器 ==========
    y_pos = add_box(y_pos - 0.4, '⑤ Transformer Encoder (8 Layers)',
                   colors['transformer'], height=1.2, fontsize=12, bold=True)

    # PreLNFFTBlock
    y_pos_block = y_pos - 0.3

    # MultiHeadAttention
    y_pos_block = add_box(y_pos_block, 'MultiHeadAttention (4 heads)',
                         colors['transformer'], height=0.7, width=7, fontsize=10)
    y_pos_block = add_subbox(y_pos_block - 0.1, [
        'Pre-LayerNorm(X)',
        'Q,K,V = Linear(256→256)',
        'Attention = softmax(QK^T/√d_k)V',
        'Output = Dropout + Residual'
    ], colors['transformer'], indent=1.5)

    # PositionwiseFeedForward
    y_pos_block = add_box(y_pos_block - 0.2, 'PositionwiseFeedForward',
                         colors['transformer'], height=0.7, width=7, fontsize=10)
    y_pos_block = add_subbox(y_pos_block - 0.1, [
        'Pre-LayerNorm(X)',
        'Conv1D(256→1024, k=9) → LeakyReLU',
        'Conv1D(1024→256, k=1)',
        'Dropout + Residual'
    ], colors['transformer'], indent=1.5)

    # 表示8层
    ax.text(x_center, y_pos_block - 0.3, '× 8 layers',
           ha='center', fontsize=11, weight='bold', color='purple',
           bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

    ax.text(9.5, y_pos - 3, 'Pre-LN\nArchitecture', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))

    y_pos = y_pos_block - 0.5
    add_arrow(y_pos + 0.2, y_pos - 0.3)

    # ========== 6. 输出层 ==========
    y_pos = add_box(y_pos - 0.4, '⑥ Output Layer', colors['output'],
                   height=0.9, fontsize=12, bold=True)

    y_pos = add_subbox(y_pos - 0.2, [
        'Linear(256 → 1)'
    ], colors['output'])

    add_arrow(y_pos + 0.2, y_pos - 0.3)

    # ========== 输出 ==========
    y_pos = add_box(y_pos - 0.4, 'Output: Speech Envelope\n[Batch, 640, 1]',
                   colors['input'], bold=True)

    # ========== 添加参数信息面板 ==========
    info_y = 3
    info_box = FancyBboxPatch(
        (0.2, info_y), 3.5, 3.5,
        boxstyle="round,pad=0.2",
        edgecolor='darkblue',
        facecolor='#F0F8FF',
        linewidth=2
    )
    ax.add_patch(info_box)

    info_text = """Model Configuration
━━━━━━━━━━━━━━━━━━━━━
• d_model = 256
• d_inner = 1024
• n_head = 4
• n_layers = 8
• dropout = 0.3
• in_channel = 64
• win_len = 10s @ 64Hz

Total Params: ~24M
"""
    ax.text(0.4, info_y + 3.3, info_text, fontsize=9, verticalalignment='top',
           family='monospace')

    # ========== 添加损失函数面板 ==========
    loss_box = FancyBboxPatch(
        (6.3, info_y), 3.5, 3.5,
        boxstyle="round,pad=0.2",
        edgecolor='darkred',
        facecolor='#FFF5EE',
        linewidth=2
    )
    ax.add_patch(loss_box)

    loss_text = """Loss Function
━━━━━━━━━━━━━━━━━━━━━
Loss = MSE + λ(Pearson)²

MSE Loss:
  mean((y_true - y_pred)²)

Pearson Loss:
  1 - corr(y_true, y_pred)

λ = 1.0 (default)
"""
    ax.text(6.5, info_y + 3.3, loss_text, fontsize=9, verticalalignment='top',
           family='monospace')

    # ========== 添加图例 ==========
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], edgecolor='black', label='Input/Output'),
        mpatches.Patch(facecolor=colors['cnn'], edgecolor='black', label='CNN Layers'),
        mpatches.Patch(facecolor=colors['attention'], edgecolor='black', label='Attention'),
        mpatches.Patch(facecolor=colors['embedding'], edgecolor='black', label='Embedding/PE'),
        mpatches.Patch(facecolor=colors['transformer'], edgecolor='black', label='Transformer'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='Output Layer'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
             framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    return fig

def create_data_flow_diagram():
    """创建数据流图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 标题
    ax.text(5, 11.5, 'Data Flow & Dimensions',
           ha='center', fontsize=16, weight='bold')

    y = 10.5
    stages = [
        ('Input EEG', '[B, 64, 640]', '#E8F4F8'),
        ('CNN Conv1', '[B, 256, 640]', '#FFE5B4'),
        ('CNN Conv2', '[B, 256, 640]', '#FFE5B4'),
        ('CNN Conv3', '[B, 256, 640]', '#FFE5B4'),
        ('SE Attention', '[B, 256, 640]', '#FFE4E1'),
        ('Transpose', '[B, 640, 256]', '#E0FFE0'),
        ('+ Sub Embedding', '[B, 640, 256]', '#E0FFE0'),
        ('+ Pos Encoding', '[B, 640, 256]', '#E0FFE0'),
        ('Transformer ×8', '[B, 640, 256]', '#E6E6FA'),
        ('Linear Output', '[B, 640, 1]', '#FAFAD2'),
        ('Output Envelope', '[B, 640, 1]', '#E8F4F8'),
    ]

    for i, (name, shape, color) in enumerate(stages):
        # 主框
        box = FancyBboxPatch((1.5, y), 4, 0.7, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(2, y + 0.35, name, ha='left', va='center', fontsize=11, weight='bold')

        # 形状标注
        shape_box = FancyBboxPatch((6, y + 0.1), 2.5, 0.5, boxstyle="round,pad=0.05",
                                  edgecolor='blue', facecolor='white', linewidth=1)
        ax.add_patch(shape_box)
        ax.text(7.25, y + 0.35, shape, ha='center', va='center',
               fontsize=10, family='monospace', color='blue')

        # 箭头
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((3.5, y), (3.5, y - 0.6),
                                  arrowstyle='->', mutation_scale=15,
                                  linewidth=2, color='black')
            ax.add_patch(arrow)

        y -= 1

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # 生成架构图
    print("生成模型架构图...")
    fig1 = create_architecture_diagram()
    fig1.savefig('model_architecture_detailed.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: model_architecture_detailed.png")

    # 生成数据流图
    print("生成数据流图...")
    fig2 = create_data_flow_diagram()
    fig2.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: data_flow_diagram.png")

    plt.show()
    print("\n完成！")
