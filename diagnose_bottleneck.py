"""
诊断脚本：分析模型性能瓶颈
检查是否存在特征坍塌、权重分布异常、以及任务天花板

用法:
    python diagnose_bottleneck.py --checkpoint <path_to_checkpoint>
    python diagnose_bottleneck.py  # 自动查找最新checkpoint
"""

import os
import glob
import torch
import torch.nn as nn
import numpy as np
import argparse
from models.FFT_block_conformer_v2 import Decoder
from util.dataset import RegressionDataset
from util.cal_pearson import pearson_correlation, pearson_loss

# ============ 配置 ============
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path')
parser.add_argument('--data_folder', type=str,
                    default="/RAID5/projects/likeyang/happy/HappyQuokka_system_for_EEG_Challenge/data/split_data")
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


def find_latest_checkpoint():
    """查找最新的checkpoint"""
    result_folders = glob.glob('test_results/conformer_v2_*')
    if not result_folders:
        return None

    latest_folder = max(result_folders, key=os.path.getmtime)
    checkpoints = glob.glob(os.path.join(latest_folder, '*.pt'))
    if not checkpoints:
        return None

    return max(checkpoints, key=os.path.getmtime)


def load_model_and_data(checkpoint_path):
    """加载模型和数据"""
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从checkpoint获取配置
    if 'args' in checkpoint:
        model_args = checkpoint['args']
    else:
        # 默认配置
        model_args = {
            'in_channel': 64, 'd_model': 256, 'd_inner': 1024,
            'n_head': 4, 'n_layers': 8, 'dropout': 0.3,
            'conv_kernel_size': 31, 'use_gated_residual': True,
            'use_mlp_head': True, 'gradient_scale': 2.0
        }

    # 创建模型
    model = Decoder(
        in_channel=model_args.get('in_channel', 64),
        d_model=model_args.get('d_model', 256),
        d_inner=model_args.get('d_inner', 1024),
        n_head=model_args.get('n_head', 4),
        n_layers=model_args.get('n_layers', 8),
        fft_conv1d_kernel=(9, 1),
        fft_conv1d_padding=(4, 0),
        dropout=model_args.get('dropout', 0.3),
        g_con=True,
        within_sub_num=71,
        conv_kernel_size=model_args.get('conv_kernel_size', 31),
        use_gated_residual=model_args.get('use_gated_residual', True),
        use_mlp_head=model_args.get('use_mlp_head', True),
        gradient_scale=model_args.get('gradient_scale', 2.0)
    ).to(device)

    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 加载测试数据
    features = ["eeg", "envelope"]
    test_files = [x for x in glob.glob(os.path.join(args.data_folder, "test_-_*"))
                 if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]

    test_dataset = RegressionDataset(test_files, 640, 64, 'test', g_con=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return model, test_loader, model_args


def analyze_feature_variance(model, val_loader, num_samples=10):
    """分析各层特征的方差，检测特征坍塌"""
    print("\n" + "="*60)
    print("1. 特征方差分析 (检测特征坍塌)")
    print("="*60)

    # 用于存储中间特征
    features = {
        'cnn_output': [],
        'conformer_input': [],
        'conformer_output': [],
        'gated_output': [],
        'final_output': []
    }

    # Hook 函数来捕获中间特征
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            features[name].append(output.detach().cpu())
        return hook

    # 修改模型以捕获中间输出
    model_to_analyze = model

    with torch.no_grad():
        for i, (eeg, envelope, sub_id) in enumerate(val_loader):
            if i >= num_samples:
                break

            eeg = eeg.squeeze(0).to(device)
            envelope = envelope.squeeze(0).to(device)
            sub_id = sub_id.to(device)

            # 手动前向传播以捕获中间特征
            # CNN部分
            x = eeg.transpose(1, 2)
            x = model.conv1(x)
            x = x.transpose(1, 2)
            x = model.norm1(x)
            x = model.act1(x)
            x = model.drop1(x)
            x = x.transpose(1, 2)

            x = model.conv2(x)
            x = x.transpose(1, 2)
            x = model.norm2(x)
            x = model.act2(x)
            x = model.drop2(x)
            x = x.transpose(1, 2)

            x = model.conv3(x)
            x = x.transpose(1, 2)
            x = model.norm3(x)
            x = model.act3(x)
            x = model.drop3(x)
            cnn_out = x.transpose(1, 2)
            features['cnn_output'].append(cnn_out.cpu())

            # SE + Subject Embedding
            dec_output = model.se(cnn_out)
            dec_output = dec_output.transpose(1, 2)

            if model.g_con:
                import torch.nn.functional as F
                sub_emb = F.one_hot(sub_id, model.within_sub_num)
                sub_emb = model.sub_proj(sub_emb.float())
                output = dec_output + sub_emb.unsqueeze(1)
            else:
                output = dec_output

            if model.pos_encoder is not None:
                output = model.pos_encoder(output)

            conformer_input = output.clone()
            features['conformer_input'].append(conformer_input.cpu())

            # Conformer layers
            for conformer_layer in model.layer_stack:
                output = conformer_layer(output)

            features['conformer_output'].append(output.cpu())

            # Gated residual
            if model.use_gated_residual:
                output = model.gated_residual(output, conformer_input)
            else:
                output = output + conformer_input

            features['gated_output'].append(output.cpu())

            # Final output
            if model.use_mlp_head:
                final = model.output_head(output)
            else:
                final = model.fc(output)

            features['final_output'].append(final.cpu())

    # 分析方差
    print("\n各层特征方差统计:")
    print("-" * 50)

    for name, feat_list in features.items():
        if feat_list:
            feat = torch.cat(feat_list, dim=0)

            # 计算各维度的方差
            var_per_dim = feat.var(dim=(0, 1))  # [d_model] 或 [1]
            var_per_sample = feat.var(dim=-1).mean()  # 每个样本的方差均值

            mean_var = var_per_dim.mean().item()
            min_var = var_per_dim.min().item()
            max_var = var_per_dim.max().item()

            # 检查是否坍塌
            collapse_warning = ""
            if mean_var < 0.01:
                collapse_warning = " ⚠️ 可能特征坍塌!"
            elif mean_var < 0.1:
                collapse_warning = " ⚠️ 方差偏低"

            print(f"{name:20s}: mean_var={mean_var:.6f}, min={min_var:.6f}, max={max_var:.6f}{collapse_warning}")

    return features


def analyze_weights(model):
    """分析模型权重分布"""
    print("\n" + "="*60)
    print("2. 权重分析")
    print("="*60)

    print("\n各层权重 norm:")
    print("-" * 50)

    layer_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            norm = param.data.norm().item()
            layer_norms[name] = norm

    # 按模块分组
    groups = {
        'CNN (conv1-3)': [],
        'SE Block': [],
        'Conformer Front (0-3)': [],
        'Conformer Back (4-7)': [],
        'Gated Residual': [],
        'Output Head': []
    }

    for name, norm in layer_norms.items():
        if 'conv1' in name or 'conv2' in name or 'conv3' in name or 'norm1' in name or 'norm2' in name or 'norm3' in name:
            groups['CNN (conv1-3)'].append((name, norm))
        elif 'se.' in name:
            groups['SE Block'].append((name, norm))
        elif 'layer_stack.0' in name or 'layer_stack.1' in name or 'layer_stack.2' in name or 'layer_stack.3' in name:
            groups['Conformer Front (0-3)'].append((name, norm))
        elif 'layer_stack.4' in name or 'layer_stack.5' in name or 'layer_stack.6' in name or 'layer_stack.7' in name:
            groups['Conformer Back (4-7)'].append((name, norm))
        elif 'gated_residual' in name:
            groups['Gated Residual'].append((name, norm))
        elif 'output_head' in name or 'fc.' in name:
            groups['Output Head'].append((name, norm))

    for group_name, params in groups.items():
        if params:
            norms = [p[1] for p in params]
            avg_norm = np.mean(norms)
            print(f"\n{group_name}:")
            print(f"  平均 norm: {avg_norm:.6f}")
            print(f"  范围: [{min(norms):.6f}, {max(norms):.6f}]")

    # 特别检查 Output Head
    print("\n" + "-" * 50)
    print("Output Head 详细分析:")
    if hasattr(model, 'output_head'):
        # 最后一层 Linear
        last_linear = model.output_head.net[-1]
        weight_norm = last_linear.weight.data.norm().item()
        bias_val = last_linear.bias.data.item() if last_linear.bias is not None else 0

        print(f"  最后 Linear 层:")
        print(f"    weight norm: {weight_norm:.6f}")
        print(f"    bias: {bias_val:.6f}")

        if abs(bias_val) > weight_norm * 10:
            print(f"  ⚠️ bias 远大于 weight，输出可能被 bias 主导!")
    elif hasattr(model, 'fc'):
        weight_norm = model.fc.weight.data.norm().item()
        bias_val = model.fc.bias.data.item() if model.fc.bias is not None else 0
        print(f"  FC层 weight norm: {weight_norm:.6f}")
        print(f"  FC层 bias: {bias_val:.6f}")


def analyze_predictions(model, val_loader, num_samples=None):
    """分析模型预测分布 (使用全部数据，与训练脚本一致)"""
    print("\n" + "="*60)
    print("3. 预测分布分析")
    print("="*60)

    all_preds = []
    all_labels = []
    all_pearsons = []

    with torch.no_grad():
        for i, (eeg, envelope, sub_id) in enumerate(val_loader):
            if num_samples is not None and i >= num_samples:
                break

            eeg = eeg.squeeze(0).to(device)
            envelope = envelope.squeeze(0).to(device)
            sub_id = sub_id.to(device)

            pred = model(eeg, sub_id)

            all_preds.append(pred.cpu())
            all_labels.append(envelope.cpu())

            # 计算 Pearson (与训练脚本一致: pearson_metric(outputs, labels))
            # pearson_metric 内部调用 pearson_correlation(y_true, y_pred)
            from util.cal_pearson import pearson_metric
            pearson = pearson_metric(pred, envelope, axis=1).mean().item()
            all_pearsons.append(pearson)

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    print(f"\n预测值统计:")
    print(f"  均值: {preds.mean().item():.6f}")
    print(f"  标准差: {preds.std().item():.6f}")
    print(f"  最小值: {preds.min().item():.6f}")
    print(f"  最大值: {preds.max().item():.6f}")

    print(f"\n真实标签统计:")
    print(f"  均值: {labels.mean().item():.6f}")
    print(f"  标准差: {labels.std().item():.6f}")
    print(f"  最小值: {labels.min().item():.6f}")
    print(f"  最大值: {labels.max().item():.6f}")

    print(f"\nPearson 相关系数:")
    print(f"  均值: {np.mean(all_pearsons):.6f}")
    print(f"  标准差: {np.std(all_pearsons):.6f}")

    # 检查预测是否接近常数
    pred_variance = preds.var().item()
    label_variance = labels.var().item()
    variance_ratio = pred_variance / label_variance if label_variance > 0 else 0

    print(f"\n方差比 (pred/label): {variance_ratio:.4f}")
    if variance_ratio < 0.5:
        print("  ⚠️ 预测方差远小于标签方差，模型可能输出近乎常数!")
    elif variance_ratio > 2.0:
        print("  ⚠️ 预测方差远大于标签方差，模型可能过度波动!")
    else:
        print("  ✓ 方差比例合理")

    return preds, labels, all_pearsons


def test_linear_baseline(val_loader, num_samples=None):
    """测试简单线性模型作为基线"""
    print("\n" + "="*60)
    print("4. 线性基线测试")
    print("="*60)

    # 收集数据
    all_eeg = []
    all_envelope = []

    for i, (eeg, envelope, sub_id) in enumerate(val_loader):
        if num_samples is not None and i >= num_samples:
            break

        eeg = eeg.squeeze(0)  # [N, T, 64] or [T, 64]
        envelope = envelope.squeeze(0)  # [N, T, 1] or [T, 1]

        # 展平为 2D
        if eeg.dim() == 3:
            eeg = eeg.reshape(-1, eeg.shape[-1])  # [N*T, 64]
            envelope = envelope.reshape(-1, envelope.shape[-1])  # [N*T, 1]

        all_eeg.append(eeg)
        all_envelope.append(envelope)

    eeg_data = torch.cat(all_eeg, dim=0)  # [total, 64]
    envelope_data = torch.cat(all_envelope, dim=0)  # [total, 1]

    # 简单线性回归: EEG[t] -> Envelope[t]
    # 使用最小二乘解
    X = eeg_data.numpy()
    y = envelope_data.numpy().flatten()

    # 添加 bias 项
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])

    # 最小二乘解
    w, residuals, rank, s = np.linalg.lstsq(X_bias, y, rcond=None)

    # 预测
    y_pred = X_bias @ w

    # 计算 Pearson
    y_mean = y.mean()
    y_pred_mean = y_pred.mean()

    numerator = np.sum((y - y_mean) * (y_pred - y_pred_mean))
    denominator = np.sqrt(np.sum((y - y_mean)**2) * np.sum((y_pred - y_pred_mean)**2))

    linear_pearson = numerator / (denominator + 1e-6)

    print(f"\n简单线性回归 (EEG -> Envelope):")
    print(f"  Pearson 相关系数: {linear_pearson:.6f}")

    if linear_pearson > 0.20:
        print(f"  ⚠️ 线性模型已达到 {linear_pearson:.3f}，复杂模型提升空间有限!")
    else:
        print(f"  ✓ 线性基线较低，复杂模型应有提升空间")

    return linear_pearson


def analyze_gate_values(model, val_loader, num_samples=10):
    """分析门控值"""
    print("\n" + "="*60)
    print("5. 门控值分析")
    print("="*60)

    if not hasattr(model, 'gated_residual') or not model.use_gated_residual:
        print("模型未使用门控残差，跳过此分析")
        return

    gate_values = []

    with torch.no_grad():
        for i, (eeg, envelope, sub_id) in enumerate(val_loader):
            if i >= num_samples:
                break

            eeg = eeg.squeeze(0).to(device)
            sub_id = sub_id.to(device)

            # 手动前向传播到门控层
            import torch.nn.functional as F

            x = eeg.transpose(1, 2)
            x = model.conv1(x)
            x = x.transpose(1, 2)
            x = model.norm1(x)
            x = model.act1(x)
            x = model.drop1(x)
            x = x.transpose(1, 2)

            x = model.conv2(x)
            x = x.transpose(1, 2)
            x = model.norm2(x)
            x = model.act2(x)
            x = model.drop2(x)
            x = x.transpose(1, 2)

            x = model.conv3(x)
            x = x.transpose(1, 2)
            x = model.norm3(x)
            x = model.act3(x)
            x = model.drop3(x)
            cnn_out = x.transpose(1, 2)

            dec_output = model.se(cnn_out)
            dec_output = dec_output.transpose(1, 2)

            if model.g_con:
                sub_emb = F.one_hot(sub_id, model.within_sub_num)
                sub_emb = model.sub_proj(sub_emb.float())
                output = dec_output + sub_emb.unsqueeze(1)
            else:
                output = dec_output

            if model.pos_encoder is not None:
                output = model.pos_encoder(output)

            conformer_input = output.clone()

            for conformer_layer in model.layer_stack:
                output = conformer_layer(output)

            # 计算门控值
            gated_residual = model.gated_residual
            residual_normed = gated_residual.norm_residual(conformer_input)
            global_feat = output.mean(dim=1)

            gate = gated_residual.gate_layer1(global_feat)
            gate = gated_residual.activation(gate)
            gate = gated_residual.gate_layer2(gate)
            gate = torch.sigmoid(gate)

            gate_values.append(gate.cpu())

    gates = torch.cat(gate_values, dim=0)

    print(f"\n门控值统计 (gate ∈ [0,1], 1=信任Conformer, 0=信任残差):")
    print(f"  均值: {gates.mean().item():.6f}")
    print(f"  标准差: {gates.std().item():.6f}")
    print(f"  最小值: {gates.min().item():.6f}")
    print(f"  最大值: {gates.max().item():.6f}")

    # 检查是否门控值接近常数
    if gates.std().item() < 0.01:
        print(f"  ⚠️ 门控值几乎是常数 ({gates.mean().item():.3f})，门控机制可能未起作用!")

    if gates.mean().item() < 0.3:
        print(f"  ⚠️ 门控值偏低，模型更信任残差连接而非Conformer输出!")
    elif gates.mean().item() > 0.7:
        print(f"  ✓ 门控值偏高，模型信任Conformer输出")


def main():
    # 查找或使用指定的 checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("错误: 未找到 checkpoint，请使用 --checkpoint 指定路径")
            return

    # 加载模型和数据
    model, val_loader, model_args = load_model_and_data(checkpoint_path)

    print("\n" + "="*60)
    print("模型性能瓶颈诊断")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"模型配置: d_model={model_args.get('d_model', 256)}, "
          f"n_layers={model_args.get('n_layers', 8)}, "
          f"n_head={model_args.get('n_head', 4)}")

    # 运行诊断
    analyze_feature_variance(model, val_loader)
    analyze_weights(model)
    analyze_predictions(model, val_loader, num_samples=None)  # 使用全部数据
    test_linear_baseline(val_loader, num_samples=None)  # 使用全部数据
    analyze_gate_values(model, val_loader)

    print("\n" + "="*60)
    print("诊断完成")
    print("="*60)


if __name__ == '__main__':
    main()
