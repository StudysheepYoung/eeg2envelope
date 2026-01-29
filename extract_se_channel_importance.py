"""Extract SE-channel attention weights and back-project to EEG channel importance."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from test_model import load_checkpoint_and_create_model
from util.testing_utils import load_split_test_samples, segment_data
from util.biosemi_64_layout import BIOSEMI_64_CHANNELS, BRAIN_REGIONS


def parse_args():
    parser = argparse.ArgumentParser(description='Extract SE channel attention importance (64ch)')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to checkpoint (must be skip_cnn=True, use_se=True).')
    parser.add_argument('--split_data_dir', type=str,
                        default='/RAID5/projects/likeyang/happy/NeuroConformer/data/split_data',
                        help='Directory containing split_data files.')
    parser.add_argument('--output_dir', type=str, default='se_channel_analysis',
                        help='Base directory to store outputs.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to run inference on.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference.')
    parser.add_argument('--save_per_sample', action='store_true',
                        help='Save raw 256-d weights and 64-d importances for each recording.')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable visualization generation.')
    return parser.parse_args()


def prepare_output_dirs(base_dir: Path) -> Dict[str, Path]:
    dirs = {
        'root': base_dir,
        'raw_se': base_dir / 'raw_se_weights',
        'proj64': base_dir / 'eeg_importance_64d',
        'aggregated': base_dir / 'aggregated',
        'brain_region': base_dir / 'brain_region_analysis',
        'visuals': base_dir / 'visualizations',
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def compute_region_importance(weights: np.ndarray) -> Dict[str, float]:
    region_scores: Dict[str, float] = {}
    for region, channels in BRAIN_REGIONS.items():
        idx = [BIOSEMI_64_CHANNELS.index(ch) for ch in channels if ch in BIOSEMI_64_CHANNELS]
        if not idx:
            continue
        region_scores[region] = float(np.mean(weights[idx]))
    return region_scores


def save_statistics(stat_path: Path,
                    global_avg: np.ndarray,
                    region_scores: Dict[str, float],
                    per_subject_avg: Dict[int, np.ndarray]):
    top_indices = np.argsort(global_avg)[::-1]
    top_channels = [BIOSEMI_64_CHANNELS[i] for i in top_indices[:5]]
    top_values = [float(global_avg[i]) for i in top_indices[:5]]

    region_sorted = sorted(region_scores.items(), key=lambda kv: kv[1], reverse=True)

    stats = {
        'global': {
            'num_channels': len(BIOSEMI_64_CHANNELS),
            'top_5_channels': top_channels,
            'top_5_importance': top_values,
            'top_region': region_sorted[0][0] if region_sorted else None,
            'region_importance': region_scores,
        },
        'subjects': {
            str(sub_id + 1): {
                'avg_importance': per_subject_avg[sub_id].tolist()
            } for sub_id in sorted(per_subject_avg.keys())
        }
    }
    with stat_path.open('w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def visualize_outputs(visual_dir: Path,
                      global_avg: np.ndarray,
                      per_subject_avg: Dict[int, np.ndarray],
                      region_scores: Dict[str, float]):
    # Channel bar chart
    plt.figure(figsize=(18, 5))
    plt.bar(range(len(BIOSEMI_64_CHANNELS)), global_avg)
    plt.xticks(range(len(BIOSEMI_64_CHANNELS)), BIOSEMI_64_CHANNELS, rotation=90)
    plt.ylabel('Importance')
    plt.title('Global EEG Channel Importance (64-d)')
    plt.tight_layout()
    plt.savefig(visual_dir / 'channel_importance_bar.png', dpi=200)
    plt.close()

    # Top-10 channels horizontal bar (use colorful palette for readability)
    top_indices = np.argsort(global_avg)[::-1][:10]
    top_channels = [BIOSEMI_64_CHANNELS[i] for i in top_indices]
    top_values = [global_avg[i] for i in top_indices]
    cmap_top = plt.get_cmap('tab20', len(top_indices))
    colors_top = cmap_top(range(len(top_indices)))
    plt.figure(figsize=(8, 6))
    plt.barh(top_channels[::-1], top_values[::-1], color=colors_top[::-1])
    plt.xlabel('Importance')
    plt.title('Top-10 Channels')
    plt.tight_layout()
    plt.savefig(visual_dir / 'top10_channels.png', dpi=200)
    plt.close()

    # Region heatmap / bar chart with coolwarm gradient + legend
    regions = list(region_scores.keys())
    values = np.array([region_scores[r] for r in regions], dtype=float)
    if len(values) == 0:
        values = np.array([0.0], dtype=float)
        regions = ['N/A']
    vmax = float(values.max()) if len(values) else 1.0
    vmin = 0.0
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm
    colors = cmap(norm(values))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(regions, values, color=colors)
    ax.set_ylabel('Importance')
    ax.set_title('Brain Region Importance (BioSemi-64)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relative importance', rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(visual_dir / 'region_heatmap.png', dpi=200)
    plt.close(fig)

    # Per-subject variation (std)
    if per_subject_avg:
        stacked = np.stack([per_subject_avg[sub] for sub in sorted(per_subject_avg.keys())])
        std_vals = stacked.std(axis=0)
        plt.figure(figsize=(18, 5))
        plt.bar(range(len(BIOSEMI_64_CHANNELS)), std_vals)
        plt.xticks(range(len(BIOSEMI_64_CHANNELS)), BIOSEMI_64_CHANNELS, rotation=90)
        plt.ylabel('Std Dev')
        plt.title('Per-subject Channel Importance Variation')
        plt.tight_layout()
        plt.savefig(visual_dir / 'per_subject_variation.png', dpi=200)
        plt.close()

    # Channel ranking table
    ranking_path = visual_dir / 'channel_ranking_table.csv'
    with ranking_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Rank', 'Channel', 'Importance'])
        sorted_pairs = sorted(
            [(BIOSEMI_64_CHANNELS[i], float(global_avg[i])) for i in range(len(global_avg))],
            key=lambda kv: kv[1], reverse=True
        )
        for rank, (ch, val) in enumerate(sorted_pairs, 1):
            writer.writerow([rank, ch, val])

    # Optional topography using MNE
    try:
        import mne
        info = mne.create_info(ch_names=BIOSEMI_64_CHANNELS, sfreq=64, ch_types='eeg')
        montage = mne.channels.make_standard_montage('biosemi64')
        info.set_montage(montage, on_missing='ignore')

        values = np.asarray(global_avg, dtype=float)
        if values.size == 0:
            values = np.zeros(len(BIOSEMI_64_CHANNELS), dtype=float)
        else:
            # Normalize and apply gamma correction to deepen reds
            values = values - values.min()
            vmax = values.max()
            if vmax < 1e-8:
                vmax = 1.0
            values = values / vmax
            values = values ** 0.35  # stronger gamma correction to highlight highs

        fig, ax = plt.subplots(figsize=(5, 4))
        im, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False,
                                     cmap='coolwarm', contours=0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Relative importance (blue → red)', rotation=270, labelpad=14)
        fig.tight_layout()
        fig.savefig(visual_dir / 'brain_topography.png', dpi=200)
        plt.close(fig)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  警告：绘制脑地形图失败（{exc}），已跳过。")


def main():
    args = parse_args()
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint {checkpoint_path} 不存在')

    model, ckpt_args = load_checkpoint_and_create_model(str(checkpoint_path), device)

    if not ckpt_args.get('skip_cnn', True):
        raise ValueError('当前checkpoint是skip_cnn=False，无法映射回64通道EEG。')
    if not ckpt_args.get('use_se', True):
        raise ValueError('当前checkpoint未启用SE模块，无法提取通道注意力。')

    se_module = getattr(model, 'se', None)
    if se_module is None or not hasattr(se_module, 'fc'):
        raise RuntimeError('模型中没有可用的SE模块。')

    experiment_name = checkpoint_path.parent.name
    base_output = Path(args.output_dir) / experiment_name
    dirs = prepare_output_dirs(base_output)

    channel_names_path = dirs['aggregated'] / 'channel_names.json'
    with channel_names_path.open('w', encoding='utf-8') as f:
        json.dump(BIOSEMI_64_CHANNELS, f, ensure_ascii=False, indent=2)

    if not hasattr(model, 'input_proj'):
        raise RuntimeError('模型缺少 input_proj 层，无法进行64通道反投影。')

    weight_matrix = model.input_proj.weight.detach().cpu()  # [256,64]
    hook_buffer = {'value': None}

    def se_hook(_, __, output):
        hook_buffer['value'] = output.detach().cpu()

    handle = se_module.fc.register_forward_hook(se_hook)

    try:
        test_samples = load_split_test_samples(args.split_data_dir)
        batch_size = args.batch_size
        input_length = 640  # 10s windows @64Hz

        subject_record_counter = defaultdict(int)
        per_subject_avg: Dict[int, List[np.ndarray]] = defaultdict(list)
        global_recordings: List[np.ndarray] = []

        for sample_idx, (eeg_data, _, sub_id) in enumerate(test_samples):
            eeg_segments = segment_data(eeg_data, input_length)
            if not eeg_segments:
                continue

            eeg_array = np.array(eeg_segments, dtype=np.float32)
            num_segments = len(eeg_segments)
            num_batches = (num_segments + batch_size - 1) // batch_size

            se_collections: List[np.ndarray] = []
            importance_collections: List[np.ndarray] = []

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_segments)
                eeg_batch = torch.FloatTensor(eeg_array[start:end]).to(device)
                sub_batch = torch.full((end - start,), sub_id, dtype=torch.long, device=device)

                hook_buffer['value'] = None
                _ = model(eeg_batch, sub_batch)
                se_weights = hook_buffer['value']
                if se_weights is None:
                    raise RuntimeError('SE hook未捕获到权重。')
                hook_buffer['value'] = None

                se_np = se_weights.numpy()  # [B, d_model]
                se_collections.append(se_np)

                se_tensor = torch.from_numpy(se_np)
                eeg_importance = torch.matmul(se_tensor, weight_matrix)
                eeg_importance = torch.abs(eeg_importance)
                eeg_importance = eeg_importance / (eeg_importance.sum(dim=1, keepdim=True) + 1e-8)
                importance_collections.append(eeg_importance.numpy())

            if not se_collections:
                continue

            se_concat = np.concatenate(se_collections, axis=0)
            eeg_importances = np.concatenate(importance_collections, axis=0)

            record_avg = eeg_importances.mean(axis=0)
            per_subject_avg[sub_id].append(record_avg)
            global_recordings.append(record_avg)

            if args.save_per_sample:
                subject_record_counter[sub_id] += 1
                rec_idx = subject_record_counter[sub_id] - 1
                tag = f"sub_{sub_id + 1:03d}_rec_{rec_idx:03d}"
                np.save(dirs['raw_se'] / f'{tag}.npy', se_concat)
                np.save(dirs['proj64'] / f'{tag}.npy', eeg_importances)

            if sample_idx % 10 == 0:
                print(f"  处理样本 {sample_idx+1}/{len(test_samples)} - sub {sub_id+1}")

        if not global_recordings:
            raise RuntimeError('未生成任何通道重要性结果，请检查输入数据。')

        global_avg = np.mean(np.stack(global_recordings), axis=0)
        per_subject_avg_vec = {
            sub_id: np.mean(np.stack(vals), axis=0)
            for sub_id, vals in per_subject_avg.items()
        }

        np.save(dirs['aggregated'] / 'global_avg_64d.npy', global_avg)
        if per_subject_avg_vec:
            np.savez(dirs['aggregated'] / 'per_subject_avg.npz',
                     **{f'sub_{sub_id + 1:03d}': arr for sub_id, arr in per_subject_avg_vec.items()})

        region_scores = compute_region_importance(global_avg)
        with (dirs['brain_region'] / 'region_importance.json').open('w', encoding='utf-8') as f:
            json.dump(region_scores, f, indent=2, ensure_ascii=False)

        region_sorted = sorted(region_scores.items(), key=lambda kv: kv[1], reverse=True)
        with (dirs['brain_region'] / 'top_regions.json').open('w', encoding='utf-8') as f:
            json.dump(region_sorted, f, indent=2, ensure_ascii=False)

        save_statistics(dirs['aggregated'] / 'statistics.json', global_avg, region_scores, per_subject_avg_vec)

        if not args.no_visualize:
            visualize_outputs(dirs['visuals'], global_avg, per_subject_avg_vec, region_scores)

        print(f"✓ 结果已保存至 {base_output}")

    finally:
        handle.remove()


if __name__ == '__main__':
    main()
