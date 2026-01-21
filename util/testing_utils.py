"""Utility helpers for loading split test data and segmenting windows."""
from __future__ import annotations

import glob
import os
from typing import List, Sequence, Tuple

import numpy as np


def load_split_test_samples(split_data_dir: str,
                             sample_rate: int = 64,
                             win_len: int = 10) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Load test samples from split_data directory only.

    Returns a list of tuples ``(eeg, envelope, sub_id)`` where ``sub_id`` is 0-based.
    """
    print("\n[1/1] 从split_data加载test文件...")
    input_length = sample_rate * win_len
    test_samples: List[Tuple[np.ndarray, np.ndarray, int]] = []

    test_files = sorted(glob.glob(os.path.join(split_data_dir, 'test_-_*')))
    if not test_files:
        print(f"  警告：未在 {split_data_dir} 找到test文件")
        return test_samples

    from itertools import groupby

    test_files_sorted = sorted(
        test_files,
        key=lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3])
    )
    test_files_grouped: List[Sequence[str]] = []

    for _, feature_paths in groupby(
        test_files_sorted,
        lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3])
    ):
        feature_list = sorted(list(feature_paths),
                              key=lambda x: "0" if "eeg" in x else x)
        test_files_grouped.append(feature_list)

    print(f"  找到 {len(test_files_grouped)} 组test文件")

    for file_group in test_files_grouped:
        eeg_file = [f for f in file_group if 'eeg.npy' in f][0]
        envelope_file = [f for f in file_group if 'envelope.npy' in f][0]

        sub_id_str = os.path.basename(eeg_file).split('_-_')[1]
        sub_id = int(sub_id_str.split('-')[1]) - 1

        eeg_data = np.load(eeg_file)
        envelope_data = np.load(envelope_file)

        # Trim to full windows to be consistent downstream
        nsegment = eeg_data.shape[0] // input_length
        if nsegment == 0:
            continue
        length = nsegment * input_length
        eeg_data = eeg_data[:length]
        envelope_data = envelope_data[:length]

        test_samples.append((eeg_data, envelope_data, sub_id))

    print(f"  ✓ 加载完成，共 {len(test_samples)} 个样本")
    return test_samples


def segment_data(data: np.ndarray, input_length: int) -> List[np.ndarray]:
    """Segment the input array into windows of ``input_length``."""
    nsegment = data.shape[0] // input_length
    if nsegment == 0:
        return []
    data = data[:int(nsegment * input_length)]
    return [data[i:i + input_length] for i in range(0, data.shape[0], input_length)]
