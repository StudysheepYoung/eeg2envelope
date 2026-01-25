"""Shared color helpers for plotting different models consistently."""

import hashlib

# Specific colors for known models (upper-case keys)
MODEL_COLOR_OVERRIDES = {
    'NEUROCONFORMER': '#E74C3C',
    'CONFORMER': '#E74C3C',
    'ADT': '#1F77B4',
    'ADT NETWORK': '#1F77B4',
    'HAPPYOUOKKA': '#FF7F0E',  # typo-safe key
    'HAPPYQUOKKA': '#FF7F0E',
    'EEGNET': '#2CA02C',
    'VLAAI': '#17BECF',
    'LINEAR': '#9467BD',
    'FCNN': '#8C564B'
}

# Display name overrides, shared across plots
MODEL_DISPLAY_OVERRIDES = {
    'NEUROCONFORMER': 'NeuroConformer',
    'CONFORMER': 'NeuroConformer',
    'ADT': 'ADT Network',
    'ADT NETWORK': 'ADT Network',
    'EEGNET': 'EEGNet'
}

# Palette used as deterministic fallback (matplotlib tab10)
COLOR_PALETTE = [
    '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
    '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'
]

DEFAULT_COLOR = '#7F7F7F'


def _normalize_name(model_name):
    """Normalize a model name for color lookup."""
    if model_name is None:
        return ''
    return str(model_name).strip().upper()


def get_display_name(model_name):
    """Return a user-friendly display name for a model."""
    if model_name is None:
        return 'Unknown'
    key = _normalize_name(model_name)
    return MODEL_DISPLAY_OVERRIDES.get(key, str(model_name))


def get_model_color(model_name, source=None):
    """Return a consistent color for a given model across plots."""
    key = _normalize_name(model_name)

    # Source priority for NeuroConformer runs
    if source and source.lower() == 'neuroconformer':
        return MODEL_COLOR_OVERRIDES['NEUROCONFORMER']

    if key in MODEL_COLOR_OVERRIDES:
        return MODEL_COLOR_OVERRIDES[key]

    if not key:
        return DEFAULT_COLOR

    # Deterministic fallback based on hash
    idx = int(hashlib.md5(key.encode('utf-8')).hexdigest(), 16)
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)]
