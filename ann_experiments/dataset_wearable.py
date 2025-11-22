"""
dataset_wearable.py

PyTorch Dataset for the processed wearable stress dataset.

Expects NPZ at:
    data/processed/stress_windows_full.npz

The NPZ must contain:
    - X: (N, C, T)
    - y: (N,)
    - subjects: (N,) array of subject IDs (strings)

Channel layout (C dimension):
    0: EDA
    1: ACC x
    2: ACC y
    3: ACC z
    4: BVP
    5: TEMP
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WearableStressDataset(Dataset):
    """
    Window-level stress dataset with optional subject and channel filtering.

    Parameters
    ----------
    npz_path : str
        Path to stress_windows_full.npz.
    subjects : list or None
        If provided, only keep windows whose subject ID is in this list.
    channels : list or None
        If provided, select a subset of channels by index, e.g. [0], [0,1,2,3].
    """

    def __init__(self, npz_path: str, subjects=None, channels=None):
        data = np.load(npz_path, allow_pickle=True)

        X = data["X"]           # (N, C, T)
        y = data["y"]           # (N,)
        subs = data["subjects"] # (N,)
        subs = subs.astype(str)

        if subjects is not None:
            subjects = [str(s) for s in subjects]
            mask = np.isin(subs, subjects)
            X = X[mask]
            y = y[mask]
            subs = subs[mask]

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.subjects = subs

        # Store selected channels (e.g. [0], [1,2,3], [0,1,2,3], etc.)
        self.channels = channels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]  # (C, T)
        if self.channels is not None:
            X = X[self.channels, :]  # select subset of channels
        y = self.y[idx]
        return X, y
