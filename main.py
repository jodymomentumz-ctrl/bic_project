"""
EEG PCA + CNN/TinyCNN/SpikingNN Training Pipeline (PyTorch)
===========================================================

Works with CSV files like `Data_S02_Sess01.csv` (columns: Time, EEG channels..., EOG, FeedBackEvent)
Optionally uses `TrainLabels.csv` if you have per-file labels.

Usage (examples):
-----------------
# Train TinyCNN with PCA to 16 components
python eeg_train.py \
  --data_dir /path/to/csvs \
  --labels_csv /path/to/TrainLabels.csv \
  --model tinycnn \
  --pca_components 16 \
  --epochs 30 --batch_size 64 --window_sec 2.0 --step_sec 0.5

# Train deeper CNN without PCA
python eeg_train.py --data_dir /path/to/csvs --model cnn --pca_components 0

# Train spiking CNN (surrogate gradient) with 10 simulation steps
python eeg_train.py --data_dir /path/to/csvs --model spiking --snn_steps 10

Notes:
------
- Sampling rate is inferred from the Time column (assumed steady).
- Labels may come from `TrainLabels.csv` or from in-file event markers (FeedBackEvent) if `--derive_labels_from_events` is set.
- Bandpass (1–40 Hz) and 50 Hz notch filters are available with `--bandpass` and `--notch`.
- PCA reduces the **channel** dimension; components are treated as channels.
- This script aims to be a strong starting baseline; tweak architecture & preprocessing for best results.
"""

import os
import glob
import math
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from scipy.signal import butter, filtfilt, iirnotch
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ----------------------------
# Utility: Signal Processing
# ----------------------------

def design_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a


def apply_bandpass(x: np.ndarray, fs: float, low=1.0, high=40.0) -> np.ndarray:
    """x: (T, C) array."""
    if not SCIPY_OK:
        return x
    b, a = design_bandpass(low, high, fs)
    return filtfilt(b, a, x, axis=0)


def apply_notch(x: np.ndarray, fs: float, freq=50.0, Q=30.0) -> np.ndarray:
    if not SCIPY_OK:
        return x
    b, a = iirnotch(w0=freq/(fs/2), Q=Q)
    return filtfilt(b, a, x, axis=0)

# ----------------------------
# Dataset & Preprocessing
# ----------------------------

EEG_COL_BLACKLIST = {"Time", "EOG", "FeedBackEvent"}

def infer_fs(time_col: np.ndarray) -> float:
    if len(time_col) < 3:
        return 200.0
    dt = np.diff(time_col)  # seconds
    dt_med = np.median(dt)
    if dt_med <= 0:
        return 200.0
    return float(round(1.0/dt_med))


def load_eeg_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in EEG_COL_BLACKLIST]
    # Keep a stable channel order
    channels = [c for c in df.columns if c not in {"Time", "EOG", "FeedBackEvent"}]
    time = df["Time"].to_numpy(dtype=float)
    fs = infer_fs(time)  # used downstream
    x = df[channels].to_numpy(dtype=np.float32)
    events = df["FeedBackEvent"].to_numpy(dtype=float) if "FeedBackEvent" in df.columns else np.zeros(len(df))
    return x, events, channels


def derive_labels_from_events(events: np.ndarray, event_threshold: float = 0.5) -> List[int]:
    """Simple derivation: any nonzero (or above threshold) event marks a positive window.
    This is a placeholder; customize to your competition's event schema.
    """
    return (events > event_threshold).astype(int).tolist()


@dataclass
class WindowingConfig:
    window_sec: float = 2.0
    step_sec: float = 0.5


class EEGWindowDataset(Dataset):
    def __init__(self,
                 file_label_items: List[Tuple[str, int]],
                 pca: Optional[PCA] = None,
                 scaler: Optional[StandardScaler] = None,
                 bandpass: bool = False,
                 notch: bool = False,
                 low_hz: float = 1.0,
                 high_hz: float = 40.0,
                 notch_hz: float = 50.0,
                 window_cfg: WindowingConfig = WindowingConfig(),
                 min_max_norm: bool = False,
                 verbose: bool = False):
        """
        file_label_items: list of (filepath, label) tuples. If label == -1, labels will be derived from events per-window.
        PCA reduces channels to components; scaler standardizes across channels for each window.
        """
        self.items: List[Tuple[np.ndarray, int]] = []
        self.verbose = verbose
        self.scaler = scaler
        self.pca = pca
        self.min_max_norm = min_max_norm
        for path, file_label in file_label_items:
            x, events, channels = load_eeg_csv(path)
            # Infer fs from Time column once more
            try:
                time = pd.read_csv(path, usecols=["Time"]).to_numpy().squeeze()
                fs = infer_fs(time)
            except Exception:
                fs = 200.0
            if bandpass:
                x = apply_bandpass(x, fs, low=low_hz, high=high_hz)
            if notch:
                x = apply_notch(x, fs, freq=notch_hz)

            win_len = int(round(window_cfg.window_sec * fs))
            step_len = int(round(window_cfg.step_sec * fs))
            if win_len <= 0:
                raise ValueError("window_sec too small")
            # Sliding windows over time
            for start in range(0, max(1, len(x) - win_len + 1), step_len):
                w = x[start:start+win_len]  # (T, C)
                if w.shape[0] < win_len:
                    continue
                # Standardize per-channel over time
                if self.min_max_norm:
                    w_min = w.min(axis=0, keepdims=True)
                    w_max = w.max(axis=0, keepdims=True) + 1e-6
                    w = (w - w_min) / (w_max - w_min)
                else:
                    w = (w - w.mean(axis=0, keepdims=True)) / (w.std(axis=0, keepdims=True) + 1e-6)
                # PCA over channels: flatten time, fit/transform features per-channel covariance
                # Reshape to (C, T) for PCA across channels using time as samples
                w_ct = w.transpose(1, 0)  # (C, T)
                if self.pca is not None:
                    # Fit transform expects (n_samples, n_features). We'll use channels as samples and time as features
                    w_p = self.pca.transform(w_ct)
                else:
                    w_p = w_ct
                # Optional global scaler across components
                if self.scaler is not None:
                    w_p = self.scaler.transform(w_p)
                # Back to (components, T)
                w_final = w_p.astype(np.float32)
                # Labeling
                if file_label >= 0:
                    y = file_label
                else:
                    # derive from event within window: if any event>0.5 mark positive
                    ev_win = events[start:start+win_len]
                    y = 1 if np.any(ev_win > 0.5) else 0
                # Save as (C, T)
                self.items.append((w_final, y))
        if self.verbose:
            print(f"Built dataset with {len(self.items)} windows")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        # Tensor shape: (C, T)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# ----------------------------
# PCA/Scaler Fit Helpers
# ----------------------------

def fit_pca_and_scaler(train_paths: List[str], pca_components: int, window_cfg: WindowingConfig,
                       bandpass: bool, notch: bool, low_hz: float, high_hz: float, notch_hz: float,
                       max_windows: int = 5000) -> Tuple[Optional[PCA], Optional[StandardScaler]]:
    if pca_components <= 0:
        return None, None
    samples = []  # collect (C, T) windows to fit PCA across channels using time as features
    collected = 0
    for path in train_paths:
        x, events, channels = load_eeg_csv(path)
        try:
            time = pd.read_csv(path, usecols=["Time"]).to_numpy().squeeze()
            fs = infer_fs(time)
        except Exception:
            fs = 200.0
        if bandpass:
            x = apply_bandpass(x, fs, low=low_hz, high=high_hz)
        if notch:
            x = apply_notch(x, fs, freq=notch_hz)
        win_len = int(round(window_cfg.window_sec * fs))
        step_len = int(round(window_cfg.step_sec * fs))
        for start in range(0, max(1, len(x) - win_len + 1), step_len):
            w = x[start:start+win_len]
            if w.shape[0] < win_len:
                continue
            w = (w - w.mean(axis=0, keepdims=True)) / (w.std(axis=0, keepdims=True) + 1e-6)
            w_ct = w.transpose(1, 0)  # (C, T)
            samples.append(w_ct)
            collected += 1
            if collected >= max_windows:
                break
        if collected >= max_windows:
            break
    if not samples:
        return None, None
    X = np.concatenate(samples, axis=0)  # stack over channels dimension: (sum_C, T)
    pca = PCA(n_components=pca_components, svd_solver="auto", whiten=False, random_state=0)
    pca.fit(X)
    scaler = StandardScaler()
    scaler.fit(pca.transform(X))
    return pca, scaler

# ----------------------------
# Models
# ----------------------------

class TinyCNN(nn.Module):
    """Very small 1D CNN. Input shape: (B, C, T)"""
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_classes)
    def forward(self, x):
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


class DeeperCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, n_classes)
    def forward(self, x):
        x = self.features(x).squeeze(-1)
        return self.classifier(x)


class EEGNet1D(nn.Module):
    """Simplified EEGNet-inspired 1D variant for CSV channel-time input.
    Reference: Lawhern et al., EEGNet (adapted to 1D)."""
    def __init__(self, in_ch: int, n_classes: int, depth_multiplier: int = 2):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv1d(in_ch, 8, kernel_size=1, bias=False),
            nn.BatchNorm1d(8)
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(8, 8*depth_multiplier, kernel_size=15, groups=8, padding=7, bias=False),
            nn.BatchNorm1d(8*depth_multiplier),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.25)
        )
        self.separable = nn.Sequential(
            nn.Conv1d(8*depth_multiplier, 16*depth_multiplier, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(16*depth_multiplier),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Linear(16*depth_multiplier, n_classes)
    def forward(self, x):
        x = self.first(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.mean(-1)
        return self.classifier(x)

# ----------------------------
# Spiking NN with Surrogate Gradient (no external deps)
# ----------------------------

class LIFLayer(nn.Module):
    def __init__(self, size, tau_mem=20.0, tau_syn=10.0, dt=1.0, v_th=1.0, v_reset=0.0):
        super().__init__()
        self.size = size
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.dt = dt
        self.v_th = v_th
        self.v_reset = v_reset
        # Buffers initialized at runtime
        self.register_buffer('v', torch.zeros(1, size))
        self.register_buffer('i_syn', torch.zeros(1, size))

    def forward(self, x_t):
        # x_t: (B, size) input current at time t
        if self.v.shape[0] != x_t.shape[0]:
            self.v = torch.zeros(x_t.shape[0], self.size, device=x_t.device)
            self.i_syn = torch.zeros_like(self.v)
        # Synaptic and membrane dynamics (Euler)
        self.i_syn = self.i_syn + self.dt * (-self.i_syn / self.tau_syn + x_t)
        self.v = self.v + self.dt * (-self.v / self.tau_mem + self.i_syn)
        # Spikes with surrogate gradient
        out = self.surrogate_heaviside(self.v - self.v_th)
        # Reset
        self.v = torch.where(out > 0, torch.full_like(self.v, self.v_reset), self.v)
        return out

    @staticmethod
    def surrogate_heaviside(x):
        class SurrogateFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return (input > 0).float()
            @staticmethod
            def backward(ctx, grad_output):
                (inp,) = ctx.saved_tensors
                sigma = 1.0
                grad = grad_output * torch.sigmoid(sigma*inp) * (1 - torch.sigmoid(sigma*inp))
                return grad
        return SurrogateFn.apply(x)

class SpikingCNN(nn.Module):
    """Rate-coded input -> conv -> linear -> LIF readout.
    We unroll over T_snn steps using identical inputs (rate code) unless `poisson_encode` is enabled.
    """
    def __init__(self, in_ch: int, n_classes: int, snn_steps: int = 10, poisson_encode: bool = False):
        super().__init__()
        self.snn_steps = snn_steps
        self.poisson_encode = poisson_encode
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, n_classes)
        self.lif = LIFLayer(size=n_classes)

    def forward(self, x):
        # x: (B, C, T), assume already normalized to ~[0, 1]
        feat = self.conv(x).squeeze(-1)  # (B, 64)
        logits = self.fc(feat)           # (B, n_cls) — used as input current
        # Rate/Poisson encode across time steps
        spk_sum = 0
        if self.poisson_encode:
            rate = torch.sigmoid(logits)
            for _ in range(self.snn_steps):
                curr = torch.bernoulli(rate)
                spk = self.lif(curr)
                spk_sum = spk_sum + spk
        else:
            curr = torch.sigmoid(logits)
            for _ in range(self.snn_steps):
                spk = self.lif(curr)
                spk_sum = spk_sum + spk
        return spk_sum / self.snn_steps

# ----------------------------
# Training/Eval
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, in_ch: int, n_classes: int, snn_steps: int) -> nn.Module:
    name = name.lower()
    if name == 'tinycnn':
        return TinyCNN(in_ch, n_classes)
    if name == 'cnn':
        return DeeperCNN(in_ch, n_classes)
    if name == 'eegnet':
        return EEGNet1D(in_ch, n_classes)
    if name == 'spiking':
        return SpikingCNN(in_ch, n_classes, snn_steps=snn_steps)
    raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out = model(xb)
        if out.ndim == 2 and out.size(1) > 1:
            loss = criterion(out, yb)
            preds = out.argmax(1)
        else:
            # binary
            out = out.view(-1)
            loss = F.binary_cross_entropy_with_logits(out, yb.float())
            preds = (torch.sigmoid(out) > 0.5).long()
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss/total, correct/total


def evaluate(model, loader, device, n_classes: int) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            if n_classes > 1:
                pred = out.argmax(1).cpu().numpy()
            else:
                out = out.view(-1)
                pred = (torch.sigmoid(out) > 0.5).long().cpu().numpy()
            ys.append(yb.numpy())
            ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return {
        'acc': accuracy_score(y, p),
        'f1': f1_score(y, p, average='macro'),
    }

# ----------------------------
# Label Loading
# ----------------------------

def load_labels_csv(labels_csv: Optional[str]) -> Dict[str, int]:
    """Expect a CSV with columns like [File, Label] or [filename,label].
    Returns dict mapping basename->label (int).
    """
    if labels_csv is None or not os.path.exists(labels_csv):
        return {}
    df = pd.read_csv(labels_csv)
    cols = [c.lower() for c in df.columns]
    if 'file' in cols:
        fcol = df.columns[cols.index('file')]
    elif 'filename' in cols:
        fcol = df.columns[cols.index('filename')]
    else:
        fcol = df.columns[0]
    if 'label' in cols:
        lcol = df.columns[cols.index('label')]
    else:
        lcol = df.columns[-1]
    labels = {}
    for _, r in df.iterrows():
        labels[os.path.basename(str(r[fcol]))] = int(r[lcol])
    return labels

# ----------------------------
# Main
# ----------------------------
"""
EEG PCA + CNN/TinyCNN/SpikingNN Training Pipeline (PyTorch)
===========================================================

Works with CSV files like `Data_S02_Sess01.csv` (columns: Time, EEG channels..., EOG, FeedBackEvent)
Optionally uses `TrainLabels.csv` if you have per-file labels.

Usage (examples):
-----------------
# Train TinyCNN with PCA to 16 components
python eeg_train.py \
  --data_dir /path/to/csvs \
  --labels_csv /path/to/TrainLabels.csv \
  --model tinycnn \
  --pca_components 16 \
  --epochs 30 --batch_size 64 --window_sec 2.0 --step_sec 0.5

# Train deeper CNN without PCA
python eeg_train.py --data_dir /path/to/csvs --model cnn --pca_components 0

# Train spiking CNN (surrogate gradient) with 10 simulation steps
python eeg_train.py --data_dir /path/to/csvs --model spiking --snn_steps 10

Notes:
------
- Sampling rate is inferred from the Time column (assumed steady).
- Labels may come from `TrainLabels.csv` or from in-file event markers (FeedBackEvent) if `--derive_labels_from_events` is set.
- Bandpass (1–40 Hz) and 50 Hz notch filters are available with `--bandpass` and `--notch`.
- PCA reduces the **channel** dimension; components are treated as channels.
- This script aims to be a strong starting baseline; tweak architecture & preprocessing for best results.
"""



# ----------------------------
# Main (no argparse; edit CONFIG and run)
# ----------------------------

CONFIG = {
    'data_dir': 'datasets//bci_challenge//train//',
    'labels_csv': ' datasets//bci_challenge//train//TrainLabels.csv',
    'derive_labels_from_events': False,
    'val_split': 0.2,
    'seed': 42,
    'bandpass': True,
    'notch': True,
    'low_hz': 1.0,
    'high_hz': 40.0,
    'notch_hz': 50.0,
    'window_sec': 2.0,
    'step_sec': 0.5,
    'pca_components': 16,
    'model': 'tinycnn',
    'epochs': 25,
    'batch_size': 64,
    'lr': 1e-3,
    'snn_steps': 10,
    'num_classes': 2,
    'min_max_norm': False,
    'save_path': 'best_model_',
}


def run_training(cfg: dict):
    set_seed(cfg['seed'])

    csv_paths = sorted(glob.glob(os.path.join(cfg['data_dir'], '*.csv')))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {cfg['data_dir']}")

    labels_map = load_labels_csv(cfg.get('labels_csv'))

    file_label_items = []
    for p in csv_paths:
        base = os.path.basename(p)
        if cfg['derive_labels_from_events']:
            file_label_items.append((p, -1))
        else:
            file_label_items.append((p, labels_map.get(base, -1)))

    window_cfg = WindowingConfig(cfg['window_sec'], cfg['step_sec'])

    train_paths, val_paths = train_test_split([p for p,_ in file_label_items], test_size=cfg['val_split'], random_state=cfg['seed'])

    pca, scaler = fit_pca_and_scaler(train_paths, cfg['pca_components'], window_cfg,
                                     bandpass=cfg['bandpass'], notch=cfg['notch'],
                                     low_hz=cfg['low_hz'], high_hz=cfg['high_hz'], notch_hz=cfg['notch_hz'])

    train_items = [(p, next((l for pp, l in file_label_items if pp == p), -1)) for p in train_paths]
    val_items = [(p, next((l for pp, l in file_label_items if pp == p), -1)) for p in val_paths]

    train_ds = EEGWindowDataset(train_items, pca=pca, scaler=scaler,
                                bandpass=cfg['bandpass'], notch=cfg['notch'],
                                low_hz=cfg['low_hz'], high_hz=cfg['high_hz'], notch_hz=cfg['notch_hz'],
                                window_cfg=window_cfg, min_max_norm=cfg['min_max_norm'])
    val_ds = EEGWindowDataset(val_items, pca=pca, scaler=scaler,
                              bandpass=cfg['bandpass'], notch=cfg['notch'],
                              low_hz=cfg['low_hz'], high_hz=cfg['high_hz'], notch_hz=cfg['notch_hz'],
                              window_cfg=window_cfg, min_max_norm=cfg['min_max_norm'])

    if len(train_ds) == 0:
        raise RuntimeError("Empty training dataset. Check window parameters and data.")
    sample_x, _ = train_ds[0]
    in_ch = sample_x.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg['model'], in_ch, cfg['num_classes'], snn_steps=cfg['snn_steps']).to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss() if cfg['num_classes'] > 1 else nn.BCEWithLogitsLoss()

    best_acc = -1.0
    for epoch in range(1, cfg['epochs']+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, criterion)
        metrics = evaluate(model, val_loader, device, cfg['num_classes'])
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val acc {metrics['acc']:.3f} f1 {metrics['f1']:.3f}")
        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            torch.save({
                'model': cfg['model'],
                'state_dict': model.state_dict(),
                'in_ch': in_ch,
                'num_classes': cfg['num_classes'],
                'pca': pca.__dict__ if pca is not None else None,
                'scaler_mean': scaler.mean_.tolist() if scaler is not None else None,
                'scaler_scale': scaler.scale_.tolist() if scaler is not None else None,
                'window_cfg': vars(window_cfg),
            }, cfg['save_path']+CONFIG["model"]+".pt")
            print(f"Saved best model -> {cfg['save_path']+CONFIG["model"]+".pt"}")

    print("Training complete.")

 
if __name__ == '__main__':
    #     if name == 'tinycnn':
    #     return TinyCNN(in_ch, n_classes)
    # if name == 'cnn':
    #     return DeeperCNN(in_ch, n_classes)
    # if name == 'eegnet':
    #     return EEGNet1D(in_ch, n_classes)
    # if name == 'spiking':

    # Tiny CNN: Epoch 025 | train loss 0.3703 acc 0.849 | val acc 0.832 f1 0.515
    # CNN: Epoch 025 | train loss 0.2714 acc 0.885 | val acc 0.792 f1 0.575 
    # EEGNet: Epoch 025 | train loss 0.4009 acc 0.841 | val acc 0.836 f1 0.470

    models= ["spiking","tinycnn"]
    for model in models:
        try:
            CONFIG["model"] = model
            run_training(CONFIG)
        except Exception as e:
            print(e)
