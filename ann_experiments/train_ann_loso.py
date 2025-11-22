"""
train_ann_loso.py

Training a compact ANN (CNN + GRU) baseline for physiological stress detection
using Leave-One-Subject-Out (LOSO) evaluation.

Reads:
    data/processed/stress_windows_full.npz

Writes:
    ann_experiments/results/ann_loso_<channel_mode>.csv  (per-subject AUROC/F1)
    ann_experiments/results/ann_loso_summary.csv         (per-channel-mode summary)
"""

import os
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from .dataset_wearable import WearableStressDataset

# -------- Device selection --------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

NPZ_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "stress_windows_full.npz")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------- Channel configs for ablation --------
# Channel layout (C dimension):
#   0: EDA
#   1: ACC x
#   2: ACC y
#   3: ACC z
#   4: BVP
#   5: TEMP
CHANNEL_CONFIGS = {
    "full":     [0, 1, 2, 3, 4, 5],
    "eda_only": [0],
    "acc_only": [1, 2, 3],
    "eda_acc":  [0, 1, 2, 3],
    "bvp_only": [4],
}

# Manually choose which channel mode to run:
#   "full", "eda_only",  "eda_acc"（optional："acc_only","bvp_only"）
CHANNEL_MODE = "eda_acc"
SELECTED_CHANNELS = CHANNEL_CONFIGS[CHANNEL_MODE]
print(f"Channel mode: {CHANNEL_MODE}, channels={SELECTED_CHANNELS}")


class StressANN(nn.Module):
    """
    Compact ANN model with:
        - 1D CNN layers
        - GRU layer
        - Fully connected classifier
    """

    def __init__(self, in_channels, num_classes, hidden_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)          # (B, 64, T')
        x = x.transpose(1, 2)     # (B, T', 64)
        out, _ = self.gru(x)      # (B, T', 2H)
        out = out.mean(dim=1)     # (B, 2H) temporal average pooling
        logits = self.fc(out)
        return logits


def train_one_fold(npz_path, train_subjects, test_subject):
    """
    Train on train_subjects and evaluate on test_subject.
    """

    train_ds = WearableStressDataset(
        npz_path,
        subjects=train_subjects,
        channels=SELECTED_CHANNELS,
    )
    test_ds = WearableStressDataset(
        npz_path,
        subjects=[test_subject],
        channels=SELECTED_CHANNELS,
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Use the number of selected channels as the model's input dimension.
    # Example:
    #   CHANNEL_MODE="full"     -> [0,1,2,3,4,5] -> 6 channels
    #   CHANNEL_MODE="eda_only" -> [0]           -> 1 channel
    #   CHANNEL_MODE="eda_acc"  -> [0,1,2,3]     -> 4 channels
    # This ensures the CNN receives the correct input shape after channel ablation.
    model = StressANN(
        in_channels=len(SELECTED_CHANNELS),
        num_classes=2,
    ).to(device)

    # Parameter count (for later comparison across models / channels)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[{test_subject}] Model parameters: {num_params / 1e3:.1f} K")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # ---- Training ----
    for epoch in range(10):
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        print(f"[{test_subject}] Epoch {epoch + 1}/10 done.")

    # ---- Evaluation ----
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds.append(prob)
            labels.append(y.numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    try:
        auroc = roc_auc_score(labels, preds)
    except Exception:
        auroc = np.nan

    y_pred = (preds >= 0.5).astype(int)
    f1 = f1_score(labels, y_pred, zero_division=0)

    print(f"[{test_subject}] AUROC={auroc:.4f}, F1={f1:.4f}")
    return auroc, f1, num_params


def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    subjects = sorted(set(data["subjects"].astype(str)))

    aurocs, f1s = [], []
    num_params_used = None  # same architecture for all folds; store once

    # Per-channel-mode results file (one row per subject)
    results_file = os.path.join(RESULTS_DIR, f"ann_loso_{CHANNEL_MODE}.csv")

    # Write CSV header
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "AUROC", "F1"])

    # LOSO loop
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        auc, f1, num_params = train_one_fold(NPZ_PATH, train_subjects, test_subject)

        aurocs.append(auc)
        f1s.append(f1)
        if num_params_used is None:
            num_params_used = num_params

        # Append per-subject result
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([test_subject, auc, f1])

    mean_auc = float(np.nanmean(aurocs))
    std_auc = float(np.nanstd(aurocs))
    mean_f1 = float(np.nanmean(f1s))
    std_f1 = float(np.nanstd(f1s))

    print("\n===== LOSO RESULTS =====")
    print(f"Channel mode = {CHANNEL_MODE}")
    print(f"AUROC mean={mean_auc:.4f}, std={std_auc:.4f}")
    print(f"F1 mean={mean_f1:.4f}, std={std_f1:.4f}")
    print("Saved per-subject results to:", results_file)

    # Append to a global summary file (across channel modes)
    summary_file = os.path.join(RESULTS_DIR, "ann_loso_summary.csv")
    file_exists = os.path.exists(summary_file)
    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "channel_mode",
                "channels",
                "num_params",
                "AUROC_mean",
                "AUROC_std",
                "F1_mean",
                "F1_std",
                "num_subjects",
            ])
        writer.writerow([
            CHANNEL_MODE,
            str(SELECTED_CHANNELS),
            num_params_used,
            mean_auc,
            std_auc,
            mean_f1,
            std_f1,
            len(subjects),
        ])


if __name__ == "__main__":
    main()
