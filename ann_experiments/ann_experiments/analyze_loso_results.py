"""
analyze_loso_results.py

Utility script to:
    - Load ann_loso_summary.csv
    - Print a neat comparison table across channel modes
    - Plot AUROC/F1 bar chart and save to PNG
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "ann_loso_summary.csv")


def load_summary(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    if not os.path.exists(SUMMARY_PATH):
        print("Summary file not found:", SUMMARY_PATH)
        print("Please run train_ann_loso.py first with different CHANNEL_MODEs.")
        return

    rows = load_summary(SUMMARY_PATH)
    if not rows:
        print("No rows in summary file.")
        return


    print("==== ANN LOSO CHANNEL COMPARISON ====")
    for row in rows:
        mode = row["channel_mode"]
        chans = row["channels"]
        auc_m = float(row["AUROC_mean"])
        auc_s = float(row["AUROC_std"])
        f1_m = float(row["F1_mean"])
        f1_s = float(row["F1_std"])
        print(
            f"{mode:10s} | chans={chans:18s} | "
            f"AUROC={auc_m:.3f}±{auc_s:.3f} | F1={f1_m:.3f}±{f1_s:.3f}"
        )


    modes = [r["channel_mode"] for r in rows]
    auc_means = [float(r["AUROC_mean"]) for r in rows]
    f1_means = [float(r["F1_mean"]) for r in rows]

    x = np.arange(len(modes))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, auc_means, width, label="AUROC")
    plt.bar(x + width / 2, f1_means, width, label="F1")

    plt.xticks(x, modes, rotation=30)
    plt.ylabel("Score")
    plt.title("ANN LOSO performance across channel modes")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "ann_loso_channels_comparison.png")
    plt.savefig(out_path, dpi=200)
    print("Saved comparison figure to:", out_path)


if __name__ == "__main__":
    main()
