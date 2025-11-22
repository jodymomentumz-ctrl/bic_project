"""
preprocess_wearable_stress.py

This script reads raw Empatica E4 STRESS session CSV files and generates
a unified window-based dataset suitable for ANN/SNN experiments.

Output:
    data/processed/stress_windows_full.npz
containing:
    X: (N, C, T)
    y: (N,)
    subjects: (N,)
"""

import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate

# Project root = parent of ann_experiments/
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

# Raw data root according to your folder layout
RAW_STRESS_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "STRESS")

# Processed output directory
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

OUTPUT_NPZ_PATH = os.path.join(PROCESSED_DIR, "stress_windows_full.npz")

# Target resampling frequency (Hz)
TARGET_FS = 4.0

# Windowing parameters
WINDOW_SEC = 30.0
STEP_SEC = 15.0  # 50% overlap


def read_empatica_signal(csv_path: str):
    """
    Load an Empatica CSV file:
    - Row 1: session start timestamp (unused)
    - Row 2: sampling rate (Hz)
    - Remaining rows: data (1 or more channels)

    Returns:
        fs (float)               - sampling rate
        data (np.ndarray)        - shape (T, C)
        time_axis (np.ndarray)   - shape (T,), in seconds
    """
    df = pd.read_csv(csv_path, header=None)
    fs = float(df.iloc[1, 0])
    data = df.iloc[2:, :].to_numpy(dtype=np.float32)

    num_samples = data.shape[0]
    duration = num_samples / fs
    time_axis = np.linspace(0.0, duration, num_samples, endpoint=False)

    return fs, data, time_axis


def resample_to_target(data, time_axis, target_fs):
    """
    Linearly resample a signal to target sampling rate.
    """
    if data.size == 0:
        return data, time_axis

    t_start = time_axis[0]
    t_end = time_axis[-1]
    duration = t_end - t_start
    if duration <= 0:
        return data, time_axis

    num_target = int(np.floor(duration * target_fs))
    if num_target <= 1:
        return data, time_axis

    t_rs = np.linspace(t_start, t_start + num_target / target_fs, num_target, endpoint=False)
    C = data.shape[1]
    data_rs = np.zeros((num_target, C), dtype=np.float32)

    for c in range(C):
        f = interpolate.interp1d(time_axis, data[:, c], fill_value="extrapolate")
        data_rs[:, c] = f(t_rs).astype(np.float32)

    return data_rs, t_rs


def build_label_intervals_for_subject(subject_id, subject_folder):
    """
    Build stress label intervals for a subject using tags.csv and EDA.csv.

    The notebook from the authors uses:
        - For v1 subjects (IDs starting with 'S'):
            stress blocks are:
                [tags[3], tags[4]]  # Stroop
                [tags[5], tags[6]]  # TMCT
                [tags[7], tags[8]]  # Real opinion
                [tags[9], tags[10]] # Opposite opinion
                [tags[11], tags[12]]# Subtract

        - For v2 subjects (IDs starting with 'f'):
            stress blocks are:
                [tags[2], tags[3]]  # TMCT
                [tags[4], tags[5]]  # Real opinion
                [tags[6], tags[7]]  # Opposite opinion
                [tags[8], tags[9]]  # Subtract

    We convert all tag timestamps to seconds relative to the session start
    (taken from the first row of EDA.csv) so that they are compatible with
    the time axis used in preprocessing.

    Returns:
        intervals: list of (start_sec, end_sec, label)
                   where label = 1 for stress segments.
                   Non-stress will be handled via the default label in
                   label_window().
    """
    import pandas as pd

    # --- read session start time from EDA.csv ---
    eda_path = os.path.join(subject_folder, "EDA.csv")
    if not os.path.exists(eda_path):
        raise FileNotFoundError(f"EDA.csv not found for subject {subject_id}")

    # Empatica format: first row = UTC start time (string), second row = fs
    eda_df = pd.read_csv(eda_path, header=None)
    session_start_str = str(eda_df.iloc[0, 0])
    session_start_ts = pd.to_datetime(session_start_str)

    # --- read tags.csv (UTC timestamps of button presses) ---
    tags_path = os.path.join(subject_folder, "tags.csv")
    if not os.path.exists(tags_path):
        raise FileNotFoundError(f"tags.csv not found for subject {subject_id}")

        # --- read tags.csv (UTC timestamps of button presses) ---
    tags_path = os.path.join(subject_folder, "tags.csv")
    if not os.path.exists(tags_path):
        raise FileNotFoundError(f"tags.csv not found for subject {subject_id}")

    # If tags.csv is empty (0 bytes), we cannot build any intervals.
    if os.path.getsize(tags_path) == 0:
        print(f"  Warning: subject {subject_id} has an EMPTY tags.csv; skipping labels.")
        return []

    tags_df = pd.read_csv(tags_path, header=None)
    if tags_df.shape[1] != 1:
        raise ValueError(f"tags.csv for {subject_id} should have 1 column, got {tags_df.shape[1]}")

    tag_ts = pd.to_datetime(tags_df[0])
    # convert to seconds from session start
    tag_secs = (tag_ts - session_start_ts).dt.total_seconds().to_numpy()

    intervals = []

    if subject_id.startswith("S"):
        # ----- Version 1 protocol -----
        if len(tag_secs) < 13:
            raise ValueError(
                f"Subject {subject_id}: expected at least 13 tags for v1, got {len(tag_secs)}"
            )

        stress_pairs = [
            (3, 4),   # Stroop
            (5, 6),   # TMCT
            (7, 8),   # Real opinion
            (9, 10),  # Opposite opinion
            (11, 12), # Subtract
        ]


    else:

        # ----- Version 2 protocol (fxx, including f14_a / f14_b) -----

        # In the ideal case there are 10 tags and we can form

        # four stress blocks: [2,3], [4,5], [6,7], [8,9].

        # However, some subjects (e.g., incomplete protocol)

        # may have fewer tags (e.g., 9). We therefore form as

        # many consecutive pairs (2k, 2k+1) as possible

        # starting from index 2.

        if len(tag_secs) < 7:
            # With fewer than 7 tags we cannot form even

            # three stress blocks; treat as malformed.

            raise ValueError(

                f"Subject {subject_id}: not enough tags for v2 "

                f"(got {len(tag_secs)})."

            )

        stress_pairs = []

        i = 2

        # build pairs (2,3), (4,5), (6,7), (8,9) ... as far as we can

        while i + 1 < len(tag_secs):
            stress_pairs.append((i, i + 1))

            i += 2

        # Optional: warn if we had to drop the last block due to missing tag

        if len(tag_secs) < 10:
            print(

                f"  Warning: subject {subject_id} has only {len(tag_secs)} tags; "

                f"using {len(stress_pairs)} stress blocks instead of 4."

            )

    # Build intervals with label=1 (stress)
    for i, j in stress_pairs:
        start_sec = float(tag_secs[i])
        end_sec = float(tag_secs[j])
        if end_sec <= start_sec:
            continue  # skip degenerate intervals
        intervals.append((start_sec, end_sec, 1))

    return intervals


def label_window(t_start, t_end, intervals, default_label=0):
    """
    Assign a label to a window by checking where its center falls.

    Windows whose center does not fall into any stress interval
    are treated as non-stress (label 0).
    """
    t_center = 0.5 * (t_start + t_end)
    for start, end, lab in intervals:
        if start <= t_center < end:
            return lab
    # outside all stress intervals → non-stress
    return default_label



def process_one_subject(subject_folder):
    """
    Process a single subject folder and return:
        X_sub: (N_sub, C, T)
        y_sub: (N_sub,)
        subjects_sub: (N_sub,)
    """
    subject_id = os.path.basename(subject_folder)
    print(f"Processing subject {subject_id} ...")

    # Required signals
    eda_path = os.path.join(subject_folder, "EDA.csv")
    bvp_path = os.path.join(subject_folder, "BVP.csv")
    temp_path = os.path.join(subject_folder, "TEMP.csv")
    acc_path = os.path.join(subject_folder, "ACC.csv")

    if not all(os.path.exists(p) for p in [eda_path, bvp_path, temp_path, acc_path]):
        print(f"  Missing signals for {subject_id}, skipping.")
        return np.empty((0,6,1)), np.empty((0,)), np.empty((0,))

    # Load signals
    _, eda, t_eda = read_empatica_signal(eda_path)
    _, bvp, t_bvp = read_empatica_signal(bvp_path)
    _, temp, t_temp = read_empatica_signal(temp_path)
    _, acc, t_acc = read_empatica_signal(acc_path)

    # Resample
    eda_rs, t_eda_rs = resample_to_target(eda,  t_eda,  TARGET_FS)
    bvp_rs, t_bvp_rs = resample_to_target(bvp,  t_bvp,  TARGET_FS)
    temp_rs,t_temp_rs= resample_to_target(temp, t_temp, TARGET_FS)
    acc_rs, t_acc_rs = resample_to_target(acc,  t_acc,  TARGET_FS)

    # Align time axis
    t_min = max(t_eda_rs[0], t_bvp_rs[0], t_temp_rs[0], t_acc_rs[0])
    t_max = min(t_eda_rs[-1],t_bvp_rs[-1],t_temp_rs[-1],t_acc_rs[-1])

    if t_max <= t_min:
        print(f"  No overlapping time range for {subject_id}, skipping.")
        return np.empty((0,6,1)), np.empty((0,)), np.empty((0,))

    duration = t_max - t_min
    num_points = int(np.floor(duration * TARGET_FS))
    t_common = np.linspace(t_min, t_min + num_points / TARGET_FS, num_points, endpoint=False)

    def interp_to_common(t_src, data_src):
        C = data_src.shape[1]
        out = np.zeros((num_points, C), dtype=np.float32)
        for c in range(C):
            f = interpolate.interp1d(t_src, data_src[:, c], fill_value="extrapolate")
            out[:, c] = f(t_common).astype(np.float32)
        return out

    eda_c = interp_to_common(t_eda_rs, eda_rs)
    acc_c = interp_to_common(t_acc_rs, acc_rs)
    bvp_c = interp_to_common(t_bvp_rs, bvp_rs)
    temp_c= interp_to_common(t_temp_rs,temp_rs)

    # Channel layout: [EDA, ACCx, ACCy, ACCz, BVP, TEMP]
    multi = np.concatenate([eda_c, acc_c, bvp_c, temp_c], axis=1)

    # Build label intervals
    intervals = build_label_intervals_for_subject(subject_id, subject_folder)

    # If no intervals are returned (e.g., empty or invalid tags),
    # skip this subject.
    if not intervals:
        print(f"  No label intervals for {subject_id}; skipping subject.")
        return np.empty((0, 6, 1)), np.empty((0,)), np.empty((0,))

    # Windowing
    win_len = int(WINDOW_SEC * TARGET_FS)
    step = int(STEP_SEC * TARGET_FS)

    X_list, y_list, subj_list = [], [], []

    for start in range(0, multi.shape[0] - win_len + 1, step):
        end = start + win_len
        t_start = t_common[start]
        t_end = t_common[end - 1]

        # label 1 = stress, 0 = non-stress
        label = label_window(t_start, t_end, intervals, default_label=0)

        window = multi[start:end].T  # to (C, T)
        X_list.append(window)
        y_list.append(label)
        subj_list.append(subject_id)

    if not X_list:
        return np.empty((0,6,1)), np.empty((0,)), np.empty((0,))

    return (
        np.stack(X_list, axis=0),
        np.array(y_list, dtype=int),
        np.array(subj_list, dtype=str)
    )


def main():
    subject_folders = sorted(
        [p for p in glob.glob(os.path.join(RAW_STRESS_ROOT, "*")) if os.path.isdir(p)]
    )
    if not subject_folders:
        raise RuntimeError(f"No subjects found at {RAW_STRESS_ROOT}")

    X_all, y_all, subj_all = [], [], []

    for sf in subject_folders:
        Xs, ys, ss = process_one_subject(sf)
        if Xs.size == 0:
            continue
        X_all.append(Xs)
        y_all.append(ys)
        subj_all.append(ss)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    subjects = np.concatenate(subj_all, axis=0)

    print("Final dataset shape:", X.shape, y.shape, subjects.shape)

    np.savez(OUTPUT_NPZ_PATH, X=X, y=y, subjects=subjects)
    print("Saved:", OUTPUT_NPZ_PATH)


if __name__ == "__main__":
    main()
