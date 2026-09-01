"""
Build an unlabeled 59-channel EEGMMIDB memmap dataset for flow-matching pretraining.

Important:
    - This dataset is used only for self-supervised EEG pretraining.
    - It does not provide artifact masks for supervised artifact segmentation.
    - The supervised artifact training/testing remains PhysioMotion only.

Input expected:
    eegmmidb_runwise_csv/
        manifest.csv
        subjects/S001/S001R03_signals.csv.gz
        subjects/S001/S001R03_metadata.json
        ...

Output:
    output_dir/
        x_float32.dat
        manifest.csv
        metadata.json

Each x window has shape:
    59 channels x 2500 samples

because:
    10 seconds x 250 Hz = 2500 samples
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_channel_name(name: str) -> str:
    """Make EEGMMIDB channel labels easier to compare/read."""
    return str(name).strip().replace(".", "").upper()


def normalize_window(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-score normalization inside one window."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + eps)


def count_windows(samples: int, window_samples: int, stride_samples: int) -> int:
    if samples < window_samples:
        return 0
    return 1 + (samples - window_samples) // stride_samples


def resample_to_target(eeg: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    """
    Resample channels x time EEG to target_sfreq using linear interpolation.

    EEGMMIDB is commonly 160 Hz, with a small subset at 128 Hz.
    Our model uses 250 Hz so every 10-second window has 2500 samples.
    """
    if abs(source_sfreq - target_sfreq) < 1e-6:
        return eeg.astype(np.float32, copy=False)

    source_samples = eeg.shape[1]
    duration_seconds = source_samples / float(source_sfreq)
    target_samples = int(round(duration_seconds * float(target_sfreq)))
    if target_samples <= 1:
        return np.empty((eeg.shape[0], 0), dtype=np.float32)

    old_t = np.arange(source_samples, dtype=np.float64) / float(source_sfreq)
    new_t = np.arange(target_samples, dtype=np.float64) / float(target_sfreq)

    out = np.empty((eeg.shape[0], target_samples), dtype=np.float32)
    for ch in range(eeg.shape[0]):
        out[ch] = np.interp(new_t, old_t, eeg[ch]).astype(np.float32)
    return out


def read_signal_csv_gz(csv_path: Path, selected_channels: list[str]) -> np.ndarray:
    """Read selected EEG channels from one compressed EEGMMIDB CSV."""
    usecols = ["sample", "time_seconds", *selected_channels]
    chunks: list[np.ndarray] = []
    for chunk in pd.read_csv(csv_path, compression="gzip", usecols=usecols, chunksize=100_000):
        arr = chunk[selected_channels].to_numpy(dtype=np.float32).T
        chunks.append(arr)
    if not chunks:
        return np.empty((len(selected_channels), 0), dtype=np.float32)
    return np.concatenate(chunks, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eegmmidb-dir",
        default=str(PROJECT_ROOT.parent / "packages" / "Final 1 EEGMMIDB" / "eegmmidb_runwise_csv"),
        help="Prepared EEGMMIDB run-wise CSV folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "outputs_v6_flowmatching_study"
            / "window_datasets"
            / "eegmmidb_59ch_10s_250hz_flow_memmap"
        ),
    )
    parser.add_argument("--target-sfreq", type=float, default=250.0)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--limit-recordings", type=int, default=0)
    args = parser.parse_args()

    eegmmidb_dir = Path(args.eegmmidb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = eegmmidb_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"EEGMMIDB manifest not found: {manifest_path}")

    manifest_in = pd.read_csv(manifest_path)
    if args.limit_recordings:
        manifest_in = manifest_in.head(args.limit_recordings).copy()

    first_csv_rel = str(manifest_in.iloc[0]["csv_file"]).replace("\\", "/")
    first_csv_path = eegmmidb_dir / first_csv_rel
    first_header = pd.read_csv(first_csv_path, compression="gzip", nrows=0).columns.tolist()
    all_signal_channels = [c for c in first_header if c not in {"sample", "time_seconds"}]
    if len(all_signal_channels) < 59:
        raise RuntimeError(f"Expected at least 59 EEG channels, found {len(all_signal_channels)}.")

    # Use the first 59 stable EEGMMIDB channels so the output shape matches PhysioMotion.
    # This is self-supervised pretraining only; no artifact labels are used.
    selected_channels = all_signal_channels[:59]

    window_samples = int(round(args.window_sec * args.target_sfreq))
    stride_samples = int(round(args.stride_sec * args.target_sfreq))

    recording_infos = []
    total_windows = 0
    for _, row in manifest_in.iterrows():
        source_sfreq = float(row["sampling_frequency_hz"])
        duration_seconds = float(row["duration_seconds"])
        target_samples = int(round(duration_seconds * args.target_sfreq))
        n_windows = count_windows(target_samples, window_samples, stride_samples)
        recording_infos.append((row, source_sfreq, n_windows))
        total_windows += n_windows

    if total_windows == 0:
        raise RuntimeError("No EEGMMIDB flow-matching windows were created.")

    x_shape = (int(total_windows), len(selected_channels), window_samples)
    x_path = output_dir / "x_float32.dat"
    x_mem = np.memmap(x_path, dtype="float32", mode="w+", shape=x_shape)

    rows = []
    skipped_recordings = []
    global_index = 0
    for row, source_sfreq, n_windows in recording_infos:
        subject = str(row["subject_id"])
        run_id = str(row["run_id"])
        run_type = str(row["run_type"])
        csv_rel = str(row["csv_file"]).replace("\\", "/")
        csv_path = eegmmidb_dir / csv_rel
        print(f"Processing EEGMMIDB {subject} {run_id} ({run_type})...", flush=True)

        try:
            eeg_source = read_signal_csv_gz(csv_path, selected_channels)
        except Exception as exc:
            skipped_recordings.append(
                {
                    "subject": subject,
                    "run": run_id,
                    "recording_kind": run_type,
                    "csv_file": str(csv_path),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            warnings.warn(
                f"Skipping unreadable EEGMMIDB recording {subject} {run_id}: {type(exc).__name__}: {exc}",
                RuntimeWarning,
            )
            continue

        eeg_250 = resample_to_target(eeg_source, source_sfreq=source_sfreq, target_sfreq=args.target_sfreq)

        written_for_recording = 0
        for window_index in range(n_windows):
            start = window_index * stride_samples
            stop = start + window_samples
            if stop > eeg_250.shape[1]:
                continue
            x_mem[global_index] = normalize_window(eeg_250[:, start:stop]).astype(np.float32)
            rows.append(
                {
                    "index": global_index,
                    "source_dataset": "eegmmidb",
                    "subject": subject,
                    "run": int(run_id.replace("R", "")),
                    "recording_id": f"{subject}{run_id}",
                    "recording_kind": run_type,
                    "window_index_in_recording": window_index,
                    "window_start_sample_250hz": start,
                    "window_stop_sample_250hz": stop,
                    "window_start_seconds": start / args.target_sfreq,
                    "window_stop_seconds": stop / args.target_sfreq,
                    "source_sampling_frequency_hz": source_sfreq,
                    "sampling_frequency_hz": args.target_sfreq,
                    "window_duration_seconds": args.window_sec,
                    "channels": len(selected_channels),
                    "has_artifact_label": False,
                    "use": "flow_matching_pretraining_only",
                }
            )
            global_index += 1
            written_for_recording += 1
        print(f"  windows written: {written_for_recording}", flush=True)
        del eeg_source, eeg_250

    x_mem.flush()

    if not rows:
        raise RuntimeError("No EEGMMIDB windows were written after skipping unreadable recordings.")

    out_manifest = output_dir / "manifest.csv"
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset_name": "EEGMMIDB 59-channel EEG windows for flow matching",
        "source_prepared_folder": str(eegmmidb_dir),
        "purpose": "external unlabeled EEG pretraining only",
        "supervised_artifact_labels_available": False,
        "x_shape": [int(global_index), len(selected_channels), window_samples],
        "x_shape_allocated_on_disk": list(x_shape),
        "x_shape_per_window": [len(selected_channels), window_samples],
        "selected_channel_names_original": selected_channels,
        "selected_channel_names_normalized": [normalize_channel_name(c) for c in selected_channels],
        "source_channel_count": len(all_signal_channels),
        "selected_channel_count": len(selected_channels),
        "target_sampling_frequency_hz": args.target_sfreq,
        "resampling_note": "EEGMMIDB recordings are linearly interpolated to 250 Hz so each 10-second window has 2500 samples.",
        "window_duration_seconds": args.window_sec,
        "stride_seconds": args.stride_sec,
        "total_windows": int(global_index),
        "total_windows_original_estimate_before_skips": int(total_windows),
        "recordings": int(len(recording_infos)),
        "recordings_used": int(len(recording_infos) - len(skipped_recordings)),
        "recordings_skipped": int(len(skipped_recordings)),
        "skipped_recordings": skipped_recordings,
        "channel_selection_limitation": (
            "The first 59 stable EEGMMIDB EEG channels are selected to match the 59-channel model shape. "
            "This is used only for self-supervised EEG pretraining, not for supervised artifact-mask training."
        ),
        "files": {
            "x_memmap": str(x_path),
            "manifest": str(out_manifest),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved EEGMMIDB flow-matching memmap dataset:", flush=True)
    print(output_dir, flush=True)
    print("x shape:", [int(global_index), len(selected_channels), window_samples], flush=True)
    print("skipped recordings:", len(skipped_recordings), flush=True)


if __name__ == "__main__":
    main()
