"""
Build an unlabeled 59-channel EEG memmap dataset from BCI Competition IV Data Set 1.

This V3 dataset is used only for self-supervised flow-matching pretraining.
It does not contain artifact masks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_window(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + eps)


def count_windows(samples_after_downsample: int, window_samples: int, stride_samples: int) -> int:
    if samples_after_downsample < window_samples:
        return 0
    return 1 + (samples_after_downsample - window_samples) // stride_samples


def read_downsampled_csv(csv_path: Path, channel_names: list[str], downsample_factor: int) -> np.ndarray:
    usecols = ["sample", *channel_names]
    chunks: list[np.ndarray] = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=200_000):
        selected = chunk.iloc[::downsample_factor]
        arr = selected[channel_names].to_numpy(dtype=np.float32).T
        chunks.append(arr)
    if not chunks:
        return np.empty((len(channel_names), 0), dtype=np.float32)
    return np.concatenate(chunks, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bci-dir",
        default=str(PROJECT_ROOT.parent / "packages" / "Final 4 - BCI IV 1" / "bci_comp_iv_ds1_1000hz_csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "outputs_v6_flowmatching_study"
            / "window_datasets"
            / "bci_iv_ds1_59ch_10s_250hz_flow_memmap"
        ),
    )
    parser.add_argument("--target-sfreq", type=float, default=250.0)
    parser.add_argument("--source-sfreq", type=float, default=1000.0)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--limit-recordings", type=int, default=0)
    args = parser.parse_args()

    bci_dir = Path(args.bci_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_in = pd.read_csv(bci_dir / "manifest.csv")
    if args.limit_recordings:
        manifest_in = manifest_in.head(args.limit_recordings).copy()

    first_meta_rel = str(manifest_in.iloc[0]["metadata_file"]).replace("\\", "/")
    first_meta_path = bci_dir / first_meta_rel
    with first_meta_path.open("r", encoding="utf-8") as f:
        first_meta = json.load(f)
    channel_names = [ch["name"] for ch in first_meta["channels"]]
    if len(channel_names) != 59:
        raise RuntimeError(f"Expected 59 BCI EEG channels, found {len(channel_names)}.")

    downsample_factor = int(round(args.source_sfreq / args.target_sfreq))
    if abs(args.source_sfreq / downsample_factor - args.target_sfreq) > 1e-6:
        raise RuntimeError("This script expects an integer downsampling factor.")

    window_samples = int(round(args.window_sec * args.target_sfreq))
    stride_samples = int(round(args.stride_sec * args.target_sfreq))

    recording_infos = []
    total_windows = 0
    for _, row in manifest_in.iterrows():
        samples_after_downsample = int(row["samples"]) // downsample_factor
        n_windows = count_windows(samples_after_downsample, window_samples, stride_samples)
        recording_infos.append((row, n_windows))
        total_windows += n_windows

    if total_windows == 0:
        raise RuntimeError("No BCI flow-matching windows were created.")

    x_shape = (int(total_windows), len(channel_names), window_samples)
    x_path = output_dir / "x_float32.dat"
    x_mem = np.memmap(x_path, dtype="float32", mode="w+", shape=x_shape)

    rows = []
    global_index = 0
    for row, n_windows in recording_infos:
        subject = str(row["subject_id"])
        recording_id = str(row["recording_id"])
        recording_kind = str(row["recording_kind"])
        csv_rel = str(row["csv_file"]).replace("\\", "/")
        csv_path = bci_dir / csv_rel
        print(f"Processing {recording_id} from {csv_path.name}...", flush=True)

        eeg = read_downsampled_csv(csv_path, channel_names, downsample_factor)
        for window_index in range(n_windows):
            start = window_index * stride_samples
            stop = start + window_samples
            x_mem[global_index] = normalize_window(eeg[:, start:stop]).astype(np.float32)
            rows.append(
                {
                    "index": global_index,
                    "source_dataset": "bci_comp_iv_ds1",
                    "subject": subject,
                    "run": 1 if recording_kind == "calibration" else 2,
                    "recording_id": recording_id,
                    "recording_kind": recording_kind,
                    "window_index_in_recording": window_index,
                    "window_start_sample_250hz": start,
                    "window_stop_sample_250hz": stop,
                    "window_start_seconds": start / args.target_sfreq,
                    "window_stop_seconds": stop / args.target_sfreq,
                    "sampling_frequency_hz": args.target_sfreq,
                    "window_duration_seconds": args.window_sec,
                    "channels": len(channel_names),
                    "has_artifact_label": False,
                    "use": "flow_matching_pretraining_only",
                }
            )
            global_index += 1
        print(f"  windows written: {n_windows}", flush=True)

    x_mem.flush()

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset_name": "BCI Competition IV Data Set 1, 59-channel EEG windows for flow matching",
        "source_prepared_folder": str(bci_dir),
        "purpose": "external unlabeled EEG pretraining only",
        "supervised_artifact_labels_available": False,
        "x_shape": list(x_shape),
        "x_shape_per_window": [len(channel_names), window_samples],
        "channel_names": channel_names,
        "source_sampling_frequency_hz": args.source_sfreq,
        "target_sampling_frequency_hz": args.target_sfreq,
        "downsample_factor": downsample_factor,
        "downsampling_note": "Every 4th sample is used to convert 1000 Hz to 250 Hz.",
        "window_duration_seconds": args.window_sec,
        "stride_seconds": args.stride_sec,
        "total_windows": int(total_windows),
        "recordings": int(len(recording_infos)),
        "files": {
            "x_memmap": str(x_path),
            "manifest": str(manifest_path),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved BCI flow-matching memmap dataset:", flush=True)
    print(output_dir, flush=True)
    print("x shape:", list(x_shape), flush=True)


if __name__ == "__main__":
    main()
