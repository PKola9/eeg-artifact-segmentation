"""
Build a curated 59-channel DEAP EEG memmap dataset for flow-matching pretraining.

Purpose:
    DEAP is used only as unlabeled EEG for self-supervised flow matching.
    It is NOT used for supervised artifact-mask training.

Input expected:
    deap_preprocessed_csv/
        manifest.csv
        subjects/S01/trials/S01_trial-01/signals.csv.gz
        ...

Output:
    output_dir/
        x_float32.dat
        manifest.csv
        metadata.json

Each window has shape:
    59 channels x 2500 samples

Notes:
    - DEAP has 32 EEG channels sampled at 128 Hz.
    - The project model expects 59 channels sampled at 250 Hz.
    - We resample DEAP to 250 Hz.
    - We place exact matching EEG channel names into the 59-channel reference layout.
    - Missing reference channels are left as zeros.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_59_CHANNELS = [
    "AF3", "AF4", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "CFC7", "CFC5", "CFC3", "CFC1", "CFC2", "CFC4", "CFC6", "CFC8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "CCP7", "CCP5", "CCP3", "CCP1", "CCP2", "CCP4", "CCP6", "CCP8",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "PO1", "PO2", "O1", "O2",
]


def normalize_window(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    valid = np.any(np.abs(x) > eps, axis=1, keepdims=True)
    mean = np.where(valid, x.mean(axis=1, keepdims=True), 0.0)
    std = np.where(valid, x.std(axis=1, keepdims=True), 1.0)
    return (x - mean) / (std + eps)


def count_windows(samples: int, window_samples: int, stride_samples: int) -> int:
    if samples < window_samples:
        return 0
    return 1 + (samples - window_samples) // stride_samples


def resample_to_target(eeg: np.ndarray, source_sfreq: float, target_sfreq: float) -> np.ndarray:
    if abs(source_sfreq - target_sfreq) < 1e-6:
        return eeg.astype(np.float32, copy=False)
    source_samples = eeg.shape[1]
    duration_seconds = source_samples / float(source_sfreq)
    target_samples = int(round(duration_seconds * float(target_sfreq)))
    old_t = np.arange(source_samples, dtype=np.float64) / float(source_sfreq)
    new_t = np.arange(target_samples, dtype=np.float64) / float(target_sfreq)
    out = np.empty((eeg.shape[0], target_samples), dtype=np.float32)
    for ch in range(eeg.shape[0]):
        out[ch] = np.interp(new_t, old_t, eeg[ch]).astype(np.float32)
    return out


def read_deap_csv_gz(csv_path: Path, available_channels: list[str]) -> np.ndarray:
    usecols = ["sample", "time_seconds", *available_channels]
    chunks: list[np.ndarray] = []
    for chunk in pd.read_csv(csv_path, compression="gzip", usecols=usecols, chunksize=100_000):
        chunks.append(chunk[available_channels].to_numpy(dtype=np.float32).T)
    if not chunks:
        return np.empty((len(available_channels), 0), dtype=np.float32)
    return np.concatenate(chunks, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deap-dir",
        default=str(PROJECT_ROOT.parent / "packages" / "Final 3 Deap" / "deap_preprocessed_csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs_v6_flowmatching_study" / "window_datasets" / "deap_59ch_10s_250hz_flow_memmap"),
    )
    parser.add_argument("--target-sfreq", type=float, default=250.0)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--limit-recordings", type=int, default=0)
    args = parser.parse_args()

    deap_dir = Path(args.deap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_in = pd.read_csv(deap_dir / "manifest.csv")
    if args.limit_recordings:
        manifest_in = manifest_in.head(args.limit_recordings).copy()

    first_csv = deap_dir / str(manifest_in.iloc[0]["csv_file"]).replace("\\", "/")
    with gzip.open(first_csv, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")

    signal_columns = [c for c in header if c not in {"sample", "time_seconds"}]
    available_eeg_channels = [c for c in signal_columns if c in REFERENCE_59_CHANNELS]
    if not available_eeg_channels:
        raise RuntimeError("No DEAP EEG channels match the 59-channel reference layout.")

    channel_to_ref_index = {ch: REFERENCE_59_CHANNELS.index(ch) for ch in available_eeg_channels}

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
        raise RuntimeError("No DEAP flow-matching windows were created.")

    x_shape = (int(total_windows), len(REFERENCE_59_CHANNELS), window_samples)
    x_path = output_dir / "x_float32.dat"
    x_mem = np.memmap(x_path, dtype="float32", mode="w+", shape=x_shape)

    rows = []
    skipped = []
    global_index = 0
    for row, source_sfreq, n_windows in recording_infos:
        subject = str(row["subject_id"])
        trial_id = str(row["trial_id"])
        csv_path = deap_dir / str(row["csv_file"]).replace("\\", "/")
        print(f"Processing DEAP {subject} {trial_id}...", flush=True)

        try:
            eeg_source = read_deap_csv_gz(csv_path, available_eeg_channels)
        except Exception as exc:
            skipped.append({"subject": subject, "trial": trial_id, "csv_file": str(csv_path), "reason": f"{type(exc).__name__}: {exc}"})
            warnings.warn(f"Skipping unreadable DEAP trial {subject} {trial_id}: {exc}", RuntimeWarning)
            continue

        eeg_250 = resample_to_target(eeg_source, source_sfreq=source_sfreq, target_sfreq=args.target_sfreq)

        written = 0
        for window_index in range(n_windows):
            start = window_index * stride_samples
            stop = start + window_samples
            if stop > eeg_250.shape[1]:
                continue
            full = np.zeros((len(REFERENCE_59_CHANNELS), window_samples), dtype=np.float32)
            for src_i, ch in enumerate(available_eeg_channels):
                full[channel_to_ref_index[ch]] = eeg_250[src_i, start:stop]
            x_mem[global_index] = normalize_window(full).astype(np.float32)
            rows.append(
                {
                    "index": global_index,
                    "source_dataset": "deap",
                    "subject": subject,
                    "run": int(str(row["trial_id"]).split("-")[-1]) if "-" in str(row["trial_id"]) else 0,
                    "recording_id": trial_id,
                    "recording_kind": "emotion_trial_unlabeled_for_flow",
                    "window_index_in_recording": window_index,
                    "window_start_sample_250hz": start,
                    "window_stop_sample_250hz": stop,
                    "window_start_seconds": start / args.target_sfreq,
                    "window_stop_seconds": stop / args.target_sfreq,
                    "source_sampling_frequency_hz": source_sfreq,
                    "sampling_frequency_hz": args.target_sfreq,
                    "window_duration_seconds": args.window_sec,
                    "channels": len(REFERENCE_59_CHANNELS),
                    "available_source_eeg_channels": len(available_eeg_channels),
                    "has_artifact_label": False,
                    "use": "flow_matching_pretraining_only",
                }
            )
            global_index += 1
            written += 1
        print(f"  windows written: {written}", flush=True)

    x_mem.flush()
    if not rows:
        raise RuntimeError("No DEAP windows were written.")

    out_manifest = output_dir / "manifest.csv"
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset_name": "DEAP curated 59-channel EEG windows for flow matching",
        "source_prepared_folder": str(deap_dir),
        "purpose": "external unlabeled EEG pretraining only",
        "supervised_artifact_labels_available": False,
        "x_shape": [int(global_index), len(REFERENCE_59_CHANNELS), window_samples],
        "x_shape_allocated_on_disk": list(x_shape),
        "x_shape_per_window": [len(REFERENCE_59_CHANNELS), window_samples],
        "reference_channel_names": REFERENCE_59_CHANNELS,
        "available_deap_channels_mapped_by_name": available_eeg_channels,
        "missing_reference_channels_filled_with_zero": [ch for ch in REFERENCE_59_CHANNELS if ch not in available_eeg_channels],
        "source_sampling_frequency_hz": "from manifest, typically 128 Hz",
        "target_sampling_frequency_hz": args.target_sfreq,
        "resampling_note": "DEAP is linearly interpolated from 128 Hz to 250 Hz.",
        "window_duration_seconds": args.window_sec,
        "stride_seconds": args.stride_sec,
        "total_windows": int(global_index),
        "recordings_used": int(len(recording_infos) - len(skipped)),
        "recordings_skipped": int(len(skipped)),
        "skipped_recordings": skipped,
        "curation_limitation": "Only exact matching EEG channel names are placed into the 59-channel reference layout; other channels remain zero. Used only for flow pretraining.",
        "files": {"x_memmap": str(x_path), "manifest": str(out_manifest)},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved DEAP flow-matching memmap dataset:", flush=True)
    print(output_dir, flush=True)
    print("x shape:", [int(global_index), len(REFERENCE_59_CHANNELS), window_samples], flush=True)
    print("mapped channels:", len(available_eeg_channels), flush=True)
    print("skipped recordings:", len(skipped), flush=True)


if __name__ == "__main__":
    main()
