"""
Build a curated 59-channel CHB-MIT EEG memmap dataset for flow-matching pretraining.

Purpose:
    CHB-MIT is used only as unlabeled EEG for self-supervised flow matching.
    It is NOT used for supervised artifact-mask training.

Important:
    CHB-MIT chb01 here has 23 bipolar EEG derivations, not the same 59-channel
    scalp montage as PhysioMotion/BCI. For flow matching only, we preserve EEG
    waveform information by placing the 23 bipolar channels into the first 23
    positions of the 59-channel reference tensor and leaving the rest as zeros.

This is intentionally described as a curated waveform-only external EEG source.
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


def read_chbmit_csv_gz(csv_path: Path, channels: list[str]) -> np.ndarray:
    usecols = ["sample", "time_seconds", *channels]
    chunks: list[np.ndarray] = []
    for chunk in pd.read_csv(csv_path, compression="gzip", usecols=usecols, chunksize=100_000):
        chunks.append(chunk[channels].to_numpy(dtype=np.float32).T)
    if not chunks:
        return np.empty((len(channels), 0), dtype=np.float32)
    return np.concatenate(chunks, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chbmit-dir",
        default=str(PROJECT_ROOT.parent / "packages" / "Final 5 CHBMIT" / "chbmit_chb01_csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs_v6_flowmatching_study" / "window_datasets" / "chbmit_chb01_59ch_10s_250hz_flow_memmap"),
    )
    parser.add_argument("--target-sfreq", type=float, default=250.0)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--limit-recordings", type=int, default=0)
    args = parser.parse_args()

    chbmit_dir = Path(args.chbmit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_in = pd.read_csv(chbmit_dir / "manifest.csv")
    if args.limit_recordings:
        manifest_in = manifest_in.head(args.limit_recordings).copy()

    first_csv = chbmit_dir / str(manifest_in.iloc[0]["csv_file"]).replace("\\", "/")
    with gzip.open(first_csv, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")
    source_channels = [c for c in header if c not in {"sample", "time_seconds"}]
    if not source_channels:
        raise RuntimeError("No CHB-MIT EEG channels found.")
    if len(source_channels) > len(REFERENCE_59_CHANNELS):
        raise RuntimeError("CHB-MIT source has more channels than the 59-channel target tensor.")

    slot_mapping = {source_ch: REFERENCE_59_CHANNELS[i] for i, source_ch in enumerate(source_channels)}

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
        raise RuntimeError("No CHB-MIT flow-matching windows were created.")

    x_shape = (int(total_windows), len(REFERENCE_59_CHANNELS), window_samples)
    x_path = output_dir / "x_float32.dat"
    x_mem = np.memmap(x_path, dtype="float32", mode="w+", shape=x_shape)

    rows = []
    skipped = []
    global_index = 0
    for row, source_sfreq, n_windows in recording_infos:
        subject = str(row["subject_id"])
        run_id = str(row["run_id"])
        csv_path = chbmit_dir / str(row["csv_file"]).replace("\\", "/")
        print(f"Processing CHB-MIT {subject} {run_id}...", flush=True)

        try:
            eeg_source = read_chbmit_csv_gz(csv_path, source_channels)
        except Exception as exc:
            skipped.append({"subject": subject, "run": run_id, "csv_file": str(csv_path), "reason": f"{type(exc).__name__}: {exc}"})
            warnings.warn(f"Skipping unreadable CHB-MIT recording {subject} {run_id}: {exc}", RuntimeWarning)
            continue

        eeg_250 = resample_to_target(eeg_source, source_sfreq=source_sfreq, target_sfreq=args.target_sfreq)

        written = 0
        for window_index in range(n_windows):
            start = window_index * stride_samples
            stop = start + window_samples
            if stop > eeg_250.shape[1]:
                continue
            full = np.zeros((len(REFERENCE_59_CHANNELS), window_samples), dtype=np.float32)
            full[: len(source_channels)] = eeg_250[:, start:stop]
            x_mem[global_index] = normalize_window(full).astype(np.float32)
            rows.append(
                {
                    "index": global_index,
                    "source_dataset": "chbmit_chb01",
                    "subject": subject,
                    "run": int(str(run_id).split("_")[-1]) if "_" in str(run_id) else 0,
                    "recording_id": run_id,
                    "recording_kind": str(row.get("run_type", "unknown")),
                    "window_index_in_recording": window_index,
                    "window_start_sample_250hz": start,
                    "window_stop_sample_250hz": stop,
                    "window_start_seconds": start / args.target_sfreq,
                    "window_stop_seconds": stop / args.target_sfreq,
                    "source_sampling_frequency_hz": source_sfreq,
                    "sampling_frequency_hz": args.target_sfreq,
                    "window_duration_seconds": args.window_sec,
                    "channels": len(REFERENCE_59_CHANNELS),
                    "available_source_bipolar_channels": len(source_channels),
                    "has_artifact_label": False,
                    "use": "flow_matching_pretraining_only",
                }
            )
            global_index += 1
            written += 1
        print(f"  windows written: {written}", flush=True)

    x_mem.flush()
    if not rows:
        raise RuntimeError("No CHB-MIT windows were written.")

    out_manifest = output_dir / "manifest.csv"
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset_name": "CHB-MIT chb01 curated 59-channel EEG windows for flow matching",
        "source_prepared_folder": str(chbmit_dir),
        "purpose": "external unlabeled EEG pretraining only",
        "supervised_artifact_labels_available": False,
        "x_shape": [int(global_index), len(REFERENCE_59_CHANNELS), window_samples],
        "x_shape_allocated_on_disk": list(x_shape),
        "x_shape_per_window": [len(REFERENCE_59_CHANNELS), window_samples],
        "reference_channel_names": REFERENCE_59_CHANNELS,
        "source_bipolar_channels": source_channels,
        "slot_mapping_source_to_reference_tensor_position": slot_mapping,
        "source_sampling_frequency_hz": "from manifest, typically 256 Hz",
        "target_sampling_frequency_hz": args.target_sfreq,
        "resampling_note": "CHB-MIT is linearly interpolated from 256 Hz to 250 Hz.",
        "window_duration_seconds": args.window_sec,
        "stride_seconds": args.stride_sec,
        "total_windows": int(global_index),
        "recordings_used": int(len(recording_infos) - len(skipped)),
        "recordings_skipped": int(len(skipped)),
        "skipped_recordings": skipped,
        "curation_limitation": "Bipolar CHB-MIT channels are slot-filled into the 59-channel tensor. Used only for flow pretraining.",
        "files": {"x_memmap": str(x_path), "manifest": str(out_manifest)},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved CHB-MIT flow-matching memmap dataset:", flush=True)
    print(output_dir, flush=True)
    print("x shape:", [int(global_index), len(REFERENCE_59_CHANNELS), window_samples], flush=True)
    print("source bipolar channels:", len(source_channels), flush=True)
    print("skipped recordings:", len(skipped), flush=True)


if __name__ == "__main__":
    main()
