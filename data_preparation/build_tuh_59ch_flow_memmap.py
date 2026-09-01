"""
Build a curated 59-channel TUH EEG memmap dataset for flow-matching pretraining.

Purpose:
    TUH is used here as an additional external EEG source for self-supervised
    flow-matching pretraining. The final supervised artifact segmentation
    training remains on the original PhysioMotion artifact-mask dataset so that
    M1/M2/M3/M4/M5 comparisons are fair.

Important:
    The TUH files in this package contain EDF signals plus artifact annotation
    CSVs. TUH uses clinical/bipolar montage names that do not exactly match the
    59-channel PhysioMotion tensor layout. For the current comparison, the TUH
    EEG waveforms are therefore slot-filled into a 59-channel tensor and used
    only for flow pretraining, not as supervised segmentation masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PREP_DIR = PROJECT_ROOT / "data_preparation"
sys.path.insert(0, str(DATA_PREP_DIR))

from build_window_mask_dataset_from_edf import read_edf_downsampled  # noqa: E402


REFERENCE_59_CHANNELS = [
    "AF3", "AF4", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "CFC7", "CFC5", "CFC3", "CFC1", "CFC2", "CFC4", "CFC6", "CFC8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "CCP7", "CCP5", "CCP3", "CCP1", "CCP2", "CCP4", "CCP6", "CCP8",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "PO1", "PO2", "O1", "O2",
]


def parse_edf_ascii(raw: bytes) -> str:
    return raw.decode("ascii", errors="ignore").strip()


def edf_duration_seconds(edf_path: Path) -> float:
    with edf_path.open("rb") as f:
        fixed = f.read(256)
    n_records = int(parse_edf_ascii(fixed[236:244]))
    record_duration = float(parse_edf_ascii(fixed[244:252]))
    return float(n_records * record_duration)


def count_windows(samples: int, window_samples: int, stride_samples: int) -> int:
    if samples < window_samples:
        return 0
    return 1 + (samples - window_samples) // stride_samples


def normalize_window(x: np.ndarray, eps: float = 1e-6, clip_value: float = 8.0) -> np.ndarray | None:
    """Robust per-channel normalization for heterogeneous TUH EDF signals.

    TUH contains recordings from different clinical systems and occasionally
    contains extreme values, infinities, or invalid numeric segments. For
    self-supervised flow pretraining, these corrupted windows should not be
    allowed to dominate MSE training. This function therefore normalizes in
    float64, rejects non-finite/corrupted windows, and clips extreme normalized
    amplitudes.
    """
    x64 = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x64).all():
        return None

    # Reject windows with implausibly large raw values before squaring/std.
    # This prevents overflow warnings and avoids poisoning the flow dataset.
    if float(np.max(np.abs(x64))) > 1e8:
        return None

    valid = np.any(np.abs(x64) > eps, axis=1, keepdims=True)
    mean = np.where(valid, x64.mean(axis=1, keepdims=True), 0.0)
    std = np.where(valid, x64.std(axis=1, keepdims=True), 1.0)
    z = (x64 - mean) / (std + eps)

    if not np.isfinite(z).all():
        return None

    z = np.clip(z, -clip_value, clip_value)
    return z.astype(np.float32)


def read_tuh_annotation_summary(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"annotation_file_exists": False, "annotation_rows": 0, "labels": {}}
    try:
        ann = pd.read_csv(csv_path, comment="#")
    except Exception as exc:
        return {
            "annotation_file_exists": True,
            "annotation_rows": 0,
            "labels": {},
            "annotation_read_error": f"{type(exc).__name__}: {exc}",
        }
    labels = {}
    if "label" in ann.columns:
        labels = {str(k): int(v) for k, v in ann["label"].value_counts().to_dict().items()}
    return {
        "annotation_file_exists": True,
        "annotation_rows": int(len(ann)),
        "labels": labels,
    }


def find_tuh_recordings(tuh_dir: Path, limit_recordings: int) -> list[tuple[Path, Path]]:
    edf_root = tuh_dir / "v3.0.1" / "edf"
    if not edf_root.exists():
        raise FileNotFoundError(f"Could not find TUH EDF root: {edf_root}")
    edfs = sorted(edf_root.rglob("*.edf"))
    if limit_recordings:
        edfs = edfs[:limit_recordings]
    return [(edf, edf.with_suffix(".csv")) for edf in edfs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tuh-dir",
        default=str(PROJECT_ROOT.parent / "packages" / "TUH"),
        help="Folder containing TUH/v3.0.1/edf or the extracted TUH folder itself.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs_v6_flowmatching_study" / "window_datasets" / "tuh_59ch_10s_250hz_flow_memmap"),
    )
    parser.add_argument("--target-sfreq", type=float, default=250.0)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--limit-recordings", type=int, default=0)
    args = parser.parse_args()

    tuh_dir = Path(args.tuh_dir)
    if (tuh_dir / "TUH").exists() and not (tuh_dir / "v3.0.1").exists():
        tuh_dir = tuh_dir / "TUH"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings = find_tuh_recordings(tuh_dir, args.limit_recordings)
    if not recordings:
        raise RuntimeError(f"No TUH EDF files found under {tuh_dir}")

    window_samples = int(round(args.window_sec * args.target_sfreq))
    stride_samples = int(round(args.stride_sec * args.target_sfreq))

    recording_infos = []
    total_windows = 0
    for edf_path, csv_path in recordings:
        try:
            duration = edf_duration_seconds(edf_path)
        except Exception as exc:
            warnings.warn(f"Skipping TUH EDF with unreadable header {edf_path}: {exc}", RuntimeWarning)
            continue
        target_samples = int(round(duration * args.target_sfreq))
        n_windows = count_windows(target_samples, window_samples, stride_samples)
        if n_windows <= 0:
            continue
        recording_infos.append((edf_path, csv_path, duration, n_windows))
        total_windows += n_windows

    if total_windows == 0:
        raise RuntimeError("No TUH flow-matching windows could be counted.")

    x_shape_allocated = (int(total_windows), len(REFERENCE_59_CHANNELS), window_samples)
    x_path = output_dir / "x_float32.dat"
    x_mem = np.memmap(x_path, dtype="float32", mode="w+", shape=x_shape_allocated)

    rows = []
    skipped = []
    label_counts: dict[str, int] = {}
    source_channel_names_first: list[str] | None = None
    global_index = 0

    for rec_i, (edf_path, csv_path, duration, n_windows) in enumerate(recording_infos, start=1):
        rel = edf_path.relative_to(tuh_dir)
        recording_id = edf_path.stem
        montage = edf_path.parent.name
        print(f"Processing TUH {rec_i}/{len(recording_infos)} {rel}...", flush=True)

        try:
            eeg, channel_names, sfreq, _duration_seconds = read_edf_downsampled(edf_path, downsample_to_hz=args.target_sfreq)
        except Exception as exc:
            skipped.append({"edf_file": str(edf_path), "reason": f"{type(exc).__name__}: {exc}"})
            warnings.warn(f"Skipping unreadable TUH EDF {edf_path}: {exc}", RuntimeWarning)
            continue

        if source_channel_names_first is None:
            source_channel_names_first = list(channel_names)

        source_channels = min(eeg.shape[0], len(REFERENCE_59_CHANNELS))
        ann_summary = read_tuh_annotation_summary(csv_path)
        for label, count in ann_summary.get("labels", {}).items():
            label_counts[label] = label_counts.get(label, 0) + int(count)

        written = 0
        skipped_bad_windows = 0
        for window_index in range(n_windows):
            start = window_index * stride_samples
            stop = start + window_samples
            if stop > eeg.shape[1]:
                continue
            full = np.zeros((len(REFERENCE_59_CHANNELS), window_samples), dtype=np.float32)
            full[:source_channels] = eeg[:source_channels, start:stop]
            normalized = normalize_window(full)
            if normalized is None:
                skipped_bad_windows += 1
                continue
            x_mem[global_index] = normalized
            rows.append(
                {
                    "index": global_index,
                    "source_dataset": "TUH_artifact_corpus",
                    "subject": f"tuh_{recording_id}",
                    "run": 0,
                    "recording_id": recording_id,
                    "recording_kind": montage,
                    "window_index_in_recording": window_index,
                    "window_start_sample_250hz": start,
                    "window_stop_sample_250hz": stop,
                    "window_start_seconds": start / args.target_sfreq,
                    "window_stop_seconds": stop / args.target_sfreq,
                    "source_sampling_frequency_hz": float(sfreq),
                    "sampling_frequency_hz": args.target_sfreq,
                    "window_duration_seconds": args.window_sec,
                    "channels": len(REFERENCE_59_CHANNELS),
                    "available_source_channels": int(eeg.shape[0]),
                    "source_channels_used": int(source_channels),
                    "annotation_file": str(csv_path),
                    "annotation_rows_in_recording": int(ann_summary.get("annotation_rows", 0)),
                    "has_artifact_label": False,
                    "use": "flow_matching_pretraining_only",
                }
            )
            global_index += 1
            written += 1
        if skipped_bad_windows:
            skipped.append(
                {
                    "edf_file": str(edf_path),
                    "reason": f"skipped {skipped_bad_windows} non-finite/extreme numeric windows",
                }
            )
        print(f"  windows written: {written}; bad windows skipped: {skipped_bad_windows}", flush=True)

    x_mem.flush()
    if not rows:
        raise RuntimeError("No TUH windows were written after skipping unreadable recordings.")

    out_manifest = output_dir / "manifest.csv"
    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset_name": "TUH artifact corpus curated 59-channel EEG windows for flow matching",
        "source_folder": str(tuh_dir),
        "purpose": "external EEG flow pretraining only for M5 comparison",
        "supervised_artifact_labels_available_in_source": True,
        "used_for_supervised_artifact_training": False,
        "reason_not_used_for_supervised_training": (
            "TUH uses a different clinical/bipolar montage and annotation scheme. "
            "For fair comparison with previous models, TUH is used only for flow pretraining; "
            "the supervised Split U-Net is still trained/evaluated on the same PhysioMotion artifact-mask dataset."
        ),
        "x_shape": [int(global_index), len(REFERENCE_59_CHANNELS), window_samples],
        "x_shape_allocated_on_disk": list(x_shape_allocated),
        "x_shape_per_window": [len(REFERENCE_59_CHANNELS), window_samples],
        "reference_channel_names": REFERENCE_59_CHANNELS,
        "first_recording_source_channel_names": source_channel_names_first or [],
        "target_sampling_frequency_hz": args.target_sfreq,
        "window_duration_seconds": args.window_sec,
        "stride_seconds": args.stride_sec,
        "total_windows": int(global_index),
        "recordings_seen": int(len(recording_infos)),
        "recordings_skipped": int(len(skipped)),
        "skipped_recordings": skipped,
        "source_annotation_label_counts_not_used_as_training_targets": dict(
            sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "curation_limitation": (
            "TUH channels are slot-filled into the 59-channel tensor. This is a waveform-only "
            "self-supervised pretraining source, not a supervised artifact-mask source."
        ),
        "files": {"x_memmap": str(x_path), "manifest": str(out_manifest)},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved TUH flow-matching memmap dataset:", flush=True)
    print(output_dir, flush=True)
    print("x shape:", [int(global_index), len(REFERENCE_59_CHANNELS), window_samples], flush=True)
    print("recordings skipped:", len(skipped), flush=True)
    print("top TUH source labels, not used as training targets:", dict(list(metadata["source_annotation_label_counts_not_used_as_training_targets"].items())[:10]), flush=True)


if __name__ == "__main__":
    main()
