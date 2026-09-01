"""
Build sample-level artifact mask dataset directly from original PhysioMotion EDF files.

Why this script exists:
    The previous CSV-based builder can only use full-recording CSVs that already exist.
    This script reads the original EDF recordings directly, so we can extend to more
    subjects and runs.

Output:
    x: windows x channels x time
    y: windows x time

Default:
    small safe expansion: a few subjects/runs, 10-second windows, 250 Hz.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = WORKSPACE_ROOT / "packages" / "Physiomotion Dataset" / "Original Dataset"
ANNOTATION_ROOT = DATASET_ROOT / "derivatives" / "Manual_Annotations"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "window_datasets"

BASELINE_LABELS = {"open_base", "close_base"}


def natural_subject_key(path: Path) -> int:
    match = re.search(r"sub-(\d+)", path.name)
    return int(match.group(1)) if match else 10**9


def subject_to_manual_name(subject: str) -> str:
    return subject.replace("-", "")


def make_paths(subject: str, run: int) -> tuple[Path, Path, Path]:
    run_bids = f"{subject}_task-artifact_run-{run:02d}"
    eeg_dir = DATASET_ROOT / subject / "eeg"
    edf = eeg_dir / f"{run_bids}_eeg.edf"
    metadata_json = eeg_dir / f"{run_bids}_eeg.json"
    manual_csv = ANNOTATION_ROOT / f"{subject_to_manual_name(subject)}_run{run:02d}.csv"
    return edf, metadata_json, manual_csv


def _parse_edf_ascii(raw: bytes) -> str:
    return raw.decode("ascii", errors="ignore").strip()


def read_edf_downsampled(edf_path: Path, downsample_to_hz: float) -> tuple[np.ndarray, list[str], float, float]:
    """
    Minimal EDF reader for PhysioMotion EDF files.

    Returns:
        eeg: channels x time, float32 physical values
        channel_names
        effective_sfreq
        duration_seconds

    This excludes the "EDF Annotations" signal.
    """
    with edf_path.open("rb") as f:
        fixed = f.read(256)
        header_bytes = int(_parse_edf_ascii(fixed[184:192]))
        n_records = int(_parse_edf_ascii(fixed[236:244]))
        record_duration = float(_parse_edf_ascii(fixed[244:252]))
        n_signals = int(_parse_edf_ascii(fixed[252:256]))

        signal_header = f.read(256 * n_signals)

        offset = 0

        def read_field(width: int) -> list[str]:
            nonlocal offset
            vals = [
                _parse_edf_ascii(signal_header[offset + i * width : offset + (i + 1) * width])
                for i in range(n_signals)
            ]
            offset += width * n_signals
            return vals

        labels = read_field(16)
        _ = read_field(80)  # transducer
        phys_dim = read_field(8)
        phys_min = np.array([float(v or 0) for v in read_field(8)], dtype=np.float64)
        phys_max = np.array([float(v or 0) for v in read_field(8)], dtype=np.float64)
        dig_min = np.array([float(v or 0) for v in read_field(8)], dtype=np.float64)
        dig_max = np.array([float(v or 0) for v in read_field(8)], dtype=np.float64)
        _ = read_field(80)  # prefilter
        samples_per_record = np.array([int(v or 0) for v in read_field(8)], dtype=np.int64)
        _ = read_field(32)  # reserved

        # Move to data start defensively.
        f.seek(header_bytes)

        eeg_indices = [
            i
            for i, label in enumerate(labels)
            if label.lower() != "edf annotations" and samples_per_record[i] > 0
        ]
        if not eeg_indices:
            raise ValueError(f"No EEG signals found in {edf_path}")

        original_sfreq = samples_per_record[eeg_indices[0]] / record_duration
        downsample_factor = max(1, int(round(original_sfreq / downsample_to_hz)))
        effective_sfreq = original_sfreq / downsample_factor
        kept_per_record = int(np.ceil(samples_per_record[eeg_indices[0]] / downsample_factor))
        total_kept = n_records * kept_per_record

        eeg = np.empty((len(eeg_indices), total_kept), dtype=np.float32)
        eeg_row_for_signal = {sig_idx: row_idx for row_idx, sig_idx in enumerate(eeg_indices)}

        scales = (phys_max - phys_min) / np.maximum(dig_max - dig_min, 1)

        write_start = 0
        for _record in range(n_records):
            record_kept = None
            for sig_idx in range(n_signals):
                n = int(samples_per_record[sig_idx])
                raw = np.frombuffer(f.read(n * 2), dtype="<i2").astype(np.float32)
                if sig_idx not in eeg_row_for_signal:
                    continue

                physical = (raw - dig_min[sig_idx]) * scales[sig_idx] + phys_min[sig_idx]
                physical = physical[::downsample_factor]
                row_idx = eeg_row_for_signal[sig_idx]
                if record_kept is None:
                    record_kept = len(physical)
                eeg[row_idx, write_start : write_start + len(physical)] = physical

            if record_kept is None:
                raise ValueError("No EEG samples read in record.")
            write_start += record_kept

        eeg = eeg[:, :write_start]
        channel_names = [labels[i] for i in eeg_indices]
        duration_seconds = n_records * record_duration
        return eeg, channel_names, effective_sfreq, duration_seconds


def create_mask(annotations: pd.DataFrame, duration_seconds: float, sfreq: float) -> np.ndarray:
    n_samples = int(np.floor(duration_seconds * sfreq))
    mask = np.zeros(n_samples, dtype=np.float32)
    artifact_rows = annotations[~annotations["label"].astype(str).isin(BASELINE_LABELS)].copy()

    for row in artifact_rows.itertuples(index=False):
        start = max(0.0, float(row.start_time))
        stop = min(duration_seconds, float(row.stop_time))
        if stop <= start:
            continue
        start_idx = max(0, min(n_samples, int(np.floor(start * sfreq))))
        stop_idx = max(0, min(n_samples, int(np.ceil(stop * sfreq))))
        mask[start_idx:stop_idx] = 1.0

    return mask


def normalize_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-8
    return (window - mean) / std


def extract_windows(
    eeg: np.ndarray,
    mask: np.ndarray,
    window_samples: int,
    stride_samples: int,
    keep_clean_fraction: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    windows_x: list[np.ndarray] = []
    windows_y: list[np.ndarray] = []
    rows: list[dict] = []

    usable = min(eeg.shape[1], mask.shape[0])
    clean_seen = 0
    clean_kept = 0

    for start in range(0, usable - window_samples + 1, stride_samples):
        stop = start + window_samples
        y = mask[start:stop]
        artifact_fraction = float(y.mean())

        if artifact_fraction == 0:
            clean_seen += 1
            if keep_clean_fraction <= 0:
                continue
            if (clean_kept / max(clean_seen, 1)) > keep_clean_fraction:
                continue
            clean_kept += 1

        x = normalize_window(eeg[:, start:stop]).astype(np.float32)
        windows_x.append(x)
        windows_y.append(y.astype(np.float32))
        rows.append(
            {
                "window_start_sample": start,
                "window_stop_sample": stop,
                "artifact_fraction": artifact_fraction,
            }
        )

    return windows_x, windows_y, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-subjects", type=int, default=3)
    parser.add_argument("--max-runs-per-subject", type=int, default=2)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--downsample-to-hz", type=float, default=250.0)
    parser.add_argument("--keep-clean-fraction", type=float, default=0.25)
    parser.add_argument("--output-name", default="")
    args = parser.parse_args()

    subject_dirs = sorted(
        [p for p in DATASET_ROOT.glob("sub-*") if p.is_dir()],
        key=natural_subject_key,
    )[: args.max_subjects]

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    manifest_rows: list[dict] = []
    channel_names: list[str] | None = None
    processed_runs = []

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        for run in range(1, args.max_runs_per_subject + 1):
            edf_path, metadata_json, manual_csv = make_paths(subject, run)
            if not edf_path.exists() or not metadata_json.exists() or not manual_csv.exists():
                print(f"Skipping missing {subject} run {run:02d}")
                continue

            print(f"Processing {subject} run {run:02d} from EDF...")
            annotations = pd.read_csv(manual_csv)
            eeg, run_channel_names, sfreq, duration_seconds = read_edf_downsampled(
                edf_path,
                args.downsample_to_hz,
            )

            if channel_names is None:
                channel_names = run_channel_names
            elif channel_names != run_channel_names:
                raise ValueError(f"Channel mismatch in {edf_path}")

            mask = create_mask(annotations, duration_seconds, sfreq)
            window_samples = int(round(args.window_sec * sfreq))
            stride_samples = int(round(args.stride_sec * sfreq))

            windows_x, windows_y, rows = extract_windows(
                eeg=eeg,
                mask=mask,
                window_samples=window_samples,
                stride_samples=stride_samples,
                keep_clean_fraction=args.keep_clean_fraction,
            )

            for local_idx, row in enumerate(rows):
                row.update(
                    {
                        "subject": subject,
                        "run": run,
                        "local_window_index": local_idx,
                        "effective_sampling_frequency_hz": sfreq,
                        "window_seconds": args.window_sec,
                        "stride_seconds": args.stride_sec,
                    }
                )
                manifest_rows.append(row)

            all_x.extend(windows_x)
            all_y.extend(windows_y)
            processed_runs.append({"subject": subject, "run": run, "windows": len(windows_x)})
            print(f"  windows kept: {len(windows_x)}")

    if not all_x:
        raise RuntimeError("No windows were created.")

    x = np.stack(all_x).astype(np.float32)
    y = np.stack(all_y).astype(np.float32)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    name = args.output_name or (
        f"edf_sub{args.max_subjects}_runs{args.max_runs_per_subject}_"
        f"win{int(args.window_sec)}s_stride{int(args.stride_sec)}s_{int(args.downsample_to_hz)}hz"
    )
    npz_path = OUTPUT_ROOT / f"{name}.npz"
    metadata_path = OUTPUT_ROOT / f"{name}_metadata.json"
    manifest_path = OUTPUT_ROOT / f"{name}_manifest.csv"

    np.savez_compressed(npz_path, x=x, y=y)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    out_meta = {
        "task": "sample-level binary artifact detection",
        "source": "original EDF files",
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "channel_names": channel_names,
        "artifact_window_count": int((y.mean(axis=1) > 0).sum()),
        "clean_window_count": int((y.mean(axis=1) == 0).sum()),
        "mean_artifact_fraction": float(y.mean()),
        "baseline_labels_excluded": sorted(BASELINE_LABELS),
        "processed_runs": processed_runs,
        "settings": vars(args),
        "files": {
            "npz": str(npz_path),
            "metadata": str(metadata_path),
            "manifest": str(manifest_path),
        },
    }
    metadata_path.write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    print("Saved dataset:")
    print(npz_path)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("artifact windows:", out_meta["artifact_window_count"])
    print("clean windows:", out_meta["clean_window_count"])
    print("mean artifact fraction:", out_meta["mean_artifact_fraction"])


if __name__ == "__main__":
    main()

