"""
Build a Version 2 channel-time EEG artifact dataset for one PhysioMotion run.

Version 1 target:
    x: windows x channels x time
    y: windows x time

Version 2 target:
    x: windows x channels x time
    y: windows x channels x time

The annotation files often use electrode-pair names such as "Fp1-F7".
For Version 2 masks, this script maps:
    Fp1-F7 -> mark Fp1 and F7 as artifact during the interval
    ALL    -> mark all channels as artifact during the interval
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = WORKSPACE_ROOT / "packages" / "Physiomotion Dataset" / "Original Dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs_v2_channel_time" / "window_datasets"

DATA_PREP_DIR = PROJECT_ROOT / "data_preparation"
sys.path.insert(0, str(DATA_PREP_DIR))

from build_window_mask_dataset_from_edf import (  # noqa: E402
    ANNOTATION_ROOT,
    BASELINE_LABELS,
    make_paths,
    normalize_window,
    read_edf_downsampled,
)


def parse_subject_number(subject: str) -> int:
    match = re.search(r"sub-(\d+)", subject)
    if not match:
        raise ValueError(f"Subject must look like sub-30, got {subject!r}")
    return int(match.group(1))


def annotation_channel_to_indices(annotation_channel: str, channel_to_index: dict[str, int]) -> list[int]:
    """
    Convert annotation channel string to EEG channel indices.

    Examples:
        "ALL"    -> all channels
        "Fp1-F7" -> [index(Fp1), index(F7)]
        "Cz"     -> [index(Cz)]
    """
    annotation_channel = str(annotation_channel).strip()
    if annotation_channel.upper() == "ALL":
        return list(channel_to_index.values())

    parts = [p.strip() for p in annotation_channel.split("-") if p.strip()]
    indices: list[int] = []
    for part in parts:
        if part in channel_to_index:
            idx = channel_to_index[part]
            if idx not in indices:
                indices.append(idx)

    if not indices:
        raise ValueError(f"Could not map annotation channel {annotation_channel!r} to EEG channels.")

    return indices


def create_channel_time_mask(
    annotations: pd.DataFrame,
    channel_names: list[str],
    duration_seconds: float,
    sfreq: float,
) -> tuple[np.ndarray, dict]:
    """
    Create y mask with shape:
        channels x samples

    0 = clean
    1 = artifact
    """
    n_samples = int(np.floor(duration_seconds * sfreq))
    mask = np.zeros((len(channel_names), n_samples), dtype=np.float32)
    channel_to_index = {name: idx for idx, name in enumerate(channel_names)}

    artifact_rows = annotations[~annotations["label"].astype(str).isin(BASELINE_LABELS)].copy()

    mapped_rows = 0
    all_rows = 0
    mapped_channel_marks = 0
    label_counts: dict[str, int] = {}
    channel_counts: dict[str, int] = {}

    for row in artifact_rows.itertuples(index=False):
        start = max(0.0, float(row.start_time))
        stop = min(duration_seconds, float(row.stop_time))
        if stop <= start:
            continue

        start_idx = max(0, min(n_samples, int(np.floor(start * sfreq))))
        stop_idx = max(0, min(n_samples, int(np.ceil(stop * sfreq))))
        if stop_idx <= start_idx:
            continue

        annotation_channel = str(row.channel)
        channel_indices = annotation_channel_to_indices(annotation_channel, channel_to_index)
        mask[channel_indices, start_idx:stop_idx] = 1.0

        mapped_rows += 1
        mapped_channel_marks += len(channel_indices)
        if annotation_channel.upper() == "ALL":
            all_rows += 1

        label = str(row.label)
        label_counts[label] = label_counts.get(label, 0) + 1
        channel_counts[annotation_channel] = channel_counts.get(annotation_channel, 0) + 1

    report = {
        "artifact_annotation_rows_used": mapped_rows,
        "all_channel_rows": all_rows,
        "channel_specific_rows": mapped_rows - all_rows,
        "mapped_channel_marks": mapped_channel_marks,
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True)),
        "top_annotation_channels": dict(
            sorted(channel_counts.items(), key=lambda item: item[1], reverse=True)[:30]
        ),
    }
    return mask, report


def extract_channel_time_windows(
    eeg: np.ndarray,
    mask: np.ndarray,
    window_samples: int,
    stride_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    windows_x: list[np.ndarray] = []
    windows_y: list[np.ndarray] = []
    rows: list[dict] = []

    usable = min(eeg.shape[1], mask.shape[1])
    for start in range(0, usable - window_samples + 1, stride_samples):
        stop = start + window_samples
        x = normalize_window(eeg[:, start:stop]).astype(np.float32)
        y = mask[:, start:stop].astype(np.float32)
        windows_x.append(x)
        windows_y.append(y)
        rows.append(
            {
                "window_start_sample": start,
                "window_stop_sample": stop,
                "artifact_fraction_channel_time": float(y.mean()),
                "artifact_channels_count": int((y.mean(axis=1) > 0).sum()),
                "is_artifact_window": bool(y.max() > 0),
            }
        )

    return np.stack(windows_x), np.stack(windows_y), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-30")
    parser.add_argument("--run", type=int, default=6)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--downsample-to-hz", type=float, default=250.0)
    parser.add_argument("--output-name", default="")
    args = parser.parse_args()

    edf_path, metadata_json, manual_csv = make_paths(args.subject, args.run)
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)
    if not manual_csv.exists():
        raise FileNotFoundError(manual_csv)

    print(f"Reading EDF: {edf_path}")
    eeg, channel_names, sfreq, duration_seconds = read_edf_downsampled(
        edf_path,
        args.downsample_to_hz,
    )

    print(f"Reading annotations: {manual_csv}")
    annotations = pd.read_csv(manual_csv)
    channel_time_mask, mask_report = create_channel_time_mask(
        annotations=annotations,
        channel_names=channel_names,
        duration_seconds=duration_seconds,
        sfreq=sfreq,
    )

    window_samples = int(round(args.window_sec * sfreq))
    stride_samples = int(round(args.stride_sec * sfreq))
    x, y, rows = extract_channel_time_windows(
        eeg=eeg,
        mask=channel_time_mask,
        window_samples=window_samples,
        stride_samples=stride_samples,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.subject}_run-{args.run:02d}_channel_time_v2_test"
    output_dir = OUTPUT_ROOT / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "dataset.npz"
    manifest_path = output_dir / "manifest.csv"
    metadata_path = output_dir / "metadata.json"

    np.savez_compressed(npz_path, x=x, y=y)

    manifest = pd.DataFrame(rows)
    manifest.insert(0, "subject", args.subject)
    manifest.insert(1, "run", args.run)
    manifest["effective_sampling_frequency_hz"] = sfreq
    manifest["window_seconds"] = args.window_sec
    manifest["stride_seconds"] = args.stride_sec
    manifest.to_csv(manifest_path, index=False)

    metadata = {
        "task": "Version 2 channel-time binary EEG artifact segmentation",
        "input_shape_description": "windows x channels x time",
        "target_shape_description": "windows x channels x time",
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "subject": args.subject,
        "run": args.run,
        "edf_path": str(edf_path),
        "manual_annotation_file": str(manual_csv),
        "channel_names": channel_names,
        "sampling_frequency_hz": sfreq,
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "window_seconds": args.window_sec,
        "stride_seconds": args.stride_sec,
        "artifact_windows": int(manifest["is_artifact_window"].sum()),
        "clean_windows": int((~manifest["is_artifact_window"]).sum()),
        "mean_artifact_fraction_channel_time": float(y.mean()),
        "mask_report": mask_report,
        "files": {
            "dataset_npz": str(npz_path),
            "manifest_csv": str(manifest_path),
            "metadata_json": str(metadata_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved Version 2 one-run channel-time dataset:")
    print(output_dir)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("artifact windows:", metadata["artifact_windows"])
    print("clean windows:", metadata["clean_windows"])
    print("mean artifact fraction channel-time:", metadata["mean_artifact_fraction_channel_time"])


if __name__ == "__main__":
    main()
