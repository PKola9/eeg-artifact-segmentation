"""
Build Version 2 channel-time EEG artifact dataset from PhysioMotion EDF files.

This is the full-dataset scalable builder.

Why sharded:
    Version 2 targets are large:
        x: windows x channels x time
        y: windows x channels x time

    Saving all runs into one giant array is memory-heavy. Therefore this script
    saves one compressed NPZ shard per subject/run and creates a global manifest.

Output per shard:
    x: windows x 59 x 2500
    y: windows x 59 x 2500
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
    make_paths,
    natural_subject_key,
    read_edf_downsampled,
)
from build_channel_time_mask_dataset_from_edf_one_run import (  # noqa: E402
    create_channel_time_mask,
    extract_channel_time_windows,
)


def parse_run_list(value: str, max_runs: int) -> list[int]:
    if value:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return list(range(1, max_runs + 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-subjects", type=int, default=30)
    parser.add_argument("--max-runs-per-subject", type=int, default=6)
    parser.add_argument("--runs", default="", help="Optional comma-separated run list, e.g. 1,2,6")
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--downsample-to-hz", type=float, default=250.0)
    parser.add_argument("--output-name", default="full_sub30_runs6_channel_time_10s_250hz_sharded")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.output_name
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(
        [p for p in DATASET_ROOT.glob("sub-*") if p.is_dir()],
        key=natural_subject_key,
    )[: args.max_subjects]
    run_list = parse_run_list(args.runs, args.max_runs_per_subject)

    manifest_rows: list[dict] = []
    processed_runs: list[dict] = []
    channel_names: list[str] | None = None
    total_windows = 0
    artifact_windows = 0
    clean_windows = 0
    total_artifact_points = 0.0
    total_channel_time_points = 0.0
    aggregate_label_counts: dict[str, int] = {}
    aggregate_top_annotation_channels: dict[str, int] = {}

    for subject_dir in subject_dirs:
        subject = subject_dir.name
        for run in run_list:
            edf_path, _metadata_json, manual_csv = make_paths(subject, run)
            if not edf_path.exists() or not manual_csv.exists():
                print(f"Skipping missing {subject} run {run:02d}")
                continue

            print(f"Processing {subject} run {run:02d} for Version 2 channel-time masks...")
            eeg, run_channel_names, sfreq, duration_seconds = read_edf_downsampled(
                edf_path,
                args.downsample_to_hz,
            )
            annotations = pd.read_csv(manual_csv)

            if channel_names is None:
                channel_names = run_channel_names
            elif channel_names != run_channel_names:
                raise ValueError(f"Channel mismatch in {edf_path}")

            channel_time_mask, mask_report = create_channel_time_mask(
                annotations=annotations,
                channel_names=run_channel_names,
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

            shard_name = f"{subject}_run-{run:02d}.npz"
            shard_path = shard_dir / shard_name
            np.savez_compressed(shard_path, x=x.astype(np.float32), y=y.astype(np.float32))

            for local_idx, row in enumerate(rows):
                row.update(
                    {
                        "subject": subject,
                        "run": run,
                        "local_window_index": local_idx,
                        "shard_name": shard_name,
                        "shard_local_index": local_idx,
                        "effective_sampling_frequency_hz": sfreq,
                        "window_seconds": args.window_sec,
                        "stride_seconds": args.stride_sec,
                    }
                )
                manifest_rows.append(row)

            run_artifact_windows = int((y.max(axis=(1, 2)) > 0).sum())
            run_clean_windows = int((y.max(axis=(1, 2)) == 0).sum())
            processed_runs.append(
                {
                    "subject": subject,
                    "run": run,
                    "windows": int(x.shape[0]),
                    "artifact_windows": run_artifact_windows,
                    "clean_windows": run_clean_windows,
                    "shard_name": shard_name,
                }
            )

            total_windows += int(x.shape[0])
            artifact_windows += run_artifact_windows
            clean_windows += run_clean_windows
            total_artifact_points += float(y.sum())
            total_channel_time_points += float(y.size)

            for label, count in mask_report["label_counts"].items():
                aggregate_label_counts[label] = aggregate_label_counts.get(label, 0) + int(count)
            for ch, count in mask_report["top_annotation_channels"].items():
                aggregate_top_annotation_channels[ch] = aggregate_top_annotation_channels.get(ch, 0) + int(count)

            print(f"  saved {shard_name}: x={tuple(x.shape)} y={tuple(y.shape)}")

            del eeg, channel_time_mask, x, y

    if not manifest_rows:
        raise RuntimeError("No Version 2 channel-time windows were created.")

    manifest_path = output_dir / "manifest.csv"
    metadata_path = output_dir / "metadata.json"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    metadata = {
        "task": "Version 2 channel-time binary EEG artifact segmentation",
        "source": "original PhysioMotion EDF files",
        "sharded": True,
        "shard_dir": str(shard_dir),
        "total_windows": total_windows,
        "artifact_windows": artifact_windows,
        "clean_windows": clean_windows,
        "x_shape_per_window": [len(channel_names or []), window_samples],
        "y_shape_per_window": [len(channel_names or []), window_samples],
        "channel_names": channel_names,
        "mean_artifact_fraction_channel_time": total_artifact_points / max(total_channel_time_points, 1.0),
        "processed_runs": processed_runs,
        "label_counts": dict(sorted(aggregate_label_counts.items(), key=lambda item: item[1], reverse=True)),
        "top_annotation_channels": dict(
            sorted(aggregate_top_annotation_channels.items(), key=lambda item: item[1], reverse=True)[:50]
        ),
        "settings": vars(args),
        "files": {
            "manifest_csv": str(manifest_path),
            "metadata_json": str(metadata_path),
            "shards": str(shard_dir),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved Version 2 sharded dataset:")
    print(output_dir)
    print("total windows:", total_windows)
    print("artifact windows:", artifact_windows)
    print("clean windows:", clean_windows)
    print("x per window:", metadata["x_shape_per_window"])
    print("y per window:", metadata["y_shape_per_window"])
    print("mean artifact fraction channel-time:", metadata["mean_artifact_fraction_channel_time"])


if __name__ == "__main__":
    main()
