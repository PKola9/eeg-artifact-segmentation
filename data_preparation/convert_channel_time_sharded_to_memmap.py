"""
Convert Version 2 channel-time sharded EEG dataset into fast memory-mapped arrays.

Why this exists:
    The sharded Version 2 dataset stores one compressed NPZ file per subject/run.
    That is good for safe storage, but slow for GPU training because every batch
    repeatedly opens and decompresses NPZ files.

    A memmap dataset stores the same x and y arrays in flat binary files:
        x_float32.dat: windows x 59 channels x 2500 samples
        y_float32.dat: windows x 59 channels x 2500 samples

    The scientific data are unchanged. Only the storage format changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs_v2_channel_time" / "window_datasets"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sharded-dir",
        default=str(OUTPUT_ROOT / "full_sub30_runs6_channel_time_10s_250hz_sharded"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT / "full_sub30_runs6_channel_time_10s_250hz_memmap"),
    )
    args = parser.parse_args()

    sharded_dir = Path(args.sharded_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(sharded_dir / "manifest.csv")
    with (sharded_dir / "metadata.json").open("r", encoding="utf-8") as f:
        source_meta = json.load(f)

    n_windows = int(len(manifest))
    x_channels, x_samples = [int(v) for v in source_meta["x_shape_per_window"]]
    y_channels, y_samples = [int(v) for v in source_meta["y_shape_per_window"]]

    x_path = output_dir / "x_float32.dat"
    y_path = output_dir / "y_float32.dat"

    x_mem = np.memmap(x_path, dtype="float32", mode="w+", shape=(n_windows, x_channels, x_samples))
    y_mem = np.memmap(y_path, dtype="float32", mode="w+", shape=(n_windows, y_channels, y_samples))

    print("Creating Version 2 channel-time memmap dataset", flush=True)
    print("source:", sharded_dir, flush=True)
    print("output:", output_dir, flush=True)
    print("x shape:", (n_windows, x_channels, x_samples), flush=True)
    print("y shape:", (n_windows, y_channels, y_samples), flush=True)

    global_start = 0
    shard_column = "shard_name" if "shard_name" in manifest.columns else "shard_file"
    local_column = "shard_local_index" if "shard_local_index" in manifest.columns else "local_window_index"

    for shard_name, group in manifest.groupby(shard_column, sort=False):
        shard_name = str(shard_name).replace("\\", "/")
        shard_path = sharded_dir / "shards" / shard_name
        if not shard_path.exists():
            shard_path = sharded_dir / shard_name
        data = np.load(shard_path)

        local_indices = group[local_column].astype(int).to_numpy()
        count = len(local_indices)
        global_stop = global_start + count

        x_mem[global_start:global_stop] = data["x"][local_indices].astype(np.float32)
        y_mem[global_start:global_stop] = data["y"][local_indices].astype(np.float32)

        print(f"copied {shard_name}: {count} windows", flush=True)
        global_start = global_stop

    x_mem.flush()
    y_mem.flush()

    manifest_out = output_dir / "manifest.csv"
    shutil.copy2(sharded_dir / "manifest.csv", manifest_out)

    metadata = dict(source_meta)
    metadata.update(
        {
            "storage": "memmap float32 arrays",
            "source_sharded_dir": str(sharded_dir),
            "dataset_dir": str(output_dir),
            "manifest": str(manifest_out),
            "x_memmap": str(x_path),
            "y_memmap": str(y_path),
            "x_shape": [n_windows, x_channels, x_samples],
            "y_shape": [n_windows, y_channels, y_samples],
            "note": "Scientific values are identical to the sharded dataset; only storage format changed for faster GPU training.",
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Version 2 memmap conversion complete.", flush=True)
    print(output_dir, flush=True)


if __name__ == "__main__":
    main()
