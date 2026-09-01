"""
Evaluate a saved flow-matching checkpoint on EEG-only memmap datasets.

Purpose
-------
This is used when a flow-matching run saved its best checkpoint but did not
save the full flowmatching_metrics.csv file. It reconstructs the same
train/validation/test split used during flow pretraining and reports MSE/RMSE
for the checkpoint.

Important
---------
Flow matching uses random Gaussian noise x0 and random time t during evaluation.
So this script fixes the random seed to make the recovered numbers repeatable.
The result should be reported as "re-evaluated checkpoint metrics".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(MODELS_DIR))

from flowmatching_channel_time import FlowMatchingChannelTimeUNet, count_parameters  # noqa: E402


class OneMemmapSource:
    def __init__(self, dataset_dir: Path, source_index: int):
        self.dataset_dir = Path(dataset_dir)
        self.source_index = source_index
        self.manifest = pd.read_csv(self.dataset_dir / "manifest.csv")
        with (self.dataset_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.x_shape = tuple(int(v) for v in self.metadata["x_shape"])
        self.x = np.memmap(self.dataset_dir / "x_float32.dat", dtype="float32", mode="r", shape=self.x_shape)

    @property
    def per_window_shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.x_shape[1:])


class MultiMemmapEEGOnlyDataset(Dataset):
    def __init__(self, dataset_dirs: list[Path], holdout_subject: str, holdout_run: int):
        self.sources = [OneMemmapSource(Path(p), i) for i, p in enumerate(dataset_dirs)]
        if not self.sources:
            raise RuntimeError("At least one dataset directory is required.")

        first_shape = self.sources[0].per_window_shape
        for src in self.sources:
            if src.per_window_shape != first_shape:
                raise RuntimeError(
                    f"Dataset shape mismatch. Expected {first_shape}, got {src.per_window_shape} from {src.dataset_dir}"
                )

        rows = []
        for src in self.sources:
            manifest = src.manifest.copy()
            if "subject" in manifest.columns and "run" in manifest.columns:
                holdout_mask = (manifest["subject"].astype(str) == holdout_subject) & (
                    manifest["run"].astype(int) == int(holdout_run)
                )
            else:
                holdout_mask = np.zeros(len(manifest), dtype=bool)

            for local_idx, row in manifest.iterrows():
                if bool(holdout_mask[local_idx]):
                    continue
                rows.append(
                    {
                        "source_index": src.source_index,
                        "local_index": int(local_idx),
                        "source_dataset_dir": str(src.dataset_dir),
                        "source_dataset": row.get("source_dataset", src.metadata.get("dataset_name", "unknown")),
                        "subject": row.get("subject", row.get("subject_id", "")),
                        "run": row.get("run", ""),
                        "recording_id": row.get("recording_id", ""),
                    }
                )
        if not rows:
            raise RuntimeError("No EEG windows available after holdout exclusion.")
        self.index_table = pd.DataFrame(rows)
        self.x_shape = (len(self.index_table), *first_shape)

    def __len__(self) -> int:
        return len(self.index_table)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.index_table.iloc[idx]
        src = self.sources[int(row["source_index"])]
        local_idx = int(row["local_index"])
        return torch.from_numpy(np.array(src.x[local_idx], dtype=np.float32, copy=True))


def make_random_split(n: int, val_fraction: float, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    test_count = int(round(n * test_fraction))
    val_count = int(round(n * val_fraction))
    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + val_count]
    train_idx = indices[test_count + val_count :]
    return train_idx, val_idx, test_idx


def flow_loss_batch(model: nn.Module, x1: torch.Tensor, device: torch.device) -> torch.Tensor:
    x1 = x1.to(device)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=device)
    xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target_velocity = x1 - x0
    pred_velocity = model(xt, t)
    return nn.functional.mse_loss(pred_velocity, target_velocity)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, seed: int) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for x1 in loader:
            loss = flow_loss_batch(model, x1, device)
            total_loss += float(loss.item()) * x1.shape[0]
            total_items += x1.shape[0]
    mse = total_loss / max(total_items, 1)
    return {"flow_mse": mse, "flow_rmse": math.sqrt(mse), "windows": int(total_items)}


def source_counts(dataset: MultiMemmapEEGOnlyDataset, indices: np.ndarray) -> dict:
    selected = dataset.index_table.iloc[indices]
    counts = selected["source_dataset_dir"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="append", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="recovered_flowmatching_checkpoint_metrics")
    parser.add_argument("--holdout-subject", default="sub-30")
    parser.add_argument("--holdout-run", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dirs = [Path(p) for p in args.dataset_dir]
    dataset = MultiMemmapEEGOnlyDataset(dataset_dirs, args.holdout_subject, args.holdout_run)
    train_idx, val_idx, test_idx = make_random_split(
        len(dataset), val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMatchingChannelTimeUNet().to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)

    print("Device:", device, flush=True)
    print("Checkpoint:", args.checkpoint, flush=True)
    print("Datasets:", [str(p) for p in dataset_dirs], flush=True)
    print("Combined x shape:", list(dataset.x_shape), flush=True)
    print("Train/validation/test windows:", len(train_idx), len(val_idx), len(test_idx), flush=True)
    print("Model parameters:", count_parameters(model), flush=True)
    print("Objective: MSE(predicted velocity, x1 - x0)", flush=True)

    train_metrics = evaluate(model, train_loader, device, seed=args.seed + 100)
    val_metrics = evaluate(model, val_loader, device, seed=args.seed + 200)
    test_metrics = evaluate(model, test_loader, device, seed=args.seed + 300)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "flowmatching_checkpoint_recovered_metrics.csv"
    summary_json = out_dir / "flowmatching_checkpoint_recovered_summary.json"

    row = {
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "train_flow_mse": train_metrics["flow_mse"],
        "train_flow_rmse": train_metrics["flow_rmse"],
        "validation_flow_mse": val_metrics["flow_mse"],
        "validation_flow_rmse": val_metrics["flow_rmse"],
        "test_flow_mse": test_metrics["flow_mse"],
        "test_flow_rmse": test_metrics["flow_rmse"],
        "train_windows": train_metrics["windows"],
        "validation_windows": val_metrics["windows"],
        "test_windows": test_metrics["windows"],
        "seed": args.seed,
    }
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    summary = {
        "task": "Recovered flow-matching checkpoint evaluation",
        "note": "This is not retraining. It re-evaluates a saved flow-matching checkpoint using fixed random seeds.",
        "run_name": args.run_name,
        "checkpoint": str(args.checkpoint),
        "datasets": [str(p) for p in dataset_dirs],
        "x_shape": list(dataset.x_shape),
        "split": {
            "train_windows": len(train_idx),
            "validation_windows": len(val_idx),
            "test_windows": len(test_idx),
            "train_source_counts": source_counts(dataset, train_idx),
            "validation_source_counts": source_counts(dataset, val_idx),
            "test_source_counts": source_counts(dataset, test_idx),
        },
        "model": "FlowMatchingChannelTimeUNet",
        "parameters": count_parameters(model),
        "objective": "MSE between predicted velocity and x1 - x0",
        "formula": "x0 = Gaussian noise; x1 = real EEG; xt = (1 - t) * x0 + t * x1; target velocity = x1 - x0",
        "metrics": row,
        "files": {
            "metrics_csv": str(metrics_csv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", flush=True)
    print(metrics_csv, flush=True)
    print(summary_json, flush=True)
    print("Recovered validation MSE:", row["validation_flow_mse"], flush=True)
    print("Recovered test MSE:", row["test_flow_mse"], flush=True)


if __name__ == "__main__":
    main()
