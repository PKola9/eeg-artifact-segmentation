"""
Flow-matching pretraining on multiple 59-channel EEG datasets.

This script uses only EEG windows. It ignores artifact masks completely.
The best checkpoint is selected by lowest validation MSE.
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
        self.dataset_dir = dataset_dir
        self.source_index = source_index
        self.manifest = pd.read_csv(dataset_dir / "manifest.csv")
        with (dataset_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.x_shape = tuple(int(v) for v in self.metadata["x_shape"])
        self.x = np.memmap(dataset_dir / "x_float32.dat", dtype="float32", mode="r", shape=self.x_shape)

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
        self.metadata = {
            "dataset_type": "multi-dataset EEG-only flow-matching dataset",
            "x_shape": list(self.x_shape),
            "x_shape_per_window": list(first_shape),
            "sources": [
                {
                    "dataset_dir": str(src.dataset_dir),
                    "original_x_shape": list(src.x_shape),
                    "metadata": src.metadata,
                }
                for src in self.sources
            ],
            "holdout_excluded_from_flow_pretraining": {
                "subject": holdout_subject,
                "run": holdout_run,
                "reason": "Avoid using the final supervised test run even during self-supervised pretraining.",
            },
        }

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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for x1 in loader:
            loss = flow_loss_batch(model, x1, device)
            total_loss += float(loss.item()) * x1.shape[0]
            total_items += x1.shape[0]
    mse = total_loss / max(total_items, 1)
    return {"flow_mse": mse, "flow_rmse": math.sqrt(mse)}


def dataset_source_counts(dataset: MultiMemmapEEGOnlyDataset, indices: np.ndarray) -> dict:
    selected = dataset.index_table.iloc[indices]
    counts = selected["source_dataset_dir"].value_counts().to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="append", required=True)
    parser.add_argument("--holdout-subject", default="sub-30")
    parser.add_argument("--holdout-run", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-features",
        type=int,
        default=8,
        help="Width of the flow-matching U-Net. Default 8 preserves all previous experiments.",
    )
    parser.add_argument("--output-name", default="combined_physio_bci59_flowmatching_earlystop")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_dirs = [Path(p) for p in args.dataset_dir]
    dataset = MultiMemmapEEGOnlyDataset(dataset_dirs, args.holdout_subject, args.holdout_run)
    train_idx, val_idx, test_idx = make_random_split(
        len(dataset), val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowMatchingChannelTimeUNet(base_features=args.base_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = PROJECT_ROOT / "outputs_v6_flowmatching_study" / "flowmatching_runs" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "flowmatching_channel_time_best_validation_mse.pt"
    final_path = output_dir / "flowmatching_channel_time_final.pt"
    metrics_path = output_dir / "flowmatching_metrics.csv"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "combined_flow_dataset_index.csv"
    dataset.index_table.to_csv(index_path, index=False)

    print("Device:", device, flush=True)
    print("Datasets:", [str(p) for p in dataset_dirs], flush=True)
    print("Combined x shape:", list(dataset.x_shape), flush=True)
    print("Train/validation/test windows:", len(train_idx), len(val_idx), len(test_idx), flush=True)
    print("Train source counts:", dataset_source_counts(dataset, train_idx), flush=True)
    print("Validation source counts:", dataset_source_counts(dataset, val_idx), flush=True)
    print("Test source counts:", dataset_source_counts(dataset, test_idx), flush=True)
    print("Base features:", args.base_features, flush=True)
    print("Model parameters:", count_parameters(model), flush=True)
    print("Objective: MSE(predicted velocity, x1 - x0)", flush=True)

    history: list[dict] = []
    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for x1 in train_loader:
            optimizer.zero_grad()
            loss = flow_loss_batch(model, x1, device)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x1.shape[0]
            total_items += x1.shape[0]

        train_mse = total_loss / max(total_items, 1)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        improved = val_metrics["flow_mse"] < (best_val_mse - args.min_delta)
        if improved:
            best_val_mse = val_metrics["flow_mse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            checkpoint_message = "saved new best validation-MSE checkpoint"
        else:
            epochs_without_improvement += 1
            checkpoint_message = f"no validation-MSE improvement ({epochs_without_improvement}/{args.patience})"

        row = {
            "epoch": epoch,
            "train_flow_mse": train_mse,
            "train_flow_rmse": math.sqrt(train_mse),
            "validation_flow_mse": val_metrics["flow_mse"],
            "validation_flow_rmse": val_metrics["flow_rmse"],
            "test_flow_mse": test_metrics["flow_mse"],
            "test_flow_rmse": test_metrics["flow_rmse"],
            "best_validation_flow_mse_so_far": best_val_mse,
            "is_best_epoch": improved,
        }
        history.append(row)

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_mse={train_mse:.5f} "
            f"val_mse={val_metrics['flow_mse']:.5f} "
            f"test_mse={test_metrics['flow_mse']:.5f} "
            f"val_rmse={val_metrics['flow_rmse']:.5f} "
            f"{checkpoint_message}",
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation MSE {best_val_mse:.5f} at epoch {best_epoch}.",
                flush=True,
            )
            break

    torch.save(model.state_dict(), final_path)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "task": "Combined PhysioMotion + BCI 59-channel EEG flow-matching pretraining",
        "purpose": "learn better EEG structure from more unlabeled EEG before supervised artifact segmentation",
        "datasets": [str(p) for p in dataset_dirs],
        "dataset_metadata": dataset.metadata,
        "x_shape": list(dataset.x_shape),
        "train_windows": len(train_idx),
        "validation_windows": len(val_idx),
        "test_windows": len(test_idx),
        "train_source_counts": dataset_source_counts(dataset, train_idx),
        "validation_source_counts": dataset_source_counts(dataset, val_idx),
        "test_source_counts": dataset_source_counts(dataset, test_idx),
      "model": "FlowMatchingChannelTimeUNet",
      "base_features": args.base_features,
      "parameters": count_parameters(model),
        "objective": "MSE between predicted velocity and x1 - x0",
        "formula": "x0 = Gaussian noise; x1 = real EEG; xt = (1 - t) * x0 + t * x1; target velocity = x1 - x0",
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "early_stopping": {
            "enabled": True,
            "monitor": "validation_flow_mse",
            "mode": "min",
            "patience": args.patience,
            "min_delta": args.min_delta,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_validation_flow_mse": best_val_mse,
        },
        "final_metrics": history[-1],
        "files": {
            "best_checkpoint_by_validation_mse": str(best_path),
            "final_checkpoint": str(final_path),
            "metrics_csv": str(metrics_path),
            "summary_json": str(summary_path),
            "combined_index_csv": str(index_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", flush=True)
    print(best_path, flush=True)
    print(final_path, flush=True)
    print(metrics_path, flush=True)
    print(summary_path, flush=True)
    print(index_path, flush=True)


if __name__ == "__main__":
    main()
