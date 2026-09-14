"""
Train Version 2 channel-time EEG artifact segmentation on a sharded dataset.

Input dataset format:
    dataset_dir/
        metadata.json
        manifest.csv
        shards/
            sub-1_run-01.npz
            ...

Each shard contains:
    x: windows x channels x time
    y: windows x channels x time

This script supports:
    BCE + Dice loss
    validation Dice checkpointing
    early stopping with patience
    held-out subject/run testing
"""

from __future__ import annotations

import argparse
import csv
import json
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

from splitunet_channel_time_segmentation import (  # noqa: E402
    SplitUNetChannelTimeSegmenter,
    count_parameters,
)
from splitunet_transformer_bottleneck_segmentation import (  # noqa: E402
    SplitUNetTransformerBottleneckSegmenter,
)
from temporal_unet_no_channel_mixing import TemporalUNetNoChannelMixing  # noqa: E402
from eeg_conformer_segmentation import EEGConformerSegmentation  # noqa: E402


class ShardedChannelTimeDataset(Dataset):
    """
    Dataset that reads one window from one NPZ shard.

    This is simple and safe. For large HPC training, a memmap version can be
    added later if speed becomes a bottleneck.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.shard_dir = dataset_dir / "shards"
        self.manifest = pd.read_csv(dataset_dir / "manifest.csv")
        with (dataset_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self._cache_name: str | None = None
        self._cache_data = None

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_shard(self, shard_name: str):
        if self._cache_name != shard_name:
            self._cache_data = np.load(self.shard_dir / shard_name)
            self._cache_name = shard_name
        return self._cache_data

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]
        shard_name = str(row["shard_name"])
        local_idx = int(row["shard_local_index"])
        data = self._load_shard(shard_name)
        x = data["x"][local_idx].astype(np.float32)
        y = data["y"][local_idx].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class MemmapChannelTimeDataset(Dataset):
    """
    Dataset that reads Version 2 windows from fast memory-mapped float32 arrays.

    This avoids repeatedly opening/decompressing NPZ shards during GPU training.
    The x/y values are the same as the sharded dataset; only the storage format
    is different.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.manifest = pd.read_csv(dataset_dir / "manifest.csv")
        with (dataset_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        x_shape = tuple(int(v) for v in self.metadata["x_shape"])
        y_shape = tuple(int(v) for v in self.metadata["y_shape"])
        self.x = np.memmap(dataset_dir / "x_float32.dat", dtype="float32", mode="r", shape=x_shape)
        self.y = np.memmap(dataset_dir / "y_float32.dat", dtype="float32", mode="r", shape=y_shape)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Copy one window into normal writable arrays before converting to torch.
        x = np.array(self.x[idx], dtype=np.float32, copy=True)
        y = np.array(self.y[idx], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_channel_time_dataset(dataset_dir: Path) -> Dataset:
    if (dataset_dir / "x_float32.dat").exists() and (dataset_dir / "y_float32.dat").exists():
        return MemmapChannelTimeDataset(dataset_dir)
    return ShardedChannelTimeDataset(dataset_dir)


def transfer_flowmatching_weights(seg_model: nn.Module, flow_checkpoint: Path) -> dict:
    """
    Copy compatible encoder/decoder weights from the V2 flow-matching model.

    The final output head is intentionally not copied:
        flow model head predicts EEG velocity
        segmentation head predicts artifact logits
    """

    flow_state = torch.load(flow_checkpoint, map_location="cpu")
    seg_state = seg_model.state_dict()
    new_state = dict(seg_state)
    copied = []
    skipped = []

    transferable_prefixes = ("down1.", "down2.", "bottleneck.", "up2.", "up1.")
    for key, seg_value in seg_state.items():
        if not key.startswith(transferable_prefixes):
            skipped.append({"key": key, "reason": "not transferred; task-specific or no matching prefix"})
            continue
        if key in flow_state and tuple(flow_state[key].shape) == tuple(seg_value.shape):
            new_state[key] = flow_state[key]
            copied.append(key)
        else:
            skipped.append({"key": key, "reason": "missing or shape mismatch in flow checkpoint"})

    seg_model.load_state_dict(new_state)
    return {
        "flow_checkpoint": str(flow_checkpoint),
        "copied_count": len(copied),
        "skipped_count": len(skipped),
        "copied_keys": copied,
        "skipped": skipped,
    }


def make_manifest_holdout_split(
    manifest: pd.DataFrame,
    holdout_subject: str,
    holdout_run: int | None,
    seed: int,
    val_fraction_from_train_pool: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subject_mask = manifest["subject"].astype(str) == holdout_subject
    if holdout_run is None:
        test_mask = subject_mask
    else:
        test_mask = subject_mask & (manifest["run"].astype(int) == int(holdout_run))
    test_idx = manifest.index[test_mask].to_numpy().copy()
    train_pool_idx = manifest.index[~test_mask].to_numpy().copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(train_pool_idx)
    val_count = int(round(len(train_pool_idx) * val_fraction_from_train_pool))
    val_idx = train_pool_idx[:val_count]
    train_idx = train_pool_idx[val_count:]
    return train_idx, val_idx, test_idx


def make_manifest_multi_subject_holdout_split(
    manifest: pd.DataFrame,
    holdout_subjects: list[str],
    seed: int,
    val_fraction_from_train_pool: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hold out every run for multiple selected subjects.

    This is used for the stricter subject-level generalization experiment:
    the test set contains complete unseen subjects, and validation is sampled
    only from the remaining training pool.
    """
    subjects = [str(subject) for subject in holdout_subjects]
    test_mask = manifest["subject"].astype(str).isin(subjects)
    test_idx = manifest.index[test_mask].to_numpy().copy()
    train_pool_idx = manifest.index[~test_mask].to_numpy().copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(train_pool_idx)
    val_count = int(round(len(train_pool_idx) * val_fraction_from_train_pool))
    val_idx = train_pool_idx[:val_count]
    train_idx = train_pool_idx[val_count:]
    return train_idx, val_idx, test_idx


def make_balanced_window_train_indices(
    manifest: pd.DataFrame,
    train_idx: np.ndarray,
    seed: int,
    clean_to_artifact_ratio: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Balance training windows at the window level.

    Artifact window:
        window has any artifact channel-time points.

    Clean window:
        window has no artifact channel-time points.

    This changes training only. Validation/test remain unchanged.
    """
    if "has_artifact" in manifest.columns:
        has_artifact = manifest["has_artifact"].astype(bool).to_numpy()
    elif "artifact_fraction_channel_time" in manifest.columns:
        has_artifact = manifest["artifact_fraction_channel_time"].astype(float).to_numpy() > 0.0
    elif "artifact_fraction" in manifest.columns:
        has_artifact = manifest["artifact_fraction"].astype(float).to_numpy() > 0.0
    else:
        raise ValueError(
            "Cannot create balanced windows: manifest needs has_artifact, "
            "artifact_fraction_channel_time, or artifact_fraction column."
        )

    train_idx = np.asarray(train_idx).copy()
    artifact_idx = train_idx[has_artifact[train_idx]]
    clean_idx = train_idx[~has_artifact[train_idx]]

    rng = np.random.default_rng(seed)
    rng.shuffle(artifact_idx)
    rng.shuffle(clean_idx)

    requested_clean_count = int(round(len(artifact_idx) * clean_to_artifact_ratio))
    clean_count = min(len(clean_idx), requested_clean_count)
    selected_clean_idx = clean_idx[:clean_count]

    balanced_idx = np.concatenate([artifact_idx, selected_clean_idx])
    rng.shuffle(balanced_idx)

    report = {
        "enabled": True,
        "clean_to_artifact_ratio_requested": clean_to_artifact_ratio,
        "original_train_windows": int(len(train_idx)),
        "original_artifact_windows": int(len(artifact_idx)),
        "original_clean_windows": int(len(clean_idx)),
        "balanced_train_windows": int(len(balanced_idx)),
        "balanced_artifact_windows": int(len(artifact_idx)),
        "balanced_clean_windows": int(len(selected_clean_idx)),
        "balanced_artifact_fraction_window_level": float(len(artifact_idx) / max(len(balanced_idx), 1)),
        "note": "Only training indices are balanced. Validation and test remain unchanged.",
    }
    return balanced_idx, report


def soft_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    intersection = (probs * target).sum()
    denominator = probs.sum() + target.sum()
    dice = (2.0 * intersection + 1e-7) / (denominator + 1e-7)
    return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * soft_dice_loss_from_logits(
            logits, target
        )


def compute_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    true = target >= 0.5

    tp = int((pred & true).sum().item())
    fp = int((pred & ~true).sum().item())
    tn = int((~pred & ~true).sum().item())
    fn = int((~pred & true).sum().item())

    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    totals = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * x.shape[0]
            total_items += x.shape[0]
            m = compute_metrics(logits.detach().cpu(), y.detach().cpu())
            for key in totals:
                totals[key] += int(m[key])

    tp, fp, tn, fn = totals["tp"], totals["fp"], totals["tn"], totals["fn"]
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "dice": dice,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def estimate_pos_weight_from_manifest(manifest: pd.DataFrame) -> float:
    if "artifact_fraction_channel_time" in manifest.columns:
        pos_fraction = float(manifest["artifact_fraction_channel_time"].mean())
    elif "artifact_fraction" in manifest.columns:
        pos_fraction = float(manifest["artifact_fraction"].mean())
    else:
        pos_fraction = 0.05
    pos_fraction = max(min(pos_fraction, 0.999), 1e-6)
    return (1.0 - pos_fraction) / pos_fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--model-arch",
        choices=[
            "split_unet",
            "split_unet_transformer_bottleneck",
            "temporal_unet_no_channel_mixing",
            "eeg_conformer_segmentation",
        ],
        default="split_unet",
        help=(
            "split_unet learns temporal patterns and EEG channel relationships; "
            "split_unet_transformer_bottleneck adds temporal attention inside the Split U-Net bottleneck; "
            "temporal_unet_no_channel_mixing processes each channel independently; "
            "eeg_conformer_segmentation is a CNN-Transformer baseline adapted for dense segmentation."
        ),
    )
    parser.add_argument("--holdout-subject", default="sub-30")
    parser.add_argument(
        "--holdout-subjects",
        nargs="+",
        default=None,
        help=(
            "Optional stricter split: hold out every run for multiple subjects, "
            "for example --holdout-subjects sub-10 sub-20 sub-30. "
            "When this is used, --holdout-run is ignored."
        ),
    )
    parser.add_argument("--holdout-run", type=int, default=6)
    parser.add_argument(
        "--holdout-all-runs",
        action="store_true",
        help="Hold out every run for the selected subject. Keeps old subject/run behavior unchanged when omitted.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-features",
        type=int,
        default=8,
        help="Width of Split U-Net models. Default 8 preserves all previous experiments.",
    )
    parser.add_argument("--output-name", default="channel_time_v2_full_training")
    parser.add_argument("--flow-checkpoint", default="", help="Optional V2 flow-matching checkpoint for initialization")
    parser.add_argument(
        "--balanced-train-windows",
        action="store_true",
        help="Use balanced artifact/clean training windows. Validation/test are unchanged.",
    )
    parser.add_argument(
        "--clean-to-artifact-ratio",
        type=float,
        default=1.0,
        help="How many clean windows to keep per artifact window when balanced training is enabled.",
    )
    parser.add_argument(
        "--pos-weight-scale",
        type=float,
        default=1.0,
        help="Scale the automatically estimated BCE positive-class weight.",
    )
    parser.add_argument(
        "--pos-weight-override",
        type=float,
        default=0.0,
        help="If > 0, use this exact BCE positive-class weight instead of the automatic estimate.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    dataset = load_channel_time_dataset(dataset_dir)
    holdout_subjects = args.holdout_subjects
    if holdout_subjects:
        holdout_run = None
        train_idx, val_idx, test_idx = make_manifest_multi_subject_holdout_split(
            dataset.manifest,
            holdout_subjects=holdout_subjects,
            seed=args.seed,
            val_fraction_from_train_pool=args.val_fraction,
        )
    else:
        holdout_run = None if args.holdout_all_runs else args.holdout_run
        holdout_subjects = [args.holdout_subject]
        train_idx, val_idx, test_idx = make_manifest_holdout_split(
            dataset.manifest,
            holdout_subject=args.holdout_subject,
            holdout_run=holdout_run,
            seed=args.seed,
            val_fraction_from_train_pool=args.val_fraction,
        )

    balance_report = {"enabled": False}
    if args.balanced_train_windows:
        train_idx, balance_report = make_balanced_window_train_indices(
            dataset.manifest,
            train_idx,
            seed=args.seed,
            clean_to_artifact_ratio=args.clean_to_artifact_ratio,
        )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_arch == "split_unet":
        model = SplitUNetChannelTimeSegmenter(base_features=args.base_features)
    elif args.model_arch == "split_unet_transformer_bottleneck":
        model = SplitUNetTransformerBottleneckSegmenter(base_features=args.base_features)
    elif args.model_arch == "temporal_unet_no_channel_mixing":
        model = TemporalUNetNoChannelMixing()
    elif args.model_arch == "eeg_conformer_segmentation":
        model = EEGConformerSegmentation()
    else:
        raise ValueError(f"Unknown model architecture: {args.model_arch}")

    transfer_report = None
    if args.flow_checkpoint:
        flow_supported_arches = {"split_unet", "split_unet_transformer_bottleneck"}
        if args.model_arch not in flow_supported_arches:
            raise ValueError(
                "Flow-matching initialization is currently supported only for "
                "--model-arch split_unet or split_unet_transformer_bottleneck."
            )
        transfer_report = transfer_flowmatching_weights(model, Path(args.flow_checkpoint))
    model = model.to(device)
    estimated_pos_weight = estimate_pos_weight_from_manifest(dataset.manifest)
    if args.pos_weight_override > 0:
        effective_pos_weight = args.pos_weight_override
        pos_weight_source = "manual override"
    else:
        effective_pos_weight = estimated_pos_weight * args.pos_weight_scale
        pos_weight_source = "estimated from manifest x scale"
    pos_weight = torch.tensor(effective_pos_weight, device=device)
    loss_fn = BCEDiceLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = PROJECT_ROOT / "outputs_v6_flowmatching_study" / "training_runs" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = output_dir / "splitunet_channel_time_final.pt"
    best_model_path = output_dir / "splitunet_channel_time_best_val_dice.pt"
    metrics_path = output_dir / "training_metrics.csv"
    summary_path = output_dir / "summary.json"

    print("Device:", device)
    print("Dataset:", dataset_dir)
    print("Total windows:", len(dataset))
    print("Train/validation/test windows:", len(train_idx), len(val_idx), len(test_idx))
    if args.balanced_train_windows:
        print("Balanced training enabled:", balance_report)
    if args.holdout_subjects:
        print("Holdout test:", ", ".join(holdout_subjects), "all runs")
    elif holdout_run is None:
        print("Holdout test:", args.holdout_subject, "all runs")
    else:
        print("Holdout test:", args.holdout_subject, f"run-{holdout_run:02d}")
    print("x per window:", dataset.metadata.get("x_shape_per_window"))
    print("y per window:", dataset.metadata.get("y_shape_per_window"))
    print("estimated_pos_weight:", float(estimated_pos_weight))
    print("pos_weight_scale:", float(args.pos_weight_scale))
    print("pos_weight_override:", float(args.pos_weight_override))
    print("effective_pos_weight:", float(pos_weight.item()))
    print("pos_weight_source:", pos_weight_source)
    print("Model architecture:", args.model_arch)
    if args.model_arch in {"split_unet", "split_unet_transformer_bottleneck"}:
        print("Base features:", args.base_features)
    print("Model parameters:", count_parameters(model))
    if transfer_report:
        print("Flow-matching initialization:", args.flow_checkpoint)
        print("Copied flow-matching weights:", transfer_report["copied_count"])

    history = []
    best_validation_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_items = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * x.shape[0]
            train_items += x.shape[0]

        train_loss = train_loss_total / max(train_items, 1)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        test_metrics = evaluate(model, test_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_metrics["loss"],
            "validation_dice": val_metrics["dice"],
            "validation_iou": val_metrics["iou"],
            "validation_f1": val_metrics["f1"],
            "validation_precision": val_metrics["precision"],
            "validation_recall_sensitivity": val_metrics["recall_sensitivity"],
            "validation_specificity": val_metrics["specificity"],
            "test_loss": test_metrics["loss"],
            "test_dice": test_metrics["dice"],
            "test_iou": test_metrics["iou"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall_sensitivity": test_metrics["recall_sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(row)

        if val_metrics["dice"] > best_validation_dice:
            best_validation_dice = val_metrics["dice"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            checkpoint_message = "saved new best checkpoint"
        else:
            epochs_without_improvement += 1
            checkpoint_message = f"no improvement ({epochs_without_improvement}/{args.patience})"

        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"test_dice={test_metrics['dice']:.4f} "
            f"test_f1={test_metrics['f1']:.4f} "
            f"{checkpoint_message}",
            flush=True,
        )

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation Dice {best_validation_dice:.4f} at epoch {best_epoch}.",
                flush=True,
            )
            break

    torch.save(model.state_dict(), final_model_path)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "task": "Version 2 channel-time EEG artifact segmentation",
        "dataset_dir": str(dataset_dir),
        "dataset_metadata": dataset.metadata,
        "total_windows": len(dataset),
        "train_windows": len(train_idx),
        "validation_windows": len(val_idx),
        "test_windows": len(test_idx),
        "balanced_training": balance_report,
        "holdout_subject": args.holdout_subject,
        "holdout_subjects": holdout_subjects,
        "holdout_run": holdout_run,
        "holdout_all_runs": bool(args.holdout_all_runs or args.holdout_subjects),
        "model": args.model_arch,
        "base_features": args.base_features
        if args.model_arch in {"split_unet", "split_unet_transformer_bottleneck"}
        else None,
        "parameters": count_parameters(model),
        "initialization": "flow-matching pretrained weights" if args.flow_checkpoint else "random supervised baseline",
        "flow_checkpoint": args.flow_checkpoint or None,
        "transfer_report": transfer_report,
        "loss": "BCEWithLogitsLoss + soft Dice loss",
        "positive_class_weight": {
            "estimated_from_manifest": estimated_pos_weight,
            "scale": args.pos_weight_scale,
            "override": args.pos_weight_override,
            "effective": float(pos_weight.item()),
            "source": pos_weight_source,
        },
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "early_stopping": {
            "enabled": True,
            "monitor": "validation_dice",
            "patience": args.patience,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_validation_dice": best_validation_dice,
        },
        "final_metrics": history[-1],
        "files": {
            "final_model": str(final_model_path),
            "best_model_by_validation_dice": str(best_model_path),
            "metrics": str(metrics_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:")
    print(final_model_path)
    print(best_model_path)
    print(metrics_path)
    print(summary_path)


if __name__ == "__main__":
    main()
