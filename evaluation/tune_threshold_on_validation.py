"""
Tune artifact probability threshold on the validation split and evaluate on test.

This is an honest evaluation step:
    - validation set chooses the threshold
    - held-out test set is evaluated after threshold is chosen

It does not retrain the model and does not change metric formulas.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.splitunet_channel_time_segmentation import SplitUNetChannelTimeSegmenter
from models.splitunet_transformer_bottleneck_segmentation import SplitUNetTransformerBottleneckSegmenter
from models.temporal_unet_no_channel_mixing import TemporalUNetNoChannelMixing
from models.eeg_conformer_segmentation import EEGConformerSegmentation
from training.train_channel_time_segmentation_sharded import (
    load_channel_time_dataset,
    make_manifest_holdout_split,
    make_manifest_multi_subject_holdout_split,
)


def load_model(checkpoint: Path, device: torch.device, model_arch: str, base_features: int = 8) -> torch.nn.Module:
    if model_arch == "split_unet":
        model = SplitUNetChannelTimeSegmenter(base_features=base_features)
    elif model_arch == "split_unet_transformer_bottleneck":
        model = SplitUNetTransformerBottleneckSegmenter(base_features=base_features)
    elif model_arch == "temporal_unet_no_channel_mixing":
        model = TemporalUNetNoChannelMixing()
    elif model_arch == "eeg_conformer_segmentation":
        model = EEGConformerSegmentation()
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_thresholds(
    model: torch.nn.Module,
    loader: DataLoader,
    thresholds: list[float],
    device: torch.device,
) -> list[dict]:
    totals = {
        threshold: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for threshold in thresholds
    }

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            probs = torch.sigmoid(model(x))
            true = y >= 0.5

            for threshold in thresholds:
                pred = probs >= threshold
                totals[threshold]["tp"] += int((pred & true).sum().item())
                totals[threshold]["fp"] += int((pred & ~true).sum().item())
                totals[threshold]["tn"] += int((~pred & ~true).sum().item())
                totals[threshold]["fn"] += int((~pred & true).sum().item())

    rows = []
    for threshold in thresholds:
        tp = totals[threshold]["tp"]
        fp = totals[threshold]["fp"]
        tn = totals[threshold]["tn"]
        fn = totals[threshold]["fn"]
        accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
        dice = (2 * tp) / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        rows.append(
            {
                "threshold": threshold,
                "dice": dice,
                "iou": iou,
                "f1": f1,
                "precision": precision,
                "recall_sensitivity": recall,
                "specificity": specificity,
                "accuracy": accuracy,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-arch",
        choices=[
            "split_unet",
            "split_unet_transformer_bottleneck",
            "temporal_unet_no_channel_mixing",
            "eeg_conformer_segmentation",
        ],
        default="split_unet",
    )
    parser.add_argument(
        "--base-features",
        type=int,
        default=8,
        help="Width of Split U-Net checkpoint. Default 8 preserves previous experiments.",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--output-dir", default="outputs_v6_flowmatching_study/threshold_tuning_default")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_path = Path(args.model)
    output_dir = PROJECT_ROOT / args.output_dir
    thresholds = [float(v.strip()) for v in args.thresholds.split(",") if v.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_channel_time_dataset(dataset_dir)
    holdout_subjects = args.holdout_subjects
    if holdout_subjects:
        holdout_run = None
        _, val_idx, test_idx = make_manifest_multi_subject_holdout_split(
            dataset.manifest,
            holdout_subjects=holdout_subjects,
            seed=args.seed,
            val_fraction_from_train_pool=args.val_fraction,
        )
    else:
        holdout_run = None if args.holdout_all_runs else args.holdout_run
        holdout_subjects = [args.holdout_subject]
        _, val_idx, test_idx = make_manifest_holdout_split(
            dataset.manifest,
            holdout_subject=args.holdout_subject,
            holdout_run=holdout_run,
            seed=args.seed,
            val_fraction_from_train_pool=args.val_fraction,
        )

    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = load_model(model_path, device, args.model_arch, args.base_features)

    validation_rows = evaluate_thresholds(model, val_loader, thresholds, device)
    best_validation = sorted(validation_rows, key=lambda row: row["dice"], reverse=True)[0]
    best_threshold = float(best_validation["threshold"])

    test_rows = evaluate_thresholds(model, test_loader, [best_threshold, 0.5], device)
    selected_test = [row for row in test_rows if abs(row["threshold"] - best_threshold) < 1e-12][0]
    threshold_05_test = [row for row in test_rows if abs(row["threshold"] - 0.5) < 1e-12][0]

    write_csv(output_dir / "validation_threshold_sweep.csv", validation_rows)
    write_csv(output_dir / "test_at_selected_and_0p5_threshold.csv", test_rows)

    summary = {
        "purpose": "Select artifact probability threshold using validation Dice, then evaluate on held-out test run.",
        "dataset_dir": str(dataset_dir),
        "model": str(model_path),
        "model_arch": args.model_arch,
        "base_features": args.base_features
        if args.model_arch in {"split_unet", "split_unet_transformer_bottleneck"}
        else None,
        "device": str(device),
        "holdout_test": {
            "subject": args.holdout_subject,
            "subjects": holdout_subjects,
            "run": holdout_run,
            "all_runs": bool(args.holdout_all_runs or args.holdout_subjects),
            "test_windows": int(len(test_idx)),
        },
        "validation_windows": int(len(val_idx)),
        "threshold_selection_rule": "choose threshold with highest validation Dice",
        "best_validation_threshold": best_threshold,
        "best_validation_metrics": best_validation,
        "test_metrics_at_best_validation_threshold": selected_test,
        "test_metrics_at_threshold_0p5": threshold_05_test,
        "important_note": "The test set was not used to choose the threshold.",
    }
    with (output_dir / "threshold_tuning_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(output_dir / "validation_threshold_sweep.csv")
    print(output_dir / "test_at_selected_and_0p5_threshold.csv")
    print(output_dir / "threshold_tuning_summary.json")
    print()
    print("Best validation threshold:", best_threshold)
    print("Validation Dice:", best_validation["dice"])
    print("Test Dice at selected threshold:", selected_test["dice"])
    print("Test Precision at selected threshold:", selected_test["precision"])
    print("Test Recall at selected threshold:", selected_test["recall_sensitivity"])
    print("Test Dice at threshold 0.5:", threshold_05_test["dice"])


if __name__ == "__main__":
    main()
