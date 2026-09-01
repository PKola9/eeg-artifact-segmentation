"""
Evaluate final V6 models with strict and time-tolerant artifact localization metrics.

This script runs on HPC, where the trained checkpoints and test dataset exist.

Why:
    Standard channel-time Dice is strict. If a predicted artifact interval is
    shifted by a small amount, strict Dice drops even when the timing is close.

    For artifact start/end localization, we also report tolerant metrics:

        A predicted artifact point is considered temporally correct if it lies
        within +/- buffer seconds of a manual artifact point on the same channel.

Important:
    Tolerant Dice does not replace strict Dice. It is an additional timing
    tolerance metric for interval localization.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.tune_threshold_on_validation import load_model  # noqa: E402
from training.train_channel_time_segmentation_sharded import (  # noqa: E402
    load_channel_time_dataset,
    make_manifest_holdout_split,
)


MODEL_CHECKPOINTS = {
    "M1_random_splitunet": "outputs_v6_flowmatching_study/training_runs/V6_CLEAN_M1_random_splitunet/splitunet_channel_time_best_val_dice.pt",
    "M2_physio_flowinit_splitunet": "outputs_v6_flowmatching_study/training_runs/V6_CLEAN_M2_physio_flowinit_splitunet/splitunet_channel_time_best_val_dice.pt",
    "M3_improved_multidataset_flowinit_splitunet": "outputs_v6_flowmatching_study/training_runs/V6_IMPROVED_M3_multidataset_flowinit_lr5e4_patience8/splitunet_channel_time_best_val_dice.pt",
    "M4_extra_dataset_flowinit_splitunet": "outputs_v6_flowmatching_study/training_runs/V6_M4_extra_deap_chbmit_flowinit_splitunet/splitunet_channel_time_best_val_dice.pt",
}


def dilate_time(mask: torch.Tensor, radius_samples: int) -> torch.Tensor:
    """Dilate boolean [B, C, T] mask along time only."""
    if radius_samples <= 0:
        return mask
    b, c, t = mask.shape
    x = mask.float().reshape(b * c, 1, t)
    kernel = 2 * radius_samples + 1
    y = F.max_pool1d(x, kernel_size=kernel, stride=1, padding=radius_samples)
    return y.reshape(b, c, t) > 0.5


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    threshold: float,
    buffers_seconds: list[float],
    sampling_frequency_hz: float,
    device: torch.device,
) -> list[dict]:
    radii = [int(round(buffer * sampling_frequency_hz)) for buffer in buffers_seconds]
    totals = {
        buffer: {
            "strict_tp": 0,
            "strict_fp": 0,
            "strict_fn": 0,
            "strict_tn": 0,
            "tol_precision_tp": 0,
            "tol_precision_fp": 0,
            "tol_recall_tp": 0,
            "tol_recall_fn": 0,
        }
        for buffer in buffers_seconds
    }

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = torch.sigmoid(model(x)) >= threshold
            true = y >= 0.5

            strict_tp = int((pred & true).sum().item())
            strict_fp = int((pred & ~true).sum().item())
            strict_fn = int((~pred & true).sum().item())
            strict_tn = int((~pred & ~true).sum().item())

            for buffer, radius in zip(buffers_seconds, radii):
                true_dilated = dilate_time(true, radius)
                pred_dilated = dilate_time(pred, radius)

                # Precision side: predicted artifact points are correct if near true artifact.
                tol_precision_tp = int((pred & true_dilated).sum().item())
                tol_precision_fp = int((pred & ~true_dilated).sum().item())

                # Recall side: true artifact points are found if near a predicted artifact.
                tol_recall_tp = int((true & pred_dilated).sum().item())
                tol_recall_fn = int((true & ~pred_dilated).sum().item())

                t = totals[buffer]
                t["strict_tp"] += strict_tp
                t["strict_fp"] += strict_fp
                t["strict_fn"] += strict_fn
                t["strict_tn"] += strict_tn
                t["tol_precision_tp"] += tol_precision_tp
                t["tol_precision_fp"] += tol_precision_fp
                t["tol_recall_tp"] += tol_recall_tp
                t["tol_recall_fn"] += tol_recall_fn

    rows = []
    for buffer in buffers_seconds:
        t = totals[buffer]
        strict_precision = safe_div(t["strict_tp"], t["strict_tp"] + t["strict_fp"])
        strict_recall = safe_div(t["strict_tp"], t["strict_tp"] + t["strict_fn"])
        strict_dice = safe_div(2 * t["strict_tp"], 2 * t["strict_tp"] + t["strict_fp"] + t["strict_fn"])
        strict_iou = safe_div(t["strict_tp"], t["strict_tp"] + t["strict_fp"] + t["strict_fn"])
        strict_accuracy = safe_div(
            t["strict_tp"] + t["strict_tn"],
            t["strict_tp"] + t["strict_fp"] + t["strict_fn"] + t["strict_tn"],
        )

        tolerant_precision = safe_div(
            t["tol_precision_tp"],
            t["tol_precision_tp"] + t["tol_precision_fp"],
        )
        tolerant_recall = safe_div(
            t["tol_recall_tp"],
            t["tol_recall_tp"] + t["tol_recall_fn"],
        )
        tolerant_dice = safe_div(
            2 * tolerant_precision * tolerant_recall,
            tolerant_precision + tolerant_recall,
        )

        rows.append(
            {
                "buffer_seconds": buffer,
                "threshold": threshold,
                "strict_dice": strict_dice,
                "strict_iou": strict_iou,
                "strict_precision": strict_precision,
                "strict_recall": strict_recall,
                "strict_accuracy": strict_accuracy,
                "tolerant_dice": tolerant_dice,
                "tolerant_precision": tolerant_precision,
                "tolerant_recall": tolerant_recall,
                "strict_tp": t["strict_tp"],
                "strict_fp": t["strict_fp"],
                "strict_fn": t["strict_fn"],
                "strict_tn": t["strict_tn"],
                "tolerant_precision_tp": t["tol_precision_tp"],
                "tolerant_precision_fp": t["tol_precision_fp"],
                "tolerant_recall_tp": t["tol_recall_tp"],
                "tolerant_recall_fn": t["tol_recall_fn"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default=str(
            PROJECT_ROOT.parent
            / "BFM2_ChannelTimeSegmentation_V2_WORKING"
            / "outputs_v2_channel_time"
            / "window_datasets"
            / "full_sub30_runs6_channel_time_10s_250hz_memmap"
        ),
    )
    parser.add_argument(
        "--comparison-csv",
        default=str(PROJECT_ROOT / "outputs_v6_flowmatching_study" / "final_tables" / "V6_three_or_four_model_clean_comparison.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs_v6_flowmatching_study" / "tolerant_interval_metrics"),
    )
    parser.add_argument("--buffers-sec", default="0,0.25,0.5,1,2,3,5,10,15")
    parser.add_argument("--holdout-subject", default="sub-30")
    parser.add_argument("--holdout-run", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sampling-frequency-hz", type=float, default=250.0)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    comparison_csv = Path(args.comparison_csv)
    output_dir = Path(args.output_dir)
    buffers = [float(v.strip()) for v in args.buffers_sec.split(",") if v.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_channel_time_dataset(dataset_dir)
    _, _, test_idx = make_manifest_holdout_split(
        dataset.manifest,
        holdout_subject=args.holdout_subject,
        holdout_run=args.holdout_run,
        seed=args.seed,
        val_fraction_from_train_pool=args.val_fraction,
    )
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)

    comparison = pd.read_csv(comparison_csv)
    all_rows = []

    print("Device:", device, flush=True)
    print("Test windows:", len(test_idx), flush=True)
    print("Buffers:", buffers, flush=True)

    for _, comparison_row in comparison.iterrows():
        model_name = str(comparison_row["model"])
        if model_name not in MODEL_CHECKPOINTS:
            print(f"Skipping {model_name}: no checkpoint path configured.", flush=True)
            continue

        checkpoint = PROJECT_ROOT / MODEL_CHECKPOINTS[model_name]
        if not checkpoint.exists():
            print(f"Skipping {model_name}: checkpoint missing: {checkpoint}", flush=True)
            continue

        threshold = float(comparison_row["selected_threshold"])
        print(f"Evaluating {model_name} at threshold {threshold}...", flush=True)
        model = load_model(checkpoint, device=device, model_arch="split_unet")
        metric_rows = evaluate_model(
            model=model,
            loader=test_loader,
            threshold=threshold,
            buffers_seconds=buffers,
            sampling_frequency_hz=args.sampling_frequency_hz,
            device=device,
        )

        for row in metric_rows:
            all_rows.append(
                {
                    "model": model_name,
                    "pretraining": comparison_row["pretraining"],
                    **row,
                }
            )

    if not all_rows:
        raise RuntimeError("No tolerant metrics were computed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "V6_tolerant_test_metrics_by_model_and_buffer.csv"
    write_csv(full_path, all_rows)

    df = pd.DataFrame(all_rows)
    best_rows = []
    for buffer, group in df.groupby("buffer_seconds"):
        best_rows.append(group.sort_values("tolerant_dice", ascending=False).iloc[0].to_dict())
    best_path = output_dir / "V6_best_model_by_tolerance_buffer.csv"
    pd.DataFrame(best_rows).to_csv(best_path, index=False)

    print("Saved:")
    print(full_path)
    print(best_path)
    print()
    print("Best model by tolerance buffer:")
    print(
        pd.DataFrame(best_rows)[
            ["buffer_seconds", "model", "strict_dice", "tolerant_dice", "tolerant_precision", "tolerant_recall"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
