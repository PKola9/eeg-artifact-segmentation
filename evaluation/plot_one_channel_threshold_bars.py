"""
Plot one-channel artifact probability curve with threshold bars.

This matches the professor's requested visualization:
    - one selected EEG channel
    - probability curve over time
    - horizontal threshold lines
    - predicted artifact intervals at multiple thresholds
    - ground-truth artifact interval/mask for the same channel

The model still predicts the full 59 x 2500 channel-time probability map.
This script only selects one channel for clear visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = WORKSPACE_ROOT / "packages" / "Physiomotion Dataset" / "Original Dataset"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PREP_DIR = PROJECT_ROOT / "data_preparation"

sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(DATA_PREP_DIR))

from splitunet_channel_time_segmentation import SplitUNetChannelTimeSegmenter  # noqa: E402
from temporal_unet_no_channel_mixing import TemporalUNetNoChannelMixing  # noqa: E402
from build_window_mask_dataset_from_edf import make_paths, normalize_window, read_edf_downsampled  # noqa: E402
from build_channel_time_mask_dataset_from_edf_one_run import create_channel_time_mask  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs_v2_channel_time" / "one_channel_threshold_bars"


def load_model(checkpoint: Path, device: torch.device, model_arch: str) -> torch.nn.Module:
    if model_arch == "split_unet":
        model = SplitUNetChannelTimeSegmenter()
    elif model_arch == "temporal_unet_no_channel_mixing":
        model = TemporalUNetNoChannelMixing()
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


def boolean_runs(mask: np.ndarray, times: np.ndarray) -> list[dict]:
    """Convert a boolean 1D mask to contiguous time intervals."""
    mask = mask.astype(bool)
    intervals = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and ((not value) or i == len(mask) - 1):
            end = i if not value else i + 1
            intervals.append(
                {
                    "start_time_seconds_in_window": float(times[start]),
                    "stop_time_seconds_in_window": float(times[min(end - 1, len(times) - 1)]),
                    "start_sample_in_window": int(start),
                    "stop_sample_in_window": int(end),
                }
            )
            start = None
    return intervals


def compute_channel_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred_b = pred.astype(bool)
    true_b = true.astype(bool)
    tp = int((pred_b & true_b).sum())
    fp = int((pred_b & ~true_b).sum())
    tn = int((~pred_b & ~true_b).sum())
    fn = int((~pred_b & true_b).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall_sensitivity": recall,
        "accuracy": accuracy,
    }


def threshold_color_map(thresholds: list[float]) -> dict[float, str]:
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    return {thr: colors[i % len(colors)] for i, thr in enumerate(thresholds)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Example: sub-30")
    parser.add_argument("--run", type=int, required=True, help="Example: 6")
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--channel", required=True, help="Example: Fp1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-arch",
        choices=["split_unet", "temporal_unet_no_channel_mixing"],
        default="split_unet",
    )
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    thresholds = [float(v.strip()) for v in args.thresholds.split(",") if v.strip()]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    edf_path, _metadata_path, annotation_path = make_paths(args.subject, args.run)
    data, channel_names, sfreq, duration_seconds = read_edf_downsampled(edf_path, downsample_to_hz=250.0)
    annotations = pd.read_csv(annotation_path)
    true_mask, _mask_report = create_channel_time_mask(
        annotations=annotations,
        channel_names=channel_names,
        duration_seconds=duration_seconds,
        sfreq=sfreq,
    )

    if args.channel not in channel_names:
        raise ValueError(f"Channel {args.channel!r} not found. Available examples: {channel_names[:10]}")
    channel_index = channel_names.index(args.channel)

    start_sample = int(round(args.start_sec * sfreq))
    stop_sample = start_sample + int(round(args.window_sec * sfreq))
    if stop_sample > data.shape[1]:
        raise ValueError("Requested window extends beyond recording.")

    x = data[:, start_sample:stop_sample].astype(np.float32)
    y = true_mask[:, start_sample:stop_sample].astype(np.float32)
    x_norm = normalize_window(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model), device, args.model_arch)

    with torch.no_grad():
        logits = model(torch.from_numpy(x_norm).unsqueeze(0).to(device))
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    times = np.arange(x.shape[1], dtype=np.float32) / float(sfreq)
    absolute_times = times + float(args.start_sec)
    channel_probs = probs[channel_index]
    channel_true = y[channel_index] >= 0.5

    prefix = f"{args.subject}_run-{args.run:02d}_{args.channel}_start-{int(args.start_sec)}s"
    csv_path = output_dir / f"{prefix}_threshold_bars.csv"
    png_path = output_dir / f"{prefix}_threshold_bars.png"
    summary_path = output_dir / f"{prefix}_threshold_bars_summary.json"

    prediction_masks = {thr: channel_probs >= thr for thr in thresholds}
    threshold_colors = threshold_color_map(thresholds)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_in_window", "time_seconds_in_window", "absolute_time_seconds", "probability", "ground_truth_mask"]
        fieldnames += [f"pred_threshold_{thr:g}" for thr in thresholds]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(times)):
            row = {
                "sample_in_window": i,
                "time_seconds_in_window": float(times[i]),
                "absolute_time_seconds": float(absolute_times[i]),
                "probability": float(channel_probs[i]),
                "ground_truth_mask": int(channel_true[i]),
            }
            for thr in thresholds:
                row[f"pred_threshold_{thr:g}"] = int(prediction_masks[thr][i])
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(times, channel_probs, color="black", linewidth=1.5, label=f"{args.channel} artifact probability")
    for thr in thresholds:
        ax.axhline(
            thr,
            linestyle="--",
            linewidth=1.2,
            color=threshold_colors[thr],
            label=f"threshold {thr:g}",
        )

    base_y = -0.12
    row_gap = 0.08
    for row_id, thr in enumerate(thresholds):
        y_level = base_y - row_id * row_gap
        for interval in boolean_runs(prediction_masks[thr], times):
            ax.hlines(
                y=y_level,
                xmin=interval["start_time_seconds_in_window"],
                xmax=interval["stop_time_seconds_in_window"],
                linewidth=7,
                color=threshold_colors[thr],
            )
        ax.text(
            -0.08,
            y_level,
            f"pred {thr:g}",
            va="center",
            ha="right",
            color=threshold_colors[thr],
            fontweight="bold",
            transform=ax.get_yaxis_transform(),
        )

    gt_y = base_y - len(thresholds) * row_gap
    ground_truth_intervals = boolean_runs(channel_true, times)
    if ground_truth_intervals:
        for interval in ground_truth_intervals:
            ax.hlines(
                y=gt_y,
                xmin=interval["start_time_seconds_in_window"],
                xmax=interval["stop_time_seconds_in_window"],
                linewidth=8,
                color="tab:green",
            )
    else:
        # Keep the ground-truth row visible even when there is no annotated artifact.
        # This avoids the visual confusion of an apparently "missing" ground-truth row.
        ax.hlines(
            y=gt_y,
            xmin=float(times[0]),
            xmax=float(times[-1]),
            linewidth=5,
            color="lightgray",
            alpha=0.9,
        )
        ax.text(
            0.5,
            gt_y + 0.018,
            "manual ground truth: clean/no artifact for this channel",
            color="gray",
            fontsize=8,
            ha="center",
        )
    ax.text(
        -0.08,
        gt_y,
        "ground truth",
        va="center",
        ha="right",
        color="tab:green" if ground_truth_intervals else "gray",
        fontweight="bold",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_title(f"{args.subject} run-{args.run:02d}, channel {args.channel}, {args.start_sec:g}s-{args.start_sec + args.window_sec:g}s")
    ax.set_xlabel("Time in selected window (seconds)")
    ax.set_ylabel("Artifact probability")
    ax.set_ylim(gt_y - 0.08, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    threshold_summaries = {}
    for thr in thresholds:
        threshold_summaries[str(thr)] = {
            "metrics": compute_channel_metrics(prediction_masks[thr], channel_true),
            "predicted_intervals": boolean_runs(prediction_masks[thr], times),
        }

    summary = {
        "subject": args.subject,
        "run": args.run,
        "channel": args.channel,
        "start_sec": args.start_sec,
        "window_sec": args.window_sec,
        "sfreq": sfreq,
        "model": str(args.model),
        "model_arch": args.model_arch,
        "input_shape_full_model": list(x.shape),
        "output_shape_full_model": list(probs.shape),
        "visualized_channel_index": channel_index,
        "thresholds": thresholds,
        "ground_truth_intervals": ground_truth_intervals,
        "ground_truth_has_artifact": bool(ground_truth_intervals),
        "threshold_summaries": threshold_summaries,
        "outputs": {
            "csv": str(csv_path),
            "plot": str(png_path),
            "summary": str(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(csv_path)
    print(png_path)
    print(summary_path)


if __name__ == "__main__":
    main()
