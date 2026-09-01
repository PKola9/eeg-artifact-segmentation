"""
Create a clean professor-style threshold visual for one EEG channel.

The figure shows:
    1. predicted artifact probability curve for one channel
    2. horizontal threshold lines
    3. colored predicted artifact bars for each threshold
    4. green ground-truth artifact bar

This is designed to be easier to explain than a dense 59-channel heatmap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PREP_DIR = PROJECT_ROOT / "data_preparation"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(DATA_PREP_DIR))

from splitunet_channel_time_segmentation import SplitUNetChannelTimeSegmenter  # noqa: E402
from temporal_unet_no_channel_mixing import TemporalUNetNoChannelMixing  # noqa: E402
from build_channel_time_mask_dataset_from_edf_one_run import (  # noqa: E402
    create_channel_time_mask,
    read_annotations,
    read_edf_downsampled,
)


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT.parent / "packages" / "Physiomotion Dataset" / "Original Dataset"
)


def load_model(checkpoint: Path, model_arch: str, device: torch.device) -> torch.nn.Module:
    if model_arch == "split_unet":
        model = SplitUNetChannelTimeSegmenter()
    elif model_arch == "temporal_unet_no_channel_mixing":
        model = TemporalUNetNoChannelMixing()
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def contiguous_segments(mask_1d: np.ndarray, time_seconds: np.ndarray) -> list[tuple[float, float]]:
    """Convert a binary 1D mask into start/end time segments."""
    mask = mask_1d.astype(bool)
    segments: list[tuple[float, float]] = []
    if mask.size == 0 or not mask.any():
        return segments

    starts = np.where(mask & np.r_[True, ~mask[:-1]])[0]
    stops = np.where(mask & np.r_[~mask[1:], True])[0]
    dt = float(np.median(np.diff(time_seconds))) if len(time_seconds) > 1 else 0.004
    for start_idx, stop_idx in zip(starts, stops):
        segments.append((float(time_seconds[start_idx]), float(time_seconds[stop_idx] + dt)))
    return segments


def draw_segments(ax, segments: list[tuple[float, float]], y: float, color: str, label: str | None = None):
    first = True
    for start, stop in segments:
        ax.broken_barh(
            [(start, stop - start)],
            (y - 0.035, 0.07),
            facecolors=color,
            edgecolors=color,
            alpha=0.95,
            label=label if first and label else None,
        )
        first = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--subject", default="sub-30")
    parser.add_argument("--run", type=int, default=6)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--channel", default="Cz")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-arch", choices=["split_unet", "temporal_unet_no_channel_mixing"], default="split_unet")
    parser.add_argument("--downsample-to-hz", type=float, default=250.0)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs_v6_flowmatching_study" / "clean_threshold_visuals"),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    subject_number = int(args.subject.replace("sub-", ""))
    subject_folder = dataset_root / args.subject
    edf_path = subject_folder / "eeg" / f"{args.subject}_task-artifact_run-{args.run:02d}_eeg.edf"
    annotation_path = dataset_root / "derivatives" / "Manual_Annotations" / f"sub{subject_number}_run{args.run:02d}.csv"

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Manual annotation file not found: {annotation_path}")

    eeg, channel_names, sfreq, duration_seconds = read_edf_downsampled(edf_path, args.downsample_to_hz)
    annotations = read_annotations(annotation_path)
    true_full_mask, mask_report = create_channel_time_mask(annotations, channel_names, duration_seconds, sfreq)

    if args.channel not in channel_names:
        raise ValueError(f"Channel {args.channel} not found. Available example channels: {channel_names[:10]}")

    start_sample = int(round(args.start_sec * sfreq))
    window_samples = int(round(args.window_sec * sfreq))
    stop_sample = start_sample + window_samples
    if start_sample < 0 or stop_sample > eeg.shape[1]:
        raise ValueError("Requested window is outside the recording.")

    raw_window = eeg[:, start_sample:stop_sample].astype(np.float32)
    true_mask = true_full_mask[:, start_sample:stop_sample].astype(np.float32)

    mean = raw_window.mean(axis=1, keepdims=True)
    std = raw_window.std(axis=1, keepdims=True)
    model_input = (raw_window - mean) / (std + 1e-6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model), args.model_arch, device)
    with torch.no_grad():
        logits = model(torch.from_numpy(model_input[None, :, :]).to(device)).cpu().numpy()[0]
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    channel_index = channel_names.index(args.channel)
    prob = probabilities[channel_index]
    truth = true_mask[channel_index]
    time_in_window = np.arange(window_samples, dtype=np.float32) / float(sfreq)
    absolute_time = args.start_sec + time_in_window
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    output_dir = Path(args.output_dir) / f"{args.subject}_run-{args.run:02d}_{args.start_sec:g}s_{args.channel}"
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        0.5: "#1f77b4",  # blue
        0.6: "#ff7f0e",  # orange
        0.7: "#9467bd",  # purple
        0.8: "#d62728",  # red
        0.85: "#8c564b",  # brown
    }
    fallback_colors = ["#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_in_window, prob, color="black", linewidth=2.0, label=f"{args.channel} artifact probability")

    for i, thr in enumerate(thresholds):
        color = colors.get(thr, fallback_colors[i % len(fallback_colors)])
        ax.axhline(thr, color=color, linestyle="--", linewidth=1.2, alpha=0.9, label=f"threshold {thr:g}")

    y_positions = {}
    base_y = -0.16
    step = -0.095
    for i, thr in enumerate(thresholds):
        y_positions[f"pred_{thr:g}"] = base_y + i * step
    ground_truth_y = base_y + len(thresholds) * step

    summary_rows = []
    for i, thr in enumerate(thresholds):
        color = colors.get(thr, fallback_colors[i % len(fallback_colors)])
        pred_mask = prob >= thr
        pred_segments = contiguous_segments(pred_mask, time_in_window)
        y = y_positions[f"pred_{thr:g}"]
        draw_segments(ax, pred_segments, y=y, color=color, label=f"predicted ≥ {thr:g}")
        summary_rows.append(
            {
                "threshold": thr,
                "predicted_artifact_samples": int(pred_mask.sum()),
                "predicted_artifact_seconds": float(pred_mask.sum() / sfreq),
                "predicted_segments": len(pred_segments),
            }
        )

    truth_segments = contiguous_segments(truth >= 0.5, time_in_window)
    draw_segments(ax, truth_segments, y=ground_truth_y, color="#2ca02c", label="ground truth")

    yticks = [*y_positions.values(), ground_truth_y]
    yticklabels = [*y_positions.keys(), "ground truth"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(ground_truth_y - 0.12, 1.05)
    ax.set_xlim(0, args.window_sec)
    ax.set_xlabel("Time inside selected window (seconds)")
    ax.set_ylabel("Artifact probability / threshold bars")
    ax.set_title(
        f"{args.subject} run-{args.run:02d}, channel {args.channel}, "
        f"{args.start_sec:g}-{args.start_sec + args.window_sec:g}s"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()

    png_path = output_dir / f"{args.subject}_run-{args.run:02d}_{args.channel}_start-{args.start_sec:g}s_clean_threshold_curve.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    curve_csv = output_dir / f"{args.subject}_run-{args.run:02d}_{args.channel}_start-{args.start_sec:g}s_curve.csv"
    pd.DataFrame(
        {
            "sample_in_window": np.arange(window_samples),
            "time_in_window_seconds": time_in_window,
            "absolute_time_seconds": absolute_time,
            "predicted_artifact_probability": prob,
            "ground_truth_mask": truth,
        }
    ).to_csv(curve_csv, index=False)

    summary_path = output_dir / f"{args.subject}_run-{args.run:02d}_{args.channel}_start-{args.start_sec:g}s_summary.json"
    summary = {
        "subject": args.subject,
        "run": args.run,
        "start_sec": args.start_sec,
        "window_sec": args.window_sec,
        "channel": args.channel,
        "model": str(args.model),
        "model_arch": args.model_arch,
        "sampling_frequency_hz": sfreq,
        "thresholds": thresholds,
        "ground_truth_artifact_samples": int((truth >= 0.5).sum()),
        "ground_truth_artifact_seconds": float((truth >= 0.5).sum() / sfreq),
        "ground_truth_segments": len(truth_segments),
        "prediction_summary_by_threshold": summary_rows,
        "mask_report": mask_report,
        "outputs": {
            "figure": str(png_path),
            "curve_csv": str(curve_csv),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Clean threshold visual saved:")
    print(png_path)
    print(curve_csv)
    print(summary_path)


if __name__ == "__main__":
    main()
