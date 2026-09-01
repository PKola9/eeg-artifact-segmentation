"""
Predict artifact intervals from one original PhysioMotion EDF window.

This is the professor-facing inference script.

Input:
    subject, run, start time, end time/window length, trained model checkpoint

Output:
    1. EEG input CSV for the selected window
    2. Probability map CSV: probability for every channel and time sample
    3. Predicted interval CSV: start/end time of detected artifacts
    4. Ground-truth interval CSV from manual annotations for comparison
    5. Summary JSON

The key output requested by the professor is:

    predicted_artifact_intervals.csv

with columns such as:

    channel, start_seconds_in_recording, end_seconds_in_recording,
    duration_seconds, threshold, mean_probability, max_probability
"""

from __future__ import annotations

import argparse
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


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "outputs_v6_flowmatching_study"
    / "training_runs"
    / "comparison_03_split_unet_flowinit_posweight1"
    / "splitunet_channel_time_best_val_dice.pt"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs_interval_predictions"


def load_model(checkpoint: Path, device: torch.device, model_arch: str) -> torch.nn.Module:
    if model_arch == "split_unet":
        model = SplitUNetChannelTimeSegmenter()
    elif model_arch == "temporal_unet_no_channel_mixing":
        model = TemporalUNetNoChannelMixing()
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compute_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    pred = pred_mask >= 0.5
    true = true_mask >= 0.5
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    tn = int((~pred & ~true).sum())
    fn = int((~pred & true).sum())

    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    dice = (2 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "dice": dice,
        "iou": iou,
    }


def merge_short_gaps(mask: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """Fill small gaps between predicted artifact regions in one 1D mask."""
    if max_gap_samples <= 0 or not mask.any():
        return mask
    mask = mask.astype(bool).copy()
    starts, stops = mask_to_segments(mask)
    if len(starts) <= 1:
        return mask
    for i in range(len(stops) - 1):
        gap_start = stops[i]
        gap_stop = starts[i + 1]
        if 0 < gap_stop - gap_start <= max_gap_samples:
            mask[gap_start:gap_stop] = True
    return mask


def mask_to_segments(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return start and exclusive-stop sample indices for contiguous True regions."""
    mask = mask.astype(bool)
    if mask.size == 0 or not mask.any():
        return np.array([], dtype=int), np.array([], dtype=int)
    starts = np.where(mask & np.r_[True, ~mask[:-1]])[0]
    stops = np.where(mask & np.r_[~mask[1:], True])[0] + 1
    return starts.astype(int), stops.astype(int)


def intervals_from_mask(
    mask: np.ndarray,
    probabilities: np.ndarray,
    channel_names: list[str],
    sfreq: float,
    absolute_window_start_sec: float,
    threshold: float,
    min_duration_sec: float,
    merge_gap_sec: float,
    source: str,
) -> pd.DataFrame:
    """Convert channel-time binary mask to human-readable artifact intervals."""
    rows: list[dict] = []
    min_samples = int(round(min_duration_sec * sfreq))
    gap_samples = int(round(merge_gap_sec * sfreq))

    for ch_idx, ch_name in enumerate(channel_names):
        one_mask = mask[ch_idx].astype(bool)
        one_mask = merge_short_gaps(one_mask, gap_samples)
        starts, stops = mask_to_segments(one_mask)
        for start, stop in zip(starts, stops):
            duration_samples = int(stop - start)
            if duration_samples < max(min_samples, 1):
                continue
            segment_probs = probabilities[ch_idx, start:stop]
            rows.append(
                {
                    "source": source,
                    "channel": ch_name,
                    "channel_index": ch_idx,
                    "start_sample_in_window": int(start),
                    "stop_sample_in_window_exclusive": int(stop),
                    "start_seconds_in_window": float(start / sfreq),
                    "end_seconds_in_window": float(stop / sfreq),
                    "start_seconds_in_recording": float(absolute_window_start_sec + start / sfreq),
                    "end_seconds_in_recording": float(absolute_window_start_sec + stop / sfreq),
                    "duration_seconds": float(duration_samples / sfreq),
                    "threshold": float(threshold),
                    "mean_probability": float(segment_probs.mean()) if segment_probs.size else None,
                    "max_probability": float(segment_probs.max()) if segment_probs.size else None,
                    "samples": duration_samples,
                }
            )

    return pd.DataFrame(rows)


def summarize_intervals(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "interval_count": 0,
            "channels_with_intervals": 0,
            "total_interval_seconds_channel_summed": 0.0,
            "top_channels": [],
        }
    by_channel = (
        df.groupby("channel")
        .agg(interval_count=("channel", "count"), total_seconds=("duration_seconds", "sum"))
        .sort_values(["total_seconds", "interval_count"], ascending=False)
        .head(15)
        .reset_index()
    )
    return {
        "interval_count": int(len(df)),
        "channels_with_intervals": int(df["channel"].nunique()),
        "total_interval_seconds_channel_summed": float(df["duration_seconds"].sum()),
        "top_channels": by_channel.to_dict(orient="records"),
    }


def save_interval_timeline(
    pred_intervals: pd.DataFrame,
    true_intervals: pd.DataFrame,
    channel_names: list[str],
    output_path: Path,
    window_sec: float,
    title: str,
) -> None:
    """Save a clean timeline plot of predicted vs true intervals."""
    channels_used = sorted(
        set(pred_intervals.get("channel", pd.Series(dtype=str)).tolist())
        | set(true_intervals.get("channel", pd.Series(dtype=str)).tolist()),
        key=lambda ch: channel_names.index(ch) if ch in channel_names else 10_000,
    )
    if not channels_used:
        channels_used = channel_names[:10]
    if len(channels_used) > 20:
        channels_used = channels_used[:20]

    fig_height = max(4, 0.35 * len(channels_used) + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    y_positions = {ch: i for i, ch in enumerate(channels_used)}

    for _, row in true_intervals.iterrows():
        ch = row["channel"]
        if ch not in y_positions:
            continue
        ax.broken_barh(
            [(row["start_seconds_in_window"], row["duration_seconds"])],
            (y_positions[ch] - 0.18, 0.16),
            facecolors="#2ca02c",
            edgecolors="#2ca02c",
            alpha=0.9,
        )

    for _, row in pred_intervals.iterrows():
        ch = row["channel"]
        if ch not in y_positions:
            continue
        ax.broken_barh(
            [(row["start_seconds_in_window"], row["duration_seconds"])],
            (y_positions[ch] + 0.02, 0.16),
            facecolors="#1f77b4",
            edgecolors="#1f77b4",
            alpha=0.9,
        )

    ax.set_xlim(0, window_sec)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(channels_used)
    ax.set_xlabel("Time inside selected window (seconds)")
    ax.set_ylabel("EEG channel")
    ax.set_title(title + "\nGreen = manual annotation, Blue = model predicted interval")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--model-arch",
        choices=["split_unet", "temporal_unet_no_channel_mixing"],
        default="split_unet",
    )
    parser.add_argument("--subject", default="sub-30")
    parser.add_argument("--run", type=int, default=6)
    parser.add_argument("--start-sec", type=float, default=306.0)
    parser.add_argument("--end-sec", type=float, default=None)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--downsample-to-hz", type=float, default=250.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration-sec", type=float, default=0.04)
    parser.add_argument("--merge-gap-sec", type=float, default=0.02)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    if args.end_sec is not None:
        if args.end_sec <= args.start_sec:
            raise ValueError("--end-sec must be greater than --start-sec")
        args.window_sec = args.end_sec - args.start_sec

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model)
    edf_path, _metadata_json, manual_csv = make_paths(args.subject, args.run)

    if dataset_root != DATASET_ROOT:
        run_bids = f"{args.subject}_task-artifact_run-{args.run:02d}"
        edf_path = dataset_root / args.subject / "eeg" / f"{run_bids}_eeg.edf"
        manual_csv = dataset_root / "derivatives" / "Manual_Annotations" / f"{args.subject.replace('-', '')}_run{args.run:02d}.csv"

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    if not manual_csv.exists():
        raise FileNotFoundError(f"Manual annotation CSV not found: {manual_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print("Reading original EDF:", edf_path)
    eeg, channel_names, sfreq, duration_seconds = read_edf_downsampled(edf_path, args.downsample_to_hz)
    annotations = pd.read_csv(manual_csv)
    true_full_mask, mask_report = create_channel_time_mask(annotations, channel_names, duration_seconds, sfreq)

    window_samples = int(round(args.window_sec * sfreq))
    start_sample = int(round(args.start_sec * sfreq))
    stop_sample = start_sample + window_samples
    if start_sample < 0 or stop_sample > eeg.shape[1]:
        raise ValueError(
            f"Requested window is outside the recording: {args.start_sec:g}-{args.start_sec + args.window_sec:g}s. "
            f"Recording duration is about {eeg.shape[1] / sfreq:.2f}s after downsampling."
        )

    raw_window = eeg[:, start_sample:stop_sample].astype(np.float32)
    model_window = normalize_window(raw_window).astype(np.float32)
    true_mask = true_full_mask[:, start_sample:stop_sample].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, args.model_arch)
    with torch.no_grad():
        batch = torch.from_numpy(model_window[None, :, :]).to(device)
        logits = model(batch)
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

    pred_mask = (probabilities >= args.threshold).astype(np.float32)
    metrics = compute_metrics(pred_mask, true_mask)

    # Convert both model output and manual labels to readable intervals.
    pred_intervals = intervals_from_mask(
        mask=pred_mask,
        probabilities=probabilities,
        channel_names=channel_names,
        sfreq=sfreq,
        absolute_window_start_sec=args.start_sec,
        threshold=args.threshold,
        min_duration_sec=args.min_duration_sec,
        merge_gap_sec=args.merge_gap_sec,
        source="model_prediction",
    )
    true_intervals = intervals_from_mask(
        mask=true_mask,
        probabilities=true_mask,
        channel_names=channel_names,
        sfreq=sfreq,
        absolute_window_start_sec=args.start_sec,
        threshold=0.5,
        min_duration_sec=0.0,
        merge_gap_sec=0.0,
        source="manual_annotation_ground_truth",
    )

    time_seconds = np.arange(window_samples) / sfreq
    absolute_time_seconds = args.start_sec + time_seconds

    output_dir = Path(args.output_dir)
    window_folder = output_dir / f"{args.subject}_run-{args.run:02d}_{args.start_sec:g}s_to_{args.start_sec + args.window_sec:g}s_thr_{args.threshold:g}"
    window_folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.subject}_run-{args.run:02d}_{args.start_sec:g}s_to_{args.start_sec + args.window_sec:g}s"

    input_csv = window_folder / f"{prefix}_INPUT_eeg_window.csv"
    probability_csv = window_folder / f"{prefix}_OUTPUT_probability_map.csv"
    pred_intervals_csv = window_folder / f"{prefix}_PREDICTED_artifact_intervals.csv"
    true_intervals_csv = window_folder / f"{prefix}_TRUE_manual_artifact_intervals.csv"
    timeline_png = window_folder / f"{prefix}_INTERVAL_timeline.png"
    summary_json = window_folder / f"{prefix}_SUMMARY.json"

    input_df = pd.DataFrame(raw_window.T, columns=channel_names)
    input_df.insert(0, "absolute_time_seconds_in_recording", absolute_time_seconds)
    input_df.insert(0, "time_seconds_in_window", time_seconds)
    input_df.insert(0, "sample_in_window", np.arange(window_samples))
    input_df.to_csv(input_csv, index=False)

    prob_df = pd.DataFrame(probabilities.T, columns=[f"prob_{ch}" for ch in channel_names])
    prob_df.insert(0, "absolute_time_seconds_in_recording", absolute_time_seconds)
    prob_df.insert(0, "time_seconds_in_window", time_seconds)
    prob_df.insert(0, "sample_in_window", np.arange(window_samples))
    prob_df.to_csv(probability_csv, index=False)

    pred_intervals.to_csv(pred_intervals_csv, index=False)
    true_intervals.to_csv(true_intervals_csv, index=False)

    save_interval_timeline(
        pred_intervals=pred_intervals,
        true_intervals=true_intervals,
        channel_names=channel_names,
        output_path=timeline_png,
        window_sec=args.window_sec,
        title=f"{args.subject} run-{args.run:02d}, {args.start_sec:g}-{args.start_sec + args.window_sec:g}s, threshold {args.threshold:g}",
    )

    summary = {
        "purpose": "Predict human-readable artifact start/end intervals from an original EDF window.",
        "input_edf": str(edf_path),
        "manual_annotation_csv": str(manual_csv),
        "model_checkpoint": str(model_path),
        "model_arch": args.model_arch,
        "subject": args.subject,
        "run": args.run,
        "start_seconds": args.start_sec,
        "end_seconds": args.start_sec + args.window_sec,
        "window_seconds": args.window_sec,
        "sampling_frequency_hz": sfreq,
        "input_shape": [len(channel_names), window_samples],
        "output_probability_shape": list(probabilities.shape),
        "threshold": args.threshold,
        "min_duration_seconds": args.min_duration_sec,
        "merge_gap_seconds": args.merge_gap_sec,
        "metrics_for_this_window": metrics,
        "predicted_interval_summary": summarize_intervals(pred_intervals),
        "true_interval_summary": summarize_intervals(true_intervals),
        "mask_report_for_full_run": mask_report,
        "created_files": {
            "input_eeg_window_csv": str(input_csv),
            "output_probability_map_csv": str(probability_csv),
            "predicted_artifact_intervals_csv": str(pred_intervals_csv),
            "true_manual_artifact_intervals_csv": str(true_intervals_csv),
            "interval_timeline_png": str(timeline_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Artifact interval prediction complete.")
    print("Window:", f"{args.subject} run-{args.run:02d} {args.start_sec:g}-{args.start_sec + args.window_sec:g}s")
    print("Input shape:", summary["input_shape"])
    print("Threshold:", args.threshold)
    print("Dice:", f"{metrics['dice']:.4f}", "Precision:", f"{metrics['precision']:.4f}", "Recall:", f"{metrics['recall_sensitivity']:.4f}")
    print("Predicted intervals:", len(pred_intervals))
    print("True intervals:", len(true_intervals))
    print("Predicted interval CSV:", pred_intervals_csv)
    print("True interval CSV:", true_intervals_csv)
    print("Timeline PNG:", timeline_png)
    print("Summary:", summary_json)


if __name__ == "__main__":
    main()
