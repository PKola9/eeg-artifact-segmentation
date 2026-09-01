"""
Professor demo: predict a Version 2 channel-time artifact map from one original EDF window.

This script shows the real input -> output behavior:

    original EDF + subject/run/start time
        -> 10-second EEG window, shape 59 x 2500
        -> trained SplitUNetChannelTimeSegmenter
        -> artifact probability map, shape 59 x 2500

Outputs:
    INPUT_eeg_window.csv
    OUTPUT_probability_map.csv
    TRUE_manual_mask.csv
    OUTPUT_channel_time_heatmap.png
    SUMMARY.json
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


DEFAULT_FLOWINIT_MODEL = (
    PROJECT_ROOT
    / "outputs_v2_channel_time"
    / "training_runs"
    / "full_sub30_runs6_channel_time_v2_flowinit_bce_dice_holdout_sub30_run6"
    / "splitunet_channel_time_best_val_dice.pt"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs_v2_channel_time" / "professor_input_output_demo"


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


def save_heatmap(
    raw_window: np.ndarray,
    probabilities: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    channel_names: list[str],
    time_seconds: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, height_ratios=[1.2, 2.5, 1.3, 1.3])

    # Show a small set of common frontal/central channels for readable input signal preview.
    preview_channels = [ch for ch in ["Fp1", "Fp2", "F7", "F8", "Cz"] if ch in channel_names]
    if not preview_channels:
        preview_channels = channel_names[:5]
    for ch in preview_channels:
        idx = channel_names.index(ch)
        signal = raw_window[idx]
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        axes[0].plot(time_seconds, signal + idx * 0.0, linewidth=0.8, label=ch)
    axes[0].set_title("Input EEG preview channels, normalized for display")
    axes[0].set_ylabel("EEG")
    axes[0].legend(loc="upper right", ncol=len(preview_channels), fontsize=8)
    axes[0].grid(True, alpha=0.2)

    im = axes[1].imshow(
        probabilities,
        aspect="auto",
        origin="lower",
        extent=[time_seconds[0], time_seconds[-1], 0, len(channel_names) - 1],
        vmin=0,
        vmax=1,
        cmap="magma",
    )
    axes[1].set_title("Model output: predicted artifact probability map (channels × time)")
    axes[1].set_ylabel("EEG channel index")
    fig.colorbar(im, ax=axes[1], label="artifact probability")

    axes[2].imshow(
        true_mask,
        aspect="auto",
        origin="lower",
        extent=[time_seconds[0], time_seconds[-1], 0, len(channel_names) - 1],
        vmin=0,
        vmax=1,
        cmap="Greens",
    )
    axes[2].set_title("Ground truth from manual annotations")
    axes[2].set_ylabel("EEG channel index")

    axes[3].imshow(
        pred_mask,
        aspect="auto",
        origin="lower",
        extent=[time_seconds[0], time_seconds[-1], 0, len(channel_names) - 1],
        vmin=0,
        vmax=1,
        cmap="Blues",
    )
    axes[3].set_title("Predicted artifact mask after threshold")
    axes[3].set_ylabel("EEG channel index")
    axes[3].set_xlabel("Time inside selected window (seconds)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--model", default=str(DEFAULT_FLOWINIT_MODEL))
    parser.add_argument(
        "--model-arch",
        choices=["split_unet", "temporal_unet_no_channel_mixing"],
        default="split_unet",
    )
    parser.add_argument("--subject", default="sub-30")
    parser.add_argument("--run", type=int, default=6)
    parser.add_argument("--start-sec", type=float, default=306.0)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--downsample-to-hz", type=float, default=250.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    model_path = Path(args.model)
    edf_path, _metadata_json, manual_csv = make_paths(args.subject, args.run)

    # make_paths uses the default project dataset path; if user supplied dataset-root, rebuild paths.
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

    time_seconds = np.arange(window_samples) / sfreq
    absolute_time_seconds = args.start_sec + time_seconds

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.subject}_run-{args.run:02d}_start-{args.start_sec:g}s_{int(args.window_sec)}s_v2_channel_time"

    input_csv = output_dir / f"{prefix}_INPUT_eeg_window.csv"
    probability_csv = output_dir / f"{prefix}_OUTPUT_probability_map.csv"
    true_mask_csv = output_dir / f"{prefix}_TRUE_manual_mask.csv"
    pred_mask_csv = output_dir / f"{prefix}_OUTPUT_threshold_mask.csv"
    heatmap_png = output_dir / f"{prefix}_OUTPUT_heatmap.png"
    summary_json = output_dir / f"{prefix}_SUMMARY.json"

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

    true_df = pd.DataFrame(true_mask.T, columns=[f"true_{ch}" for ch in channel_names])
    true_df.insert(0, "absolute_time_seconds_in_recording", absolute_time_seconds)
    true_df.insert(0, "time_seconds_in_window", time_seconds)
    true_df.insert(0, "sample_in_window", np.arange(window_samples))
    true_df.to_csv(true_mask_csv, index=False)

    pred_df = pd.DataFrame(pred_mask.T, columns=[f"pred_{ch}" for ch in channel_names])
    pred_df.insert(0, "absolute_time_seconds_in_recording", absolute_time_seconds)
    pred_df.insert(0, "time_seconds_in_window", time_seconds)
    pred_df.insert(0, "sample_in_window", np.arange(window_samples))
    pred_df.to_csv(pred_mask_csv, index=False)

    save_heatmap(
        raw_window=raw_window,
        probabilities=probabilities,
        true_mask=true_mask,
        pred_mask=pred_mask,
        channel_names=channel_names,
        time_seconds=time_seconds,
        output_path=heatmap_png,
        title=f"{args.subject} run-{args.run:02d}, {args.start_sec:g}-{args.start_sec + args.window_sec:g}s",
    )

    summary = {
        "purpose": "Professor demo: original EDF input window to Version 2 channel-time artifact probability output",
        "input_edf": str(edf_path),
        "manual_annotation_csv": str(manual_csv),
        "model_checkpoint": str(model_path),
        "model_arch": args.model_arch,
        "subject": args.subject,
        "run": args.run,
        "start_seconds": args.start_sec,
        "window_seconds": args.window_sec,
        "sampling_frequency_hz": sfreq,
        "input_shape": [len(channel_names), window_samples],
        "output_probability_shape": list(probabilities.shape),
        "true_mask_shape": list(true_mask.shape),
        "threshold": args.threshold,
        "metrics_for_this_window": metrics,
        "true_artifact_channel_time_points": int(true_mask.sum()),
        "predicted_artifact_channel_time_points": int(pred_mask.sum()),
        "mask_report_for_full_run": mask_report,
        "created_files": {
            "input_eeg_window_csv": str(input_csv),
            "output_probability_map_csv": str(probability_csv),
            "true_manual_mask_csv": str(true_mask_csv),
            "predicted_threshold_mask_csv": str(pred_mask_csv),
            "heatmap_png": str(heatmap_png),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Version 2 channel-time prediction complete.")
    print("Input shape:", summary["input_shape"])
    print("Output probability shape:", summary["output_probability_shape"])
    print("Dice:", f"{metrics['dice']:.4f}", "IoU:", f"{metrics['iou']:.4f}")
    print("Precision:", f"{metrics['precision']:.4f}", "Recall:", f"{metrics['recall_sensitivity']:.4f}")
    print("Input CSV:", input_csv)
    print("Probability CSV:", probability_csv)
    print("True mask CSV:", true_mask_csv)
    print("Predicted mask CSV:", pred_mask_csv)
    print("Heatmap:", heatmap_png)
    print("Summary:", summary_json)


if __name__ == "__main__":
    main()
