"""
Final artifact-wise evaluation for the V6 EEG artifact segmentation model.

This script evaluates a trained binary channel-time artifact segmentation model
on the held-out PhysioMotion test recording and reports:

1. Overall test-set metrics.
2. Per raw annotation-label metrics.
3. Temporal boundary Hausdorff distance in samples, seconds, and milliseconds.

Important interpretation:
    The trained model predicts artifact vs clean. It does not predict artifact
    class names. Therefore, artifact-wise evaluation is computed as a
    one-vs-rest analysis: for each annotation label/category, a separate
    ground-truth mask is built and compared with the binary model prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
DATA_PREP_DIR = PROJECT_ROOT / "data_preparation"
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(DATA_PREP_DIR))
sys.path.insert(0, str(MODELS_DIR))

from build_channel_time_mask_dataset_from_edf_one_run import annotation_channel_to_indices  # noqa: E402
from build_window_mask_dataset_from_edf import (  # noqa: E402
    BASELINE_LABELS,
    make_paths,
    normalize_window,
    read_edf_downsampled,
)
from splitunet_channel_time_segmentation import SplitUNetChannelTimeSegmenter  # noqa: E402
from temporal_unet_no_channel_mixing import TemporalUNetNoChannelMixing  # noqa: E402


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "outputs_v6_flowmatching_study"
    / "training_runs"
    / "V6_CLEAN_M3_multidataset_flowinit_splitunet"
    / "splitunet_channel_time_best_val_dice.pt"
)


GROUP_RULES = {
    "eye_blink": ["blink"],
    "eye_movement": ["eyem"],
    "head_movement": ["headm"],
    "facial_muscle": ["eyebrow", "chew"],
    "mouth_tongue_swallow": ["tongue", "swallow"],
}


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


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def confusion_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    tn = int((~pred & ~true).sum())
    fn = int((~pred & true).sum())

    artifact_iou = safe_div(tp, tp + fp + fn)
    clean_iou = safe_div(tn, tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)

    return {
        "dice": dice,
        "iou_jaccard_artifact": artifact_iou,
        "iou_clean": clean_iou,
        "miou_binary_clean_artifact": (artifact_iou + clean_iou) / 2.0,
        "precision": precision,
        "recall_sensitivity": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "true_positive_samples": int(true.sum()),
        "predicted_positive_samples": int(pred.sum()),
    }


def mask_to_segments(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    mask = mask_1d.astype(bool)
    if mask.size == 0 or not mask.any():
        return []
    starts = np.where(mask & np.r_[True, ~mask[:-1]])[0]
    stops = np.where(mask & np.r_[~mask[1:], True])[0] + 1
    return [(int(s), int(e)) for s, e in zip(starts, stops)]


def boundary_points(mask: np.ndarray) -> np.ndarray:
    points: list[int] = []
    for ch in range(mask.shape[0]):
        for start, stop in mask_to_segments(mask[ch]):
            points.append(start)
            points.append(max(start, stop - 1))
    return np.asarray(points, dtype=np.int64)


def symmetric_hausdorff_1d(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred_points = boundary_points(pred_mask)
    true_points = boundary_points(true_mask)
    if pred_points.size == 0 and true_points.size == 0:
        return 0.0
    if pred_points.size == 0 or true_points.size == 0:
        return math.nan

    pred_points = np.sort(pred_points)
    true_points = np.sort(true_points)

    def directed(a: np.ndarray, b: np.ndarray) -> int:
        idx = np.searchsorted(b, a)
        left = np.where(idx > 0, np.abs(a - b[np.maximum(idx - 1, 0)]), np.iinfo(np.int64).max)
        right = np.where(idx < b.size, np.abs(a - b[np.minimum(idx, b.size - 1)]), np.iinfo(np.int64).max)
        return int(np.minimum(left, right).max())

    return float(max(directed(pred_points, true_points), directed(true_points, pred_points)))


def add_hausdorff(metrics: dict, pred_mask: np.ndarray, true_mask: np.ndarray, sfreq: float) -> dict:
    hd_samples = symmetric_hausdorff_1d(pred_mask, true_mask)
    metrics = dict(metrics)
    metrics["hausdorff_boundary_samples"] = hd_samples
    metrics["hausdorff_boundary_seconds"] = float(hd_samples / sfreq) if not math.isnan(hd_samples) else math.nan
    metrics["hausdorff_boundary_milliseconds"] = (
        float(1000.0 * hd_samples / sfreq) if not math.isnan(hd_samples) else math.nan
    )
    return metrics


def label_to_group(label: str) -> str:
    label = str(label)
    for group, tokens in GROUP_RULES.items():
        if any(token in label for token in tokens):
            return group
    return "other_artifact"


def create_masks_by_label(
    annotations: pd.DataFrame,
    channel_names: list[str],
    duration_seconds: float,
    sfreq: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n_samples = int(np.floor(duration_seconds * sfreq))
    channel_to_index = {name: idx for idx, name in enumerate(channel_names)}

    artifact_rows = annotations[~annotations["label"].astype(str).isin(BASELINE_LABELS)].copy()
    raw_labels = sorted(artifact_rows["label"].astype(str).unique().tolist())
    groups = sorted({label_to_group(label) for label in raw_labels})

    label_masks = {
        label: np.zeros((len(channel_names), n_samples), dtype=bool)
        for label in raw_labels
    }
    group_masks = {
        group: np.zeros((len(channel_names), n_samples), dtype=bool)
        for group in groups
    }

    for row in artifact_rows.itertuples(index=False):
        label = str(row.label)
        group = label_to_group(label)
        start = max(0.0, float(row.start_time))
        stop = min(duration_seconds, float(row.stop_time))
        if stop <= start:
            continue
        start_idx = max(0, min(n_samples, int(np.floor(start * sfreq))))
        stop_idx = max(0, min(n_samples, int(np.ceil(stop * sfreq))))
        if stop_idx <= start_idx:
            continue
        try:
            channel_indices = annotation_channel_to_indices(str(row.channel), channel_to_index)
        except ValueError:
            continue
        label_masks[label][channel_indices, start_idx:stop_idx] = True
        group_masks[group][channel_indices, start_idx:stop_idx] = True

    return label_masks, group_masks


def predict_full_recording(
    model: torch.nn.Module,
    eeg: np.ndarray,
    sfreq: float,
    device: torch.device,
    threshold: float,
    window_sec: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    window_samples = int(round(window_sec * sfreq))
    n_channels, n_samples = eeg.shape
    n_windows = n_samples // window_samples
    usable_samples = n_windows * window_samples

    probability = np.zeros((n_channels, usable_samples), dtype=np.float32)

    starts = list(range(0, usable_samples, window_samples))
    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            batch = np.stack(
                [
                    normalize_window(eeg[:, start : start + window_samples]).astype(np.float32)
                    for start in batch_starts
                ]
            )
            x = torch.from_numpy(batch).to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            for start, prob in zip(batch_starts, probs):
                probability[:, start : start + window_samples] = prob

    pred_mask = probability >= threshold
    return probability, pred_mask


def crop_mask(mask: np.ndarray, usable_samples: int) -> np.ndarray:
    return mask[:, :usable_samples].astype(bool)


def artifact_window_scope(true_mask: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Build an evaluation scope for artifact-wise metrics.

    For a specific artifact type, metrics are evaluated only on windows where
    that artifact type is present. This avoids treating predictions in unrelated
    artifact windows as false positives for the selected artifact type.
    """
    n_channels, n_samples = true_mask.shape
    scope = np.zeros((n_channels, n_samples), dtype=bool)
    n_windows = n_samples // window_samples
    for w in range(n_windows):
        start = w * window_samples
        stop = start + window_samples
        if true_mask[:, start:stop].any():
            scope[:, start:stop] = True
    return scope


def apply_scope(mask: np.ndarray, scope: np.ndarray) -> np.ndarray:
    return mask.astype(bool) & scope.astype(bool)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_bar(path: Path, df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str) -> None:
    if df.empty:
        return
    plot_df = df.sort_values(y_col, ascending=False)
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(plot_df)), 5.5))
    bars = ax.bar(plot_df[x_col].astype(str), plot_df[y_col], color="#2563eb", edgecolor="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, float(plot_df[y_col].max()) * 1.15 if np.isfinite(plot_df[y_col]).any() else 1.0))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, plot_df[y_col]):
        if pd.notna(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-30")
    parser.add_argument("--run", type=int, default=6)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-arch", default="split_unet", choices=["split_unet", "temporal_unet_no_channel_mixing"])
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--downsample-to-hz", type=float, default=250.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="06_FINAL_RESULTS_TO_SHOW/FINAL_ARTIFACTWISE_EVALUATION",
    )
    args = parser.parse_args()

    edf_path, _, manual_csv = make_paths(args.subject, args.run)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    output_dir = PROJECT_ROOT / args.output_dir

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    if not manual_csv.exists():
        raise FileNotFoundError(f"Manual annotation CSV not found: {manual_csv}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Model:", model_path)
    print("EDF:", edf_path)
    print("Annotations:", manual_csv)

    eeg, channel_names, sfreq, duration_seconds = read_edf_downsampled(edf_path, args.downsample_to_hz)
    annotations = pd.read_csv(manual_csv)
    label_masks, group_masks = create_masks_by_label(annotations, channel_names, duration_seconds, sfreq)

    model = load_model(model_path, device, args.model_arch)
    probability, pred_mask = predict_full_recording(
        model=model,
        eeg=eeg,
        sfreq=sfreq,
        device=device,
        threshold=args.threshold,
        window_sec=args.window_sec,
        batch_size=args.batch_size,
    )

    usable_samples = pred_mask.shape[1]
    window_samples = int(round(args.window_sec * sfreq))
    overall_true = np.zeros_like(pred_mask, dtype=bool)
    for mask in label_masks.values():
        overall_true |= crop_mask(mask, usable_samples)

    overall = add_hausdorff(confusion_metrics(pred_mask, overall_true), pred_mask, overall_true, sfreq)
    overall_row = {
        "level": "overall_test_recording",
        "subject": args.subject,
        "run": args.run,
        "threshold": args.threshold,
        "sampling_frequency_hz": sfreq,
        "evaluated_duration_seconds": usable_samples / sfreq,
        **overall,
    }

    label_rows: list[dict] = []
    for label, full_mask in label_masks.items():
        true_mask = crop_mask(full_mask, usable_samples)
        scope = artifact_window_scope(true_mask, window_samples)
        scoped_pred = apply_scope(pred_mask, scope)
        scoped_true = apply_scope(true_mask, scope)
        metrics = add_hausdorff(confusion_metrics(scoped_pred, scoped_true), scoped_pred, scoped_true, sfreq)
        label_rows.append(
            {
                "artifact_label": label,
                "evaluation_scope": "only_nonoverlapping_10s_test_windows_containing_this_artifact_label",
                "scoped_duration_seconds_channel_summed": float(scope.sum() / sfreq),
                "threshold": args.threshold,
                **metrics,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "01_overall_test_metrics.csv", [overall_row])
    write_csv(output_dir / "02_per_artifact_label_metrics.csv", label_rows)

    label_df = pd.DataFrame(label_rows)
    plot_bar(output_dir / "06_per_artifact_label_dice.png", label_df, "artifact_label", "dice", "Per-artifact label Dice", "Dice")

    definitions = f"""# Final artifact-wise evaluation metrics

Model checkpoint:
`{model_path}`

Test recording:
`{edf_path}`

Manual annotations:
`{manual_csv}`

Threshold used:
`{args.threshold}`

Sampling frequency:
`{sfreq} Hz`

## Metric categories

### Segmentation / overlap metrics

- Dice coefficient: overlap between predicted artifact mask and manual artifact mask.
- IoU / Jaccard index: intersection divided by union of predicted and manual artifact masks.
- mIoU: mean of artifact-class IoU and clean-class IoU for this binary segmentation task.

### Detection metrics

- Precision: among samples predicted as artifact, how many were truly artifact.
- Recall / sensitivity: among true artifact samples, how many were detected.
- F1-score: harmonic mean of precision and recall.
- Accuracy: overall correct artifact/clean decisions.
- Specificity: among true clean samples, how many were correctly kept clean.

### Temporal boundary metric

- Hausdorff distance is calculated using start/end boundary points of predicted and
  manual artifact intervals. It is reported in samples, seconds, and milliseconds.
  This directly measures how far the predicted artifact start/end timing can be
  from the manual annotation boundary.

## Artifact-wise interpretation

The trained model is binary: artifact vs clean. It does not output artifact class
names. Therefore, per-artifact results are artifact-conditioned evaluations.
For each individual manual annotation label, a separate ground-truth mask is
created. Metrics are then calculated only on non-overlapping 10-second test
windows where that artifact label is present. This answers: when this artifact
label appears in the test recording, how well does the binary artifact detector
cover it?
"""
    (output_dir / "05_metric_definitions_for_thesis.md").write_text(definitions, encoding="utf-8")

    summary = {
        "purpose": "Overall and per-artifact evaluation of final EEG artifact segmentation model.",
        "model_checkpoint": str(model_path),
        "test_recording": str(edf_path),
        "manual_annotation_csv": str(manual_csv),
        "threshold": args.threshold,
        "sampling_frequency_hz": sfreq,
        "overall_metrics": overall_row,
        "artifact_labels": sorted(label_masks.keys()),
        "created_files": {
            "overall_csv": str(output_dir / "01_overall_test_metrics.csv"),
            "per_artifact_label_csv": str(output_dir / "02_per_artifact_label_metrics.csv"),
            "definitions_md": str(output_dir / "05_metric_definitions_for_thesis.md"),
        },
    }
    (output_dir / "04_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved final artifact-wise evaluation to:")
    print(output_dir)
    print("Overall Dice:", f"{overall_row['dice']:.4f}")
    print("Overall IoU:", f"{overall_row['iou_jaccard_artifact']:.4f}")
    print("Overall mIoU:", f"{overall_row['miou_binary_clean_artifact']:.4f}")
    print("Overall Hausdorff seconds:", overall_row["hausdorff_boundary_seconds"])


if __name__ == "__main__":
    main()
