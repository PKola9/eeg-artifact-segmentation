"""
Artifact-wise evaluation on the saved PhysioMotion test-window dataset.

This is the clearest professor-facing version because it uses the same saved
10-second test windows that were used for the segmentation experiments.

The trained model is binary: artifact vs clean. Artifact-wise metrics are
computed by constructing label-specific manual masks for the same saved windows
and comparing the binary model prediction with each label/group mask.
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
DATA_PREP_DIR = PROJECT_ROOT / "data_preparation"
MODELS_DIR = PROJECT_ROOT / "models"
sys.path.insert(0, str(DATA_PREP_DIR))
sys.path.insert(0, str(MODELS_DIR))

from build_channel_time_mask_dataset_from_edf_one_run import annotation_channel_to_indices  # noqa: E402
from build_window_mask_dataset_from_edf import BASELINE_LABELS  # noqa: E402
from splitunet_channel_time_segmentation import SplitUNetChannelTimeSegmenter  # noqa: E402
from temporal_unet_no_channel_mixing import TemporalUNetNoChannelMixing  # noqa: E402


DEFAULT_DATASET = (
    WORKSPACE_ROOT
    / "BFM2_ChannelTimeSegmentation_V2_WORKING"
    / "outputs_v2_channel_time"
    / "window_datasets"
    / "sub30_run06_channel_time_v2_test"
)

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


def label_to_group(label: str) -> str:
    for group, tokens in GROUP_RULES.items():
        if any(token in str(label) for token in tokens):
            return group
    return "other_artifact"


def load_model(checkpoint: Path, device: torch.device, model_arch: str) -> torch.nn.Module:
    model = SplitUNetChannelTimeSegmenter() if model_arch == "split_unet" else TemporalUNetNoChannelMixing()
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


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    artifact_iou = safe_div(tp, tp + fp + fn)
    clean_iou = safe_div(tn, tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    false_positive_rate = safe_div(fp, fp + tn)
    false_negative_rate = safe_div(fn, fn + tp)
    false_discovery_rate = safe_div(fp, fp + tp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    balanced_accuracy = (recall + specificity) / 2.0
    prevalence = safe_div(tp + fn, tp + fp + tn + fn)
    predicted_positive_fraction = safe_div(tp + fp, tp + fp + tn + fn)
    denom_mcc = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    matthews_corrcoef = safe_div(tp * tn - fp * fn, denom_mcc)
    total = tp + fp + tn + fn
    expected_accuracy = safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), total * total)
    cohen_kappa = safe_div(accuracy - expected_accuracy, 1.0 - expected_accuracy)
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
        "balanced_accuracy": balanced_accuracy,
        "negative_predictive_value": npv,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "false_discovery_rate": false_discovery_rate,
        "matthews_corrcoef": matthews_corrcoef,
        "cohen_kappa": cohen_kappa,
        "artifact_prevalence": prevalence,
        "predicted_artifact_fraction": predicted_positive_fraction,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def confusion_metrics(pred: np.ndarray, true: np.ndarray, scope: np.ndarray | None = None) -> dict:
    pred = pred.astype(bool)
    true = true.astype(bool)
    if scope is not None:
        scope = scope.astype(bool)
        pred = pred[scope]
        true = true[scope]
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    tn = int((~pred & ~true).sum())
    fn = int((~pred & true).sum())
    out = metrics_from_counts(tp, fp, tn, fn)
    out["true_positive_samples"] = int(true.sum())
    out["predicted_positive_samples"] = int(pred.sum())
    out["evaluated_channel_time_points"] = int(true.size)
    return out


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool).ravel()
    scores = scores.astype(np.float64).ravel()
    positives = int(y_true.sum())
    negatives = int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_true = y_true[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    n = len(scores)
    while start < n:
        stop = start + 1
        while stop < n and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    positive_rank_sum = ranks[y_true].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision_score_np(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool).ravel()
    scores = scores.astype(np.float64).ravel()
    positives = int(y_true.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_true = y_true[order]
    tp_cum = np.cumsum(sorted_true)
    precision_at_k = tp_cum / (np.arange(len(sorted_true)) + 1)
    return float(precision_at_k[sorted_true].sum() / positives)


def probability_metrics(probs: np.ndarray, true: np.ndarray, scope: np.ndarray | None = None) -> dict:
    true_bool = true.astype(bool)
    probs = probs.astype(np.float64)
    if scope is not None:
        scope = scope.astype(bool)
        true_bool = true_bool[scope]
        probs = probs[scope]
    y = true_bool.ravel()
    p = np.clip(probs.ravel(), 1e-7, 1 - 1e-7)
    if y.size == 0:
        return {
            "brier_score": math.nan,
            "binary_cross_entropy": math.nan,
            "auroc": math.nan,
            "average_precision_pr_auc": math.nan,
            "mean_predicted_probability": math.nan,
            "mean_probability_true_artifact": math.nan,
            "mean_probability_true_clean": math.nan,
        }
    return {
        "brier_score": float(np.mean((p - y.astype(np.float64)) ** 2)),
        "binary_cross_entropy": float(-np.mean(y * np.log(p) + (~y) * np.log(1 - p))),
        "auroc": binary_auc(y, p),
        "average_precision_pr_auc": average_precision_score_np(y, p),
        "mean_predicted_probability": float(np.mean(p)),
        "mean_probability_true_artifact": float(np.mean(p[y])) if y.any() else math.nan,
        "mean_probability_true_clean": float(np.mean(p[~y])) if (~y).any() else math.nan,
    }


def mask_to_segments(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    mask = mask_1d.astype(bool)
    if mask.size == 0 or not mask.any():
        return []
    starts = np.where(mask & np.r_[True, ~mask[:-1]])[0]
    stops = np.where(mask & np.r_[~mask[1:], True])[0] + 1
    return [(int(s), int(e)) for s, e in zip(starts, stops)]


def boundary_points_window_stack(mask: np.ndarray) -> np.ndarray:
    points: list[int] = []
    n_windows, n_channels, n_samples = mask.shape
    for w in range(n_windows):
        base = w * n_samples
        for ch in range(n_channels):
            for start, stop in mask_to_segments(mask[w, ch]):
                points.append(base + start)
                points.append(base + max(start, stop - 1))
    return np.asarray(points, dtype=np.int64)


def symmetric_hausdorff_samples(pred: np.ndarray, true: np.ndarray) -> float:
    pred_points = np.sort(boundary_points_window_stack(pred))
    true_points = np.sort(boundary_points_window_stack(true))
    if pred_points.size == 0 and true_points.size == 0:
        return 0.0
    if pred_points.size == 0 or true_points.size == 0:
        return math.nan

    def directed(a: np.ndarray, b: np.ndarray) -> int:
        idx = np.searchsorted(b, a)
        left = np.where(idx > 0, np.abs(a - b[np.maximum(idx - 1, 0)]), np.iinfo(np.int64).max)
        right = np.where(idx < b.size, np.abs(a - b[np.minimum(idx, b.size - 1)]), np.iinfo(np.int64).max)
        return int(np.minimum(left, right).max())

    return float(max(directed(pred_points, true_points), directed(true_points, pred_points)))


def add_hausdorff(row: dict, pred: np.ndarray, true: np.ndarray, sfreq: float) -> dict:
    hd = symmetric_hausdorff_samples(pred, true)
    row = dict(row)
    row["hausdorff_boundary_samples"] = hd
    row["hausdorff_boundary_seconds"] = float(hd / sfreq) if not math.isnan(hd) else math.nan
    row["hausdorff_boundary_milliseconds"] = float(1000 * hd / sfreq) if not math.isnan(hd) else math.nan
    return row


def predict_windows(model: torch.nn.Module, x: np.ndarray, threshold: float, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    probs_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size].astype(np.float32)).to(device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            probs_all.append(probs)
    probs = np.concatenate(probs_all, axis=0)
    return probs, probs >= threshold


def build_label_window_masks(
    annotations: pd.DataFrame,
    manifest: pd.DataFrame,
    channel_names: list[str],
    sfreq: float,
    window_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    artifact_rows = annotations[~annotations["label"].astype(str).isin(BASELINE_LABELS)].copy()
    labels = sorted(artifact_rows["label"].astype(str).unique())
    groups = sorted({label_to_group(label) for label in labels})
    shape = (len(manifest), len(channel_names), window_samples)
    label_masks = {label: np.zeros(shape, dtype=bool) for label in labels}
    group_masks = {group: np.zeros(shape, dtype=bool) for group in groups}
    channel_to_index = {name: idx for idx, name in enumerate(channel_names)}

    for w, win in manifest.reset_index(drop=True).iterrows():
        win_start = int(win["window_start_sample"])
        win_stop = int(win["window_stop_sample"])
        for row in artifact_rows.itertuples(index=False):
            label = str(row.label)
            group = label_to_group(label)
            ann_start = int(np.floor(float(row.start_time) * sfreq))
            ann_stop = int(np.ceil(float(row.stop_time) * sfreq))
            start = max(win_start, ann_start)
            stop = min(win_stop, ann_stop)
            if stop <= start:
                continue
            try:
                ch_idx = annotation_channel_to_indices(str(row.channel), channel_to_index)
            except ValueError:
                continue
            local_start = start - win_start
            local_stop = stop - win_start
            label_masks[label][w, ch_idx, local_start:local_stop] = True
            group_masks[group][w, ch_idx, local_start:local_stop] = True
    return label_masks, group_masks


def artifact_window_scope(true_mask: np.ndarray) -> np.ndarray:
    present = true_mask.reshape(true_mask.shape[0], -1).any(axis=1)
    scope = np.zeros_like(true_mask, dtype=bool)
    scope[present] = True
    return scope


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_bar(path: Path, rows: list[dict], label_col: str, metric_col: str, title: str) -> None:
    df = pd.DataFrame(rows).sort_values(metric_col, ascending=False)
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(df)), 5.5))
    bars = ax.bar(df[label_col].astype(str), df[metric_col].astype(float), color="#16a34a", edgecolor="black")
    ax.set_title(title)
    ax.set_ylabel(metric_col)
    ax.set_ylim(0, max(1.0, float(df[metric_col].max()) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    for bar, value in zip(bars, df[metric_col]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value), f"{float(value):.3f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--model-arch", choices=["split_unet", "temporal_unet_no_channel_mixing"], default="split_unet")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="06_FINAL_RESULTS_TO_SHOW/FINAL_ARTIFACTWISE_EVALUATION_WINDOW_BASED")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    model_path = Path(args.model)
    output_dir = PROJECT_ROOT / args.output_dir
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(dataset_dir / "manifest.csv")
    arrays = np.load(dataset_dir / "dataset.npz")
    x = arrays["x"].astype(np.float32)
    y = arrays["y"].astype(bool)
    channel_names = metadata["channel_names"]
    sfreq = float(metadata["sampling_frequency_hz"])
    window_samples = int(metadata["window_samples"])
    annotations = pd.read_csv(metadata["manual_annotation_file"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", dataset_dir)
    print("Model:", model_path)
    print("Windows:", x.shape)

    model = load_model(model_path, device, args.model_arch)
    probs, pred = predict_windows(model, x, args.threshold, device, args.batch_size)

    label_masks, _ = build_label_window_masks(annotations, manifest, channel_names, sfreq, window_samples)

    overall = {
        **confusion_metrics(pred, y),
        **probability_metrics(probs, y),
    }
    overall = add_hausdorff(overall, pred, y, sfreq)
    overall_row = {
        "level": "saved_test_windows_overall",
        "dataset_dir": str(dataset_dir),
        "model": str(model_path),
        "threshold": args.threshold,
        "windows": len(x),
        "sampling_frequency_hz": sfreq,
        **overall,
    }

    label_rows: list[dict] = []
    for label, true_mask in label_masks.items():
        scope = artifact_window_scope(true_mask)
        scoped_pred = pred & scope
        scoped_true = true_mask & scope
        row = {
            **confusion_metrics(pred, true_mask, scope),
            **probability_metrics(probs, true_mask, scope),
        }
        row = add_hausdorff(row, scoped_pred, scoped_true, sfreq)
        label_rows.append({"artifact_label": label, "threshold": args.threshold, **row})

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "01_overall_saved_test_window_metrics.csv", [overall_row])
    write_csv(output_dir / "02_per_artifact_label_saved_test_window_metrics.csv", label_rows)
    plot_bar(output_dir / "05_per_artifact_label_dice_saved_windows.png", label_rows, "artifact_label", "dice", "Raw annotation-label Dice on saved test windows")

    note = f"""# Window-based artifact-wise evaluation

This folder evaluates artifact-wise metrics on the saved 10-second test windows:

`{dataset_dir}`

This is easier to compare with the model's existing test-window Dice because it
uses the same prepared test-window format.

The model is binary artifact-vs-clean. Artifact-wise metrics are calculated by
creating a manual mask for each individual annotation label on the same saved
test windows, then comparing the binary model prediction with that label mask.

Threshold: `{args.threshold}`

Overall saved-window Dice: `{overall_row['dice']:.4f}`
Overall saved-window IoU: `{overall_row['iou_jaccard_artifact']:.4f}`
Overall saved-window mIoU: `{overall_row['miou_binary_clean_artifact']:.4f}`
"""
    (output_dir / "00_OPEN_THIS_FIRST_WINDOW_BASED_ARTIFACTWISE.md").write_text(note, encoding="utf-8")
    (output_dir / "06_summary.json").write_text(json.dumps({"overall": overall_row, "artifact_labels": label_rows}, indent=2), encoding="utf-8")

    print("Saved:", output_dir)
    print("Overall saved-window Dice:", f"{overall_row['dice']:.4f}")


if __name__ == "__main__":
    main()
