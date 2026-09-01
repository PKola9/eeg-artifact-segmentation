"""
Evaluate the selected M3 improved model as artifact start/end intervals.

This script is intentionally separate from the previous result files.
It creates a clean professor-facing folder for:

    M3 improved = Split U-Net initialized from flow matching trained on
    PhysioMotion + BCI IV + EEGMMIDB.

It evaluates the saved held-out test windows and reports:
    1. strict sample-level segmentation metrics
    2. predicted and manual artifact intervals per channel/window
    3. one-to-one interval matches
    4. missed manual intervals and false predicted intervals
    5. tolerance sweep for start/end interval agreement
    6. summary figures

The existing result files are not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.splitunet_channel_time_segmentation import SplitUNetChannelTimeSegmenter


@dataclass(frozen=True)
class Interval:
    interval_id: str
    source: str
    window_index: int
    channel_index: int
    channel: str
    start_sample_in_window: int
    stop_sample_in_window_exclusive: int
    start_seconds_in_window: float
    end_seconds_in_window: float
    start_seconds_in_recording: float
    end_seconds_in_recording: float
    duration_seconds: float
    mean_probability: float | None = None
    max_probability: float | None = None


def contiguous_regions(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop_exclusive) segments where mask is True."""
    mask = mask_1d.astype(bool)
    if mask.size == 0 or not mask.any():
        return []
    changes = np.diff(mask.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    stops = np.where(changes == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        stops = np.r_[stops, mask.size]
    return [(int(s), int(e)) for s, e in zip(starts, stops)]


def intervals_from_masks(
    binary_masks: np.ndarray,
    probabilities: np.ndarray | None,
    manifest: pd.DataFrame,
    channel_names: list[str],
    sfreq: float,
    source: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    n_windows, n_channels, _ = binary_masks.shape

    for w in range(n_windows):
        window_start_sample = int(manifest.loc[w, "window_start_sample"])
        window_start_seconds = window_start_sample / sfreq

        for c in range(n_channels):
            for seg_i, (start, stop) in enumerate(contiguous_regions(binary_masks[w, c] > 0)):
                start_win_sec = start / sfreq
                end_win_sec = stop / sfreq
                start_rec_sec = window_start_seconds + start_win_sec
                end_rec_sec = window_start_seconds + end_win_sec
                duration = end_win_sec - start_win_sec
                interval_probs = None if probabilities is None else probabilities[w, c, start:stop]

                rows.append(
                    {
                        "interval_id": f"{source}_w{w:04d}_c{c:02d}_i{seg_i:03d}",
                        "source": source,
                        "window_index": w,
                        "channel_index": c,
                        "channel": channel_names[c],
                        "start_sample_in_window": start,
                        "stop_sample_in_window_exclusive": stop,
                        "start_seconds_in_window": start_win_sec,
                        "end_seconds_in_window": end_win_sec,
                        "start_seconds_in_recording": start_rec_sec,
                        "end_seconds_in_recording": end_rec_sec,
                        "duration_seconds": duration,
                        "mean_probability": float(np.mean(interval_probs)) if interval_probs is not None and interval_probs.size else np.nan,
                        "max_probability": float(np.max(interval_probs)) if interval_probs is not None and interval_probs.size else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def interval_union(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(a_end, b_end) - min(a_start, b_start)


def match_intervals(manual_df: pd.DataFrame, pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Greedy one-to-one strict interval matching by same window/channel and highest IoU."""
    if manual_df.empty or pred_df.empty:
        return pd.DataFrame(), manual_df.copy(), pred_df.copy()

    used_manual: set[str] = set()
    used_pred: set[str] = set()
    selected_rows: list[dict] = []

    pred_groups = {key: group.reset_index(drop=True) for key, group in pred_df.groupby(["window_index", "channel_index"])}

    for key, manual_group_raw in manual_df.groupby(["window_index", "channel_index"]):
        if key not in pred_groups:
            continue

        manual_group = manual_group_raw.reset_index(drop=True)
        pred_group = pred_groups[key]

        ms = manual_group["start_seconds_in_recording"].to_numpy()[:, None]
        me = manual_group["end_seconds_in_recording"].to_numpy()[:, None]
        ps = pred_group["start_seconds_in_recording"].to_numpy()[None, :]
        pe = pred_group["end_seconds_in_recording"].to_numpy()[None, :]

        overlap = np.maximum(0.0, np.minimum(me, pe) - np.maximum(ms, ps))
        positive = np.argwhere(overlap > 0)
        if positive.size == 0:
            continue

        union = np.maximum(me, pe) - np.minimum(ms, ps)
        mdur = manual_group["duration_seconds"].to_numpy()[:, None]
        pdur = pred_group["duration_seconds"].to_numpy()[None, :]
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        dice = np.divide(2.0 * overlap, mdur + pdur, out=np.zeros_like(overlap), where=(mdur + pdur) > 0)

        order = sorted(positive.tolist(), key=lambda ij: (iou[ij[0], ij[1]], dice[ij[0], ij[1]]), reverse=True)
        local_manual_used: set[int] = set()
        local_pred_used: set[int] = set()

        for mi, pi in order:
            if mi in local_manual_used or pi in local_pred_used:
                continue
            m = manual_group.iloc[mi]
            p = pred_group.iloc[pi]
            m_id = m["interval_id"]
            p_id = p["interval_id"]
            if m_id in used_manual or p_id in used_pred:
                continue
            local_manual_used.add(mi)
            local_pred_used.add(pi)
            used_manual.add(m_id)
            used_pred.add(p_id)
            selected_rows.append(
                {
                    "manual_interval_id": m_id,
                    "predicted_interval_id": p_id,
                    "window_index": int(key[0]),
                    "channel_index": int(key[1]),
                    "channel": m["channel"],
                    "manual_start_seconds_in_recording": m["start_seconds_in_recording"],
                    "manual_end_seconds_in_recording": m["end_seconds_in_recording"],
                    "predicted_start_seconds_in_recording": p["start_seconds_in_recording"],
                    "predicted_end_seconds_in_recording": p["end_seconds_in_recording"],
                    "manual_duration_seconds": m["duration_seconds"],
                    "predicted_duration_seconds": p["duration_seconds"],
                    "overlap_seconds": float(overlap[mi, pi]),
                    "union_seconds": float(union[mi, pi]),
                    "interval_iou": float(iou[mi, pi]),
                    "interval_dice": float(dice[mi, pi]),
                    "start_error_seconds": abs(p["start_seconds_in_recording"] - m["start_seconds_in_recording"]),
                    "end_error_seconds": abs(p["end_seconds_in_recording"] - m["end_seconds_in_recording"]),
                    "center_error_seconds": abs(
                        ((p["start_seconds_in_recording"] + p["end_seconds_in_recording"]) / 2.0)
                        - ((m["start_seconds_in_recording"] + m["end_seconds_in_recording"]) / 2.0)
                    ),
                    "duration_error_seconds": abs(p["duration_seconds"] - m["duration_seconds"]),
                    "predicted_mean_probability": p.get("mean_probability", np.nan),
                    "predicted_max_probability": p.get("max_probability", np.nan),
                }
            )

    matched_df = pd.DataFrame(selected_rows)
    missed_df = manual_df[~manual_df["interval_id"].isin(used_manual)].copy()
    false_df = pred_df[~pred_df["interval_id"].isin(used_pred)].copy()
    return matched_df, missed_df, false_df


def tolerant_interval_scores(manual_df: pd.DataFrame, pred_df: pd.DataFrame, buffers: list[float]) -> pd.DataFrame:
    rows: list[dict] = []
    manual_n = len(manual_df)
    pred_n = len(pred_df)
    manual_groups = {key: group.reset_index(drop=True) for key, group in manual_df.groupby(["window_index", "channel_index"])}
    pred_groups = {key: group.reset_index(drop=True) for key, group in pred_df.groupby(["window_index", "channel_index"])}

    for buffer_sec in buffers:
        detected_manual = 0
        true_pred = 0

        for key, manual_group in manual_groups.items():
            pred_group = pred_groups.get(key)
            if pred_group is None or pred_group.empty:
                continue
            ms = manual_group["start_seconds_in_recording"].to_numpy()[:, None] - buffer_sec
            me = manual_group["end_seconds_in_recording"].to_numpy()[:, None] + buffer_sec
            ps = pred_group["start_seconds_in_recording"].to_numpy()[None, :]
            pe = pred_group["end_seconds_in_recording"].to_numpy()[None, :]
            hit_matrix = np.maximum(0.0, np.minimum(me, pe) - np.maximum(ms, ps)) > 0
            detected_manual += int(hit_matrix.any(axis=1).sum())

        for key, pred_group in pred_groups.items():
            manual_group = manual_groups.get(key)
            if manual_group is None or manual_group.empty:
                continue
            ps = pred_group["start_seconds_in_recording"].to_numpy()[:, None] - buffer_sec
            pe = pred_group["end_seconds_in_recording"].to_numpy()[:, None] + buffer_sec
            ms = manual_group["start_seconds_in_recording"].to_numpy()[None, :]
            me = manual_group["end_seconds_in_recording"].to_numpy()[None, :]
            hit_matrix = np.maximum(0.0, np.minimum(pe, me) - np.maximum(ps, ms)) > 0
            true_pred += int(hit_matrix.any(axis=1).sum())

        precision = true_pred / pred_n if pred_n else 0.0
        recall = detected_manual / manual_n if manual_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        rows.append(
            {
                "buffer_seconds": buffer_sec,
                "manual_intervals": manual_n,
                "predicted_intervals": pred_n,
                "manual_intervals_detected": detected_manual,
                "predicted_intervals_with_manual_match": true_pred,
                "interval_precision": precision,
                "interval_recall": recall,
                "interval_f1_dice_style": f1,
            }
        )

    return pd.DataFrame(rows)


def sample_level_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray, sfreq: float) -> dict:
    y = y_true.astype(bool).ravel()
    p = pred.astype(bool).ravel()
    prob_flat = prob.astype(np.float32).ravel()
    tp = int(np.logical_and(p, y).sum())
    fp = int(np.logical_and(p, ~y).sum())
    tn = int(np.logical_and(~p, ~y).sum())
    fn = int(np.logical_and(~p, y).sum())

    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou_art = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    iou_clean = tn / (tn + fp + fn) if (tn + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = dice
    balanced_accuracy = (recall + specificity) / 2.0

    true_boundaries: list[int] = []
    pred_boundaries: list[int] = []
    # Boundary locations in flattened sample units are not clinically meaningful because flattening
    # crosses channels/windows. Instead, calculate Hausdorff after interval extraction in seconds.

    return {
        "dice": dice,
        "iou_jaccard_artifact": iou_art,
        "iou_clean": iou_clean,
        "miou_binary_clean_artifact": (iou_art + iou_clean) / 2.0,
        "precision": precision,
        "recall_sensitivity": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "artifact_prevalence": float(y.mean()),
        "predicted_artifact_fraction": float(p.mean()),
        "mean_predicted_probability": float(prob_flat.mean()),
        "mean_probability_true_artifact": float(prob_flat[y].mean()) if y.any() else np.nan,
        "mean_probability_true_clean": float(prob_flat[~y].mean()) if (~y).any() else np.nan,
        "evaluated_channel_time_points": int(y.size),
    }


def hausdorff_boundary_seconds(manual_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    true_boundaries = np.r_[
        manual_df["start_seconds_in_recording"].to_numpy(),
        manual_df["end_seconds_in_recording"].to_numpy(),
    ]
    pred_boundaries = np.r_[
        pred_df["start_seconds_in_recording"].to_numpy(),
        pred_df["end_seconds_in_recording"].to_numpy(),
    ]
    if len(true_boundaries) == 0 or len(pred_boundaries) == 0:
        return float("nan")
    # Directed maximum of nearest-neighbor distances in both directions.
    true_to_pred = np.max([np.min(np.abs(pred_boundaries - t)) for t in true_boundaries])
    pred_to_true = np.max([np.min(np.abs(true_boundaries - p)) for p in pred_boundaries])
    return float(max(true_to_pred, pred_to_true))


def save_bar_plot(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, color: str = "#1f77b4"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=color, edgecolor="black")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_tolerance_plot(path: Path, tol_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(tol_df["buffer_seconds"], tol_df["interval_f1_dice_style"], marker="o", linewidth=2.5, label="Interval Dice/F1")
    ax.plot(tol_df["buffer_seconds"], tol_df["interval_precision"], marker="o", linewidth=2.0, label="Interval precision")
    ax.plot(tol_df["buffer_seconds"], tol_df["interval_recall"], marker="o", linewidth=2.0, label="Interval recall")
    ax.set_title("M3 improved interval agreement across boundary buffers", fontsize=15, fontweight="bold")
    ax.set_xlabel("Boundary buffer in seconds")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buffers", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sub in [
        "01_SAMPLE_LEVEL_METRICS",
        "02_INTERVAL_TABLES",
        "03_TOLERANCE_SWEEP",
        "04_PER_CHANNEL",
        "05_FIGURES",
    ]:
        (args.output_dir / sub).mkdir(exist_ok=True)

    with (args.dataset_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    channel_names = metadata["channel_names"]
    sfreq = float(metadata["sampling_frequency_hz"])
    manifest = pd.read_csv(args.dataset_dir / "manifest.csv")

    data = np.load(args.dataset_dir / "dataset.npz")
    x = data["x"].astype(np.float32)
    y = (data["y"] > 0.5).astype(np.uint8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SplitUNetChannelTimeSegmenter()
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    probabilities = np.zeros_like(x, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            xb = torch.from_numpy(x[start : start + args.batch_size]).to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            probabilities[start : start + len(probs)] = probs

    pred = (probabilities >= args.threshold).astype(np.uint8)

    sample_metrics = sample_level_metrics(y, probabilities, pred, sfreq)
    manual_df = intervals_from_masks(y, None, manifest, channel_names, sfreq, "manual_ground_truth")
    pred_df = intervals_from_masks(pred, probabilities, manifest, channel_names, sfreq, "m3_improved_prediction")
    matched_df, missed_df, false_df = match_intervals(manual_df, pred_df)
    tolerance_df = tolerant_interval_scores(manual_df, pred_df, args.buffers)

    haus = hausdorff_boundary_seconds(manual_df, pred_df)
    sample_metrics.update(
        {
            "model_name": "M3 improved multi-dataset flow-init Split U-Net",
            "flow_pretraining": "FM2: PhysioMotion + BCI IV + EEGMMIDB",
            "threshold": args.threshold,
            "windows": int(x.shape[0]),
            "channels": int(x.shape[1]),
            "time_samples_per_window": int(x.shape[2]),
            "sampling_frequency_hz": sfreq,
            "manual_intervals": int(len(manual_df)),
            "predicted_intervals": int(len(pred_df)),
            "strict_matched_intervals": int(len(matched_df)),
            "strict_missed_manual_intervals": int(len(missed_df)),
            "strict_false_predicted_intervals": int(len(false_df)),
            "hausdorff_boundary_seconds": haus,
            "hausdorff_boundary_milliseconds": haus * 1000.0 if not math.isnan(haus) else np.nan,
        }
    )

    if not matched_df.empty:
        for metric in ["interval_dice", "interval_iou", "start_error_seconds", "end_error_seconds", "center_error_seconds", "duration_error_seconds"]:
            sample_metrics[f"matched_mean_{metric}"] = float(matched_df[metric].mean())
            sample_metrics[f"matched_median_{metric}"] = float(matched_df[metric].median())
    else:
        for metric in ["interval_dice", "interval_iou", "start_error_seconds", "end_error_seconds", "center_error_seconds", "duration_error_seconds"]:
            sample_metrics[f"matched_mean_{metric}"] = np.nan
            sample_metrics[f"matched_median_{metric}"] = np.nan

    sample_df = pd.DataFrame([sample_metrics])
    sample_df.to_csv(args.output_dir / "01_SAMPLE_LEVEL_METRICS" / "m3_improved_overall_sample_metrics.csv", index=False)
    manual_df.to_csv(args.output_dir / "02_INTERVAL_TABLES" / "manual_artifact_intervals.csv", index=False)
    pred_df.to_csv(args.output_dir / "02_INTERVAL_TABLES" / "m3_improved_predicted_artifact_intervals.csv", index=False)
    matched_df.to_csv(args.output_dir / "02_INTERVAL_TABLES" / "m3_improved_matched_intervals_strict.csv", index=False)
    missed_df.to_csv(args.output_dir / "02_INTERVAL_TABLES" / "m3_improved_missed_manual_intervals_strict.csv", index=False)
    false_df.to_csv(args.output_dir / "02_INTERVAL_TABLES" / "m3_improved_false_predicted_intervals_strict.csv", index=False)
    tolerance_df.to_csv(args.output_dir / "03_TOLERANCE_SWEEP" / "m3_improved_interval_tolerance_sweep.csv", index=False)

    if not matched_df.empty:
        per_channel = (
            matched_df.groupby(["channel_index", "channel"])
            .agg(
                matched_intervals=("manual_interval_id", "count"),
                mean_interval_dice=("interval_dice", "mean"),
                mean_interval_iou=("interval_iou", "mean"),
                median_start_error_seconds=("start_error_seconds", "median"),
                median_end_error_seconds=("end_error_seconds", "median"),
                mean_start_error_seconds=("start_error_seconds", "mean"),
                mean_end_error_seconds=("end_error_seconds", "mean"),
            )
            .reset_index()
            .sort_values("matched_intervals", ascending=False)
        )
    else:
        per_channel = pd.DataFrame()
    per_channel.to_csv(args.output_dir / "04_PER_CHANNEL" / "m3_improved_per_channel_interval_summary.csv", index=False)

    # Figures
    save_bar_plot(
        args.output_dir / "05_FIGURES" / "m3_improved_main_overlap_metrics.png",
        ["Dice", "IoU", "mIoU", "Precision", "Recall", "Accuracy", "Specificity"],
        [
            sample_metrics["dice"],
            sample_metrics["iou_jaccard_artifact"],
            sample_metrics["miou_binary_clean_artifact"],
            sample_metrics["precision"],
            sample_metrics["recall_sensitivity"],
            sample_metrics["accuracy"],
            sample_metrics["specificity"],
        ],
        "M3 improved: overall saved test-window metrics",
        "Score",
        "#2b83ba",
    )
    save_tolerance_plot(args.output_dir / "05_FIGURES" / "m3_improved_interval_tolerance_sweep.png", tolerance_df)

    if not matched_df.empty:
        error_values = [
            sample_metrics["matched_median_start_error_seconds"],
            sample_metrics["matched_median_end_error_seconds"],
            sample_metrics["matched_median_center_error_seconds"],
            sample_metrics["matched_median_duration_error_seconds"],
        ]
        save_bar_plot(
            args.output_dir / "05_FIGURES" / "m3_improved_interval_boundary_errors_seconds.png",
            ["Start error", "End error", "Center error", "Duration error"],
            error_values,
            "M3 improved: median interval boundary errors",
            "Seconds",
            "#f28e2b",
        )

    readme = f"""# M3 improved interval-matching results

This folder evaluates only the selected best flow-initialized Split U-Net:

- Model: M3 improved multi-dataset flow-init Split U-Net
- Flow pretraining: FM2 = PhysioMotion + BCI IV + EEGMMIDB
- Checkpoint: `{args.model}`
- Threshold: `{args.threshold}`
- Dataset: saved held-out test windows from subject sub-30 run-06
- Window shape: 59 channels × 2500 samples = 10 seconds at 250 Hz

Nothing in the old result folders was changed.

## What to open first

1. `01_SAMPLE_LEVEL_METRICS/m3_improved_overall_sample_metrics.csv`
   - Overall Dice, IoU, mIoU, precision, recall, accuracy, specificity, and interval counts.

2. `02_INTERVAL_TABLES/m3_improved_predicted_artifact_intervals.csv`
   - Every predicted artifact interval with channel, start time, end time, duration, and probability.

3. `02_INTERVAL_TABLES/manual_artifact_intervals.csv`
   - Every manual ground-truth interval extracted from the saved test masks.

4. `02_INTERVAL_TABLES/m3_improved_matched_intervals_strict.csv`
   - One-to-one predicted-vs-manual interval matches on the same window and channel.

5. `03_TOLERANCE_SWEEP/m3_improved_interval_tolerance_sweep.csv`
   - Interval detection precision/recall/F1 under different boundary buffers.

6. `05_FIGURES/`
   - Ready-to-show graphs.

## Important interpretation

Sample-level Dice measures overlap between the full predicted binary mask and the full manual binary mask across all channels and time samples.

Interval matching converts those masks into start/end segments and asks whether predicted artifact intervals align with manual artifact intervals on the same channel and window.
"""
    (args.output_dir / "00_READ_ME_FIRST.md").write_text(readme, encoding="utf-8")

    print("Saved clean M3 improved interval results in:")
    print(args.output_dir)
    print("\nMain metrics:")
    print(sample_df[["dice", "iou_jaccard_artifact", "miou_binary_clean_artifact", "precision", "recall_sensitivity", "accuracy", "specificity", "manual_intervals", "predicted_intervals", "strict_matched_intervals"]].to_string(index=False))
    print("\nTolerance sweep:")
    print(tolerance_df.to_string(index=False))


if __name__ == "__main__":
    main()
