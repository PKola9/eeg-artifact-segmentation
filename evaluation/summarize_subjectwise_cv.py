"""
Summarize subject-wise cross-validation results.

This script reads the five subject-wise folds for:
    - random Split U-Net
    - M3 flow-initialized Split U-Net
    - M3 flow-initialized Split U-Net with validation-selected smoothing

It writes two CSV files:
    1. per-fold results
    2. mean/std summary by method
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs_v6_flowmatching_study"

FOLDS = [
    ("fold0", "sub-1, sub-10, sub-15, sub-20, sub-25, sub-30"),
    ("fold1", "sub-2, sub-11, sub-16, sub-21, sub-26, sub-29"),
    ("fold2", "sub-3, sub-12, sub-17, sub-22, sub-27"),
    ("fold3", "sub-5, sub-13, sub-18, sub-23, sub-28"),
    ("fold4", "sub-7, sub-9, sub-14, sub-19, sub-24"),
]

METHODS = [
    {
        "method": "random_split_unet",
        "experiment_prefix": "CV_RANDOM",
        "summary_type": "threshold",
        "description": "Random-initialized Split U-Net",
    },
    {
        "method": "m3_flowinit_split_unet",
        "experiment_prefix": "CV_M3",
        "summary_type": "threshold",
        "description": "Split U-Net initialized from M3 flow-matching checkpoint",
    },
    {
        "method": "m3_flowinit_split_unet_smoothing",
        "experiment_prefix": "CV_M3",
        "summary_type": "smoothing",
        "description": "M3 flow-initialized Split U-Net plus validation-selected temporal smoothing",
    },
]


METRICS = [
    "test_dice",
    "test_iou",
    "test_precision",
    "test_recall",
    "test_accuracy",
    "validation_dice",
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def threshold_row(method: dict, fold_name: str, subjects: str) -> dict:
    experiment = f"{method['experiment_prefix']}_{fold_name}_subjectwise"
    path = OUTPUT_ROOT / "threshold_sweeps" / experiment / "threshold_tuning_summary.json"
    summary = load_json(path)
    if summary is None:
        return missing_row(method, experiment, fold_name, subjects, f"Missing {path}")

    test = summary["test_metrics_at_best_validation_threshold"]
    validation = summary["best_validation_metrics"]
    holdout = summary["holdout_test"]
    return {
        "method": method["method"],
        "description": method["description"],
        "experiment": experiment,
        "fold": fold_name,
        "heldout_subjects": subjects,
        "postprocessing": "threshold only",
        "selected_threshold": summary["best_validation_threshold"],
        "selected_smoothing_seconds": "",
        "validation_dice": validation["dice"],
        "test_dice": test["dice"],
        "test_iou": test["iou"],
        "test_precision": test["precision"],
        "test_recall": test["recall_sensitivity"],
        "test_accuracy": test["accuracy"],
        "test_windows": holdout["test_windows"],
        "status": "available",
        "notes": "Threshold selected on validation only.",
    }


def smoothing_row(method: dict, fold_name: str, subjects: str) -> dict:
    experiment = f"{method['experiment_prefix']}_{fold_name}_subjectwise"
    path = OUTPUT_ROOT / "postprocessing_smoothing" / experiment / "smoothing_threshold_summary.json"
    summary = load_json(path)
    if summary is None:
        return missing_row(method, experiment, fold_name, subjects, f"Missing {path}")

    test = summary["test_metrics_at_best_validation_smoothing_threshold"]
    validation = summary["best_validation_metrics"]
    holdout = summary["holdout_test"]
    return {
        "method": method["method"],
        "description": method["description"],
        "experiment": experiment,
        "fold": fold_name,
        "heldout_subjects": subjects,
        "postprocessing": "validation-selected smoothing + threshold",
        "selected_threshold": summary["best_validation_threshold"],
        "selected_smoothing_seconds": summary["best_validation_smoothing_window_seconds"],
        "validation_dice": validation["dice"],
        "test_dice": test["dice"],
        "test_iou": test["iou"],
        "test_precision": test["precision"],
        "test_recall": test["recall_sensitivity"],
        "test_accuracy": test["accuracy"],
        "test_windows": holdout["test_windows"],
        "status": "available",
        "notes": "Smoothing window and threshold selected on validation only.",
    }


def missing_row(method: dict, experiment: str, fold_name: str, subjects: str, reason: str) -> dict:
    return {
        "method": method["method"],
        "description": method["description"],
        "experiment": experiment,
        "fold": fold_name,
        "heldout_subjects": subjects,
        "postprocessing": "",
        "selected_threshold": "",
        "selected_smoothing_seconds": "",
        "validation_dice": "",
        "test_dice": "",
        "test_iou": "",
        "test_precision": "",
        "test_recall": "",
        "test_accuracy": "",
        "test_windows": "",
        "status": "missing",
        "notes": reason,
    }


def make_row(method: dict, fold_name: str, subjects: str) -> dict:
    if method["summary_type"] == "threshold":
        return threshold_row(method, fold_name, subjects)
    if method["summary_type"] == "smoothing":
        return smoothing_row(method, fold_name, subjects)
    raise ValueError(f"Unknown summary_type: {method['summary_type']}")


def mean_std(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def make_method_summary(rows: list[dict]) -> list[dict]:
    summary_rows = []
    for method in METHODS:
        available = [row for row in rows if row["method"] == method["method"] and row["status"] == "available"]
        out = {
            "method": method["method"],
            "description": method["description"],
            "available_folds": len(available),
            "expected_folds": len(FOLDS),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in available if row[metric] != ""]
            mean, std = mean_std(values)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output_dir = OUTPUT_ROOT / "subjectwise_cv_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in METHODS:
        for fold_name, subjects in FOLDS:
            rows.append(make_row(method, fold_name, subjects))

    summary_rows = make_method_summary(rows)

    per_fold_path = output_dir / "subjectwise_cv_per_fold_results.csv"
    summary_path = output_dir / "subjectwise_cv_mean_std_summary.csv"
    write_csv(per_fold_path, rows)
    write_csv(summary_path, summary_rows)

    print("Saved:")
    print(per_fold_path)
    print(summary_path)
    print()
    for row in summary_rows:
        dice_mean = row["test_dice_mean"]
        dice_std = row["test_dice_std"]
        print(
            f"{row['method']}: folds {row['available_folds']}/{row['expected_folds']}, "
            f"test Dice mean={dice_mean}, std={dice_std}"
        )


if __name__ == "__main__":
    main()
