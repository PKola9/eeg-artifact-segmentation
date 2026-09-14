"""
Summarize the post-defense three-subject holdout experiments.

This script reads the threshold tuning summaries for each held-out subject set and
creates one CSV table that is easy to send to a supervisor. It also includes the
original thesis M3 result as a reference row.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


THESIS_REFERENCE = {
    "experiment": "thesis_reference_M3_single_recording",
    "heldout_subjects": "sub-30",
    "heldout_scope": "single recording: run-06",
    "selected_threshold": 0.35,
    "test_dice": 0.4328,
    "test_iou": 0.2761,
    "test_precision": 0.3563,
    "test_recall": 0.5508,
    "test_accuracy": 0.935,
    "validation_dice": "",
    "test_windows": "",
    "notes": "Original thesis result; test was one held-out recording, not full subject-level holdout.",
}


EXPERIMENTS = [
    {
        "experiment": "M3_three_subject_holdout_setA_sub10_sub20_sub30",
        "heldout_subjects": "sub-10, sub-20, sub-30",
    },
    {
        "experiment": "M3_three_subject_holdout_setB_sub1_sub15_sub25",
        "heldout_subjects": "sub-1, sub-15, sub-25",
    },
    {
        "experiment": "M3_three_subject_holdout_setC_sub5_sub18_sub27",
        "heldout_subjects": "sub-5, sub-18, sub-27",
    },
]


def read_summary(experiment: str) -> dict | None:
    path = (
        PROJECT_ROOT
        / "outputs_v6_flowmatching_study"
        / "threshold_sweeps"
        / experiment
        / "threshold_tuning_summary.json"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_row(experiment: dict) -> dict:
    summary = read_summary(experiment["experiment"])
    if summary is None:
        return {
            "experiment": experiment["experiment"],
            "heldout_subjects": experiment["heldout_subjects"],
            "heldout_scope": "three complete subjects, all runs",
            "selected_threshold": "",
            "test_dice": "",
            "test_iou": "",
            "test_precision": "",
            "test_recall": "",
            "test_accuracy": "",
            "validation_dice": "",
            "test_windows": "",
            "notes": "Missing threshold_tuning_summary.json; training/evaluation may still be running or failed.",
        }

    test = summary["test_metrics_at_best_validation_threshold"]
    validation = summary["best_validation_metrics"]
    holdout = summary["holdout_test"]
    subjects = holdout.get("subjects") or [holdout.get("subject", "")]

    return {
        "experiment": experiment["experiment"],
        "heldout_subjects": ", ".join(subjects),
        "heldout_scope": "three complete subjects, all runs",
        "selected_threshold": summary["best_validation_threshold"],
        "test_dice": test["dice"],
        "test_iou": test["iou"],
        "test_precision": test["precision"],
        "test_recall": test["recall_sensitivity"],
        "test_accuracy": test["accuracy"],
        "validation_dice": validation["dice"],
        "test_windows": holdout["test_windows"],
        "notes": "Threshold selected on validation only; held-out test subjects were not used for threshold selection.",
    }


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs_v6_flowmatching_study" / "three_subject_holdout_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "m3_three_subject_holdout_comparison.csv"

    rows = [THESIS_REFERENCE] + [make_row(experiment) for experiment in EXPERIMENTS]
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved summary:")
    print(output_path)
    print()
    for row in rows:
        dice = row["test_dice"] if row["test_dice"] != "" else "not available yet"
        print(f"{row['experiment']}: test Dice = {dice}")


if __name__ == "__main__":
    main()
