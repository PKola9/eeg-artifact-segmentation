"""
Run Version 2 channel-time model on many original EDF windows.

This is for meeting/demo analysis:
    - run several selected 10-second windows
    - save full input/output files for each window
    - collect metrics into one summary CSV

Each individual window is processed by:
    predict_original_edf_window_v2_channel_time.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "outputs_v2_channel_time"
    / "training_runs"
    / "balanced_50_50_flowinit_bce_dice"
    / "splitunet_channel_time_best_val_dice.pt"
)
DEFAULT_WINDOWS_CSV = PROJECT_ROOT / "evaluation" / "demo_windows_to_predict.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "many_window_demo_outputs"


def create_default_windows_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        # subject, run, start_sec, window_sec, reason
        ("sub-30", 6, 135, 10, "open_base interval; likely mostly non-artifact baseline label"),
        ("sub-30", 6, 306, 10, "horizontal eye movement annotation around 306.5-314.9s"),
        ("sub-30", 6, 316, 10, "after horizontal eye movement window"),
        ("sub-30", 6, 520, 10, "additional test window"),
        ("sub-30", 6, 700, 10, "additional test window"),
        ("sub-1", 1, 306, 10, "different subject/run comparison"),
        ("sub-1", 1, 713, 10, "manual-artifact-style time used in earlier demo discussions"),
        ("sub-2", 3, 306, 10, "different subject/run comparison"),
        ("sub-10", 4, 306, 10, "different subject/run comparison"),
        ("sub-20", 5, 306, 10, "different subject/run comparison"),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "run", "start_sec", "window_sec", "reason"])
        writer.writerows(rows)


def load_windows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "subject": row["subject"],
                    "run": int(row["run"]),
                    "start_sec": float(row["start_sec"]),
                    "window_sec": float(row.get("window_sec", 10) or 10),
                    "reason": row.get("reason", ""),
                }
            )
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-csv", default=str(DEFAULT_WINDOWS_CSV))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--model-arch",
        choices=["split_unet", "temporal_unet_no_channel_mixing"],
        default="split_unet",
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--create-template-only", action="store_true")
    args = parser.parse_args()

    windows_csv = Path(args.windows_csv)
    if not windows_csv.is_absolute():
        windows_csv = PROJECT_ROOT / windows_csv

    if not windows_csv.exists():
        create_default_windows_csv(windows_csv)
        print("Created default windows CSV:")
        print(windows_csv)
        if args.create_template_only:
            return

    if args.create_template_only:
        print("Template already exists:")
        print(windows_csv)
        return

    model = Path(args.model)
    if not model.is_absolute():
        model = PROJECT_ROOT / model
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    windows = load_windows(windows_csv)
    predictor = PROJECT_ROOT / "evaluation" / "predict_original_edf_window_v2_channel_time.py"

    for idx, row in enumerate(windows, start=1):
        window_out = output_dir / f"window_{idx:03d}_{row['subject']}_run-{row['run']:02d}_start-{int(row['start_sec'])}s"
        window_out.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(windows)}] Predicting {row['subject']} run {row['run']:02d}, start {row['start_sec']}s")

        cmd = [
            sys.executable,
            str(predictor),
            "--subject",
            row["subject"],
            "--run",
            str(row["run"]),
            "--start-sec",
            str(row["start_sec"]),
            "--window-sec",
            str(row["window_sec"]),
            "--threshold",
            str(args.threshold),
            "--model",
            str(model),
            "--model-arch",
            args.model_arch,
            "--output-dir",
            str(window_out),
        ]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
        if result.returncode != 0:
            summary_rows.append(
                {
                    **row,
                    "status": "FAILED",
                    "error": result.stderr.strip().replace("\n", " | "),
                }
            )
            print("  FAILED")
            print(result.stderr)
            continue

        summary_files = list(window_out.glob("*_SUMMARY.json"))
        if not summary_files:
            summary_rows.append({**row, "status": "FAILED", "error": "summary JSON not found"})
            continue

        summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
        metrics = summary["metrics_for_this_window"]
        created = summary["created_files"]
        summary_rows.append(
            {
                **row,
                "status": "OK",
                "threshold": args.threshold,
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall_sensitivity": metrics["recall_sensitivity"],
                "specificity": metrics["specificity"],
                "accuracy": metrics["accuracy"],
                "true_artifact_points": summary["true_artifact_channel_time_points"],
                "predicted_artifact_points": summary["predicted_artifact_channel_time_points"],
                "heatmap_png": created["heatmap_png"],
                "summary_json": created["summary_json"],
                "probability_csv": created["output_probability_map_csv"],
                "true_mask_csv": created["true_manual_mask_csv"],
                "predicted_mask_csv": created["predicted_threshold_mask_csv"],
            }
        )
        print(f"  OK dice={metrics['dice']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall_sensitivity']:.4f}")

    summary_csv = output_dir / "many_window_prediction_summary.csv"
    fieldnames = sorted({key for row in summary_rows for key in row.keys()})
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Saved combined summary:")
    print(summary_csv)


if __name__ == "__main__":
    main()
