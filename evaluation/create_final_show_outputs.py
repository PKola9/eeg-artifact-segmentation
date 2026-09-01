"""
Create final organized outputs for thesis/professor presentation.

This script should be run AFTER the V4 training job finishes.

It creates:
    outputs_v6_flowmatching_study/FINAL_RESULTS_TO_SHOW/
        01_HIGHLIGHT_splitunet_vs_flowinit/
        02_FULL_three_model_comparison/
        03_SAME_WINDOW_OUTPUTS/

The purpose is to keep the storytelling clean:
    - highlight the main comparison separately:
        Split U-Net random baseline vs Flow-initialized Split U-Net
    - keep the full three-model comparison separately:
        Temporal U-Net no channel mixing vs Split U-Net vs Flow-initialized Split U-Net
    - generate the same input-window demos for every model.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs_v6_flowmatching_study"
FINAL_ROOT = OUTPUT_ROOT / "FINAL_RESULTS_TO_SHOW"


MODELS = [
    {
        "model_key": "temporal_unet_no_channel_mixing",
        "display_name": "Temporal U-Net without channel mixing",
        "arch": "temporal_unet_no_channel_mixing",
        "training_folder": "comparison_01_temporal_unet_no_channel_mixing_posweight1",
        "threshold_folder": "comparison_01_temporal_unet_no_channel_mixing_posweight1",
        "highlight": False,
    },
    {
        "model_key": "split_unet_random",
        "display_name": "Split U-Net random baseline",
        "arch": "split_unet",
        "training_folder": "comparison_02_split_unet_random_posweight1",
        "threshold_folder": "comparison_02_split_unet_random_posweight1",
        "highlight": True,
    },
    {
        "model_key": "split_unet_flowinit",
        "display_name": "Flow-initialized Split U-Net",
        "arch": "split_unet",
        "training_folder": "comparison_03_split_unet_flowinit_posweight1",
        "threshold_folder": "comparison_03_split_unet_flowinit_posweight1",
        "highlight": True,
    },
]


DEFAULT_WINDOWS = [
    {
        "window_id": "window_01_sub30_run06_135s",
        "subject": "sub-30",
        "run": 6,
        "start_sec": 135.0,
        "window_sec": 10.0,
        "channels_for_bars": ["Cz", "Fp1", "Fp2"],
        "reason": "Held-out test run example window.",
    },
    {
        "window_id": "window_02_sub30_run06_306s",
        "subject": "sub-30",
        "run": 6,
        "start_sec": 306.0,
        "window_sec": 10.0,
        "channels_for_bars": ["Cz", "Fp1", "Fp2"],
        "reason": "Held-out test run window used in previous professor demo.",
    },
    {
        "window_id": "window_03_sub30_run06_520s",
        "subject": "sub-30",
        "run": 6,
        "start_sec": 520.0,
        "window_sec": 10.0,
        "channels_for_bars": ["Cz", "F7", "F8"],
        "reason": "Additional held-out test run window for visual comparison.",
    },
]


def read_threshold_summary(model_info: dict) -> dict:
    path = OUTPUT_ROOT / "threshold_tuning" / model_info["threshold_folder"] / "threshold_tuning_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing threshold tuning summary for {model_info['display_name']}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def model_checkpoint(model_info: dict) -> Path:
    path = (
        OUTPUT_ROOT
        / "training_runs"
        / model_info["training_folder"]
        / "splitunet_channel_time_best_val_dice.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing model checkpoint for {model_info['display_name']}: {path}")
    return path


def comparison_row(model_info: dict, summary: dict) -> dict:
    val = summary["best_validation_metrics"]
    test = summary["test_metrics_at_best_validation_threshold"]
    return {
        "model_key": model_info["model_key"],
        "model_name": model_info["display_name"],
        "model_arch": model_info["arch"],
        "best_validation_threshold": summary["best_validation_threshold"],
        "validation_dice": val["dice"],
        "validation_iou": val["iou"],
        "validation_precision": val["precision"],
        "validation_recall": val["recall_sensitivity"],
        "test_dice": test["dice"],
        "test_iou": test["iou"],
        "test_precision": test["precision"],
        "test_recall": test["recall_sensitivity"],
        "test_accuracy": test["accuracy"],
        "test_tp": test["tp"],
        "test_fp": test["fp"],
        "test_tn": test["tn"],
        "test_fn": test["fn"],
    }


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(command)}")


def create_window_outputs(dataset_root: str, thresholds: str) -> None:
    predictor = PROJECT_ROOT / "evaluation" / "predict_original_edf_window_v2_channel_time.py"
    bar_plotter = PROJECT_ROOT / "evaluation" / "plot_one_channel_threshold_bars.py"
    same_window_root = FINAL_ROOT / "03_SAME_WINDOW_OUTPUTS"
    same_window_root.mkdir(parents=True, exist_ok=True)

    for window in DEFAULT_WINDOWS:
        window_dir = same_window_root / window["window_id"]
        window_dir.mkdir(parents=True, exist_ok=True)
        (window_dir / "window_reason.txt").write_text(window["reason"], encoding="utf-8")

        for model_info in MODELS:
            threshold_summary = read_threshold_summary(model_info)
            selected_threshold = float(threshold_summary["best_validation_threshold"])
            checkpoint = model_checkpoint(model_info)

            model_dir = window_dir / model_info["model_key"]
            bars_dir = model_dir / "bars"
            model_dir.mkdir(parents=True, exist_ok=True)
            bars_dir.mkdir(parents=True, exist_ok=True)

            run_command(
                [
                    sys.executable,
                    str(predictor),
                    "--dataset-root",
                    dataset_root,
                    "--subject",
                    window["subject"],
                    "--run",
                    str(window["run"]),
                    "--start-sec",
                    str(window["start_sec"]),
                    "--window-sec",
                    str(window["window_sec"]),
                    "--threshold",
                    str(selected_threshold),
                    "--model",
                    str(checkpoint),
                    "--model-arch",
                    model_info["arch"],
                    "--output-dir",
                    str(model_dir),
                ]
            )

            for channel in window["channels_for_bars"]:
                run_command(
                    [
                        sys.executable,
                        str(bar_plotter),
                        "--subject",
                        window["subject"],
                        "--run",
                        str(window["run"]),
                        "--start-sec",
                        str(window["start_sec"]),
                        "--window-sec",
                        str(window["window_sec"]),
                        "--channel",
                        channel,
                        "--thresholds",
                        thresholds,
                        "--model",
                        str(checkpoint),
                        "--model-arch",
                        model_info["arch"],
                        "--output-dir",
                        str(bars_dir),
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT.parent / "packages" / "Physiomotion Dataset" / "Original Dataset"),
    )
    parser.add_argument("--thresholds-for-bars", default="0.5,0.6,0.7,0.8,0.85")
    parser.add_argument("--skip-window-outputs", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()

    if FINAL_ROOT.exists() and args.replace_existing:
        shutil.rmtree(FINAL_ROOT)
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)

    rows = [comparison_row(model, read_threshold_summary(model)) for model in MODELS]

    highlighted_rows = [row for row, model in zip(rows, MODELS) if model["highlight"]]
    highlight_dir = FINAL_ROOT / "01_HIGHLIGHT_splitunet_vs_flowinit"
    full_dir = FINAL_ROOT / "02_FULL_three_model_comparison"

    write_rows_csv(highlight_dir / "highlight_splitunet_vs_flowinit.csv", highlighted_rows)
    (highlight_dir / "highlight_splitunet_vs_flowinit.json").write_text(
        json.dumps(highlighted_rows, indent=2),
        encoding="utf-8",
    )

    write_rows_csv(full_dir / "full_three_model_comparison.csv", rows)
    (full_dir / "full_three_model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if not args.skip_window_outputs:
        create_window_outputs(dataset_root=args.dataset_root, thresholds=args.thresholds_for_bars)

    manifest = {
        "purpose": "Final organized thesis/professor outputs.",
        "main_highlight": "Split U-Net random baseline vs flow-matching initialized Split U-Net.",
        "full_comparison": "Temporal U-Net no channel mixing vs Split U-Net random vs Split U-Net flow-init.",
        "selected_windows": DEFAULT_WINDOWS,
        "created_folders": {
            "highlighted_two_model_comparison": str(highlight_dir),
            "full_three_model_comparison": str(full_dir),
            "same_window_outputs": str(FINAL_ROOT / "03_SAME_WINDOW_OUTPUTS"),
        },
    }
    (FINAL_ROOT / "README_FINAL_OUTPUTS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Final organized outputs saved in:")
    print(FINAL_ROOT)
    print("Highlighted comparison:")
    print(highlight_dir / "highlight_splitunet_vs_flowinit.csv")
    print("Full comparison:")
    print(full_dir / "full_three_model_comparison.csv")


if __name__ == "__main__":
    main()

