"""
Create one clean CSV/JSON summary for the three-model thesis comparison.

It reads the threshold tuning summaries produced for:
    1. Temporal U-Net without channel mixing
    2. Split U-Net random baseline
    3. Flow-initialized Split U-Net

Output:
    outputs_v6_flowmatching_study/comparison_summary/three_model_comparison.csv
    outputs_v6_flowmatching_study/comparison_summary/three_model_comparison.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs_v6_flowmatching_study"


MODELS = [
    {
        "model_id": "01_temporal_unet_no_channel_mixing",
        "model_name": "Temporal U-Net, no channel mixing",
        "folder": "comparison_01_temporal_unet_no_channel_mixing_posweight1",
        "main_question": "What happens if each EEG channel is processed independently?",
    },
    {
        "model_id": "02_split_unet_random",
        "model_name": "Split U-Net random baseline",
        "folder": "comparison_02_split_unet_random_posweight1",
        "main_question": "What is gained by modeling EEG channel relationships?",
    },
    {
        "model_id": "03_split_unet_flowinit",
        "model_name": "Split U-Net with flow-matching initialization",
        "folder": "comparison_03_split_unet_flowinit_posweight1",
        "main_question": "What is gained by adding self-supervised EEG pretraining?",
    },
]


def main() -> None:
    rows = []
    for model in MODELS:
        path = OUTPUT_ROOT / "threshold_tuning" / model["folder"] / "threshold_tuning_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing threshold summary: {path}")

        summary = json.loads(path.read_text(encoding="utf-8"))
        val = summary["best_validation_metrics"]
        test = summary["test_metrics_at_best_validation_threshold"]

        rows.append(
            {
                "model_id": model["model_id"],
                "model_name": model["model_name"],
                "main_question": model["main_question"],
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
        )

    out_dir = OUTPUT_ROOT / "comparison_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "three_model_comparison.csv"
    json_path = out_dir / "three_model_comparison.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("Saved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()

