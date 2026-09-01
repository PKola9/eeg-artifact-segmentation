"""
Create the clean final comparison table directly from saved threshold-sweep files.

This avoids manual copying mistakes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "outputs_v6_flowmatching_study"


MODELS = [
    (
        "M1_random_splitunet",
        "none",
        ROOT / "threshold_sweeps" / "V6_CLEAN_M1_random_splitunet",
    ),
    (
        "M2_physio_flowinit_splitunet",
        "PhysioMotion_only_flow_matching",
        ROOT / "threshold_sweeps" / "V6_CLEAN_M2_physio_flowinit_splitunet",
    ),
    (
        "M3_improved_multidataset_flowinit_splitunet",
        "PhysioMotion_plus_BCI_IV_plus_EEGMMIDB_flow_matching",
        ROOT / "threshold_sweeps" / "V6_IMPROVED_M3_multidataset_flowinit_lr5e4_patience8",
    ),
    (
        "M4_extra_dataset_flowinit_splitunet",
        "PhysioMotion_plus_BCI_IV_plus_EEGMMIDB_plus_DEAP_plus_CHBMIT_flow_matching",
        ROOT / "threshold_sweeps" / "V6_M4_extra_deap_chbmit_flowinit_splitunet",
    ),
    (
        "M5_TUH_extra_flowinit_splitunet",
        "PhysioMotion_plus_BCI_IV_plus_EEGMMIDB_plus_DEAP_plus_CHBMIT_plus_TUH_flow_matching",
        ROOT / "threshold_sweeps" / "V6_M5_tuh_extra_flowinit_splitunet",
    ),
]


def read_one(model: str, pretraining: str, sweep_dir: Path) -> dict | None:
    summary_path = sweep_dir / "threshold_tuning_summary.json"
    test_path = sweep_dir / "test_at_selected_and_0p5_threshold.csv"
    if not summary_path.exists() or not test_path.exists():
        print(f"Skipping missing model results: {model}")
        return None

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test_df = pd.read_csv(test_path)

    selected_threshold = float(summary["best_validation_threshold"])
    validation_dice = float(summary["best_validation_metrics"]["dice"])

    selected = test_df[test_df["threshold"].round(6) == round(selected_threshold, 6)]
    if selected.empty:
        raise RuntimeError(f"Could not find selected threshold {selected_threshold} in {test_path}")
    row_selected = selected.iloc[0]

    fixed = test_df[test_df["threshold"].round(6) == 0.5]
    row_05 = fixed.iloc[0] if not fixed.empty else row_selected

    return {
        "model": model,
        "pretraining": pretraining,
        "selected_threshold": selected_threshold,
        "validation_dice": validation_dice,
        "test_dice_selected_threshold": float(row_selected["dice"]),
        "test_precision_selected_threshold": float(row_selected["precision"]),
        "test_recall_selected_threshold": float(row_selected["recall_sensitivity"]),
        "test_dice_threshold_0p5": float(row_05["dice"]),
    }


def main() -> None:
    rows = []
    for model, pretraining, sweep_dir in MODELS:
        row = read_one(model, pretraining, sweep_dir)
        if row is not None:
            rows.append(row)

    out_dir = ROOT / "final_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "V6_all_available_model_clean_comparison.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    # Also keep the earlier filename so there is one familiar place to look.
    four_model_path = out_dir / "V6_three_or_four_model_clean_comparison.csv"
    pd.DataFrame(rows).to_csv(four_model_path, index=False)
    familiar_path = out_dir / "V6_three_model_clean_comparison.csv"
    pd.DataFrame(rows).to_csv(familiar_path, index=False)

    print("Saved final comparison tables:")
    print(out_path)
    print(four_model_path)
    print(familiar_path)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
