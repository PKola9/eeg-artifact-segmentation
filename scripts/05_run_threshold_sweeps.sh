#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs_v6_flowmatching_study/logs

PHYSIO_DATASET="$HOME/thesis_eeg/BFM2_ChannelTimeSegmentation_V2_WORKING/outputs_v2_channel_time/window_datasets/full_sub30_runs6_channel_time_10s_250hz_memmap"

run_sweep () {
  local name="$1"
  local model="$2"
  local outdir="$3"
  echo "=== Threshold sweep: $name ==="
  python evaluation/tune_threshold_on_validation.py \
    --dataset-dir "$PHYSIO_DATASET" \
    --model "$model" \
    --batch-size 8 \
    --output-dir "$outdir"
}

run_sweep \
  "M1 random Split U-Net" \
  "outputs_v6_flowmatching_study/training_runs/M1_random_splitunet_balanced50_posweight1/splitunet_channel_time_best_val_dice.pt" \
  "outputs_v6_flowmatching_study/threshold_sweeps/M1_random_splitunet"

run_sweep \
  "M2 PhysioMotion flow-initialized Split U-Net" \
  "outputs_v6_flowmatching_study/training_runs/M2_physio_flowinit_splitunet_balanced50_posweight1/splitunet_channel_time_best_val_dice.pt" \
  "outputs_v6_flowmatching_study/threshold_sweeps/M2_physio_flowinit_splitunet"

run_sweep \
  "M3 Multi-dataset flow-initialized Split U-Net" \
  "outputs_v6_flowmatching_study/training_runs/M3_multidataset_flowinit_splitunet_balanced50_posweight1/splitunet_channel_time_best_val_dice.pt" \
  "outputs_v6_flowmatching_study/threshold_sweeps/M3_multidataset_flowinit_splitunet"

echo
echo "All threshold sweeps complete."

