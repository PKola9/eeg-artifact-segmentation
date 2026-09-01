#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs_v6_flowmatching_study/logs

PHYSIO_DATASET="$HOME/thesis_eeg/BFM2_ChannelTimeSegmentation_V2_WORKING/outputs_v2_channel_time/window_datasets/full_sub30_runs6_channel_time_10s_250hz_memmap"

echo "=== Random Split U-Net baseline ==="
python training/train_channel_time_segmentation_sharded.py \
  --dataset-dir "$PHYSIO_DATASET" \
  --model-arch split_unet \
  --balanced-train-windows \
  --clean-to-artifact-ratio 1.0 \
  --pos-weight-override 1.0 \
  --epochs 20 \
  --batch-size 8 \
  --patience 5 \
  --output-name "M1_random_splitunet_balanced50_posweight1" \
  2>&1 | tee outputs_v6_flowmatching_study/logs/03_random_splitunet_baseline.log

echo
echo "Random Split U-Net baseline complete."

