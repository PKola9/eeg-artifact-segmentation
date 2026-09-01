#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs_v6_flowmatching_study/logs

PHYSIO_DATASET="$HOME/thesis_eeg/BFM2_ChannelTimeSegmentation_V2_WORKING/outputs_v2_channel_time/window_datasets/full_sub30_runs6_channel_time_10s_250hz_memmap"
BCI_FLOW_DATASET="outputs_v6_flowmatching_study/window_datasets/bci_iv_ds1_59ch_10s_250hz_flow_memmap"
EEGMMIDB_FLOW_DATASET="outputs_v6_flowmatching_study/window_datasets/eegmmidb_59ch_10s_250hz_flow_memmap"

echo "=== Stage 1: Multi-dataset flow matching ==="
python training/pretrain_flowmatching_channel_time_multidataset_earlystop.py \
  --dataset-dir "$PHYSIO_DATASET" \
  --dataset-dir "$BCI_FLOW_DATASET" \
  --dataset-dir "$EEGMMIDB_FLOW_DATASET" \
  --epochs 30 \
  --batch-size 8 \
  --patience 5 \
  --output-name "02_physio_bci_eegmmidb_flowmatching_earlystop" \
  2>&1 | tee outputs_v6_flowmatching_study/logs/04_multidataset_flowmatching.log

FLOW_CKPT="outputs_v6_flowmatching_study/flowmatching_runs/02_physio_bci_eegmmidb_flowmatching_earlystop/flowmatching_channel_time_best_validation_mse.pt"

echo
echo "=== Stage 2: Split U-Net initialized from multi-dataset flow matching ==="
python training/train_channel_time_segmentation_sharded.py \
  --dataset-dir "$PHYSIO_DATASET" \
  --model-arch split_unet \
  --balanced-train-windows \
  --clean-to-artifact-ratio 1.0 \
  --pos-weight-override 1.0 \
  --flow-checkpoint "$FLOW_CKPT" \
  --epochs 20 \
  --batch-size 8 \
  --patience 5 \
  --output-name "M3_multidataset_flowinit_splitunet_balanced50_posweight1" \
  2>&1 | tee outputs_v6_flowmatching_study/logs/05_multidataset_flowinit_segmentation.log

echo
echo "Multi-dataset flow-initialized segmentation complete."

