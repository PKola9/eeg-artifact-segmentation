#!/usr/bin/env bash
set -euo pipefail

echo "=== V6 M5: add TUH to flow initialization experiment ==="
echo "Working directory: $(pwd)"

PHYSIO_DATASET="$HOME/thesis_eeg/BFM2_ChannelTimeSegmentation_V2_WORKING/outputs_v2_channel_time/window_datasets/full_sub30_runs6_channel_time_10s_250hz_memmap"
BCI_FLOW_DATASET="outputs_v6_flowmatching_study/window_datasets/bci_iv_ds1_59ch_10s_250hz_flow_memmap"
EEGMMIDB_FLOW_DATASET="outputs_v6_flowmatching_study/window_datasets/eegmmidb_59ch_10s_250hz_flow_memmap"
DEAP_FLOW_DATASET="outputs_v6_flowmatching_study/window_datasets/deap_59ch_10s_250hz_flow_memmap"
CHBMIT_FLOW_DATASET="outputs_v6_flowmatching_study/window_datasets/chbmit_chb01_59ch_10s_250hz_flow_memmap"
TUH_FLOW_DATASET="outputs_v6_flowmatching_study/window_datasets/tuh_59ch_10s_250hz_flow_memmap"

echo
echo "STEP 1: Build curated TUH flow-matching dataset"
python data_preparation/build_tuh_59ch_flow_memmap.py \
  --tuh-dir "$HOME/thesis_eeg/packages/TUH" \
  --output-dir "$TUH_FLOW_DATASET"

echo
echo "STEP 2: M5 flow-matching pretraining on PhysioMotion + BCI IV + EEGMMIDB + DEAP + CHB-MIT + TUH"
python training/pretrain_flowmatching_channel_time_multidataset_earlystop.py \
  --dataset-dir "$PHYSIO_DATASET" \
  --dataset-dir "$BCI_FLOW_DATASET" \
  --dataset-dir "$EEGMMIDB_FLOW_DATASET" \
  --dataset-dir "$DEAP_FLOW_DATASET" \
  --dataset-dir "$CHBMIT_FLOW_DATASET" \
  --dataset-dir "$TUH_FLOW_DATASET" \
  --epochs 20 \
  --batch-size 8 \
  --patience 5 \
  --output-name "V6_M5_physio_bci_eegmmidb_deap_chbmit_tuh_flowmatching_20ep"

M5_FLOW_CKPT="outputs_v6_flowmatching_study/flowmatching_runs/V6_M5_physio_bci_eegmmidb_deap_chbmit_tuh_flowmatching_20ep/flowmatching_channel_time_best_validation_mse.pt"

echo
echo "STEP 3: Train Split U-Net with M5 TUH-added flow initialization"
python training/train_channel_time_segmentation_sharded.py \
  --dataset-dir "$PHYSIO_DATASET" \
  --model-arch split_unet \
  --balanced-train-windows \
  --clean-to-artifact-ratio 1.0 \
  --pos-weight-override 1.0 \
  --flow-checkpoint "$M5_FLOW_CKPT" \
  --epochs 30 \
  --batch-size 8 \
  --lr 5e-4 \
  --patience 8 \
  --output-name "V6_M5_tuh_extra_flowinit_splitunet"

echo
echo "STEP 4: Threshold sweep for M5"
python evaluation/tune_threshold_on_validation.py \
  --dataset-dir "$PHYSIO_DATASET" \
  --model "outputs_v6_flowmatching_study/training_runs/V6_M5_tuh_extra_flowinit_splitunet/splitunet_channel_time_best_val_dice.pt" \
  --batch-size 8 \
  --output-dir "outputs_v6_flowmatching_study/threshold_sweeps/V6_M5_tuh_extra_flowinit_splitunet"

echo
echo "STEP 5: Update final comparison table directly from saved threshold sweeps"
python evaluation/update_v6_final_comparison_from_sweeps.py

echo
echo "M5 TUH experiment complete."
echo "Final comparison:"
echo "outputs_v6_flowmatching_study/final_tables/V6_three_or_four_model_clean_comparison.csv"
