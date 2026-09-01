#!/usr/bin/env bash
set -euo pipefail

echo "=== V6 environment check ==="
pwd
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo
echo "=== Dataset checks ==="

PHYSIO_DATASET="$HOME/thesis_eeg/BFM2_ChannelTimeSegmentation_V2_WORKING/outputs_v2_channel_time/window_datasets/full_sub30_runs6_channel_time_10s_250hz_memmap"
BCI_DIR="$HOME/thesis_eeg/packages/Final 4 - BCI IV 1/bci_comp_iv_ds1_1000hz_csv"
EEGMMIDB_DIR="$HOME/thesis_eeg/packages/Final 1 EEGMMIDB/eegmmidb_runwise_csv"

test -d "$PHYSIO_DATASET" && echo "PHYSIOMOTION MEMMAP OK: $PHYSIO_DATASET" || { echo "PHYSIOMOTION MEMMAP MISSING: $PHYSIO_DATASET"; exit 1; }
test -d "$BCI_DIR" && echo "BCI IV OK: $BCI_DIR" || { echo "BCI IV MISSING: $BCI_DIR"; exit 1; }
test -d "$EEGMMIDB_DIR" && echo "EEGMMIDB OK: $EEGMMIDB_DIR" || { echo "EEGMMIDB MISSING: $EEGMMIDB_DIR"; exit 1; }

echo
echo "=== Required code files ==="
test -f training/pretrain_flowmatching_channel_time_multidataset_earlystop.py && echo "OK flow pretraining"
test -f training/train_channel_time_segmentation_sharded.py && echo "OK segmentation training"
test -f evaluation/tune_threshold_on_validation.py && echo "OK threshold sweep"
test -f evaluation/predict_artifact_intervals_from_edf_window.py && echo "OK interval prediction"
test -f data_preparation/build_bci_iv_ds1_59ch_flow_memmap.py && echo "OK BCI builder"
test -f data_preparation/build_eegmmidb_59ch_flow_memmap.py && echo "OK EEGMMIDB builder"

echo
echo "V6 check complete."

