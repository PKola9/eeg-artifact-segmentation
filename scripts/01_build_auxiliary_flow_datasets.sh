#!/usr/bin/env bash
set -euo pipefail

echo "=== Build external unlabeled EEG datasets for flow matching ==="
mkdir -p outputs_v6_flowmatching_study/logs

echo "Building BCI IV Dataset 1 59-channel flow memmap..."
python data_preparation/build_bci_iv_ds1_59ch_flow_memmap.py 2>&1 | tee outputs_v6_flowmatching_study/logs/build_bci_iv_flow_memmap.log

echo
echo "Building EEGMMIDB 59-channel flow memmap..."
python data_preparation/build_eegmmidb_59ch_flow_memmap.py 2>&1 | tee outputs_v6_flowmatching_study/logs/build_eegmmidb_flow_memmap.log

echo
echo "External flow datasets ready."

