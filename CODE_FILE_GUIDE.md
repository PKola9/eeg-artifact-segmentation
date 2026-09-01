# Code File Guide

This file explains the purpose of each main code file in the repository. It is intended to make the project easy to inspect, reproduce, and extend.

## Data preparation

### `data_preparation/build_channel_time_mask_dataset_from_edf_sharded.py`

Builds the main supervised PhysioMotion channel-time dataset from EDF recordings and manual artifact annotations. It creates EEG windows and aligned binary artifact masks.

### `data_preparation/convert_channel_time_sharded_to_memmap.py`

Converts the sharded PhysioMotion window dataset into memory-mapped arrays for faster training on the HPC system.

### `data_preparation/build_channel_time_mask_dataset_from_edf_one_run.py`

Creates a channel-time dataset for a single subject/run. This is useful for checking preprocessing and for focused debugging.

### `data_preparation/build_window_mask_dataset_from_edf.py`

Earlier window/mask preparation script kept for reproducibility of the development path.

### `data_preparation/build_bci_iv_ds1_59ch_flow_memmap.py`

Converts BCI Competition IV Dataset 1 into 59-channel, 10-second EEG windows for unlabeled flow-matching pretraining.

### `data_preparation/build_eegmmidb_59ch_flow_memmap.py`

Converts EEGMMIDB recordings into 59-channel, 10-second EEG windows for unlabeled flow-matching pretraining.

### `data_preparation/build_deap_59ch_flow_memmap.py`

Converts DEAP EEG recordings into shape-compatible unlabeled windows for additional flow-matching ablation experiments.

### `data_preparation/build_chbmit_59ch_flow_memmap.py`

Converts CHB-MIT EEG recordings into shape-compatible unlabeled windows for additional flow-matching ablation experiments.

### `data_preparation/build_tuh_59ch_flow_memmap.py`

Converts TUH EEG recordings into shape-compatible unlabeled windows for an additional flow-matching ablation experiment.

## Models

### `models/flowmatching_channel_time.py`

Defines the flow-matching network. It learns to predict the velocity from Gaussian noise toward real EEG windows. The learned compatible U-Net body weights are later transferred into the supervised Split U-Net.

### `models/splitunet_channel_time_segmentation.py`

Defines the main Split U-Net segmentation model. It predicts a dense artifact probability map with shape 59 x 2500 for each 10-second EEG window.

### `models/temporal_unet_no_channel_mixing.py`

Defines a comparison model that focuses on temporal processing without explicitly modeling channel relationships in the same way as the Split U-Net.

### `models/eeg_conformer_segmentation.py`

Defines the attention-based EEG-Conformer-style segmentation baseline used as an additional comparison model.

## Training

### `training/pretrain_flowmatching_channel_time_multidataset_earlystop.py`

Runs self-supervised flow-matching pretraining on one or more unlabeled EEG datasets. It saves MSE curves, the best validation checkpoint, and training summaries.

### `training/train_channel_time_segmentation_sharded.py`

Runs supervised segmentation training. It supports random initialization, flow-matching initialization, balanced training windows, different model architectures, early stopping, and checkpoint saving.

## Evaluation

### `evaluation/tune_threshold_on_validation.py`

Sweeps probability thresholds on the validation set, selects the threshold with the highest validation Dice, and reports final test-set performance at both the selected threshold and 0.5.

### `evaluation/evaluate_v6_tolerant_test_metrics.py`

Evaluates interval-aware/tolerant metrics for the selected model. This is used to study whether predicted artifact intervals are close to manual intervals even when boundaries do not overlap perfectly.

### `evaluation/evaluate_m3_improved_interval_matching_saved_test_windows.py`

Evaluates interval matching for the improved M3 model on saved test windows.

### `evaluation/evaluate_artifactwise_on_saved_test_windows.py`

Computes artifact-label-specific metrics by comparing the model prediction with masks created from manual annotations for each artifact label.

### `evaluation/evaluate_final_artifactwise_metrics.py`

Creates final artifact-wise summaries and plots used for thesis reporting.

### `evaluation/evaluate_flowmatching_checkpoint_mse.py`

Re-evaluates a saved flow-matching checkpoint to recover validation/test MSE metrics when an epoch-level metrics file is missing.

### `evaluation/predict_artifact_intervals_from_edf_window.py`

Runs the trained model on a selected EDF time window and converts probability maps into readable predicted artifact intervals with channel, start time, end time, duration, and probability statistics.

### `evaluation/predict_original_edf_window_v2_channel_time.py`

Runs prediction on a selected original EDF window using the channel-time segmentation model.

### `evaluation/predict_many_original_edf_windows_v2.py`

Runs prediction for multiple EDF windows and organizes output examples.

### `evaluation/plot_one_channel_threshold_bars.py`

Creates one-channel probability and threshold visualizations for showing how probability curves become predicted artifact intervals.

### `evaluation/create_clean_threshold_curve_visual.py`

Creates cleaner threshold-selection visualizations for presentation/report use.

### `evaluation/create_final_show_outputs.py`

Collects and organizes the final result outputs into presentation-ready folders.

### `evaluation/summarize_three_model_comparison.py`

Summarizes model-level comparison results across the Split U-Net variants.

### `evaluation/update_v6_final_comparison_from_sweeps.py`

Builds or updates final comparison tables directly from saved threshold-sweep outputs.

## Scripts

### `scripts/00_check_environment_and_data.sh`

Checks that the environment, GPU, required datasets, and required code files are available.

### `scripts/01_build_auxiliary_flow_datasets.sh`

Builds auxiliary flow-matching datasets from external EEG sources.

### `scripts/02_train_physio_flow_pretraining.sh`

Runs PhysioMotion-only flow-matching pretraining.

### `scripts/03_train_random_splitunet_baseline.sh`

Trains the randomly initialized Split U-Net baseline.

### `scripts/04_train_multidataset_flowinit_splitunet.sh`

Trains the main multi-dataset flow-initialized Split U-Net segmentation model.

### `scripts/05_run_threshold_sweeps.sh`

Runs threshold sweeps for trained segmentation models.

### `scripts/06_train_deap_chbmit_ablation.sh`

Runs the additional DEAP and CHB-MIT flow-matching ablation experiment.

### `scripts/07_train_tuh_ablation.sh`

Runs the additional TUH flow-matching ablation experiment.

### `scripts/train_tuh_ablation.slurm`

SLURM batch script for running the TUH ablation on the HPC cluster.

## Results

The `results/` folder contains selected CSV, JSON, PNG, and small checkpoint files. Large generated memory-map datasets are intentionally excluded.

Important result areas:

- `results/flow_pretraining_metrics/`: flow-matching MSE/RMSE summaries and figures.
- `results/model_comparison/`: comparison across Split U-Net variants.
- `results/threshold_selection/`: validation threshold-selection plots.
- `results/overall_test_metrics/`: final held-out test metrics.
- `results/artifact_label_analysis/`: per-artifact-label evaluation.
- `results/extended_best_model_metrics/`: additional metrics for the selected best Split U-Net model.
- `results/interval_demo/`: selected channel-level start/end interval examples.
- `results/probability_outputs/`: probability-map outputs used to derive artifact intervals.
- `results/training_summaries/`: raw experiment summaries, logs, threshold sweeps, and small checkpoints.
