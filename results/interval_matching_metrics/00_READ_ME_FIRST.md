# M3 improved interval-matching results

This folder evaluates only the selected best flow-initialized Split U-Net:

- Model: M3 improved multi-dataset flow-init Split U-Net
- Flow pretraining: FM2 = PhysioMotion + BCI IV + EEGMMIDB
- Checkpoint: `C:\Users\prana\OneDrive\Documents\DEEPLEARNING MODELS FOR REMOVING ARTIFACTS FROM EEG DATA\BFM2_V6_FLOWMATCHING_STUDY_FULL_PIPELINE\LOCAL_SHORT_MODEL_PATHS\improved_m3_best.pt`
- Threshold: `0.35`
- Dataset: saved held-out test windows from subject sub-30 run-06
- Window shape: 59 channels × 2500 samples = 10 seconds at 250 Hz

Nothing in the old result folders was changed.

## What to open first

1. `01_SAMPLE_LEVEL_METRICS/m3_improved_overall_sample_metrics.csv`
   - Overall Dice, IoU, mIoU, precision, recall, accuracy, specificity, and interval counts.

2. `02_INTERVAL_TABLES/m3_improved_predicted_artifact_intervals.csv`
   - Every predicted artifact interval with channel, start time, end time, duration, and probability.

3. `02_INTERVAL_TABLES/manual_artifact_intervals.csv`
   - Every manual ground-truth interval extracted from the saved test masks.

4. `02_INTERVAL_TABLES/m3_improved_matched_intervals_strict.csv`
   - One-to-one predicted-vs-manual interval matches on the same window and channel.

5. `03_TOLERANCE_SWEEP/m3_improved_interval_tolerance_sweep.csv`
   - Interval detection precision/recall/F1 under different boundary buffers.

6. `05_FIGURES/`
   - Ready-to-show graphs.

## Important interpretation

Sample-level Dice measures overlap between the full predicted binary mask and the full manual binary mask across all channels and time samples.

Interval matching converts those masks into start/end segments and asks whether predicted artifact intervals align with manual artifact intervals on the same channel and window.
