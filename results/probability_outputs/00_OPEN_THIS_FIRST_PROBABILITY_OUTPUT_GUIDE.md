# V6 probability-output results - open this first

This folder contains the final professor-facing probability-output examples generated using the improved V6 M3 model:

**Split U-Net initialized from multi-dataset flow-matching pretraining.**

Flow-matching pretraining used:

- PhysioMotion
- BCI Competition IV Dataset 1
- EEGMMIDB

The segmentation model was then fine-tuned on PhysioMotion manual artifact annotations.

## Best window to show first

Open this folder first:

`W1_306_316_best`

This is the cleanest and strongest probability-output example.

- Subject/run: `sub-30`, run `06`
- Recording time: `306s-316s`
- Threshold used for M3: `0.35`
- Window Dice: `0.8063`
- Window precision: `0.8565`
- Window recall: `0.7616`

## New clearer visual to show

Inside each window folder, open:

`model_comparison_channel_bars.png`

This is the new clean visual. It shows:

1. One selected EEG channel signal.
2. Manual ground-truth artifact bar.
3. M1 random Split U-Net prediction bar.
4. M2 Physio-only flow-init Split U-Net prediction bar.
5. M3 improved multi-dataset flow-init Split U-Net prediction bar.

This is easier to explain than the full 59-channel timeline because it focuses on one channel and directly compares the three models.

## Ranked probability-output windows

| Rank | Folder | Dice | Precision | Recall | Best use |
|---:|---|---:|---:|---:|---|
| 1 | `W1_306_316_best` | `0.8063` | `0.8565` | `0.7616` | Show first; strongest overlap. |
| 2 | `W2_310_320_strong` | `0.7732` | `0.7903` | `0.7568` | Second strong example. |
| 3 | `W3_840_850_high_recall` | `0.7393` | `0.6932` | `0.7919` | Good high-recall example. |
| 4 | `W4_850_860_high_recall` | `0.7314` | `0.6471` | `0.8409` | Good high-recall example. |
| 5 | `W5_760_770_optional` | `0.6235` | `0.7862` | `0.5166` | Optional example. |

## What each window folder contains

Each window folder contains:

- `input_eeg_window.csv`  
  The original 10-second EEG input window.

- `output_probability_map.csv`  
  The model output probability map for every channel and time sample.

- `predicted_artifact_intervals.csv`  
  The model-predicted artifact start and end times per channel.

- `true_manual_artifact_intervals.csv`  
  The manual annotation start/end intervals.

- `interval_timeline.png`  
  Timeline comparing predicted artifact intervals with manual intervals.

- `model_comparison_channel_bars.png`  
  Clean one-channel visualization comparing M1, M2, and M3 prediction bars.

- `summary.json`  
  Summary of the selected window, threshold, metrics, and generated file paths.

## What to say while showing the best window

"Here I select a 10-second EEG window from the held-out test recording. The model receives the EEG signal with shape `59 x 2500`, meaning 59 channels and 2500 samples after downsampling to 250 Hz. The Split U-Net outputs a probability map with the same shape. After applying the selected threshold, I convert that probability map into predicted artifact regions. Then I extract artifact start and stop times per affected channel and compare those predicted intervals with the manual annotations."

