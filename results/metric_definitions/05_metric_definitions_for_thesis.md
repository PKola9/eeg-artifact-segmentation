# Final artifact-wise evaluation metrics

Model checkpoint:
`C:\Users\prana\OneDrive\Documents\DEEPLEARNING MODELS FOR REMOVING ARTIFACTS FROM EEG DATA\BFM2_V6_FLOWMATCHING_STUDY_FULL_PIPELINE\outputs_v6_flowmatching_study\training_runs\V6_CLEAN_M3_multidataset_flowinit_splitunet\splitunet_channel_time_best_val_dice.pt`

Test recording:
`C:\Users\prana\OneDrive\Documents\DEEPLEARNING MODELS FOR REMOVING ARTIFACTS FROM EEG DATA\packages\Physiomotion Dataset\Original Dataset\sub-30\eeg\sub-30_task-artifact_run-06_eeg.edf`

Manual annotations:
`C:\Users\prana\OneDrive\Documents\DEEPLEARNING MODELS FOR REMOVING ARTIFACTS FROM EEG DATA\packages\Physiomotion Dataset\Original Dataset\derivatives\Manual_Annotations\sub30_run06.csv`

Threshold used:
`0.35`

Sampling frequency:
`250.0 Hz`

## Metric categories

### Segmentation / overlap metrics

- Dice coefficient: overlap between predicted artifact mask and manual artifact mask.
- IoU / Jaccard index: intersection divided by union of predicted and manual artifact masks.
- mIoU: mean of artifact-class IoU and clean-class IoU for this binary segmentation task.

### Detection metrics

- Precision: among samples predicted as artifact, how many were truly artifact.
- Recall / sensitivity: among true artifact samples, how many were detected.
- F1-score: harmonic mean of precision and recall.
- Accuracy: overall correct artifact/clean decisions.
- Specificity: among true clean samples, how many were correctly kept clean.

### Temporal boundary metric

- Hausdorff distance is calculated using start/end boundary points of predicted and
  manual artifact intervals. It is reported in samples, seconds, and milliseconds.
  This directly measures how far the predicted artifact start/end timing can be
  from the manual annotation boundary.

## Artifact-wise interpretation

The trained model is binary: artifact vs clean. It does not output artifact class
names. Therefore, per-artifact results are artifact-conditioned evaluations.
For each manual annotation label or group, a separate ground-truth mask is
created. Metrics are then calculated only on non-overlapping 10-second test
windows where that artifact label/group is present. This answers: when this
artifact type appears in the test recording, how well does the binary artifact
detector cover it?
