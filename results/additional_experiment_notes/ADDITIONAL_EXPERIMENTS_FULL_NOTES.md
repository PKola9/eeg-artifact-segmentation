# Additional Experiments and Extended Ablation Notes

This file records the additional experiments conducted after the main proposed flow-initialized Split U-Net pipeline. These experiments should be included in the thesis as **additional experiments / extended ablations**, not as replacements for the main method.

The main proposed method remains:

> Multi-dataset flow-matching initialized Split U-Net for dense EEG artifact probability mapping and start/end-time localization.

The additional experiments were used to answer follow-up questions:

1. Does adding more unlabeled EEG data improve flow-matching pretraining?
2. Does better flow-matching MSE automatically improve supervised artifact segmentation?
3. How does the proposed flow-initialized Split U-Net compare with a high-capacity transformer-style baseline?
4. Is model capacity a possible reason why the transformer performs better?

---

## 1. Main result context before additional experiments

The main controlled comparison was built around the same Split U-Net architecture.

### Main models

| Model | Description |
|---|---|
| M1 | Randomly initialized Split U-Net baseline |
| M2 | PhysioMotion-only flow-initialized Split U-Net |
| M3 | Multi-dataset flow-initialized Split U-Net using PhysioMotion + BCI Competition IV Dataset 1 + EEGMMIDB |

The important point is that M1, M2, and M3 are the fair flow-matching ablation because the supervised segmentation architecture is the same. Only the initialization/pretraining changes.

### Main M3 strict saved-window test result

The best proposed flow-initialized Split U-Net is the improved M3 model.

| Metric | Value |
|---|---:|
| Selected threshold | 0.35 |
| Dice | 0.4328 |
| IoU / Jaccard | 0.2761 |
| mIoU | 0.6048 |
| Precision | 0.3564 |
| Recall / Sensitivity | 0.5508 |
| F1 | 0.4328 |
| Accuracy | 0.9351 |
| Specificity | 0.9532 |
| AUROC | 0.9044 |
| PR-AUC | 0.3683 |

### Main M3 interval-aware/start-end result

For the same improved M3 model, a 2-second interval-aware start/end evaluation was also used to evaluate practical localization agreement.

| Metric | Value |
|---|---:|
| Dice | approximately 0.5813 |
| Precision | approximately 0.4336 |
| Recall | approximately 0.8816 |

This is useful because the practical goal is not only sample-level overlap but also identifying artifact intervals with start and end times.

---

## 2. M4: Flow matching with additional EEG datasets

### Purpose

M4 tested whether adding more unlabeled EEG datasets to flow-matching pretraining would improve downstream artifact segmentation.

### Flow-matching pretraining data

M4 used:

- PhysioMotion
- BCI Competition IV Dataset 1
- EEGMMIDB
- DEAP
- CHB-MIT

### Reason for the experiment

Flow matching is self-supervised and does not need artifact labels. Therefore, additional EEG datasets can be used to teach the model a broader representation of EEG structure.

The expectation was:

> More EEG data may help the flow model learn better EEG dynamics and therefore provide better initialization for supervised artifact segmentation.

### Outcome

M4 improved the flow-matching objective compared with some earlier runs, but it did not improve the final downstream segmentation compared with the improved M3 model.

### Interpretation

This showed an important methodological point:

> Lower flow-matching MSE does not automatically guarantee better artifact segmentation.

The reason is that flow matching learns general EEG structure, while the supervised task requires accurate artifact localization on PhysioMotion. If additional datasets differ strongly in montage, recording condition, preprocessing, amplitude distribution, or artifact characteristics, they may not transfer cleanly to the target segmentation task.

Therefore, M4 supports the conclusion that **dataset compatibility matters** in self-supervised EEG pretraining.

---

## 3. M5: TUH EEG added to flow-matching pretraining

### Purpose

M5 tested whether adding TUH EEG to the flow-matching corpus improves final artifact segmentation.

TUH was added because it is a large and diverse EEG dataset. The hypothesis was that this could improve flow-matching pretraining by exposing the model to broader EEG structure.

### TUH preprocessing

TUH EDF recordings were converted into the same model-compatible representation:

| Property | Value |
|---|---:|
| Channels | 59 |
| Window length | 10 seconds |
| Sampling rate | 250 Hz |
| Samples per window | 2500 |
| Window type | EEG-only flow-matching windows |

Because TUH is large and numerically heterogeneous, preprocessing was made robust:

- non-finite windows were skipped,
- extreme-valued windows were skipped,
- z-scored values were clipped,
- a 10-second non-overlapping stride was used to control storage and memory.

Final TUH flow dataset:

| Item | Value |
|---|---:|
| TUH windows | 35,783 |
| Channels | 59 |
| Samples per window | 2500 |
| Skipped recordings | 1 |

The TUH source labels were recorded for description but were not used as supervised targets during flow matching.

### Flow-matching pretraining data

M5 used:

- PhysioMotion
- BCI Competition IV Dataset 1
- EEGMMIDB
- DEAP
- CHB-MIT
- TUH

### M5 flow-matching result

The TUH-extended flow model improved the flow-matching MSE.

| Metric | Value |
|---|---:|
| Validation MSE | 0.4432 |
| Test MSE | 0.4424 |
| Test RMSE | 0.6651 |
| Best epoch | 20 |
| Epochs completed | 20 |

### Local result files for M5 flow matching

```text
outputs_v6_flowmatching_study/flowmatching_runs/V6_M5_TUH_full_stride10_flowmatching_20ep/flowmatching_metrics.csv
outputs_v6_flowmatching_study/flowmatching_runs/V6_M5_TUH_full_stride10_flowmatching_20ep/summary.json
outputs_v6_flowmatching_study/flowmatching_runs/V6_M5_TUH_full_stride10_flowmatching_20ep/flowmatching_channel_time_best_validation_mse.pt
outputs_v6_flowmatching_study/flowmatching_runs/V6_M5_TUH_full_stride10_flowmatching_20ep/flowmatching_channel_time_final.pt
```

### M5 downstream segmentation attempt 1

Settings:

| Setting | Value |
|---|---:|
| Learning rate | 5e-4 |
| Patience | 8 |

Threshold tuning result:

| Metric | Value |
|---|---:|
| Best validation threshold | 0.45 |
| Validation Dice | 0.4091 |
| Test Dice | 0.3924 |
| Test Precision | 0.3317 |
| Test Recall | 0.4801 |
| Test Dice at threshold 0.5 | 0.3961 |

Local files still need to be downloaded if not already present:

```text
outputs_v6_flowmatching_study/training_runs/V6_M5_TUH_full_flowinit_splitunet_lr5e4_patience8
outputs_v6_flowmatching_study/threshold_sweeps/V6_M5_TUH_full_flowinit_splitunet_lr5e4_patience8
```

### M5 downstream segmentation attempt 2

Settings:

| Setting | Value |
|---|---:|
| Learning rate | 2e-4 |
| Patience | 10 |

Threshold tuning result:

| Metric | Value |
|---|---:|
| Best validation threshold | 0.30 |
| Validation Dice | 0.3639 |
| Test Dice | 0.3568 |
| Test Precision | 0.2873 |
| Test Recall | 0.4704 |
| Test Dice at threshold 0.5 | 0.3613 |

Local result files:

```text
outputs_v6_flowmatching_study/training_runs/V6_M5_TUH_full_flowinit_splitunet_lr2e4_patience10/training_metrics.csv
outputs_v6_flowmatching_study/training_runs/V6_M5_TUH_full_flowinit_splitunet_lr2e4_patience10/summary.json
outputs_v6_flowmatching_study/threshold_sweeps/V6_M5_TUH_full_flowinit_splitunet_lr2e4_patience10/validation_threshold_sweep.csv
outputs_v6_flowmatching_study/threshold_sweeps/V6_M5_TUH_full_flowinit_splitunet_lr2e4_patience10/test_at_selected_and_0p5_threshold.csv
outputs_v6_flowmatching_study/threshold_sweeps/V6_M5_TUH_full_flowinit_splitunet_lr2e4_patience10/threshold_tuning_summary.json
```

### Interpretation of M5

M5 is important because it improved the flow-matching objective but did not improve segmentation.

This supports the thesis discussion point:

> A lower self-supervised flow-matching MSE means the model is better at predicting noise-to-EEG velocity, but downstream segmentation depends on whether the learned representation transfers to the artifact localization task.

Therefore, M5 should be reported as an additional experiment showing that larger pretraining data is not automatically better unless it is compatible with the target dataset and task.

---

## 4. M6: EEG-Conformer-style transformer segmentation baseline

### Purpose

M6 was added as an external architecture baseline. It tests how a transformer-style model performs on the same dense EEG artifact segmentation task.

This was included because transformer/conformer models are common modern baselines in EEG deep learning.

### Architecture

M6 is an EEG-Conformer-style dense segmentation model using:

- temporal convolution for local EEG feature extraction,
- Transformer encoder layers for longer-range temporal dependency modeling,
- an upsampling/decoder head to return dense channel-time artifact logits.

Input:

```text
59 x 2500 EEG window
```

Output:

```text
59 x 2500 artifact logit map
```

### Fairness note

M6 is not a direct flow-matching ablation because it has a different architecture and many more parameters.

| Model | Approximate parameters |
|---|---:|
| Small Split U-Net / M3 | 26,609 |
| M6 EEG-Conformer-style baseline | 297,083 |

M6 is approximately 11 times larger than the original Split U-Net. Therefore, it is best treated as a high-capacity transformer baseline, not as a direct replacement for the controlled flow-matching comparison.

### Training result

| Item | Value |
|---|---:|
| Best validation Dice during training | 0.6272 |
| Best epoch | 19 |
| Early stopping epoch | 27 |

### Threshold tuning result

| Metric | Value |
|---|---:|
| Best validation threshold | 0.65 |
| Validation Dice | 0.6305 |
| Test Dice | 0.6704 |
| Test Precision | 0.6223 |
| Test Recall | 0.7266 |
| Test Dice at threshold 0.5 | 0.6431 |

### Local/HPC result files

```text
outputs_v6_flowmatching_study/training_runs/V6_M6_eeg_conformer_segmentation_baseline/training_metrics.csv
outputs_v6_flowmatching_study/training_runs/V6_M6_eeg_conformer_segmentation_baseline/summary.json
outputs_v6_flowmatching_study/threshold_sweeps/V6_M6_eeg_conformer_segmentation_baseline/validation_threshold_sweep.csv
outputs_v6_flowmatching_study/threshold_sweeps/V6_M6_eeg_conformer_segmentation_baseline/test_at_selected_and_0p5_threshold.csv
outputs_v6_flowmatching_study/threshold_sweeps/V6_M6_eeg_conformer_segmentation_baseline/threshold_tuning_summary.json
```

### Interpretation of M6

M6 achieved the strongest absolute segmentation performance so far. This suggests that attention-based temporal modeling is highly effective for dense EEG artifact localization.

However, this does not invalidate the flow-matching contribution. The fair flow-matching ablation remains M1 vs M2 vs M3, where the architecture is fixed.

A careful thesis interpretation is:

> The controlled Split U-Net experiments show that flow-matching initialization improves the same segmentation architecture. The transformer baseline shows that high-capacity attention-based architectures can further improve artifact segmentation, motivating future work on flow-initialized transformer segmentation.

---

## 5. Planned / optional M7: Large flow-initialized Split U-Net

### Purpose

M7 was prepared to test whether the M6 transformer advantage is partly due to model capacity.

The original flow-initialized Split U-Net was very small compared with M6. Therefore, M7 increases the Split U-Net width.

### Architecture size

| Model | Base features | Approximate parameters |
|---|---:|---:|
| Original Split U-Net | 8 | 26,609 |
| Large Split U-Net | 32 | 419,009 |
| Large flow-matching U-Net | 32 | 419,058 |

### Dataset choice

M7 should use the same pretraining data as the best M3 flow model:

- PhysioMotion
- BCI Competition IV Dataset 1
- EEGMMIDB

The reason is that M4/M5 showed that adding more datasets can improve flow MSE but does not necessarily improve segmentation. Therefore, M7 should isolate model capacity, not change both dataset and architecture at the same time.

### Scientific question

M7 asks:

> If the flow-initialized Split U-Net is given a parameter count comparable to the transformer baseline, can it close the performance gap?

This is an optional additional experiment. It is not required to explain the current completed project.

---

## 6. Overall conclusion from additional experiments

The additional experiments make the thesis stronger because they show that the project did not simply train one model. It investigated the relationship between:

- self-supervised pretraining data,
- dataset compatibility,
- flow-matching objective quality,
- downstream segmentation transfer,
- architecture capacity,
- transformer-based attention modeling,
- practical start/end-time artifact localization.

Main conclusions:

1. Flow matching improves Split U-Net when compared within the same architecture family.
2. More EEG datasets can improve flow MSE, but do not automatically improve artifact segmentation.
3. TUH improved the flow-matching objective, but did not improve downstream segmentation.
4. Dataset compatibility matters for self-supervised EEG transfer.
5. The transformer baseline is much stronger in absolute Dice, but it is also much larger and architecturally different.
6. Future work should explore flow-matching pretraining combined with transformer-style segmentation.

This gives a scientifically honest direction:

> The proposed flow-matching initialization is useful for improving Split U-Net artifact localization, while the extended transformer baseline shows a promising future direction for high-capacity EEG artifact segmentation.
