# Professor-ready final results package

This folder is the clean folder to show and explain. It is not a dustbin.

The full backup of older/existing result folders is kept separately here:

`BFM2_V6_FLOWMATCHING_STUDY_FULL_PIPELINE/BACKUP_RESULTS`

## Main story

The project evaluates EEG artifact segmentation using a Split U-Net model with flow-matching initialization.

The model receives a 10-second EEG window:

`59 channels × 2500 samples`

and predicts:

`artifact probability for every channel and time sample`

After thresholding, the probability map becomes a binary artifact mask. From this mask, artifact start and end intervals can be extracted.

## What to show in order

### 1. Flow matching pretraining

Folder:

`01_FLOW`

Use this to explain how flow matching learns EEG structure before supervised artifact segmentation.

Key idea:

`noise/intermediate EEG → predicted velocity toward real EEG`

Metric:

`MSE / RMSE`

Lower MSE means the flow model learned the EEG velocity field better.

### 2. Model comparison

Folder:

`02_MODEL_COMPARE`

Use this to compare:

1. Random Split U-Net
2. PhysioMotion flow-init Split U-Net
3. Multi-dataset flow-init Split U-Net

This answers:

“Does flow-matching initialization improve the segmentation model?”

### 3. Threshold sweep

Folder:

`03_THRESHOLDS`

Use this to explain that the probability threshold was selected using validation Dice, then final performance was evaluated on the held-out test set.

Important:

Validation is used for choosing the threshold/model.
Test is used for final reporting.

### 4. Overall test metrics

Folder:

`04_OVERALL_TEST`

This contains the overall saved-test-window metrics:

- Dice
- IoU / Jaccard
- mIoU
- Precision
- Recall / sensitivity
- F1-score
- Accuracy
- Specificity
- Hausdorff boundary distance

### 5. Individual artifact-label metrics

Folder:

`05_ARTIFACT_LABELS`

This is important for the professor’s new request.

It contains metrics separately for each raw manual annotation label, such as:

- `hor_eyem`
- `blink`
- `blink_hor_headm`
- `eyebrow`
- `chew`
- `tongue`
- `hor_headm`

Use:

`10_ALL_INDIVIDUAL_ARTIFACT_LABEL_RESULTS.csv`

and:

`11_ALL_INDIVIDUAL_ARTIFACT_LABEL_DICE.png`

### 6. Start/end interval demo

Folder:

`07_INTERVAL_DEMO`

Use this to show that the project is not only producing a Dice score. It can output:

- predicted artifact start time
- predicted artifact end time
- manual interval comparison
- probability curve

This is the most professor-facing practical output.

### 7. Probability outputs

Folder:

`08_PROB_OUTPUT`

This contains probability-map related outputs. Use only if he asks how the model output looks before thresholding.

### 8. Metric definitions and scripts

Folder:

`09_DEFINITIONS`

Use this if he asks:

- what Dice means
- what IoU means
- what mIoU means
- what Hausdorff distance means
- how artifact-wise evaluation was calculated

### 10. Code used

Folder:

`10_CODE`

This contains the main evaluation scripts used to generate the final tables.

## Important explanation about different Dice values

There are different Dice values because they answer different questions.

| Dice type | Meaning |
|---|---|
| Validation Dice | Used to choose checkpoint/threshold |
| Strict test Dice | Exact channel-time overlap on saved test windows |
| Tolerant interval Dice | Start/end interval overlap with timing tolerance |
| Artifact-wise Dice | Strict Dice for each manual annotation label/category |

Do not mix these as if they are the same number.

## The short explanation to say

“First, I evaluated the overall segmentation model on the held-out test windows. Then I extended the evaluation to artifact-wise analysis by using the manual annotation labels. Since the model is binary artifact-versus-clean, I created a separate ground-truth mask for each artifact label and compared the model’s binary artifact prediction against that label’s mask on the saved test windows. This gives both overall test-set performance and per-artifact performance.”
