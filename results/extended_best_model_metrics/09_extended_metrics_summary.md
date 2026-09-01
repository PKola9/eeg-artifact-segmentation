# Improved M3 extended metric summary

This folder evaluates the final improved multi-dataset flow-initialized Split U-Net checkpoint on the saved 10-second test windows.

## Main overlap metrics

- Dice: 0.4328
- IoU / Jaccard: 0.2761
- Mean IoU, clean + artifact: 0.6048

## Detection metrics

- Precision: 0.3564
- Recall / sensitivity: 0.5508
- F1-score: 0.4328
- Accuracy: 0.9351
- Specificity: 0.9532
- Balanced accuracy: 0.7520
- Negative predictive value: 0.9783
- Matthews correlation coefficient: 0.4107
- Cohen's kappa: 0.4000

## Error and probability metrics

- False positive rate: 0.0468
- False negative rate: 0.4492
- False discovery rate: 0.6436
- Brier score: 0.0417
- Binary cross-entropy: 0.1467
- AUROC: 0.9044
- Average precision / PR-AUC: 0.3683

## Temporal boundary metric

- Hausdorff boundary distance: 606.504 seconds

Important: Hausdorff distance is a strict worst-case boundary metric. It is sensitive to isolated unmatched intervals, so it should not be described as an average start/end-time error.
