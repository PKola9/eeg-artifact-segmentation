# Artifact-wise segmentation interpretation

Source: `ARTIFACT_SPECIFIC_METRIC_BARS/00_source_per_artifact_label_metrics.csv`.

The model is a binary artifact segmenter. It does not classify artifact type. For this analysis, each manual artifact label is isolated and compared against the same binary predicted artifact mask.

| Artifact label | Band | Dice | IoU | Precision | Recall | PR-AUC | Hausdorff (s) | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hor_eyem | strong | 0.767 | 0.621 | 0.779 | 0.754 | 0.794 | 6.48 | Best class: spatially broad, sustained horizontal eye movement pattern; high precision and high recall. |
| chew | strong | 0.687 | 0.524 | 0.655 | 0.723 | 0.763 | 7.91 | Clear high-amplitude artifact morphology; very high precision, but recall below precision means the model detects confident cores more than full duration. |
| blink | strong | 0.677 | 0.511 | 0.925 | 0.534 | 0.902 | 7.61 | Clear high-amplitude artifact morphology; very high precision, but recall below precision means the model detects confident cores more than full duration. |
| eyebrow | strong | 0.668 | 0.502 | 0.928 | 0.522 | 0.782 | 5.74 | Clear high-amplitude artifact morphology; very high precision, but recall below precision means the model detects confident cores more than full duration. |
| blink_hor_headm | strong | 0.614 | 0.443 | 0.580 | 0.652 | 0.658 | 6.72 | Mixed artifact label; detected moderately well, but overlap is harder because the event combines multiple sources. |
| blink_eyebrow | moderate | 0.556 | 0.385 | 0.422 | 0.813 | 0.530 | 6.17 | Mixed artifact label; detected moderately well, but overlap is harder because the event combines multiple sources. |
| tongue | weak | 0.348 | 0.211 | 0.295 | 0.424 | 0.245 | 8.74 | Shorter or less stereotyped artifact pattern; more false positives/missed boundary portions reduce Dice. |
| hor_headm | weak | 0.297 | 0.174 | 0.770 | 0.184 | 0.583 | 6.58 | Head-movement/vertical movement labels are weakest; likely more variable, lower sample support, and harder to separate from baseline drift. |
| swallow_eyebrow | weak | 0.199 | 0.110 | 0.124 | 0.491 | 0.128 | 7.78 | Shorter or less stereotyped artifact pattern; more false positives/missed boundary portions reduce Dice. |
| blink_ver_headm | weak | 0.095 | 0.050 | 0.129 | 0.075 | 0.081 | 9.41 | Head-movement/vertical movement labels are weakest; likely more variable, lower sample support, and harder to separate from baseline drift. |
| ver_headm | weak | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 | 8.11 | Head-movement/vertical movement labels are weakest; likely more variable, lower sample support, and harder to separate from baseline drift. |