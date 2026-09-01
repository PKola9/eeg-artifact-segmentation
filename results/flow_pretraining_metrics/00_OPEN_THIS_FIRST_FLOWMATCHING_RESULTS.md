# Flow-matching results to show

This folder summarizes how the self-supervised flow-matching stage behaved before supervised Split U-Net artifact segmentation.

## Main files

1. `01_flowmatching_summary_all_runs.csv`  
   Clean table containing FM1, FM2, and FM3/M4 flow-matching MSE/RMSE values.

2. `05_flowmatching_validation_test_mse_bar.png`  
   Best plot to show flow-matching validation/test MSE. Lower is better.

3. `06_flowmatching_best_validation_mse_all_runs.png`  
   Simple single-metric bar chart for validation MSE.

4. `07_flowmatching_and_segmentation_story_bars.png`  
   Side-by-side story: flow pretraining error and downstream segmentation Dice.

## Important note for FM2

FM2 had a saved best checkpoint, but the original epoch-by-epoch `flowmatching_metrics.csv` was missing. Therefore FM2 is shown using recovered checkpoint evaluation:

- `flowmatching_checkpoint_recovered_metrics.csv`
- `flowmatching_checkpoint_recovered_summary.json`

This means FM2 is still valid as a checkpoint evaluation, but it does not have a full epoch curve.

## Scientific interpretation

Lower flow-matching MSE means the flow model predicts the noise-to-EEG velocity more accurately. The final segmentation quality is reported separately using Dice, precision, and recall on the PhysioMotion artifact segmentation test set.
