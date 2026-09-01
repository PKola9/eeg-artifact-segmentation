# Threshold sweep results

This folder contains the validation-threshold sweep files.

How to read these files:

- `validation_threshold_sweep.csv` shows metrics at each tested threshold on the validation set.
- `threshold_tuning_summary.json` shows which threshold was selected using validation Dice.
- `test_at_selected_and_0p5_threshold.csv` shows final test metrics at the selected threshold and at threshold 0.5.

Important explanation:

The threshold is selected using validation Dice. The final test result is then reported on the held-out test set.
