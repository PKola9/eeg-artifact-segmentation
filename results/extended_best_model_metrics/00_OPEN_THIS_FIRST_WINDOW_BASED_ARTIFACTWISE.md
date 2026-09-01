# Window-based artifact-wise evaluation

This folder evaluates artifact-wise metrics on the saved 10-second test windows:

`C:\Users\prana\OneDrive\Documents\DEEPLEARNING MODELS FOR REMOVING ARTIFACTS FROM EEG DATA\BFM2_ChannelTimeSegmentation_V2_WORKING\outputs_v2_channel_time\window_datasets\sub30_run06_channel_time_v2_test`

This is easier to compare with the model's existing test-window Dice because it
uses the same prepared test-window format.

The model is binary artifact-vs-clean. Artifact-wise metrics are calculated by
creating a manual mask for each annotation label/group on the same saved test
windows, then comparing the binary model prediction with that label/group mask.

Threshold: `0.35`

Overall saved-window Dice: `0.4328`
Overall saved-window IoU: `0.2761`
Overall saved-window mIoU: `0.6048`
