# Dense EEG Artifact Segmentation with Flow-Matching Pretraining

This repository contains the final code and selected result artifacts for a Master's thesis project on dense EEG artifact segmentation.

The project treats artifact handling as a channel-time segmentation problem. Each 10-second EEG window is represented as a 59 x 2500 tensor, corresponding to 59 EEG channels sampled at 250 Hz. The model predicts an artifact probability for every channel-time point, rather than assigning a single label to an entire EEG window.

## Project overview

The main research question is whether self-supervised flow-matching pretraining can provide useful EEG-aware initialization for a supervised Split U-Net artifact segmentation model.

The final pipeline has two main stages:

1. **Self-supervised flow-matching pretraining**
   - Uses unlabeled EEG windows.
   - Learns a noise-to-EEG velocity field.
   - Selects the best checkpoint using validation MSE.

2. **Supervised artifact segmentation**
   - Uses PhysioMotion manual artifact annotations.
   - Trains a Split U-Net to predict dense artifact probability maps.
   - Evaluates overlap, probability quality, threshold behavior, artifact-label behavior, and interval start/end outputs.

The repository also includes additional comparison experiments, including auxiliary dataset ablations, an attention-based EEG-Conformer-style baseline, and a post-presentation attention-flow follow-up experiment.

## Repository structure

```text
data_preparation/      Dataset conversion scripts
models/                Model definitions
training/              Flow-matching and segmentation training scripts
evaluation/            Evaluation, threshold tuning, interval extraction, and plotting scripts
scripts/               Reproducible shell scripts used for major experiment stages
docs/                  Supporting notes and documentation
results/               Selected CSVs, JSON summaries, plots, and small checkpoints
requirements.txt       Python dependency list
CODE_FILE_GUIDE.md     Explanation of the purpose of each code file
```

## What is included

This release includes:

- all source code used for data preparation, model training, evaluation, and visualization;
- selected final result tables and figures;
- small model checkpoints and training summaries needed to document the completed experiments;
- scripts showing the commands used to reproduce the major stages.

## What is not included

Large raw datasets and generated memory-mapped training arrays are not included in this repository because they are too large for GitHub and should be managed separately.

Excluded examples:

- raw EDF/CSV EEG datasets;
- generated `x_float32.dat` memory-map files;
- full local/HPC dataset folders;
- temporary notebooks, staging folders, and presentation-only drafts.

Some saved CSV/JSON result files contain the original local or HPC paths used during the experiments. These paths are kept as provenance metadata only. They do not mean that the same absolute paths must exist on another machine unless the full experiments are being reproduced from raw data.

## Dataset usage

The supervised segmentation task uses the PhysioMotion artifact annotations. Auxiliary EEG datasets are used only for self-supervised flow-matching pretraining experiments, where labels are not required.

The main dataset roles are:

| Dataset | Role |
|---|---|
| PhysioMotion | Supervised artifact segmentation and flow-matching pretraining |
| BCI Competition IV Dataset 1 | Auxiliary unlabeled EEG for flow-matching pretraining |
| EEGMMIDB | Auxiliary unlabeled EEG for flow-matching pretraining |
| DEAP | Auxiliary unlabeled EEG for additional pretraining ablation |
| CHB-MIT | Auxiliary unlabeled EEG for additional pretraining ablation |
| TUH | Auxiliary unlabeled EEG for additional pretraining ablation |

## Main model variants

| Model | Description |
|---|---|
| M1 | Randomly initialized Split U-Net baseline |
| M2 | Split U-Net initialized from PhysioMotion-only flow-matching pretraining |
| M3 | Split U-Net initialized from multi-dataset flow-matching pretraining using PhysioMotion, BCI IV, and EEGMMIDB |
| M4 | Additional flow-matching ablation with DEAP and CHB-MIT |
| M5 | Additional flow-matching ablation with TUH |
| M6 | Attention-based EEG-Conformer-style baseline |
| Attention-flow follow-up | Split U-Net with a Transformer bottleneck and M3 flow-matching initialization for compatible convolutional layers |

The thesis focuses mainly on the controlled Split U-Net comparison from M1 to M3. M4, M5, and M6 are included as additional experiments and future-direction evidence.

## Post-presentation update

After the thesis presentation, the evaluation was extended to stricter subject-level testing:

- three-subject held-out test sets;
- five-fold subject-wise cross-validation;
- an attention-flow follow-up model combining the Split U-Net segmentation pipeline, M3 flow initialization, and a Transformer bottleneck.

The main post-presentation files are:

```text
docs/post_presentation_update/post_presentation_evaluation_report.docx
docs/post_presentation_update/attention_flow_methodology_report.docx
results/post_presentation_update/
models/splitunet_transformer_bottleneck_segmentation.py
scripts/08_train_attention_flow_subjectwise_cv.slurm
```

The attention-flow model was evaluated on the same five subject-wise folds as the previous baselines. The comparison files in `results/post_presentation_update/` contain side-by-side metrics for Dice, IoU, precision, recall, accuracy, and fold-level Dice gains.

## Key result locations

```text
results/flow_pretraining_metrics/
results/model_comparison/
results/threshold_selection/
results/overall_test_metrics/
results/artifact_label_analysis/
results/extended_best_model_metrics/
results/interval_demo/
results/probability_outputs/
results/interval_matching_metrics/
results/training_summaries/
results/post_presentation_update/
```

The most important result folders are:

- `results/model_comparison/` for model-level Dice, precision, and recall comparisons;
- `results/overall_test_metrics/` for final held-out test-set metrics;
- `results/artifact_label_analysis/` for per-artifact-label analysis;
- `results/interval_demo/` and `results/probability_outputs/` for channel-specific start/end interval examples;
- `results/training_summaries/` for raw experiment summaries, threshold sweeps, logs, and small checkpoints.

## Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

The full training pipeline is designed for an HPC/GPU environment. The shell scripts in `scripts/` document the main stages:

```text
00_check_environment_and_data.sh
01_build_auxiliary_flow_datasets.sh
02_train_physio_flow_pretraining.sh
03_train_random_splitunet_baseline.sh
04_train_multidataset_flowinit_splitunet.sh
05_run_threshold_sweeps.sh
06_train_deap_chbmit_ablation.sh
07_train_tuh_ablation.sh
08_train_attention_flow_subjectwise_cv.slurm
train_tuh_ablation.slurm
```

Paths inside the scripts may need to be adjusted to match the local or HPC dataset location.

## Final note

This repository is organized as a clean research-code release. It keeps the source code, important result artifacts, and experiment provenance, while excluding large generated datasets and temporary working files.
