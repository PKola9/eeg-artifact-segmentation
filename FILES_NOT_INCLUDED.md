# Files Not Included in This GitHub Release

This repository is a clean code and selected-results release. It intentionally does not include very large raw or generated data files.

## Not uploaded

The following file categories are not included:

1. **Raw EEG datasets**
   - PhysioMotion raw EDF/annotation folders
   - BCI Competition IV raw/preprocessed data
   - EEGMMIDB raw/preprocessed data
   - DEAP raw/preprocessed data
   - CHB-MIT raw/preprocessed data
   - TUH raw EDF data

2. **Generated memory-mapped training arrays**
   - Large `x_float32.dat` files
   - Large generated `window_datasets/` folders
   - Other intermediate arrays that can be regenerated from the preprocessing scripts

3. **Temporary working files**
   - ZIP transfer files
   - staging folders
   - presentation drafts
   - Python cache folders

## Why they are not included

These files are excluded because they are too large for GitHub and because many raw EEG datasets have separate access or licensing conditions. The repository instead includes the source code, scripts, selected result files, small checkpoints, and documentation needed to understand and reproduce the work when the datasets are available.

## What is included instead

The repository includes:

- data preparation scripts;
- model definitions;
- training scripts;
- evaluation scripts;
- threshold-selection code;
- interval extraction code;
- selected final CSV/JSON result summaries;
- selected final figures;
- small saved checkpoints used for provenance;
- reproducibility scripts.

## Reproducing the full pipeline

To fully reproduce training from raw data, the datasets must be placed in local or HPC paths matching the scripts, or the paths inside the scripts must be edited to match the new machine.

The expected workflow is:

1. prepare the raw datasets separately;
2. run the data preparation scripts;
3. run flow-matching pretraining;
4. train the segmentation models;
5. run threshold selection and evaluation;
6. generate interval and result visualizations.
