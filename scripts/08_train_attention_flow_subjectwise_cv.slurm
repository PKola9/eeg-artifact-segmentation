#!/usr/bin/env bash
#SBATCH --job-name=cv_attn_flow
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=56G
#SBATCH --time=14:00:00
#SBATCH --array=0-4
#SBATCH --output=outputs_v6_flowmatching_study/logs/cv_attention_flow_%A_%a.out
#SBATCH --error=outputs_v6_flowmatching_study/logs/cv_attention_flow_%A_%a.err

set -eo pipefail

cd "$HOME/thesis_eeg/BFM2_V6_FLOWMATCHING_STUDY_FULL_PIPELINE"
mkdir -p outputs_v6_flowmatching_study/logs

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate eeg_bfm2_gpu
set -u

PHYSIO_DATASET="$HOME/thesis_eeg/BFM2_ChannelTimeSegmentation_V2_WORKING/outputs_v2_channel_time/window_datasets/full_sub30_runs6_channel_time_10s_250hz_memmap"
M3_FLOW_CKPT="outputs_v6_flowmatching_study/flowmatching_runs/V6_CLEAN_02_physio_bci_eegmmidb_flowmatching_20ep/flowmatching_channel_time_best_validation_mse.pt"

FOLD_NAMES=(
  "fold0"
  "fold1"
  "fold2"
  "fold3"
  "fold4"
)

FOLD_SUBJECTS=(
  "sub-1 sub-10 sub-15 sub-20 sub-25 sub-30"
  "sub-2 sub-11 sub-16 sub-21 sub-26 sub-29"
  "sub-3 sub-12 sub-17 sub-22 sub-27"
  "sub-5 sub-13 sub-18 sub-23 sub-28"
  "sub-7 sub-9 sub-14 sub-19 sub-24"
)

FOLD_NAME="${FOLD_NAMES[$SLURM_ARRAY_TASK_ID]}"
read -r -a SUBJECTS <<< "${FOLD_SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

OUTPUT_NAME="CV_ATTENTION_FLOWINIT_${FOLD_NAME}_subjectwise"
THRESHOLD_DIR="outputs_v6_flowmatching_study/threshold_sweeps/${OUTPUT_NAME}"

echo "Running subject-wise CV"
echo "Method: attention_flowinit_split_unet"
echo "Fold: ${FOLD_NAME}"
echo "Held-out subjects: ${SUBJECTS[*]}"
echo "Output: ${OUTPUT_NAME}"

python training/train_channel_time_segmentation_sharded.py \
  --dataset-dir "$PHYSIO_DATASET" \
  --model-arch split_unet_transformer_bottleneck \
  --holdout-subjects "${SUBJECTS[@]}" \
  --balanced-train-windows \
  --clean-to-artifact-ratio 1.0 \
  --pos-weight-override 1.0 \
  --flow-checkpoint "$M3_FLOW_CKPT" \
  --epochs 30 \
  --batch-size 8 \
  --lr 5e-4 \
  --patience 8 \
  --output-name "$OUTPUT_NAME"

python evaluation/tune_threshold_on_validation.py \
  --dataset-dir "$PHYSIO_DATASET" \
  --model "outputs_v6_flowmatching_study/training_runs/${OUTPUT_NAME}/splitunet_channel_time_best_val_dice.pt" \
  --model-arch split_unet_transformer_bottleneck \
  --holdout-subjects "${SUBJECTS[@]}" \
  --batch-size 8 \
  --output-dir "$THRESHOLD_DIR"

echo "Completed subject-wise CV attention_flowinit ${FOLD_NAME}"
