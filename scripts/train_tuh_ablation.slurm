#!/usr/bin/env bash
#SBATCH --job-name=eeg_v6_m5_tuh
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=outputs_v6_flowmatching_study/logs/v6_m5_tuh_full_%j.out
#SBATCH --error=outputs_v6_flowmatching_study/logs/v6_m5_tuh_full_%j.err

set -eo pipefail

cd "$HOME/thesis_eeg/BFM2_V6_FLOWMATCHING_STUDY_FULL_PIPELINE"
mkdir -p outputs_v6_flowmatching_study/logs

# Avoid conda activate.d unbound variable failure seen earlier.
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate eeg_bfm2_gpu
set -u

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

bash RUN_V6_07_TRAIN_M5_TUH_EXTRA.sh
