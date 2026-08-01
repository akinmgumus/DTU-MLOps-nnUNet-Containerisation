#!/bin/bash
set -e

# Create cache directories with write permissions
mkdir -p /tmp/matplotlib /tmp/torch_cache
chmod -R 777 /tmp/matplotlib /tmp/torch_cache 2>/dev/null || true

# KAGGLE_* and WANDB_* arrive from the environment (docker run --env-file .env),
# never from a file baked into the image. The nnUNet_* paths are deliberately not
# read from the environment — they are pinned to container paths below, so a host
# .env cannot leak host paths in.
echo "KAGGLE_USERNAME set: ${KAGGLE_USERNAME:+yes}"
echo "WANDB_API_KEY set:   ${WANDB_API_KEY:+yes}"

if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "Error: KAGGLE_USERNAME and KAGGLE_KEY are required to download the dataset."
    echo "Run with: docker run --env-file .env ..."
    exit 1
fi

# Set nnUNet environment variables to container paths
export nnUNet_raw="/app/nnUNet_raw"
export nnUNet_preprocessed="/app/nnUNet_preprocessed"
export nnUNet_results="/app/nnUNet_results"

# Set cache directories
export HOME="/tmp"
export MPLCONFIGDIR="/tmp/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache"

# W&B Login (if WANDB_API_KEY is set, this will use it)
if [ ! -z "$WANDB_API_KEY" ]; then
    echo "--- W&B Login ---"
    wandb login "$WANDB_API_KEY"
fi

echo "--- Step 1: Data Preparation ---"
# Run the 'main' command from data.py (Download + Export)
python3 -m dtu_mlops_111.data main

echo "--- Step 2: nnU-Net Planning ---"
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity

echo "--- Step 3: Training (5 Epochs) ---"
# Call custom trainer class using -tr parameter
nnUNetv2_train 101 2d 0 -tr nnUNetTrainer_5epochs_custom --npz

# In case you want to train all 5 folds for ensemble uncomment the following lines:
# for fold in 0 1 2 3 4; do
#     echo "Training Fold $fold..."
#     nnUNetv2_train 101 2d $fold -tr nnUNetTrainer_5epochs_custom --npz
# done

# echo "--- All 5 folds trained successfully! ---"

echo "--- Process Completed Successfully! ---"