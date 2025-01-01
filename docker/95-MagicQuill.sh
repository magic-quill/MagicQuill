#!/usr/bin/env bash
set -e

echo -e "\n[INFO] Running $0 script..."

# Activate conda environment
echo -e "\n[INFO] Activating conda environment..."
source ~/miniconda3/bin/activate
conda activate MagicQuill
conda info --envs

# Run nvidia and cuda checks
echo -e "\n[INFO] Checking NVIDIA GPU..."
nvidia-smi
nvcc --version

# Run PyTorch checks
echo -e "\n[INFO] Checking PyTorch..."
python -c "import torch; print(torch.cuda.is_available())"

# Run MagicQuill
echo -e "\n[INFO] Running MagicQuill..."
python /home/quill/MagicQuill/gradio_run.py || {
    echo "[ERROR] Failed to run MagicQuill."
    exit 1
}
