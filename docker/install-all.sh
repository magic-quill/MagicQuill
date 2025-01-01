#!/usr/bin/env bash

#
# Command line arguments:
#   --install-miniconda     Install Miniconda if not already installed
#   --use-local-wheels      Use local PyTorch wheels for installation
#   --help                  Display help message

# Default values
INSTALL_MINICONDA=false
USE_LOCAL_WHEELS=false

MINICONDA_PATH=~/miniconda3

show_help() {
    echo "Usage: $0 [--install-miniconda] [--use-local-wheels] [--help]"
}

# Function to detect if Conda is installed.
detect_conda() {
    # Check if Conda is in PATH
    if command -v conda &> /dev/null; then
        local conda_version=$(conda --version | cut -d' ' -f2)
        echo "Conda v${conda_version} detected in PATH"
        return 0
    fi
    
    return 1
}

# Function to install Miniconda
install_miniconda() {
    if [ ! -d "$MINICONDA_PATH" ]; then
        echo "[INFO] Installing Miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $MINICONDA_PATH -u
        rm -f miniconda.sh
    fi

    source ${MINICONDA_PATH}/bin/activate
    conda init --all
    
    MINICONDA_VERSION=$(conda --version)
    echo "[INFO] Mini${MINICONDA_VERSION} installed."
}

# Function to create and initialise the MagicQuill conda environment
create_magicquill_env() {
    # Initialize conda
    CONDA_PATH=$(conda info --base)
    MAGICQUILL_ENV=MagicQuill

    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "${MAGICQUILL_ENV}"; then
        echo "[INFO] Creating conda environment: ${MAGICQUILL_ENV}"
        conda create -n "${MAGICQUILL_ENV}" python=3.10 -y

        # Initialize conda after creating new environment
        echo "[INFO] Initializing conda..."
        conda init bash
    fi

    # Activate conda environment
    echo "Activating conda environment: ${MAGICQUILL_ENV}"
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    source ~/miniconda3/bin/activate
    conda activate "${MAGICQUILL_ENV}"
}

# Function to install PyTorch with CUDA 11.8
install_torch() {
    echo "[INFO] Installing PyTorch 2.1.2 with CUDA 11.8..."
    if [ "$USE_LOCAL_WHEELS" = "true" ]; then
        echo "Using local wheels... for debugging"
        pip install ./torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl
        pip install ./torchaudio-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl
        pip install ./torchvision-0.16.2%2Bcu118-cp310-cp310-linux_x86_64.whl
    else
        pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
    fi

    # Verify PyTorch installation and CUDA availability
    python -c "import torch; print('PyTorch version:', torch.version); print('CUDA available:', torch.cuda.is_available())"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install or verify PyTorch"
        exit 1
    fi
}

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --install-miniconda)
            INSTALL_MINICONDA=true
            shift
            ;;
        --use-local-wheels)
            USE_LOCAL_WHEELS=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done



# Check if Conda is installed
if ! detect_conda; then
    if [ "$INSTALL_MINICONDA" = "true" ]; then
        install_miniconda
    else
        echo "[ERROR] Conda is required. Please install Miniconda or use --install-miniconda option."
        exit 1
    fi
fi

# Create and activate MagicQuill conda environment
create_magicquill_env

# Install PyTorch with CUDA 11.8
install_torch

# Install the required interface package
echo "[INFO] Installing gradio_magicquill..."
pip install ./gradio_magicquill-0.0.1-py3-none-any.whl

# Install the llava environment
echo "Setting up LLaVA environment..."
if [ ! -d "./MagicQuill/MagicQuill/LLaVA" ]; then
    echo "[ERROR] Directory MagicQuill/LLaVA does not exist. Ensure the folder structure is correct."
    exit 1
fi
cp -f pyproject.toml ./MagicQuill/MagicQuill/LLaVA/
pip install -e ./MagicQuill/MagicQuill/LLaVA/

# Install remaining dependencies
echo "[INFO] Installing additional requirements..."
pip install -r requirements.txt

# Clean up
echo "[INFO] Cleaning up..."
rm -f ./gradio_magicquill-0.0.1-py3-none-any.whl
rm -f ./torch*.whl
rm -f ./requirements.txt
