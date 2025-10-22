#!/bin/bash

# GraspMAS Installation Script
# This script automates the installation of GraspMAS for zero-shot grasp detection
# Usage: bash install_graspmas.sh YOUR_OPENAI_API_KEY

set -e  # Exit on error

echo "============================================================"
echo "GraspMAS Installation Script"
echo "IROS 2025 - Zero-Shot Language-driven Grasp Detection"
echo "============================================================"
echo ""

# Check if API key is provided
if [ -z "$1" ]; then
    echo "ERROR: OpenAI API key required"
    echo "Usage: bash install_graspmas.sh YOUR_OPENAI_API_KEY"
    echo ""
    echo "Get your API key from: https://platform.openai.com/api-keys"
    exit 1
fi

API_KEY=$1
BASE_DIR="/home/dhruv/via"
GRASPMAS_DIR="$BASE_DIR/GraspMAS"

echo "Configuration:"
echo "  Base directory: $BASE_DIR"
echo "  Install to: $GRASPMAS_DIR"
echo "  GPU: Device 4"
echo "  CUDA: 11.8"
echo ""

# Step 1: Clone repository
echo "Step 1/8: Cloning GraspMAS repository..."
cd "$BASE_DIR"
if [ -d "$GRASPMAS_DIR" ]; then
    echo "  ⚠ GraspMAS directory already exists. Pulling latest changes..."
    cd "$GRASPMAS_DIR"
    git pull
    git submodule update --init --recursive
else
    git clone --recurse-submodules https://github.com/Fsoft-AIC/GraspMAS.git
    cd "$GRASPMAS_DIR"
fi
echo "  ✓ Repository ready"
echo ""

# Step 2: Create API key file
echo "Step 2/8: Setting up OpenAI API key..."
echo "$API_KEY" > api.key
chmod 600 api.key  # Secure the key file
echo "  ✓ API key saved to api.key"
echo ""

# Step 3: Create conda environment
echo "Step 3/8: Creating conda environment 'graspmas'..."
if conda env list | grep -q "^graspmas "; then
    echo "  ⚠ Environment 'graspmas' already exists. Skipping creation."
else
    conda create -n graspmas python=3.9 -y
    echo "  ✓ Environment created"
fi
echo ""

# Step 4: Install CUDA toolkit
echo "Step 4/8: Installing CUDA toolkit 11.8..."
eval "$(conda shell.bash hook)"
conda activate graspmas
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
echo "  ✓ CUDA toolkit installed"
echo ""

# Step 5: Install PyTorch
echo "Step 5/8: Installing PyTorch 2.2.0 with CUDA 11.8..."
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
echo "  ✓ PyTorch installed"
echo ""

# Step 6: Install Python requirements
echo "Step 6/8: Installing Python dependencies..."
pip install -r requirements.txt
echo "  ✓ Requirements installed"
echo ""

# Step 7: Install Detectron2
echo "Step 7/8: Installing Detectron2..."
cd detectron2

# Set CUDA paths to use conda CUDA 11.8 instead of system CUDA 12.3
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install ninja for faster builds
echo "  Installing ninja build system..."
conda install ninja -y

# Force use of conda CUDA 11.8
echo "  Building Detectron2 with CUDA 11.8..."
TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" pip install -e . --no-build-isolation

cd ..
echo "  ✓ Detectron2 installed"
echo ""

# Step 8: Download pretrained models
echo "Step 8/8: Downloading pretrained models..."
echo "  This may take 10-20 minutes depending on your connection..."
bash download.sh
echo "  ✓ Models downloaded"
echo ""

# Verification
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

python -c "
import torch
import sys

print('Python version:', sys.version.split()[0])
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU count:', torch.cuda.device_count())
    
# Test imports
try:
    from image_patch import ImagePatch
    print('ImagePatch import: ✓')
except Exception as e:
    print('ImagePatch import: ✗ -', e)
    sys.exit(1)

# Check API key
try:
    with open('api.key', 'r') as f:
        key = f.read().strip()
    print('API key loaded: ✓')
    print('Key preview:', key[:8] + '...' + key[-4:])
except:
    print('API key: ✗')
    sys.exit(1)

print('\\nInstallation verification: ✓ SUCCESS')
"

VERIFICATION_STATUS=$?

echo ""
echo "============================================================"
if [ $VERIFICATION_STATUS -eq 0 ]; then
    echo "✓ INSTALLATION COMPLETE"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Activate environment: conda activate graspmas"
    echo "  2. Set GPU: export CUDA_VISIBLE_DEVICES=4"
    echo "  3. Test installation: jupyter notebook test_graspmas.ipynb"
    echo ""
    echo "Resources:"
    echo "  - Setup guide: $BASE_DIR/GRASPMAS_SETUP.md"
    echo "  - Test notebook: $BASE_DIR/test_graspmas.ipynb"
    echo "  - Comparison: $BASE_DIR/MODEL_COMPARISON_AND_PIPELINE.md"
    echo ""
    echo "Quick test:"
    echo "  cd $GRASPMAS_DIR"
    echo "  conda activate graspmas"
    echo "  python main_simple.py --query 'Grasp the object' --image-path test.jpg --save-folder output/"
    echo ""
else
    echo "✗ INSTALLATION FAILED"
    echo "============================================================"
    echo ""
    echo "Please check the error messages above and:"
    echo "  1. Ensure conda is properly installed"
    echo "  2. Check internet connection for downloads"
    echo "  3. Verify CUDA toolkit compatibility"
    echo "  4. Check Python package dependencies"
    echo ""
fi

echo "Installation log saved to: $GRASPMAS_DIR/install.log"
