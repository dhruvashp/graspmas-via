#!/bin/bash

# Fix Detectron2 Installation with CUDA 11.8
# This script resolves the CUDA version mismatch error

set -e

echo "============================================================"
echo "Detectron2 CUDA Fix Script"
echo "Resolving CUDA 12.3 vs 11.8 mismatch"
echo "============================================================"
echo ""

# Activate environment
echo "Activating graspmas environment..."
eval "$(conda shell.bash hook)"
conda activate graspmas

# Navigate to Detectron2 directory
cd /home/dhruv/via/GraspMAS/detectron2

echo ""
echo "Current CUDA detection:"
echo "  System CUDA: 12.3"
echo "  PyTorch CUDA: 11.8"
echo "  Solution: Force use of conda CUDA 11.8"
echo ""

# Clean previous build attempts
echo "Cleaning previous build artifacts..."
rm -rf build/ dist/ detectron2.egg-info/
pip uninstall detectron2 -y 2>/dev/null || true

# Set environment to use conda CUDA 11.8
echo "Setting CUDA environment to use conda CUDA 11.8..."
export CUDA_HOME=$CONDA_PREFIX
export CUDA_PATH=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA_PATH: $CUDA_PATH"

# Install ninja for faster compilation
echo ""
echo "Installing ninja build system..."
conda install ninja -y

# Verify ninja installation
which ninja && echo "  ✓ Ninja installed" || echo "  ⚠ Ninja not found, will use slower build"

# Build and install Detectron2 with correct CUDA
echo ""
echo "Building Detectron2 with CUDA 11.8..."
echo "This may take 5-10 minutes..."
echo ""

# Use TORCH_CUDA_ARCH_LIST to specify GPU architectures
# --no-build-isolation ensures our CUDA paths are used
TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
FORCE_CUDA=1 \
pip install -e . --no-build-isolation -v

BUILD_STATUS=$?

echo ""
echo "============================================================"
if [ $BUILD_STATUS -eq 0 ]; then
    echo "✓ DETECTRON2 INSTALLATION SUCCESSFUL"
    echo "============================================================"
    echo ""
    
    # Verify installation
    echo "Verifying installation..."
    python -c "
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
print('Detectron2 version:', detectron2.__version__)
print('✓ Import successful')
"
    
    VERIFY_STATUS=$?
    
    if [ $VERIFY_STATUS -eq 0 ]; then
        echo ""
        echo "✓ Detectron2 is working correctly!"
        echo ""
        echo "Next steps:"
        echo "  1. Continue with: bash download.sh"
        echo "  2. Then test: jupyter notebook test_graspmas.ipynb"
    else
        echo ""
        echo "⚠ Installation succeeded but import failed"
        echo "Please check the error above"
    fi
else
    echo "✗ DETECTRON2 INSTALLATION FAILED"
    echo "============================================================"
    echo ""
    echo "Troubleshooting steps:"
    echo "  1. Check that conda environment is activated"
    echo "  2. Verify PyTorch installation: python -c 'import torch; print(torch.__version__)'"
    echo "  3. Check CUDA toolkit: conda list | grep cuda"
    echo "  4. Try manual installation:"
    echo "     cd /home/dhruv/via/GraspMAS/detectron2"
    echo "     export CUDA_HOME=\$CONDA_PREFIX"
    echo "     pip install -e . --no-build-isolation"
fi

echo ""
