# GraspMAS Setup and Installation Guide

This guide will help you set up GraspMAS alongside your existing GraspAnything++ environment.

## Directory Structure

```
/home/dhruv/via/
├── LGD/                          # Existing GraspAnything++
├── GraspMAS/                     # New GraspMAS (to be created)
├── test_data/                    # Your NPZ test files
└── conda environments:
    ├── grasp_anything            # Existing (PyTorch 2.0.1 + CUDA 11.8)
    └── graspmas                  # New (PyTorch 2.2.0 + CUDA 11.8)
```

## Installation Steps

### 1. Clone GraspMAS Repository

```bash
cd /home/dhruv/via
git clone --recurse-submodules https://github.com/Fsoft-AIC/GraspMAS.git
cd GraspMAS
```

**Note**: The `--recurse-submodules` flag is crucial as GraspMAS includes:
- `detectron2` (object detection)
- `vlpart` (part detection)
- `OCID_VLG` (dataset)

### 2. Setup OpenAI API Key

```bash
# Create api.key file with your OpenAI API key
echo "YOUR_OPENAI_API_KEY_HERE" > api.key
```

**Important**: 
- Get your API key from: https://platform.openai.com/api-keys
- Keep `api.key` file secure (already in .gitignore)
- Typical cost: $0.02-0.05 per grasp query

### 3. Create Conda Environment

```bash
conda create -n graspmas python=3.9 -y
conda activate graspmas
```

### 4. Install CUDA Toolkit

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

**Note**: Uses CUDA 11.8 (same as your existing setup), compatible with your CUDA 12.3 system.

### 5. Install PyTorch

```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**Differences from GraspAnything++**:
- PyTorch 2.2.0 (vs 2.0.1) - newer version
- Same CUDA 11.8 support

### 6. Install GraspMAS Requirements

```bash
pip install -r requirements.txt
```

**This installs**:
- OpenAI Python SDK
- Transformers
- GroundingDINO dependencies
- SAM (Segment Anything Model)
- BLIP for captioning
- VLPart for part detection
- Image processing libraries

### 7. Install Detectron2

```bash
cd detectron2
pip install -e .
cd ..
```

**Note**: Detectron2 is Meta's object detection framework, used by several tools in GraspMAS.

### 8. Download Pretrained Models

```bash
bash download.sh
```

**This downloads**:
- GroundingDINO weights
- SAM checkpoint
- VLPart models
- Other vision model weights

**Expected download size**: ~2-5 GB

---

## Verification Steps

### Check Installation

```python
import torch
import openai
from PIL import Image
import numpy as np

# 1. Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")

# 2. Check OpenAI API key
with open('api.key', 'r') as f:
    api_key = f.read().strip()
    print(f"\\nAPI key loaded: {api_key[:8]}...{api_key[-4:]}")

# 3. Check if main modules can be imported
try:
    from image_patch import ImagePatch
    print("\\n✓ GraspMAS modules loaded successfully")
except Exception as e:
    print(f"\\n✗ Error loading GraspMAS: {e}")
```

### Test GPU on Device 4

```python
import os
import torch

# Set GPU device 4 (same as your GraspAnything++ setup)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Verify GPU is accessible
print(f"Selected GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test tensor creation
x = torch.randn(1000, 1000).cuda()
print(f"\\n✓ GPU tensor creation successful")
print(f"Tensor device: {x.device}")
```

---

## File Structure Overview

```
GraspMAS/
├── agents/                      # Multi-agent system code
│   ├── planner.py              # Planner agent
│   ├── coder.py                # Coder agent
│   └── observer.py             # Observer agent
├── grasp/                      # Grasp detection models
├── vlpart/                     # VLPart (submodule)
├── detectron2/                 # Detectron2 (submodule)
├── OCID_VLG/                   # Dataset (submodule)
├── image_patch.py              # Main ImagePatch class
├── main_simple.py              # CLI inference script
├── simple_demo.ipynb           # Simple demo notebook
├── Maniskill_demo.ipynb        # ManiSkill robot simulation
├── requirements.txt            # Python dependencies
├── download.sh                 # Download pretrained models
└── api.key                     # Your OpenAI API key
```

---

## Key Configuration Files

### `image_patch.py` - Main Interface

This file contains:
- `ImagePatch` class - main entry point
- Tool definitions (SAM, GroundingDINO, VLPart, etc.)
- Code generation and execution logic

**Important parameters**:
- `max_round`: Maximum refinement iterations (default: 5)
- `api_key`: Path to OpenAI API key file
- Tools can be added/removed here

### `requirements.txt` - Dependencies

Key packages:
```
openai>=1.0.0           # OpenAI API
transformers            # HuggingFace models
torch                   # PyTorch
torchvision            # Vision utilities
groundingdino          # Text-to-bbox grounding
segment-anything       # SAM
opencv-python          # Image processing
pillow                 # Image I/O
numpy                  # Numerical computing
```

---

## GPU and Memory Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (for running all tools)
- **RAM**: 16GB system memory
- **Storage**: 10GB for models and code

### Recommended
- **GPU**: 12GB+ VRAM (RTX 3090, A5000, etc.)
- **RAM**: 32GB system memory
- **Storage**: 20GB

### Your Setup (8 GPUs, CUDA 12.3)
- ✅ More than adequate
- Using GPU 4 as before
- CUDA 11.8 toolkit compatible with CUDA 12.3 system

---

## Expected Installation Time

| Step | Time | Notes |
|------|------|-------|
| Clone repository | 1-2 min | Includes submodules |
| Create conda env | 1-2 min | |
| Install PyTorch | 3-5 min | ~2GB download |
| Install requirements | 5-10 min | Many dependencies |
| Install Detectron2 | 2-3 min | Build from source |
| Download models | 10-20 min | 2-5GB download |
| **Total** | **25-45 min** | Depends on network speed |

---

## Troubleshooting

### Issue: Submodules not cloned
```bash
# If you forgot --recurse-submodules
cd GraspMAS
git submodule update --init --recursive
```

### Issue: CUDA out of memory
```python
# Reduce batch size or use CPU for some tools
# Edit image_patch.py to adjust device allocation
```

### Issue: OpenAI API errors
```bash
# Verify API key is valid
python -c "import openai; openai.api_key='YOUR_KEY'; print(openai.Model.list())"
```

### Issue: Detectron2 build fails
```bash
# Install build dependencies
conda install gcc_linux-64 gxx_linux-64
pip install ninja
cd detectron2
pip install -e .
```

---

## Next Steps

After installation:

1. ✅ **Run simple_demo.ipynb** - Test on example images
2. ✅ **Test on your NPZ data** - Convert and test on your robot images
3. ✅ **Compare with GraspAnything++** - Benchmark performance
4. ✅ **Customize tools** - Add/remove vision models as needed
5. ✅ **Production deployment** - Optimize for your use case

---

## Quick Start Command Summary

```bash
# Full installation in one go
cd /home/dhruv/via
git clone --recurse-submodules https://github.com/Fsoft-AIC/GraspMAS.git
cd GraspMAS
echo "YOUR_OPENAI_API_KEY" > api.key
conda create -n graspmas python=3.9 -y
conda activate graspmas
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
cd detectron2 && pip install -e . && cd ..
bash download.sh
```

Then test:
```bash
# Activate environment
conda activate graspmas
export CUDA_VISIBLE_DEVICES=4

# Run simple demo
jupyter notebook simple_demo.ipynb
```
