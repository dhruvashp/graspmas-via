# GraspMAS Testing Environment

A complete testing environment for **GraspMAS** (Multi-Agent System for Zero-Shot Grasp Detection) with ready-to-run Jupyter notebook and automated setup scripts.

## 🎯 What is GraspMAS?

GraspMAS is a state-of-the-art grasp detection system that uses three specialized AI agents:
- **Planner**: Strategizes complex grasp queries
- **Coder**: Generates Python code using vision tools  
- **Observer**: Evaluates results and provides feedback

**Key Features:**
- ✅ Excellent zero-shot performance on novel objects
- ✅ Handles complex natural language queries
- ✅ Built-in object detection (SAM, GroundingDINO, VLPart)
- ✅ Iterative refinement for better results
- ✅ Works with any RGB image

## 📋 Prerequisites

- **Hardware**: NVIDIA GPU with 8GB+ VRAM
- **Software**: 
  - Anaconda/Miniconda
  - CUDA 11.8+ 
  - OpenAI API key (GPT-4 access)

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
git clone <your-repo-url>
cd graspmas-testing

# Run automated installation (recommended)
bash install_graspmas.sh

# OR create environment manually:
conda create -n graspmas python=3.9 -y
conda activate graspmas
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install GraspMAS requirements
cd GraspMAS
pip install -r requirements.txt

# If you get Detectron2 CUDA errors, run:
bash ../fix_detectron2.sh
```

### 3. Setup OpenAI API Key

```bash
cd GraspMAS
echo "your-openai-api-key-here" > api.key
```

### 4. Download Pretrained Models

```bash
# Inside GraspMAS directory
bash download.sh
```

### 5. Run the Testing Notebook

```bash
# Start Jupyter in the repository root
jupyter notebook test_graspmas.ipynb
```

## 📝 Testing Notebook Overview

The `test_graspmas.ipynb` notebook includes:

1. **Environment Setup** - GPU configuration and imports
2. **Installation Verification** - Check all dependencies
3. **Image Loading** - Load rubber duck test image
4. **Basic Grasp Detection** - Simple "Grasp the rubber duck" query
5. **Grasp Analysis** - Detailed explanation of output format
6. **Robot Coordinate Conversion** - Convert 2D grasp to 3D robot pose
7. **Advanced Queries** - Part-level and conditional grasping
8. **Batch Processing** - Multiple images/queries
9. **Cost Estimation** - OpenAI API usage costs

## 🎯 Grasp Output Format

GraspMAS returns: `[quality, center_x, center_y, width, height, angle]`

- **Coordinates**: Top-left origin (0,0), standard OpenCV convention
- **Units**: Pixels in original image size (auto-scaled from internal 416×416)
- **Quality**: Confidence score 0.0-1.0
- **Angle**: Degrees, 0° = horizontal

## 🔧 Common Issues & Solutions

### CUDA Version Mismatch
```bash
bash fix_detectron2.sh
```

### NumPy 2.x Compatibility
```bash
pip install "numpy<2"
```

### Import Errors
```bash
# Restart Jupyter kernel after installing packages
# Kernel → Restart Kernel
```

### Missing OpenCV
```bash
pip install opencv-python
```

## 📊 Example Usage

```python
from agents.graspmas import GraspMAS

# Initialize
graspmas = GraspMAS(api_file='api.key', max_round=3)

# Run query
save_path, grasp_pose = await graspmas.query(
    "Grasp the rubber duck by its head", 
    "rubber_ducky.png"
)

print(f"Grasp pose: {grasp_pose}")
# Output: [0.9999, 995.33, 629.38, 336.53, 36.88, 5.46]
```

## 📚 Documentation

- **`GRASPMAS_SETUP.md`** - Detailed installation guide
- **`GRASPMAS_QUICKSTART.md`** - Quick reference
- **`test_graspmas.ipynb`** - Complete testing notebook with examples

## 💰 Cost Information

- **API Cost**: ~$0.02-0.05 per grasp query (OpenAI GPT-4)
- **Research usage**: ~$0.30/day (10 queries)
- **Development**: ~$1.50/day (50 queries)

## 🏗️ Repository Structure

```
graspmas-testing/
├── README.md                 # This file
├── test_graspmas.ipynb      # Main testing notebook
├── rubber_ducky.png         # Test image
├── install_graspmas.sh      # Automated setup script
├── fix_detectron2.sh        # CUDA fix script
├── GRASPMAS_SETUP.md        # Detailed setup guide
├── GRASPMAS_QUICKSTART.md   # Quick reference
└── GraspMAS/                # GraspMAS source code
    ├── agents/              # Multi-agent system
    ├── grasp/              # Grasp detection models
    ├── requirements.txt    # Python dependencies
    ├── download.sh         # Model download script
    └── api.key            # OpenAI API key (create this)
```

## 📖 Citation

```bibtex
@article{graspmas2025,
  title={GraspMAS: Multi-Agent System for Zero-Shot Grasp Detection},
  booktitle={IROS 2025},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test with the provided notebook
4. Submit a pull request

## 📄 License

This project follows the original GraspMAS licensing terms.

---

**Quick Test:** After setup, run cell 5 in the notebook with query "Grasp the rubber duck" to verify everything works! 🦆