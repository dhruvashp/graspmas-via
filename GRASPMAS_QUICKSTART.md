# GraspMAS Quick Start Guide

## 🚀 Installation (One Command)

```bash
cd /home/dhruv/via
bash install_graspmas.sh YOUR_OPENAI_API_KEY
```

**Time**: 25-45 minutes (depending on network speed)

---

## 📋 What You Get

After installation:

```
/home/dhruv/via/
├── GraspMAS/                    # GraspMAS repository
│   ├── agents/                  # Multi-agent system
│   ├── grasp/                   # Grasp detection models
│   ├── image_patch.py          # Main interface
│   ├── simple_demo.ipynb       # Official demo
│   ├── api.key                 # Your OpenAI key
│   └── ...
├── test_graspmas.ipynb         # Your custom test notebook
├── GRASPMAS_SETUP.md           # Detailed setup guide
├── MODEL_COMPARISON_AND_PIPELINE.md  # Model comparison
└── install_graspmas.sh         # Installation script
```

**Conda environment**: `graspmas` (Python 3.9, PyTorch 2.2.0, CUDA 11.8)

---

## ⚡ Quick Test (3 Ways)

### Option 1: Using the Test Notebook (Recommended)
```bash
conda activate graspmas
export CUDA_VISIBLE_DEVICES=4
cd /home/dhruv/via
jupyter notebook test_graspmas.ipynb
```

### Option 2: Using Official Demo
```bash
conda activate graspmas
export CUDA_VISIBLE_DEVICES=4
cd /home/dhruv/via/GraspMAS
jupyter notebook simple_demo.ipynb
```

### Option 3: Command Line
```bash
conda activate graspmas
export CUDA_VISIBLE_DEVICES=4
cd /home/dhruv/via/GraspMAS

# Test on an image
python main_simple.py \
    --api-file "api.key" \
    --max-round 5 \
    --query "Grasp the knife at its handle" \
    --image-path path/to/image.jpg \
    --save-folder output/
```

---

## 💻 Basic Usage in Python

```python
from image_patch import ImagePatch

# Initialize with image path
patch = ImagePatch('path/to/image.jpg', max_round=5)

# Execute grasp detection with natural language
result = patch.execute("Grasp the red cup by its handle")

# Result contains grasp pose information
print(result)
```

---

## 🎯 Example Queries

GraspMAS understands complex natural language:

**Simple**:
- `"Grasp the object"`
- `"Find a grasp pose"`

**Object-specific**:
- `"Grasp the knife"`
- `"Grasp the red cup"`
- `"Grasp the largest object"`

**Part-specific**:
- `"Grasp the knife at its handle"`
- `"Grasp the cup by the rim"`
- `"Grasp the bottle cap"`

**Complex reasoning**:
- `"Find the leftmost object and grasp it"`
- `"Grasp the metallic object on the table"`
- `"If there is a knife, grasp it by the handle, otherwise grasp any object"`

---

## 🔧 Configuration

### Adjust Refinement Rounds
```python
# More rounds = better quality, slower inference
patch = ImagePatch('image.jpg', max_round=3)  # Fast (5-15s)
patch = ImagePatch('image.jpg', max_round=5)  # Balanced (10-30s)
patch = ImagePatch('image.jpg', max_round=10) # Thorough (20-60s)
```

### Modify Tools (Advanced)

Edit `/home/dhruv/via/GraspMAS/image_patch.py` to:
- Add custom vision models
- Remove unused tools
- Adjust parameters

**Available tools**:
- SAM (Segment Anything Model)
- GroundingDINO (text-to-bbox)
- VLPart (part detection)
- BLIP (image captioning)
- Depth estimation
- Grasp detection models

---

## 📊 Performance Comparison

| Metric | GraspAnything++ | GraspMAS |
|--------|----------------|----------|
| **Speed** | 10-30s | 5-30s (adaptive) |
| **Zero-shot** | ❌ Poor | ✅ Excellent |
| **Complex queries** | ⚠️ Limited | ✅ Excellent |
| **Object detection** | ❌ None | ✅ Built-in |
| **Refinement** | ❌ No | ✅ Yes (iterative) |
| **Cost** | Free (GPU) | $0.02-0.05/query |

---

## 💰 Cost Breakdown

**OpenAI API Costs** (GPT-4 Turbo):
- Per query: $0.02-0.05 (typical)
- 10 queries/day: ~$1/day, ~$365/year
- 100 queries/day: ~$3/day, ~$1,095/year

**Tips to reduce costs**:
1. Use fewer refinement rounds (`max_round=3` instead of 5)
2. Batch similar queries together
3. Cache results for repeated images
4. Use GPT-4-mini if available (cheaper)

---

## 🐛 Troubleshooting

### Installation Issues

**Problem**: Git submodules not cloned
```bash
cd /home/dhruv/via/GraspMAS
git submodule update --init --recursive
```

**Problem**: CUDA out of memory
```python
# Use a different GPU
export CUDA_VISIBLE_DEVICES=0  # or 1, 2, 3, 5, 6, 7
```

**Problem**: OpenAI API errors
```bash
# Verify API key
cat /home/dhruv/via/GraspMAS/api.key

# Test API connection
python -c "
import openai
openai.api_key = open('api.key').read().strip()
print(openai.Model.list())
"
```

### Runtime Issues

**Problem**: Import errors
```bash
# Ensure environment is activated
conda activate graspmas

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Problem**: Slow inference
- Reduce `max_round` parameter
- Use faster GPU
- Check internet connection (API calls)

**Problem**: Poor results
- Increase `max_round` for more refinement
- Make query more specific
- Ensure image quality is good

---

## 📚 Additional Resources

**Documentation**:
- [GraspMAS Paper](https://arxiv.org/abs/2506.18448)
- [Project Website](https://zquang2202.github.io/GraspMAS/)
- [GitHub Repository](https://github.com/Fsoft-AIC/GraspMAS)

**Your Local Guides**:
- Detailed setup: `/home/dhruv/via/GRASPMAS_SETUP.md`
- Model comparison: `/home/dhruv/via/MODEL_COMPARISON_AND_PIPELINE.md`
- Test notebook: `/home/dhruv/via/test_graspmas.ipynb`

**Community**:
- GitHub Issues: Report bugs or ask questions
- Same team as GraspAnything++ (Fsoft-AIC)
- Updated 2 hours ago (active development!)

---

## 🎓 Key Concepts

### Multi-Agent Architecture

1. **Planner Agent**
   - Receives natural language query
   - Creates step-by-step strategy
   - Breaks down complex tasks

2. **Coder Agent**
   - Generates Python code dynamically
   - Selects appropriate vision tools
   - Executes generated code

3. **Observer Agent**
   - Evaluates execution results
   - Provides feedback for improvement
   - Triggers refinement if needed

### Why This Works

- **Zero-shot**: No training needed for new objects/queries
- **Adaptive**: Different tools for different queries
- **Self-correcting**: Iterative refinement improves results
- **Extensible**: Easy to add new tools/capabilities

---

## ✅ Next Steps

1. **Run installation**: `bash install_graspmas.sh YOUR_API_KEY`
2. **Test basic query**: Open `test_graspmas.ipynb`
3. **Compare models**: Test same query on both GraspAnything++ and GraspMAS
4. **Evaluate performance**: Check speed, quality, and cost
5. **Deploy**: Integrate with your robot control system

---

## 🔄 Migration from GraspAnything++

### Side-by-side comparison script

```python
import time
from pathlib import Path

# GraspAnything++ inference
import sys
sys.path.insert(0, '/home/dhruv/via/LGD')
# ... load model and run inference ...
start = time.time()
# result_plusplus = ...
time_plusplus = time.time() - start

# GraspMAS inference
sys.path.insert(0, '/home/dhruv/via/GraspMAS')
from image_patch import ImagePatch
start = time.time()
patch = ImagePatch('image.jpg', max_round=5)
result_mas = patch.execute("Grasp the object")
time_mas = time.time() - start

print(f"GraspAnything++: {time_plusplus:.2f}s")
print(f"GraspMAS: {time_mas:.2f}s")
```

### Recommended workflow

1. Week 1: Install and test GraspMAS
2. Week 2: Benchmark on your dataset
3. Week 3: Evaluate cost and performance
4. Week 4: Choose production model

---

## 📞 Support

If you encounter issues:

1. Check `/home/dhruv/via/GRASPMAS_SETUP.md` for detailed setup
2. Review error messages carefully
3. Check GitHub issues: https://github.com/Fsoft-AIC/GraspMAS/issues
4. Verify API key and quota: https://platform.openai.com/usage

---

**Last Updated**: October 15, 2025  
**GraspMAS Version**: Latest (IROS 2025)  
**Your Setup**: 8 GPUs, CUDA 12.3, Using GPU 4
