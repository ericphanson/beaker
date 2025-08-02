# Bird Head Detector

A simple YOLOv8n-based bird head detection model trained on the CUB-200-2011 dataset. The model attempts to identify bird head regions in images, though performance varies significantly with image quality, bird pose, and species.

The code was largely written by Claude Sonnet 4 via GitHub copilot.

## 1. License & Usage

| Origin | Original terms | What that means for these weights |
|--------|----------------|-----------------------------------|
| **Dataset:** Caltech-UCSD Birds-200-2011 | "Images are for **non-commercial research and educational purposes only**." | ➜ **No commercial use** of the weights or any derivative work. |
| **Training code:** Ultralytics YOLOv8 | Source and official models released under **GNU AGPL-3.0** | ➜ If you **redistribute or serve** the weights, you must also release the full source & weights **under AGPL-3.0**. |

### Summary  
Because the weights were trained on CUB images *and* with AGPL-licensed code, they are provided **solely for non-commercial research/education** under **AGPL-3.0**.  
Commercial use would require **separate rights to the images** *and* a **non-AGPL licence from Ultralytics**.

*No warranty. Provided "as is."*

## 2. Quick Start Inference

You'll need `git` and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

To run detection on images using a pre-trained model:

```bash
# Clone the repository
git clone https://github.com/ericphanson/bird-head-detector

cd bird-head-detector

# Install dependencies
uv sync

# Basic inference (downloads model automatically)
uv run python infer.py --source image.jpg --show

# Save results
uv run python infer.py --source image.jpg --save

# Create square crops around detected heads
uv run python infer.py --source image.jpg --crop
```

**Limitations:**
- Works best on clear, well-lit images of single birds
- Performance degrades with poor lighting, motion blur, or multiple birds
- May struggle with unusual poses or partially occluded heads
- False positives possible on non-bird objects

## 3. Model Card

**Architecture:** YOLOv8n (nano) - optimized for speed over accuracy  
**Dataset:** CUB-200-2011 bird parts (head regions only)  
**Training images:** ~6,000 train, ~6,000 validation  
**Classes:** 1 (bird_head)  
**Input size:** 640×640 pixels  

**Expected performance:**
- Generally reliable on clear photos of common bird species
- May miss small or distant birds
- Accuracy not evaluated on real-world deployment scenarios
- Model size: ~6MB (nano variant prioritizes speed and portability)

## 4. Development & Training

### 4.1. Data

Download and prepare the CUB-200-2011 dataset:

1. **Download CUB-200-2011:**
   ```bash
   # Download from Caltech (requires accepting terms)
   # http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
   # Extract to: data/CUB_200_2011/
   ```

2. **Convert to YOLO format:**
   ```bash
   uv run python convert_to_yolo.py
   ```
   
   This creates `data/yolo/` with train/val splits and YOLO-format labels. The conversion extracts head-related parts (beak, crown, forehead, eyes, nape, throat) and creates bounding boxes around them.

### 4.2. Install Dependencies

```bash
uv sync
```

Requires Python 3.12+. On M1/M2 Macs, verify MPS is available:
```bash
uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### 4.3. Training

**Basic training:**
```bash
uv run python train.py
```

**Debug mode** (faster, less data):
Edit `train.py` and set `'debug_run': True` in `TRAINING_CONFIG`.

**Comet.ml tracking** (optional):
```bash
export COMET_API_KEY="your-api-key"
uv run python train.py
```

**Training notes:**
- Expects ~2-4 hours on M1 MacBook Pro for full training
- Model converges quickly but benefits from longer training
- Batch size may need adjustment based on available memory
- No hyperparameter tuning has been performed

### 4.4. Releases

Create GitHub releases with trained models:

```bash
# Ensure clean repository state
git add . && git commit -m "Update before release"

# Create release
uv run python release.py

# Follow prompts for version number and model selection
```

The script uploads the selected model as `bird-head-detector.pt` along with training artifacts (plots, configs, results).
