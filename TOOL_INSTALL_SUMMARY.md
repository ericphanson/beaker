# Installation & Tool Configuration Summary

## What was configured:

### 1. Python Package Structure
- Created `bird_head_detector/` package directory
- Added `__init__.py` with version info
- Moved and adapted `infer.py` to `bird_head_detector/infer.py`

### 2. PyProject.toml Configuration
- Added `[project.scripts]` section defining `bird-head-detector` console command
- **Separated dependencies into groups:**
  - **Base**: Core inference only (numpy, opencv-python, ultralytics, platformdirs)
  - **Training**: Experiment tracking (comet-ml, pyyaml)
  - **Preprocessing**: Data conversion (pillow, polars, tqdm)
  - **Dev**: All dependencies combined
- Added project URLs for homepage and repository
- Added build system configuration using hatchling

### 3. Cache Directory Support
- Added `get_cache_dir()` function using `platformdirs.user_cache_dir()`
- Cache location: `~/Library/Caches/bird-head-detector/models/` (on macOS)
- Models downloaded to cache are persistent across runs
- Tool works independently of source repository location

### 4. Installation Methods
- **Tool installation**: `uv tool install git+https://github.com/ericphanson/bird-head-detector.git`
- **Local development**: Clone repo and use `uv run python infer.py`
- Both methods work identically for end users

### 5. Model Download Behavior
- When installed as tool: Downloads to system cache directory
- When run from source: Still checks local `runs/detect/` first, then cache
- Automatic GitHub release download if no local model found
- Models persist between tool runs (no re-downloading)

## Installation Commands:

For end users on any computer (inference only - lightweight):
```bash
# Install the tool (40 packages vs 77 with full deps)
uv tool install git+https://github.com/ericphanson/bird-head-detector.git

# Use the tool
bird-head-detector --source image.jpg --crop
```

For development with training/preprocessing:
```bash
# Clone and develop with all dependencies
git clone https://github.com/ericphanson/bird-head-detector
cd bird-head-detector
uv sync --extra dev
uv run python train.py
uv run python convert_to_yolo.py
```

For development with inference only:
```bash
# Clone and develop with minimal dependencies  
git clone https://github.com/ericphanson/bird-head-detector
cd bird-head-detector
uv sync
uv run python infer.py --source image.jpg --crop
```

## File Structure Created:
```
bird-head-detector/
├── pyproject.toml          # Package configuration with console script
├── bird_head_detector/     # Python package
│   ├── __init__.py        # Package initialization
│   └── infer.py           # Main inference tool (adapted for packaging)
├── infer.py               # Original script (still works for development)
└── README.md              # Updated with installation instructions
```
