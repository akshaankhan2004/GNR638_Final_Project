#!/usr/bin/env bash
# =============================================================================
# setup.bash  –  GNR638 Project
#
# GRADER FLOW (how this is called):
#   cd ./your_directory        ← grader cds into the cloned repo
#   bash setup.bash            ← THIS FILE runs from inside the repo
#   conda activate gnr_project_env
#   python inference.py --test_dir <absolute_path>
#
# What this script does (internet IS available at this point):
#   1. Create conda env  gnr_project_env  (Python 3.11)
#   2. Install PyTorch (CUDA 12.1 wheels — forward-compatible with CUDA 12.6)
#   3. Install all Python requirements from requirements.txt
#   4. Download Qwen2.5-VL-3B-Instruct weights (~7 GB, no HF token needed)
#
# NOTE: inference.py is already present in this directory (cloned by grader).
#       We do NOT clone the repo again inside this script.
# =============================================================================

set -e          # exit immediately on any error
set -o pipefail # catch errors in pipes

ENV_NAME="gnr_project_env"
PYTHON_VER="3.11"
MODEL_HF_ID="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_DIR="$HOME/models/qwen25vl-3b"

echo "============================================================"
echo " GNR638 Project Setup"
echo " Working dir : $(pwd)"
echo " Conda env   : $ENV_NAME  (Python $PYTHON_VER)"
echo " Model       : $MODEL_HF_ID"
echo " Model dir   : $MODEL_DIR"
echo " CUDA_VISIBLE: ${CUDA_VISIBLE_DEVICES:-all}"
echo "============================================================"

# ── Verify we are in the right directory ─────────────────────────────────────
if [ ! -f "inference.py" ]; then
    echo "ERROR: inference.py not found in $(pwd)"
    echo "       Make sure setup.bash is run from the project root directory."
    exit 1
fi
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found in $(pwd)"
    exit 1
fi
echo "  Found inference.py and requirements.txt in $(pwd)"

# ── Step 1: Create conda environment ─────────────────────────────────────────
echo ""
echo "[1/4] Creating conda environment: $ENV_NAME (Python $PYTHON_VER) ..."

# Remove previous env if exists (makes script idempotent)
conda remove --name "$ENV_NAME" --all -y 2>/dev/null || true

conda create -y -n "$ENV_NAME" python="$PYTHON_VER"

# Activate the environment inside this shell session
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "  Active env : $CONDA_DEFAULT_ENV"
echo "  Python     : $(python --version)"

# ── Step 2: Install PyTorch with CUDA support ─────────────────────────────────
echo ""
echo "[2/4] Installing PyTorch 2.4.1 (cu121 wheels, compatible with CUDA 12.6) ..."

pip install --upgrade pip --quiet

# cu121 wheels work on CUDA 12.6 (backward-compatible ABI)
pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

python -c "
import torch
print(f'  torch          : {torch.__version__}')
print(f'  CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU            : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"

# ── Step 3: Install all project requirements ──────────────────────────────────
echo ""
echo "[3/4] Installing requirements from requirements.txt ..."

pip install -r requirements.txt

# Sanity check key imports
python -c "
import cv2, numpy, pandas, PIL, transformers
print(f'  transformers   : {transformers.__version__}')
print(f'  opencv         : {cv2.__version__}')
print(f'  numpy          : {numpy.__version__}')
print(f'  pandas         : {pandas.__version__}')
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print(f'  Qwen2_5_VL     : OK')
"

# ── Step 4: Download model weights from HuggingFace ──────────────────────────
echo ""
echo "[4/4] Downloading $MODEL_HF_ID to $MODEL_DIR ..."
echo "  (first-time download is ~7 GB – expect 10-20 min)"

mkdir -p "$MODEL_DIR"

python - <<PYEOF
from huggingface_hub import snapshot_download
from pathlib import Path
import os

model_id  = "$MODEL_HF_ID"
local_dir = "$MODEL_DIR"

print(f"  Starting download: {model_id}")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    # Skip non-PyTorch formats to save space/time
    ignore_patterns=[
        "*.msgpack",
        "flax_model*",
        "tf_model*",
        "rust_model*",
        "*.ot",
    ],
)
print(f"  Download complete: {local_dir}")

# ── Integrity check ────────────────────────────────────────────────────────
p = Path(local_dir)
required_files = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
]
for req in required_files:
    fpath = p / req
    assert fpath.exists(), f"MISSING: {req}"
    print(f"  ✓ {req}  ({os.path.getsize(fpath)/1e3:.0f} KB)")

# List weight shards
shards = sorted(p.glob("model*.safetensors"))
total_gb = sum(f.stat().st_size for f in shards) / 1e9
print(f"  ✓ {len(shards)} weight shard(s)  ({total_gb:.2f} GB total)")
print("  Integrity check PASSED")
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Grader will now run:"
echo "   conda activate $ENV_NAME"
echo "   python inference.py --test_dir <absolute_path_to_test_dir>"
echo ""
echo " Output: ./submission.csv  (in this directory)"
echo "============================================================"
