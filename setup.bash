#!/usr/bin/env bash
# =============================================================================
# setup.bash  –  GNR638 Project
#
# GRADER FLOW:
#   [Grader extracts zip → gets a folder containing only this setup.bash]
#   cd ./your_directory          ← cd into that folder
#   bash setup.bash              ← THIS FILE runs here (internet available)
#   conda activate gnr_project_env
#   python inference.py --test_dir <absolute_path_to_test_dir>
#   python grading_script.py --submission_file submission.csv
#   conda remove --name gnr_project_env --all -y
#
# What this script does (internet IS available):
#   1. Clone the GitHub repo into the current directory (so inference.py lands here)
#   2. Create conda env gnr_project_env (Python 3.11)
#   3. Install PyTorch 2.4.1 with CUDA 12.1 wheels (forward-compatible with 12.6)
#   4. Install all requirements from requirements.txt
#   5. Download Qwen2.5-VL-3B-Instruct weights (~7 GB, no HF token needed)
# =============================================================================

set -e
set -o pipefail

# ── CONFIG – only line you need to change ────────────────────────────────────
REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"   # ← CHANGE THIS
# ─────────────────────────────────────────────────────────────────────────────

ENV_NAME="gnr_project_env"
PYTHON_VER="3.11"
MODEL_HF_ID="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_DIR="$HOME/models/qwen25vl-3b"

echo "============================================================"
echo " GNR638 Project Setup"
echo " Working dir : $(pwd)"
echo " Repo URL    : $REPO_URL"
echo " Conda env   : $ENV_NAME  (Python $PYTHON_VER)"
echo " Model       : $MODEL_HF_ID  →  $MODEL_DIR"
echo "============================================================"

# ── Step 1: Clone repo into current directory ─────────────────────────────────
# We clone into a temp folder then move contents up so that inference.py
# ends up in the SAME directory as this setup.bash (i.e. ./your_directory).
# The grader runs:  python inference.py  (no subdirectory), so this is required.
echo ""
echo "[1/5] Cloning repository ..."

CLONE_TMP="_repo_tmp"
rm -rf "$CLONE_TMP"
git clone "$REPO_URL" "$CLONE_TMP"

# Move everything from the cloned repo into the current directory
# (excluding the .git folder — keep our own)
cp -r "$CLONE_TMP"/. .
rm -rf "$CLONE_TMP"

echo "  Files now in $(pwd):"
ls -1

# Verify inference.py landed here
if [ ! -f "inference.py" ]; then
    echo "ERROR: inference.py not found after clone. Check your repo structure."
    exit 1
fi
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found after clone. Check your repo structure."
    exit 1
fi
echo "  inference.py and requirements.txt found ✓"

# ── Step 2: Create conda environment ─────────────────────────────────────────
echo ""
echo "[2/5] Creating conda environment: $ENV_NAME (Python $PYTHON_VER) ..."

conda remove --name "$ENV_NAME" --all -y 2>/dev/null || true
conda create -y -n "$ENV_NAME" python="$PYTHON_VER"

# Activate inside this shell
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "  Active env : $CONDA_DEFAULT_ENV"
echo "  Python     : $(python --version)"

# ── Step 3: Install PyTorch with CUDA support ─────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch 2.4.1 (cu121, forward-compatible with CUDA 12.6) ..."

pip install --upgrade pip --quiet

# cu121 wheels are ABI-compatible with CUDA 12.6 on the grader machine
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

# ── Step 4: Install all project requirements ──────────────────────────────────
echo ""
echo "[4/5] Installing requirements from requirements.txt ..."

pip install -r requirements.txt

python -c "
import cv2, numpy, pandas, PIL, transformers
print(f'  transformers   : {transformers.__version__}')
print(f'  opencv         : {cv2.__version__}')
print(f'  numpy          : {numpy.__version__}')
print(f'  pandas         : {pandas.__version__}')
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print(f'  Qwen2_5_VL     : import OK')
"

# ── Step 5: Download model weights from HuggingFace ──────────────────────────
echo ""
echo "[5/5] Downloading $MODEL_HF_ID ..."
echo "  Target : $MODEL_DIR"
echo "  Size   : ~7 GB  (expect 10-20 min first time)"

mkdir -p "$MODEL_DIR"

python - <<PYEOF
from huggingface_hub import snapshot_download
from pathlib import Path
import os

model_id  = "$MODEL_HF_ID"
local_dir = "$MODEL_DIR"

print(f"  Downloading {model_id} ...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=[
        "*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.ot",
    ],
)
print(f"  Download complete → {local_dir}")

# Integrity check
p = Path(local_dir)
for req in ["config.json", "tokenizer_config.json", "tokenizer.json"]:
    assert (p / req).exists(), f"MISSING: {req}"
    print(f"  ✓ {req}")
shards    = sorted(p.glob("model*.safetensors"))
total_gb  = sum(f.stat().st_size for f in shards) / 1e9
print(f"  ✓ {len(shards)} weight shard(s)  ({total_gb:.2f} GB)")
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
echo "   → writes submission.csv in $(pwd)"
echo "============================================================"
