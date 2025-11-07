#!/bin/bash
# Setup script for cloud deployment (RunPod, Vast.ai, Lambda Labs, etc.)
# Run this script after spinning up a GPU instance

set -e

echo "=========================================="
echo "DOTS OCR Cloud Setup"
echo "=========================================="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi

# Update system
echo ""
echo "Updating system packages..."
apt-get update -qq
apt-get install -y git wget curl

# Install Python if needed
if ! command -v python3 &> /dev/null; then
    echo "Installing Python..."
    apt-get install -y python3 python3-pip python3-venv
fi

# Create working directory
WORK_DIR="/workspace/diario-lisbon-ocr"
mkdir -p $WORK_DIR
cd $WORK_DIR

echo ""
echo "Working directory: $WORK_DIR"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
echo ""
echo "Installing PyTorch..."
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install DOTS OCR
echo ""
echo "Installing DOTS OCR..."
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
pip install -e .

# Install additional dependencies
echo ""
echo "Installing additional dependencies..."
pip install vllm>=0.11.0
pip install tqdm psutil python-dotenv

# Download model weights
echo ""
echo "Downloading model weights..."
cd $WORK_DIR
python3 << 'PYTHON_SCRIPT'
from huggingface_hub import snapshot_download
import os

model_id = "rednote-hilab/dots.ocr"
cache_dir = "./models"

print(f"Downloading {model_id}...")
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    local_dir=os.path.join(cache_dir, "DotsOCR"),
    local_dir_use_symlinks=False
)
print("Model download complete!")
PYTHON_SCRIPT

# Create data directory
mkdir -p $WORK_DIR/data
mkdir -p $WORK_DIR/ocr_output

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your newspaper scans to: $WORK_DIR/data"
echo "2. Start vLLM server:"
echo "   vllm serve $WORK_DIR/models/DotsOCR --trust-remote-code --async-scheduling --gpu-memory-utilization 0.95"
echo ""
echo "3. In another terminal, run OCR processing:"
echo "   python ocr_processor.py --data-dir ./data --output-dir ./ocr_output"
echo ""
echo "Or use Transformers directly (no server needed):"
echo "   python ocr_processor.py --data-dir ./data --output-dir ./ocr_output --use-transformers --model-path ./models/DotsOCR"
echo ""
