#!/bin/bash
# Quick setup script for vLLM concurrent batch processing on vast.ai
# Run this on your vast.ai instance

set -e

echo "========================================"
echo "Setting up vLLM for batch OCR"
echo "========================================"

# Go to workspace
cd /workspace/diario-lisbon-ocr || cd /workspace && mkdir -p diario-lisbon-ocr && cd diario-lisbon-ocr

# Activate venv (should already exist from previous setup)
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found, creating new one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
fi

# Check if openai is installed (required for batch_ocr_vllm.py)
if ! python3 -c "import openai" 2>/dev/null; then
    echo "📦 Installing openai package..."
    pip install openai>=1.0.0
else
    echo "✓ openai package already installed"
fi

# Check if vllm is installed
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "📦 Installing vllm..."
    pip install vllm>=0.11.0
else
    echo "✓ vllm already installed"
fi

# Check if tqdm is installed
if ! python3 -c "import tqdm" 2>/dev/null; then
    echo "📦 Installing tqdm..."
    pip install tqdm
else
    echo "✓ tqdm already installed"
fi

# Check model
MODEL_DIR="models/DotsOCR_Patched"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model not found at $MODEL_DIR"
    echo "Run the full setup from VASTAI_SETUP_REPORT.md first"
    exit 1
else
    echo "✓ Model found: $MODEL_DIR"
fi

# Create output directory
mkdir -p ocr_output
echo "✓ Output directory ready"

echo ""
echo "========================================"
echo "✅ Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Upload batch_ocr_vllm.py:"
echo "   scp -P 29576 batch_ocr_vllm.py root@79.112.58.103:/workspace/diario-lisbon-ocr/"
echo "   # Or using SSH alias:"
echo "   scp batch_ocr_vllm.py vastai-ocr:/workspace/diario-lisbon-ocr/"
echo ""
echo "2. Start vLLM server (in tmux/screen):"
echo "   tmux new -s vllm"
echo "   cd /workspace/diario-lisbon-ocr"
echo "   source venv/bin/activate"
echo "   vllm serve models/DotsOCR_Patched \\"
echo "     --gpu-memory-utilization 0.90 \\"
echo "     --trust-remote-code \\"
echo "     --port 8000"
echo ""
echo "3. In another terminal, run batch processor:"
echo "   ssh vastai-ocr"
echo "   cd /workspace/diario-lisbon-ocr"
echo "   source venv/bin/activate"
echo "   python3 batch_ocr_vllm.py data/ --concurrency 8"
echo ""
