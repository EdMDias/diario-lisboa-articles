# OCR Processing with DOTS OCR

## Three Ways to Process

### Option 1: Google Colab (Recommended - Free GPU!)

**Best for**: Free processing with GPU, no local setup

1. Upload your data to Google Drive
2. Open `colab_ocr_notebook.ipynb` in Colab
3. Enable GPU (Runtime > Change runtime > GPU)
4. Run all cells

**Cost**: FREE (12h sessions) or $10/month Colab Pro

**With colab-cli (terminal integration)**:
```bash
# Setup colab-cli (one-time)
python colab_helpers.py cli-setup

# Open notebook in Colab from terminal
colab-cli open-nb colab_ocr_notebook.ipynb
```

**Helper commands**:
```bash
# Estimate time
python colab_helpers.py estimate 1000 --gpu T4

# Get full setup instructions
python colab_helpers.py setup
```

### Option 2: Local Processing

**Best for**: You have NVIDIA GPU with 12GB+ VRAM

```bash
# Check system
python check_system.py

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-ocr.txt
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr && pip install -e . && cd ..

# Process
python ocr_processor.py --data-dir ./data --output-dir ./ocr_output
```

### Option 3: Cloud GPU (RunPod, Vast.ai)

**Best for**: Large batches, faster processing

```bash
# On cloud instance
bash setup_cloud.sh
python ocr_processor.py --data-dir ./data --output-dir ./ocr_output
```

**Cost**: $33-125 for full archive (~150K pages)

### Option 4: Concurrent Batch Processing (RECOMMENDED for Production)

**Best for**: Maximum GPU utilization, processing large archives (150K+ images)

**Features**:
- 🚀 **3-5x faster** than sequential processing
- ⚡ Async/concurrent requests to vLLM with automatic batching
- 🔄 Auto-resume: skips already processed images
- 📊 Real-time progress tracking
- 💾 90% GPU utilization on RTX 3090

**Setup**:

```bash
# 1. Start vLLM server (90% GPU memory)
vllm serve prithivMLmods/Dots.OCR-Latest-BF16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code \
  --port 8000

# 2. Run batch processor in another terminal
python batch_ocr_vllm.py /path/to/images \
  --output-dir ./ocr_output \
  --concurrency 8
```

**Usage**:

```bash
# Basic usage (default: 8 concurrent requests)
python batch_ocr_vllm.py ./data/1921

# Tune concurrency based on GPU memory
python batch_ocr_vllm.py ./data --concurrency 6  # Conservative
python batch_ocr_vllm.py ./data --concurrency 12 # Aggressive (24GB+ VRAM)

# Test with limited images
python batch_ocr_vllm.py ./data --limit 20

# Custom vLLM server
python batch_ocr_vllm.py ./data --vllm-url http://localhost:8000/v1
```

**Tuning Concurrency**:
- **RTX 3090 (24GB)**: Start with 8, test up to 10-12
- **RTX 4090 (24GB)**: Try 10-14
- **A100 (40GB)**: Try 16-20
- **A100 (80GB)**: Try 24-32

Monitor GPU memory with `nvidia-smi` and adjust if you see OOM errors.

**Expected Performance**:
- Sequential: 3-5 images/minute
- Concurrent (8x): **15-25 images/minute**
- Full archive (150K images): ~100-170 hours (~4-7 days)

## Output Files

### Sequential Processing (Options 1-3)
Each page produces:
- `page_001_ocr.json` - Structured data with layout
- `page_001_ocr.txt` - Plain text
- `batch_results.json` - Summary

### Concurrent Batch Processing (Option 4)
Each page produces:
- `page_001_ocr.json` - Structured data with layout and bboxes

## More Info

- Model: https://github.com/rednote-hilab/dots.ocr
- Colab setup: `python colab_helpers.py setup`
