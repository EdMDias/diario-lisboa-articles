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

## Output Files

Each page produces:
- `page_001_ocr.json` - Structured data with layout
- `page_001_ocr.txt` - Plain text
- `batch_results.json` - Summary

## More Info

- Model: https://github.com/rednote-hilab/dots.ocr
- Colab setup: `python colab_helpers.py setup`
