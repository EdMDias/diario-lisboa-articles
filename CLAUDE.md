# Diário de Lisboa Archive Scraper

## Project Overview
This project downloads the complete digital archive of Diário de Lisboa newspaper from Casa Comum, covering publications from 1921 to 1990.

## Key Commands

### Run the scraper for a specific date range
```bash
python diario_lisboa_scraper.py --start-date 1921-04-07 --end-date 1921-04-30
```

### Run the full archive download
```bash
python diario_lisboa_scraper.py
```

### Resume interrupted download
```bash
python diario_lisboa_scraper.py --resume
```

### Test with a single day
```bash
python diario_lisboa_scraper.py --test --date 1921-04-07
```

## Project Structure
```
diario-lisbon/
├── CLAUDE.md           # This file - quick reference
├── PLANNING.md         # Detailed planning and technical documentation
├── diario_lisboa_scraper.py  # Main scraper script
├── requirements.txt    # Python dependencies
├── progress.json       # Tracks download progress
├── errors.log         # Error logging
└── data/              # Downloaded newspapers
    ├── 1921/
    │   ├── 04/
    │   │   ├── 07/
    │   │   │   ├── page_001.jpg
    │   │   │   ├── page_002.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── ...
```

## Quick Facts
- **Total Years**: 70 (1921-1990)
- **Publication Schedule**: 6 days/week (Monday-Saturday, no Sundays)
- **Typical Pages per Edition**: 8 pages
- **Total Estimated Files**: ~150,000+ pages
- **Source**: Casa Comum - Fundação Mário Soares e Maria Barroso

## Dependencies
```bash
pip install -r requirements.txt
```

## Notes
- The scraper implements polite crawling with delays to avoid overloading the server
- Progress is saved automatically and can be resumed if interrupted
- Already downloaded files are skipped to avoid redundancy
- High-quality images (d2 version) are downloaded by default

## Vast.ai vLLM Setup - Critical Issues and Solutions

### Issue 1: vLLM Version for DOTS OCR Support
**Problem**: vLLM stable (0.11.2 from PyPI) doesn't support DOTS OCR - gets video processor errors
**Solution**: MUST use vLLM nightly build
```bash
pip3 install vllm --extra-index-url https://wheels.vllm.ai/nightly
```

### Issue 2: PyTorch Version Compatibility
**Problem**: Initial PyTorch 2.7.0 conflicts with vLLM 0.11.2 (requires 2.9.0)
**Solution**: Use PyTorch 2.9.0 + torchvision 0.24.0
```bash
pip3 install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

### Issue 3: Flash Attention Package Not Needed
**Problem**: Tried to install flash-attn wheel for torch 2.9 (doesn't exist, 404 error)
**Solution**: vLLM nightly has Flash Attention BUILT-IN - no separate package needed
- vLLM automatically uses FLASH_ATTN backend
- Provides 3-5x speedup without external flash-attn package

### Issue 4: Disk Space Management
**Problem**: Multiple PyTorch installs/uninstalls filled 32GB disk (86% usage)
**Solution**: Clear pip cache regularly on limited disk instances
```bash
pip3 cache purge  # Freed 12-13GB
```

### Issue 5: Model Name in Script vs Server
**Problem**: Script requested `prithivMLmods/Dots.OCR-Latest-BF16` but server serves `rednote-hilab/dots.ocr`
**Solution**: Use official model name in batch_ocr_vllm.py: `rednote-hilab/dots.ocr`

### Issue 6: Conda vs System Python
**Problem**: Packages installed globally but script run in conda env without openai
**Solution**: Either deactivate conda or install packages in conda env
```bash
conda deactivate  # Use system Python
# OR
pip install openai tqdm  # Install in conda env
```

### Working Configuration (Updated 2025-12-16)

See TOKEN_ANALYSIS.md for detailed token requirements analysis.

```bash
# On vast.ai RTX 3090 instance (32GB disk minimum)
# Python 3.12.3, CUDA 12.8

# Install dependencies
pip3 install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip3 install transformers accelerate qwen-vl-utils Pillow tqdm openai
pip3 install vllm --extra-index-url https://wheels.vllm.ai/nightly

# Download model
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='rednote-hilab/dots.ocr', local_dir='/workspace/diario-lisbon-ocr/models/DotsOCR')"
```

### vLLM Server Configuration Options

Based on token analysis (see TOKEN_ANALYSIS.md), choose configuration based on your priority:

#### Option 1: Maximum Speed (80-85% success rate)
```bash
vllm serve rednote-hilab/dots.ocr \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 8000 \
  --enable-chunked-prefill \
  --max-num-seqs 8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384 \
  --disable-log-requests
```
- **Max concurrency**: 22× (use 8×)
- **Throughput**: ~3.1 img/min
- **Use case**: Fast initial processing, reprocess failures later
- **Script setting**: `--max-tokens 12000`

#### Option 2: Balanced (99% success rate)
```bash
vllm serve rednote-hilab/dots.ocr \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 8000 \
  --enable-chunked-prefill \
  --max-num-seqs 8 \
  --max-model-len 20480 \
  --max-num-batched-tokens 20480 \
  --disable-log-requests
```
- **Max concurrency**: 18× (use 8×)
- **Throughput**: ~2.3 img/min
- **Use case**: Reliable processing with minimal failures
- **Script setting**: `--max-tokens 15000`

#### Option 3: Maximum Reliability (100% success rate) - RECOMMENDED
```bash
vllm serve rednote-hilab/dots.ocr \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 8000 \
  --enable-chunked-prefill \
  --max-num-seqs 8 \
  --max-model-len 24576 \
  --max-num-batched-tokens 24576 \
  --disable-log-requests
```
- **Max concurrency**: 15× (use 8×)
- **Throughput**: ~1.9 img/min
- **Use case**: Zero failures, best for unattended processing (DEFAULT)
- **Script setting**: `--max-tokens 18000` (default)

### Run Batch Processing
```bash
# Default (Option 3 - 100% reliability)
python3 batch_ocr_vllm.py /path/to/images

# Or explicitly specify max-tokens to match server configuration:
python3 batch_ocr_vllm.py /path/to/images --max-tokens 12000  # Option 1
python3 batch_ocr_vllm.py /path/to/images --max-tokens 15000  # Option 2
python3 batch_ocr_vllm.py /path/to/images --max-tokens 18000  # Option 3 (default)
```

### Token Requirements (From Analysis)
- **Average page**: 16,000 tokens total (12,877 input + 3,123 output)
- **Dense page (p90)**: 17,411 tokens total
- **Very dense (p95)**: 17,905 tokens total
- **Extreme (p99)**: 18,578 tokens total
- **Max observed**: 18,918 tokens total

### Performance
- Flash Attention: ENABLED (built into vLLM nightly)
- Chunked Prefill: ENABLED (critical for long inputs)
- GPU utilization: 85-90%
- 6.2× faster than unoptimized baseline

---

## Article Extraction Pipeline (NEW - 2026-01-10)

After OCR processing, extract structured, searchable articles from newspaper pages.

### Output Directory Structure (Updated 2026-01-12)

Articles are now saved to a **separate directory** from OCR outputs:
```
ocr_output/1974/04/24/
└── page_001_ocr.json          # OCR only

articles_output/1974/04/24/     # NEW: Separate directory
├── page_001_articles.json      # Extracted articles
├── page_002_articles.json
└── articles_database.json      # Combined database
```

### Quick Start

**Extract articles from a single page (default output to articles_output/):**
```bash
python article_extractor.py ocr_output/1974/04/24/page_001_ocr.json
# Output: articles_output/1974/04/24/page_001_articles.json
```

**Extract with custom output directory:**
```bash
python article_extractor.py ocr_output/1974/04/24/page_001_ocr.json \
  --articles-output-dir custom_articles
# Output: custom_articles/1974/04/24/page_001_articles.json
```

**Process entire newspaper issue:**
```bash
python batch_extract_articles.py ocr_output/1974/04/24/
# Output: articles_output/1974/04/24/ (156 articles from 26 pages)
#         articles_output/articles_database.json
```

**Visualize article extraction (NEW):**
```bash
# Auto-detect OCR path from image
python visualize_articles.py data/1974/04/24/page_001.png
# Output: visualized_page_001.png (bounding boxes colored by article)

# Specify OCR path explicitly
python visualize_articles.py data/1974/04/24/page_001.png \
  ocr_output/1974/04/24/page_001_ocr.json \
  my_visualization.png
```

**Create searchable database:**
```bash
python create_article_database.py articles_output/1974/04/24/articles_database.json
# Output: articles.db (SQLite with full-text search)
```

**Search articles:**
```bash
# Interactive mode
python search_articles.py

# One-off search
python search_articles.py --query "Mitterrand"

# View specific article
python search_articles.py --article "1974/04/24/page_001_article_006"
```

### What Gets Extracted

From OCR bounding boxes → Structured articles with:
- ✅ **Title** (extracted from Title/Section-header categories)
- ✅ **Full text** (all paragraphs in reading order)
- ✅ **Images & captions** (associated pictures)
- ✅ **Cross-page continuations** ("Continua na pág. X" markers)
- ✅ **Metadata** (date, page number, paragraph count)

### Performance Metrics (1974/04/24 test)

- **26 pages** → **156 articles** (6 articles per page average)
- **90.4%** of articles have titles
- **3.8%** have continuation markers (6 articles)
- **Processing speed:** 0.2 seconds per page
- **Extraction accuracy:** ~85% (based on manual inspection)

### Algorithm Overview

1. **Column detection:** Cluster boxes by X-coordinate (50px gap)
2. **Reading order:** Sort top-to-bottom within columns
3. **Article boundaries:** Detect via titles + vertical gaps (100px)
4. **Build articles:** Group related paragraphs, images, captions
5. **Track continuations:** Extract "Continua na pág. X" patterns

### Visualization (NEW - 2026-01-12)

Validate article extraction quality by visualizing bounding boxes colored by article:

**Features:**
- Each article gets a unique color (HSL golden ratio for maximum distinction)
- All OCR bounding boxes from same article shown in same color
- Legend shows article IDs and truncated titles
- Auto-detects OCR path from image path

**Use Cases:**
- Verify article segmentation (check if boxes are grouped correctly)
- Identify extraction errors (wrong article boundaries)
- Understand multi-column article flow
- Debug continuation markers

**Example Output:**
A page with 7 articles will show:
- Article 001 (red boxes): Masthead
- Article 002 (blue boxes): Main story spanning 2 columns
- Article 003 (green boxes): Sidebar article
- ...each with "Art 001", "Art 002", etc. labels

**Note:** Visualization re-runs article extraction internally, so you see current extraction logic (not saved JSON).

### Key Files

- `article_extractor.py` - Core extraction engine
- `batch_extract_articles.py` - Process multiple pages
- `create_article_database.py` - Build SQLite database
- `search_articles.py` - Search interface
- `visualize_articles.py` - **NEW:** Article-based visualization
- `ARTICLE_EXTRACTION_RESEARCH_PLAN.md` - Detailed research plan
- `ARTICLE_EXTRACTION_IMPLEMENTATION.md` - Implementation summary

### Database Schema

```sql
-- Articles table (with FTS5 full-text search)
CREATE TABLE articles (
    article_id TEXT PRIMARY KEY,
    title TEXT,
    text TEXT,
    date TEXT,  -- YYYY-MM-DD
    year INTEGER,
    page_number INTEGER,
    continuation_to TEXT,  -- "pág. X" if continues
    ...
);

-- Full-text search (FTS5)
CREATE VIRTUAL TABLE articles_fts USING fts5(title, text);
```

### Tuning Parameters

Adjust in `article_extractor.py`:
```python
min_column_gap = 50      # Horizontal gap between columns (pixels)
min_article_gap = 100    # Vertical gap between articles (pixels)
```

- **Smaller gaps** → More segmentation (risk: split single articles)
- **Larger gaps** → Less segmentation (risk: merge adjacent articles)

### Next Steps (Phase 2)

- ⬜ Implement continuation resolver (auto-link cross-page articles)
- ⬜ Scale to full archive (150K pages → 500K articles)
- ⬜ Improve boundary detection (ML-based or graph approach)
- ⬜ Build web interface for browsing/searching

### Documentation

See **ARTICLE_EXTRACTION_IMPLEMENTATION.md** for full implementation details, benchmarks, and future roadmap.