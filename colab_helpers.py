#!/usr/bin/env python3
"""
Colab Helpers - Utilities for Google Colab Integration
Adapted for Diário de Lisboa OCR processing
"""

import json
from pathlib import Path
from typing import List
import argparse


def estimate_colab_time(num_images: int, gpu_type: str = "T4"):
    """
    Estimate OCR processing time on Colab GPU.

    Args:
        num_images: Number of images to process
        gpu_type: GPU type (T4, V100, A100)
    """
    # Processing speed estimates (images per minute)
    speed_map = {
        'T4': 2.0,      # Colab free tier
        'V100': 3.5,    # Colab Pro
        'A100': 5.0,    # Colab Pro+
    }

    images_per_minute = speed_map.get(gpu_type, 2.0)
    total_minutes = num_images / images_per_minute
    total_hours = total_minutes / 60

    print(f"📊 Estimated Colab Processing Time:")
    print(f"   GPU: {gpu_type}")
    print(f"   Images: {num_images:,}")
    print(f"   Speed: ~{images_per_minute} images/minute")
    print(f"   ⏱️  Estimated time: {total_hours:.1f} hours ({total_minutes:.0f} minutes)")

    # Session limits
    free_limit_hours = 12
    pro_limit_hours = 24

    if total_hours > pro_limit_hours:
        print(f"   ⚠️  WARNING: Exceeds Colab Pro+ limit ({pro_limit_hours}h)")
        print(f"   Process in {int(total_hours / pro_limit_hours) + 1} batches")
    elif total_hours > free_limit_hours:
        print(f"   ⚠️  WARNING: Exceeds free Colab limit ({free_limit_hours}h)")
        print(f"   You'll need Colab Pro or process in batches")
    else:
        print(f"   ✅ Fits within Colab free tier session limit")

    # Batch suggestions
    if total_hours > 2:
        images_per_batch = int(images_per_minute * 60 * 10)  # 10 hours per batch
        num_batches = (num_images + images_per_batch - 1) // images_per_batch
        print(f"\n   💡 Suggested batches: {num_batches} batches of ~{images_per_batch} images")


def count_images(data_dir: Path) -> int:
    """Count JPG images in directory"""
    images = list(data_dir.glob('**/*.jpg'))
    return len(images)


def print_colab_cli_setup():
    """Print instructions for colab-cli terminal integration"""
    instructions = """
╔═══════════════════════════════════════════════════════════════╗
║              colab-cli - Terminal Integration                  ║
╚═══════════════════════════════════════════════════════════════╝

## Install colab-cli

pip install colab-cli
# or with uv:
uv pip install colab-cli

## Setup (One-time)

1. Get Google Drive API credentials:
   - Go to: https://console.developers.google.com/
   - Create new project or select existing
   - Enable "Google Drive API"
   - Create OAuth 2.0 credentials (Desktop app)
   - Download as client_secrets.json

2. Configure colab-cli:
   cd /path/to/credentials
   colab-cli set-config client_secrets.json
   colab-cli set-auth-user 0

## Usage

# Upload notebook to Colab and open in browser
colab-cli open-nb colab_ocr_notebook.ipynb

# After editing in Colab, pull changes back
colab-cli pull-nb colab_ocr_notebook.ipynb

# List your Colab notebooks
colab-cli list-nb

═══════════════════════════════════════════════════════════════
📖 Docs: https://github.com/Akshay090/colab-cli
"""
    print(instructions)


def print_colab_setup_instructions():
    """Print instructions for using the Colab notebook"""
    instructions = """
╔═══════════════════════════════════════════════════════════════╗
║     Google Colab OCR Processing - Setup Instructions          ║
╚═══════════════════════════════════════════════════════════════╝

## Quick Start with colab-cli (Recommended!)

If you have colab-cli installed:
   colab-cli open-nb colab_ocr_notebook.ipynb

Or run: python colab_helpers.py cli-setup

## Step 1: Upload Data to Google Drive

1. Create folder structure in Google Drive:

   MyDrive/
   └── diario-lisbon/
       ├── data/              ← Upload your newspaper scans here
       │   └── 1921/
       │       └── 04/
       │           └── 07/
       │               ├── page_001.jpg
       │               ├── page_002.jpg
       │               └── ...
       └── ocr_output/        ← Results will be saved here (auto-created)

2. Upload your newspaper scans to the data folder

## Step 2: Upload Notebook to Colab

**Option A: Direct Upload**
1. Go to: https://colab.research.google.com
2. File > Upload notebook
3. Select: colab_ocr_notebook.ipynb

**Option B: From Google Drive**
1. Upload colab_ocr_notebook.ipynb to Google Drive
2. Double-click to open in Colab

## Step 3: Configure GPU

1. In Colab: Runtime > Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (free) or **V100/A100** (Pro)
4. Click Save

## Step 4: Run Notebook

1. Update paths in Cell 2 (Mount Google Drive) to match your structure
2. Adjust BATCH_LIMIT in Cell 7 if needed (default: 100 images)
3. Run all cells: Runtime > Run all
4. Monitor progress in the output

## Step 5: Access Results

Results are automatically saved to:
   Google Drive: MyDrive/diario-lisbon/ocr_output/

Each image produces:
   - page_XXX_ocr.json  (structured data with layout)
   - page_XXX_ocr.txt   (plain text)
   - batch_results.json (summary of all processing)

## Tips for Large Batches

1. **Process in batches**: Set BATCH_LIMIT to process N images at a time
2. **Monitor session**: Colab free tier has 12-hour limit
3. **Save progress**: Results are saved every 10 images
4. **Continue processing**: Re-run notebook with different BATCH_LIMIT ranges

## Cost Comparison

- **Colab Free**: $0 (T4 GPU, 12h sessions)
- **Colab Pro**: $10/month (V100 GPU, 24h sessions, faster)
- **Colab Pro+**: $50/month (A100 GPU, 24h sessions, fastest)

For ~150,000 images:
- Free tier: Process in ~15 sessions over time
- Pro: Process in ~7 sessions
- Pro+: Process in ~5 sessions

═══════════════════════════════════════════════════════════════

📖 DOTS OCR Documentation: https://github.com/rednote-hilab/dots.ocr
"""
    print(instructions)


def create_batch_config(total_images: int, batch_size: int = 1000):
    """
    Create configuration for processing large datasets in batches.

    Args:
        total_images: Total number of images
        batch_size: Images per batch
    """
    num_batches = (total_images + batch_size - 1) // batch_size

    config = {
        "total_images": total_images,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "batches": []
    }

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_images)
        config["batches"].append({
            "batch_number": i + 1,
            "start_index": start_idx,
            "end_index": end_idx,
            "num_images": end_idx - start_idx,
            "completed": False
        })

    # Save configuration
    config_file = Path("batch_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Created batch configuration: {config_file}")
    print(f"\nBatch Plan:")
    print(f"  Total images: {total_images:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")
    print(f"\nIn Colab notebook, set:")
    print(f"  BATCH_LIMIT = {batch_size}")
    print(f"  And process batches 1-{num_batches} sequentially")

    return config


def main():
    parser = argparse.ArgumentParser(description="Colab OCR helpers")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # estimate command
    est = subparsers.add_parser('estimate', help='Estimate processing time')
    est.add_argument('num_images', type=int, help='Number of images to process')
    est.add_argument('--gpu', choices=['T4', 'V100', 'A100'], default='T4',
                     help='GPU type (default: T4 for free tier)')

    # count command
    count = subparsers.add_parser('count', help='Count images in directory')
    count.add_argument('data_dir', type=Path, help='Data directory path')

    # setup command
    subparsers.add_parser('setup', help='Print setup instructions')

    # cli-setup command
    subparsers.add_parser('cli-setup', help='Print colab-cli setup instructions')

    # batch command
    batch = subparsers.add_parser('batch', help='Create batch processing configuration')
    batch.add_argument('total_images', type=int, help='Total number of images')
    batch.add_argument('--size', type=int, default=1000, help='Images per batch')

    args = parser.parse_args()

    if args.command == 'estimate':
        estimate_colab_time(args.num_images, args.gpu)
    elif args.command == 'count':
        num_images = count_images(args.data_dir)
        print(f"Found {num_images:,} images in {args.data_dir}")
        if num_images > 0:
            print(f"\nRun: python colab_helpers.py estimate {num_images}")
    elif args.command == 'setup':
        print_colab_setup_instructions()
    elif args.command == 'cli-setup':
        print_colab_cli_setup()
    elif args.command == 'batch':
        create_batch_config(args.total_images, args.size)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
