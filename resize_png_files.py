#!/usr/bin/env python3
"""
Resize PNG newspaper scans to optimal DOTS OCR size
One-time preprocessing before batch OCR processing

Resizes images exceeding 11,289,600 pixels (11.3 MP) to ~10 MP
while preserving aspect ratio and using high-quality resampling.
"""

import argparse
from PIL import Image
from pathlib import Path
import logging

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    tqdm = lambda x, **kwargs: x

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DOTS OCR optimal size
MAX_PIXELS = 11_000_000  # Stay safely under 11.3MP limit


def resize_image(img_path, max_pixels=MAX_PIXELS, backup=False, dry_run=False):
    """
    Resize image to stay under pixel limit while preserving aspect ratio

    Args:
        img_path: Path to image file
        max_pixels: Maximum pixel count
        backup: Create .original backup before resizing
        dry_run: Don't actually resize, just report what would be done

    Returns:
        True if resized, False if already optimal
    """
    try:
        img = Image.open(img_path)
        width, height = img.size
        current_pixels = width * height

        if current_pixels <= max_pixels:
            logger.debug(f'{img_path.name}: Already optimal ({width}×{height} = {current_pixels:,} pixels)')
            return False

        # Calculate new dimensions
        scale = (max_pixels / current_pixels) ** 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        new_pixels = new_width * new_height

        logger.info(f'{img_path.name}: {width}×{height} ({current_pixels:,}) → {new_width}×{new_height} ({new_pixels:,})')

        if dry_run:
            return True

        # Backup original if requested
        if backup:
            backup_path = img_path.with_suffix(img_path.suffix + '.original')
            if not backup_path.exists():
                # Copy instead of rename to preserve original
                img.save(backup_path, quality=100)
                logger.debug(f'  Backup created: {backup_path.name}')

        # Resize with high-quality LANCZOS filter
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save with high quality
        if img_path.suffix.lower() == '.png':
            img_resized.save(img_path, optimize=True)
        else:
            img_resized.save(img_path, quality=95, optimize=True)

        return True

    except Exception as e:
        logger.error(f'Error processing {img_path}: {e}')
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Resize PNG newspaper scans to optimal DOTS OCR size'
    )

    parser.add_argument(
        'data_dir',
        type=str,
        nargs='?',
        default='./data',
        help='Directory containing images to resize (default: ./data)'
    )

    parser.add_argument(
        '--max-pixels',
        type=int,
        default=MAX_PIXELS,
        help=f'Maximum pixel count (default: {MAX_PIXELS:,})'
    )

    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create .original backups before resizing'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be resized without actually doing it'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='**/*.png',
        help='File pattern to match (default: **/*.png)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error(f'Data directory not found: {data_dir}')
        return

    # Find all matching files
    files = list(data_dir.glob(args.pattern))
    logger.info(f'Found {len(files)} files matching {args.pattern}')

    if not files:
        logger.warning(f'No files found matching {args.pattern} in {data_dir}')
        return

    if args.dry_run:
        logger.info('DRY RUN MODE - no files will be modified')

    # Resize with progress bar
    resized_count = 0
    total_before = 0
    total_after = 0

    for file_path in tqdm(files, desc='Processing files'):
        img = Image.open(file_path)
        total_before += img.size[0] * img.size[1]

        if resize_image(file_path, args.max_pixels, args.backup, args.dry_run):
            resized_count += 1
            if not args.dry_run:
                img_after = Image.open(file_path)
                total_after += img_after.size[0] * img_after.size[1]
        else:
            total_after += img.size[0] * img.size[1]

    # Summary
    logger.info('=' * 60)
    logger.info('SUMMARY')
    logger.info('=' * 60)
    logger.info(f'Total files: {len(files)}')
    logger.info(f'Resized: {resized_count}')
    logger.info(f'Already optimal: {len(files) - resized_count}')

    if not args.dry_run and resized_count > 0:
        reduction = (1 - total_after / total_before) * 100
        logger.info(f'Total pixel reduction: {reduction:.1f}%')

    if args.dry_run:
        logger.info('\nRun without --dry-run to actually resize files')

    logger.info('=' * 60)


if __name__ == '__main__':
    main()
