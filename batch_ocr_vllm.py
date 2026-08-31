#!/usr/bin/env python3
"""
Concurrent Batch OCR Processor for Diário de Lisboa
Uses vLLM with async/concurrent processing for maximum GPU utilization
"""

import os
import json
import argparse
import logging
import asyncio
import base64
from pathlib import Path
from typing import List, Dict, Set, Optional
from datetime import datetime
from dataclasses import dataclass
import time

try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
except ImportError:
    async_tqdm = lambda x, **kwargs: x
    tqdm = lambda x, **kwargs: x

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_ocr.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ImageProcessingMetrics:
    """Metrics for a single image processing operation"""
    image_path: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"  # pending, success, failed
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def total_duration(self) -> Optional[float]:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None


def calculate_percentiles(durations: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles"""
    if not durations:
        return {}

    sorted_durations = sorted(durations)
    n = len(sorted_durations)

    return {
        'p50': sorted_durations[int(n * 0.50)],
        'p90': sorted_durations[int(n * 0.90)],
        'p95': sorted_durations[int(n * 0.95)],
        'p99': sorted_durations[int(n * 0.99)],
        'min': sorted_durations[0],
        'max': sorted_durations[-1],
        'mean': sum(sorted_durations) / n
    }


def get_ocr_prompt() -> str:
    """Get Portuguese OCR prompt for structured output with bboxes"""
    return """Please output the layout information from this newspaper page image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title', 'Crossword-puzzle'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - Crossword-puzzle: Extract the crossword puzzle grid structure, clues (horizontal and vertical), and any filled-in answers. Format as structured text or JSON.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.
    - This is a Portuguese newspaper, preserve the original Portuguese text.
    - If you detect a crossword puzzle, categorize it as 'Crossword-puzzle' and extract both the grid and clues.

5. Final Output: The entire output must be a single JSON object."""


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 encoding"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def find_all_images(directory: Path) -> List[Path]:
    """
    Recursively find all image files in directory

    Args:
        directory: Root directory to search

    Returns:
        List of image file paths
    """
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = []

    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))

    return sorted(images)


def get_output_path(image_path: Path, output_dir: Path, directory: Optional[Path] = None) -> Path:
    """
    Get output JSON path for an image

    Args:
        image_path: Input image path
        output_dir: Base output directory
        directory: Root input directory (to preserve subdirectory structure)

    Returns:
        Path where JSON output should be saved
    """
    # Preserve directory structure relative to input
    if directory:
        relative_path = image_path.relative_to(directory)
        # Change page_001.jpg -> page_001_ocr.json while preserving all subdirs
        output_file = f"{relative_path.stem}_ocr.json"
        return output_dir / relative_path.parent / output_file

    return output_dir / f"{image_path.stem}_ocr.json"


def filter_unprocessed(image_paths: List[Path], output_dir: Path, directory: Optional[Path]) -> List[Path]:
    """
    Filter out images that have already been processed

    Args:
        image_paths: List of all image paths
        output_dir: Output directory to check for existing results

    Returns:
        List of unprocessed image paths
    """
    unprocessed = []

    for img_path in image_paths:
        output_path = get_output_path(img_path, output_dir, directory)
        if not output_path.exists():
            unprocessed.append(img_path)

    return unprocessed


class ConcurrentOCRProcessor:
    """Async OCR processor with concurrent request handling"""

    def __init__(self,
                 model_path: str = "rednote-hilab/dots.ocr",
                 vllm_url: str = "http://localhost:8000/v1",
                 concurrency: int = 8,
                 max_tokens: int = 18000):
        """
        Initialize concurrent OCR processor

        Args:
            model_path: Model identifier for vLLM
            vllm_url: vLLM server URL
            concurrency: Maximum concurrent requests
            max_tokens: Maximum tokens for OCR output (match to vLLM server's max-model-len)
        """
        self.model_path = model_path
        self.vllm_url = vllm_url
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.prompt = get_ocr_prompt()
        self.client = None
        self.semaphore = asyncio.Semaphore(concurrency)

        # Statistics
        self.processed = 0
        self.failed = 0
        self.start_time = None

        # Detailed metrics tracking
        self.metrics: List[ImageProcessingMetrics] = []
        self.active_requests: int = 0
        self.active_requests_lock = asyncio.Lock()
        self.peak_concurrent_requests: int = 0

        # Error categorization counters
        self.error_counts = {
            'timeout': 0,
            'api_error': 0,
            'file_error': 0,
            'unknown': 0
        }

        logger.info(f"Initialized processor with concurrency={concurrency}, max_tokens={max_tokens}")

    async def initialize(self):
        """Initialize async vLLM client"""
        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                base_url=self.vllm_url,
                api_key="dummy",  # vLLM doesn't need real API key
                timeout=600.0,  # 10 minute timeout for long OCR tasks
                max_retries=0  # No retries - fail once and move on
            )

            logger.info(f"AsyncOpenAI client initialized at {self.vllm_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vLLM client: {e}")
            return False

    async def process_single_image(self, image_path: Path, output_dir: Path, input_dir: Path) -> Dict:
        """
        Process a single image with OCR

        Args:
            image_path: Path to image file
            output_dir: Directory to save results
            input_dir: Input root directory (for preserving subdirs)

        Returns:
            Result dictionary with status and metadata
        """
        # Create metrics object (start_time will be set after semaphore)
        metrics = ImageProcessingMetrics(
            image_path=str(image_path),
            start_time=0.0  # Placeholder, set after semaphore
        )

        async with self.semaphore:
            # Track active requests
            async with self.active_requests_lock:
                self.active_requests += 1
                self.peak_concurrent_requests = max(
                    self.peak_concurrent_requests,
                    self.active_requests
                )
                current_active = self.active_requests

            # Set start time NOW - after semaphore acquired, actual processing start
            metrics.start_time = time.time()

            # Log start with active requests
            logger.info(
                f"[START] {image_path.name} | Active: {current_active}/{self.concurrency}"
            )

            try:
                # Prepare request
                image_b64 = image_to_base64(str(image_path))

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": self.prompt
                            }
                        ]
                    }
                ]

                # Make async request to vLLM
                response = await self.client.chat.completions.create(
                    model=self.model_path,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.0
                )

                # Parse result
                ocr_text = response.choices[0].message.content

                # Validate and repair JSON if needed
                try:
                    json.loads(ocr_text)  # Try parsing as-is
                    repaired = False
                except json.JSONDecodeError:
                    # Try to repair the JSON
                    try:
                        from json_repair import repair_json
                        ocr_text = repair_json(ocr_text)
                        json.loads(ocr_text)  # Verify it's now valid
                        repaired = True
                        logger.warning(f"[REPAIRED JSON] {image_path.name}")
                    except Exception as e:
                        # Can't repair - treat as failure
                        metrics.status = "failed"
                        metrics.error_type = "invalid_json"
                        metrics.error_message = f"Invalid JSON, repair failed: {str(e)}"
                        metrics.end_time = time.time()
                        self.error_counts['invalid_json'] += 1
                        self.failed += 1

                        logger.error(
                            f"[INVALID JSON] {image_path.name} | Error: {str(e)} | "
                            f"Response preview: {ocr_text[:200]}..."
                        )

                        raise ValueError(f"Invalid JSON response: {str(e)}")

                result = {
                    "image_path": str(image_path),
                    "ocr_result": ocr_text,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_path,
                    "status": "success",
                    "json_repaired": repaired
                }

                # Save result
                output_path = get_output_path(image_path, output_dir, input_dir)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                # Update metrics
                metrics.status = "success"
                metrics.end_time = time.time()

                self.processed += 1

                # Log completion with timing
                logger.info(
                    f"[DONE] {image_path.name} | "
                    f"Time: {metrics.total_duration:.2f}s | "
                    f"Active: {self.active_requests}/{self.concurrency}"
                )

                return result

            except asyncio.TimeoutError as e:
                # Timeout error
                metrics.status = "failed"
                metrics.error_type = "timeout"
                metrics.error_message = str(e)
                metrics.end_time = time.time()
                self.error_counts['timeout'] += 1
                self.failed += 1

                logger.error(
                    f"[TIMEOUT] {image_path.name} | "
                    f"After: {metrics.total_duration:.2f}s"
                )

                return {
                    "image_path": str(image_path),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed"
                }

            except Exception as e:
                # Other errors - categorize them
                metrics.status = "failed"
                metrics.end_time = time.time()

                # Categorize error type
                error_str = str(e).lower()
                if "api" in error_str or "connection" in error_str or "openai" in error_str:
                    metrics.error_type = "api_error"
                    self.error_counts['api_error'] += 1
                elif "filenotfound" in str(type(e).__name__).lower():
                    metrics.error_type = "file_error"
                    self.error_counts['file_error'] += 1
                else:
                    metrics.error_type = "unknown"
                    self.error_counts['unknown'] += 1

                metrics.error_message = str(e)
                self.failed += 1

                logger.error(
                    f"[FAILED:{metrics.error_type.upper()}] {image_path.name} | "
                    f"Time: {metrics.total_duration:.2f}s | "
                    f"Error: {str(e)[:80]}"
                )

                return {
                    "image_path": str(image_path),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed"
                }

            finally:
                # Decrement active requests
                async with self.active_requests_lock:
                    self.active_requests -= 1

                # Store metrics
                self.metrics.append(metrics)

    async def process_batch(self, image_paths: List[Path], output_dir: Path, input_dir: Path) -> List[Dict]:
        """
        Process batch of images concurrently

        Args:
            image_paths: List of image paths to process
            output_dir: Directory to save results
            input_dir: Input root directory (for preserving subdirs)

        Returns:
            List of result dictionaries
        """
        self.start_time = time.time()

        logger.info(f"Processing {len(image_paths)} images with concurrency={self.concurrency}")

        # Create tasks for all images
        tasks = [
            self.process_single_image(img_path, output_dir, input_dir)
            for img_path in image_paths
        ]

        # Process with progress bar
        results = []
        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

                # Update progress stats
                elapsed = time.time() - self.start_time
                rate = self.processed / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'rate': f'{rate:.2f} img/s',
                    'failed': self.failed
                })

        return results

    def print_summary(self, total_images: int, elapsed_time: float):
        """Print comprehensive processing summary with detailed statistics"""

        # Calculate latency percentiles
        successful_metrics = [m for m in self.metrics if m.status == "success"]
        durations = [m.total_duration for m in successful_metrics if m.total_duration]

        percentiles = {}
        if durations:
            percentiles = calculate_percentiles(durations)

        logger.info("=" * 80)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 80)

        # Overall Statistics
        logger.info("Overall Statistics:")
        logger.info(f"  Total images: {total_images}")
        logger.info(f"  Successfully processed: {self.processed}")
        logger.info(f"  Failed: {self.failed}")
        if total_images > 0:
            logger.info(f"  Success rate: {(self.processed/total_images*100):.1f}%")
        logger.info(f"  Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}min)")

        if elapsed_time > 0:
            rate = total_images / elapsed_time
            logger.info(f"  Average rate: {rate:.2f} img/s ({rate*60:.1f} img/min)")

        # Concurrency Statistics
        logger.info("")
        logger.info("Concurrency Statistics:")
        logger.info(f"  Configured concurrency: {self.concurrency}")
        logger.info(f"  Configured max_tokens: {self.max_tokens}")
        logger.info(f"  Peak concurrent requests: {self.peak_concurrent_requests}")
        if self.concurrency > 0:
            utilization = (self.peak_concurrent_requests / self.concurrency * 100)
            logger.info(f"  Peak utilization: {utilization:.1f}%")

        # Latency Distribution
        if percentiles:
            logger.info("")
            logger.info("Latency Distribution (seconds):")
            logger.info(f"  p50 (median): {percentiles.get('p50', 0):.2f}s")
            logger.info(f"  p90: {percentiles.get('p90', 0):.2f}s")
            logger.info(f"  p95: {percentiles.get('p95', 0):.2f}s")
            logger.info(f"  p99: {percentiles.get('p99', 0):.2f}s")
            logger.info(f"  Min: {min(durations):.2f}s")
            logger.info(f"  Max: {max(durations):.2f}s")

        # Error Breakdown
        if self.failed > 0:
            logger.info("")
            logger.info("Error Breakdown:")
            for error_type, count in self.error_counts.items():
                if count > 0:
                    pct = (count/self.failed*100) if self.failed > 0 else 0
                    logger.info(f"  {error_type}: {count} ({pct:.1f}%)")

        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description="Concurrent batch OCR processing for newspaper scans using vLLM"
    )

    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing images to process (recursively)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ocr_output',
        help='Directory to save OCR results (default: ./ocr_output)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='rednote-hilab/dots.ocr',
        help='Model path/ID for vLLM (default: rednote-hilab/dots.ocr)'
    )

    parser.add_argument(
        '--vllm-url',
        type=str,
        default='http://localhost:8000/v1',
        help='vLLM server URL (default: http://localhost:8000/v1)'
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        default=8,
        help='Maximum concurrent requests (default: 8, tune based on GPU memory)'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=18000,
        help='Max tokens for output (default: 18000 for 100%% reliability). Use 12000 for 16K server, 15000 for 20K server, 18000 for 24K server. See TOKEN_ANALYSIS.md'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of images to process (for testing)'
    )

    args = parser.parse_args()

    # Validate paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    logger.info(f"Searching for images in {input_dir}...")
    all_images = find_all_images(input_dir)
    logger.info(f"Found {len(all_images)} total images")

    if not all_images:
        logger.error("No images found to process")
        return

    # Filter out already processed images
    logger.info("Checking for already processed images...")
    unprocessed_images = filter_unprocessed(all_images, output_dir, input_dir)
    already_processed = len(all_images) - len(unprocessed_images)

    logger.info(f"Already processed: {already_processed}")
    logger.info(f"Remaining to process: {len(unprocessed_images)}")

    if not unprocessed_images:
        logger.info("All images have already been processed!")
        return

    # Apply limit if specified
    if args.limit:
        unprocessed_images = unprocessed_images[:args.limit]
        logger.info(f"Limited to {args.limit} images for this run")

    # Initialize processor
    processor = ConcurrentOCRProcessor(
        model_path=args.model_path,
        vllm_url=args.vllm_url,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens
    )

    if not await processor.initialize():
        logger.error("Failed to initialize processor")
        return

    # Process images
    logger.info(f"Starting concurrent OCR processing...")
    logger.info(f"Concurrency level: {args.concurrency}")
    logger.info(f"Output directory: {output_dir}")

    start_time = time.time()
    results = await processor.process_batch(unprocessed_images, output_dir, input_dir)
    elapsed_time = time.time() - start_time

    # Print summary
    processor.print_summary(len(unprocessed_images), elapsed_time)

    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
