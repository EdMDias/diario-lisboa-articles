#!/usr/bin/env python3
"""
Diário de Lisboa OCR Processor
Uses DOTS OCR model to extract text from scanned newspaper pages
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    tqdm = lambda x, **kwargs: x

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """Process newspaper scans with DOTS OCR model"""

    def __init__(self,
                 use_vllm: bool = True,
                 model_path: str = "rednote-hilab/dots.ocr",
                 vllm_url: str = "http://localhost:8000/v1",
                 gpu_memory_utilization: float = 0.95):
        """
        Initialize OCR processor

        Args:
            use_vllm: If True, use vLLM server (faster), else use Transformers
            model_path: Path to model or HuggingFace model ID
            vllm_url: URL for vLLM server
            gpu_memory_utilization: GPU memory to use (for vLLM)
        """
        self.use_vllm = use_vllm
        self.model_path = model_path
        self.vllm_url = vllm_url
        self.gpu_memory_utilization = gpu_memory_utilization
        self.model = None
        self.processor = None

        logger.info(f"Initializing OCR processor with {'vLLM' if use_vllm else 'Transformers'}")

    def initialize_vllm(self):
        """Initialize vLLM client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.vllm_url,
                api_key="dummy"  # vLLM doesn't need real API key
            )
            logger.info(f"vLLM client initialized at {self.vllm_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vLLM client: {e}")
            return False

    def initialize_transformers(self):
        """Initialize Transformers model"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info(f"Loading model from {self.model_path}")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_default_prompt(self, mode: str = "full") -> str:
        """
        Get default prompt for OCR

        Args:
            mode: 'full' for complete parsing, 'text_only' for text extraction only
        """
        if mode == "text_only":
            return """Please extract all the text from this newspaper page image.

1. Extract text in reading order (top to bottom, left to right).
2. Preserve the original language (Portuguese).
3. Format the output as clean text, preserving paragraph structure.
4. Skip headers, footers, and page numbers.
5. For multi-column layouts, read left column completely, then right column.

Output only the extracted text in reading order."""

        else:  # full mode
            return """Please output the layout information from this newspaper page image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.
    - This is a Portuguese newspaper, preserve the original Portuguese text.

5. Final Output: The entire output must be a single JSON object."""

    def process_image_vllm(self, image_path: str, prompt: Optional[str] = None) -> Dict:
        """
        Process single image using vLLM

        Args:
            image_path: Path to image file
            prompt: Custom prompt (optional)

        Returns:
            Dictionary with OCR results
        """
        if not hasattr(self, 'client'):
            if not self.initialize_vllm():
                raise RuntimeError("vLLM client not initialized")

        if prompt is None:
            prompt = self.get_default_prompt()

        try:
            # Prepare the request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{self._image_to_base64(image_path)}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                max_tokens=24000,
                temperature=0.0
            )

            result = {
                "image_path": image_path,
                "text": response.choices[0].message.content,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_path,
                "method": "vllm"
            }

            return result

        except Exception as e:
            logger.error(f"Error processing {image_path} with vLLM: {e}")
            return {
                "image_path": image_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def process_image_transformers(self, image_path: str, prompt: Optional[str] = None) -> Dict:
        """
        Process single image using Transformers

        Args:
            image_path: Path to image file
            prompt: Custom prompt (optional)

        Returns:
            Dictionary with OCR results
        """
        if self.model is None or self.processor is None:
            if not self.initialize_transformers():
                raise RuntimeError("Model not initialized")

        if prompt is None:
            prompt = self.get_default_prompt()

        try:
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda")

            # Inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            result = {
                "image_path": image_path,
                "text": output_text[0],
                "timestamp": datetime.now().isoformat(),
                "model": self.model_path,
                "method": "transformers"
            }

            return result

        except Exception as e:
            logger.error(f"Error processing {image_path} with Transformers: {e}")
            return {
                "image_path": image_path,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def process_image(self, image_path: str, prompt: Optional[str] = None) -> Dict:
        """Process image using selected method"""
        if self.use_vllm:
            return self.process_image_vllm(image_path, prompt)
        else:
            return self.process_image_transformers(image_path, prompt)

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64"""
        import base64
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def process_batch(self,
                     image_paths: List[str],
                     output_dir: str,
                     prompt: Optional[str] = None,
                     save_individual: bool = True) -> List[Dict]:
        """
        Process batch of images

        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            prompt: Custom prompt (optional)
            save_individual: Save individual JSON for each image

        Returns:
            List of result dictionaries
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []

        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                result = self.process_image(image_path, prompt)
                results.append(result)

                # Save individual result
                if save_individual and "error" not in result:
                    image_name = Path(image_path).stem
                    json_path = output_path / f"{image_name}_ocr.json"

                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

                    # Also save just the text
                    txt_path = output_path / f"{image_name}_ocr.txt"
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(result.get('text', ''))

                # Small delay to avoid overwhelming the system
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # Save batch results
        batch_path = output_path / "batch_results.json"
        with open(batch_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Processed {len(results)} images. Results saved to {output_dir}")

        return results


def find_newspaper_images(data_dir: str,
                         year: Optional[int] = None,
                         month: Optional[int] = None,
                         day: Optional[int] = None) -> List[str]:
    """
    Find newspaper images in data directory

    Args:
        data_dir: Root data directory
        year: Filter by year (optional)
        month: Filter by month (optional)
        day: Filter by day (optional)

    Returns:
        List of image file paths
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return []

    # Build search pattern
    if year and month and day:
        pattern = f"{year:04d}/{month:02d}/{day:02d}/*.jpg"
    elif year and month:
        pattern = f"{year:04d}/{month:02d}/*/*.jpg"
    elif year:
        pattern = f"{year:04d}/*/*/*.jpg"
    else:
        pattern = "*/*/*/*/*.jpg"

    images = sorted(data_path.glob(pattern))
    logger.info(f"Found {len(images)} images")

    return [str(img) for img in images]


def main():
    parser = argparse.ArgumentParser(
        description="Process Diário de Lisboa newspaper scans with DOTS OCR"
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory containing newspaper scans'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ocr_output',
        help='Directory to save OCR results'
    )

    parser.add_argument(
        '--year',
        type=int,
        help='Process only specific year'
    )

    parser.add_argument(
        '--month',
        type=int,
        help='Process only specific month'
    )

    parser.add_argument(
        '--day',
        type=int,
        help='Process only specific day'
    )

    parser.add_argument(
        '--use-transformers',
        action='store_true',
        help='Use Transformers instead of vLLM (slower but may work on more systems)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='rednote-hilab/dots.ocr',
        help='Path to model or HuggingFace model ID'
    )

    parser.add_argument(
        '--vllm-url',
        type=str,
        default='http://localhost:8000/v1',
        help='URL for vLLM server'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'text_only'],
        default='full',
        help='OCR mode: full (structured) or text_only (plain text)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of images to process (for testing)'
    )

    args = parser.parse_args()

    # Find images to process
    logger.info("Searching for images...")
    image_paths = find_newspaper_images(
        args.data_dir,
        year=args.year,
        month=args.month,
        day=args.day
    )

    if not image_paths:
        logger.error("No images found to process")
        return

    if args.limit:
        image_paths = image_paths[:args.limit]
        logger.info(f"Limited to {args.limit} images")

    # Initialize processor
    processor = OCRProcessor(
        use_vllm=not args.use_transformers,
        model_path=args.model_path,
        vllm_url=args.vllm_url
    )

    # Get prompt based on mode
    prompt = processor.get_default_prompt(mode=args.mode)

    # Process images
    logger.info(f"Starting OCR processing of {len(image_paths)} images...")
    start_time = time.time()

    results = processor.process_batch(
        image_paths=image_paths,
        output_dir=args.output_dir,
        prompt=prompt
    )

    elapsed_time = time.time() - start_time

    # Summary
    successful = sum(1 for r in results if "error" not in r)
    failed = len(results) - successful

    logger.info(f"Processing complete!")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Average time per image: {elapsed_time/len(results):.2f} seconds")
    logger.info(f"Successfully processed: {successful}/{len(results)}")
    if failed > 0:
        logger.warning(f"Failed: {failed}/{len(results)}")

    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
