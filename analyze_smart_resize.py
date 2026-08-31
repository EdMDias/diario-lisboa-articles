#!/usr/bin/env python3
"""
Analyze the Qwen2-VL smart_resize algorithm to understand exactly how
an input image of 2733×3250 gets resized to 2588×3091.

This reproduces the exact logic from:
- qwen-vl-utils/vision_process.py (QwenLM)
- transformers/models/qwen2_vl/image_processing_qwen2_vl.py (HuggingFace)
"""

import math

# Constants from qwen-vl-utils
IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
SPATIAL_MERGE_SIZE = 2

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize_qwen_vl_utils(
    height: int,
    width: int,
    factor: int,
    min_pixels: int = None,
    max_pixels: int = None
) -> tuple[int, int]:
    """
    Original implementation from qwen-vl-utils/vision_process.py

    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    max_pixels = max_pixels if max_pixels is not None else (IMAGE_MAX_TOKEN_NUM * factor ** 2)
    min_pixels = min_pixels if min_pixels is not None else (IMAGE_MIN_TOKEN_NUM * factor ** 2)

    assert max_pixels >= min_pixels, "The max_pixels of image must be greater than or equal to min_pixels."

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    # Step 1: Round to nearest multiple of factor
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    print(f"\n=== STEP 1: Initial rounding to factor={factor} ===")
    print(f"Original: {height} × {width} = {height * width:,} pixels")
    print(f"Rounded:  {h_bar} × {w_bar} = {h_bar * w_bar:,} pixels")

    # Step 2: Check if exceeds max_pixels
    if h_bar * w_bar > max_pixels:
        print(f"\n=== STEP 2: Exceeds max_pixels ({max_pixels:,}) - DOWNSCALING ===")
        beta = math.sqrt((height * width) / max_pixels)
        print(f"Beta (scale factor): sqrt({height * width:,} / {max_pixels:,}) = {beta:.6f}")

        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)

        print(f"After downscale: {h_bar} × {w_bar} = {h_bar * w_bar:,} pixels")

    # Step 3: Check if below min_pixels
    elif h_bar * w_bar < min_pixels:
        print(f"\n=== STEP 2: Below min_pixels ({min_pixels:,}) - UPSCALING ===")
        beta = math.sqrt(min_pixels / (height * width))
        print(f"Beta (scale factor): sqrt({min_pixels:,} / {height * width:,}) = {beta:.6f}")

        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

        print(f"After upscale: {h_bar} × {w_bar} = {h_bar * w_bar:,} pixels")
    else:
        print(f"\n=== STEP 2: Within range [{min_pixels:,}, {max_pixels:,}] - NO SCALING ===")

    return h_bar, w_bar

def smart_resize_transformers(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280
) -> tuple[int, int]:
    """
    HuggingFace transformers implementation.
    Identical logic but uses math.floor/ceil directly instead of helper functions.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar

def analyze_resize_case(height: int, width: int, image_patch_size: int = 14):
    """Analyze a specific resize case with detailed output."""

    # Calculate parameters as used in fetch_image()
    patch_factor = image_patch_size * SPATIAL_MERGE_SIZE
    min_pixels = IMAGE_MIN_TOKEN_NUM * patch_factor ** 2
    max_pixels = IMAGE_MAX_TOKEN_NUM * patch_factor ** 2

    print("=" * 80)
    print(f"QWEN2-VL SMART_RESIZE ANALYSIS")
    print("=" * 80)
    print(f"\nInput Image Size: {height} × {width}")
    print(f"Aspect Ratio: {width/height:.6f} ({width}:{height})")
    print(f"\nConfiguration:")
    print(f"  - image_patch_size: {image_patch_size}")
    print(f"  - SPATIAL_MERGE_SIZE: {SPATIAL_MERGE_SIZE}")
    print(f"  - patch_factor (factor): {patch_factor}")
    print(f"  - IMAGE_MIN_TOKEN_NUM: {IMAGE_MIN_TOKEN_NUM}")
    print(f"  - IMAGE_MAX_TOKEN_NUM: {IMAGE_MAX_TOKEN_NUM}")
    print(f"\nPixel Constraints:")
    print(f"  - min_pixels: {IMAGE_MIN_TOKEN_NUM} × {patch_factor}² = {min_pixels:,}")
    print(f"  - max_pixels: {IMAGE_MAX_TOKEN_NUM} × {patch_factor}² = {max_pixels:,}")

    # Run the algorithm
    result_h, result_w = smart_resize_qwen_vl_utils(
        height, width, patch_factor, min_pixels, max_pixels
    )

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULT: {height} × {width} → {result_h} × {result_w}")
    print(f"{'=' * 80}")
    print(f"Scale factors: height={result_h/height:.6f}, width={result_w/width:.6f}")
    print(f"Final aspect ratio: {result_w/result_h:.6f}")
    print(f"Final total pixels: {result_h * result_w:,}")
    print(f"Divisible by {patch_factor}: height={result_h % patch_factor == 0}, width={result_w % patch_factor == 0}")

    # Verify with transformers implementation
    print(f"\n{'=' * 80}")
    print("VERIFICATION WITH HUGGINGFACE TRANSFORMERS IMPLEMENTATION:")
    print('=' * 80)
    result_h_tf, result_w_tf = smart_resize_transformers(
        height, width, patch_factor, min_pixels, max_pixels
    )
    print(f"Transformers result: {result_h_tf} × {result_w_tf}")

    if result_h == result_h_tf and result_w == result_w_tf:
        print("✓ Both implementations produce IDENTICAL results")
    else:
        print("✗ MISMATCH between implementations!")

    return result_h, result_w

if __name__ == "__main__":
    # Analyze the specific case from the user's question
    print("\n" + "=" * 80)
    print("CASE 1: User's example - 2733×3250")
    print("=" * 80)
    analyze_resize_case(2733, 3250)

    print("\n\n" + "=" * 80)
    print("CASE 2: Another example - 1000×1000 (square)")
    print("=" * 80)
    analyze_resize_case(1000, 1000)

    print("\n\n" + "=" * 80)
    print("CASE 3: Small image - 500×600")
    print("=" * 80)
    analyze_resize_case(500, 600)
