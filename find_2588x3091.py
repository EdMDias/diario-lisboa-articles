#!/usr/bin/env python3
"""
Find what parameters would cause 2733×3250 to resize to 2588×3091
"""

import math

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Smart resize algorithm"""
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar

# Input and expected output
height, width = 2733, 3250
expected_h, expected_w = 2588, 3091

print(f"Looking for parameters that transform {height}×{width} → {expected_h}×{expected_w}")
print(f"\nExpected output analysis:")
print(f"  - Both divisible by 28? h:{expected_h % 28 == 0}, w:{expected_w % 28 == 0}")
print(f"  - Total pixels: {expected_h * expected_w:,}")
print(f"  - Scale factor: {expected_h / height:.6f} (h), {expected_w / width:.6f} (w)")

# Check if it's divisible by different factors
for factor in [14, 28, 56]:
    if expected_h % factor == 0 and expected_w % factor == 0:
        print(f"  - Divisible by {factor}: YES")
    else:
        print(f"  - Divisible by {factor}: NO")

# Calculate what beta would be needed
actual_pixels = expected_h * expected_w
original_pixels = height * width
beta_calc = math.sqrt(original_pixels / actual_pixels)

print(f"\nCalculated beta (downscale factor): {beta_calc:.6f}")
print(f"This suggests max_pixels = {actual_pixels:,}")

# Try to find the factor
print("\n" + "="*80)
print("Testing different factors:")
print("="*80)

for factor in [14, 28, 56]:
    print(f"\nFactor = {factor}:")

    # What would be the max_pixels if we got this result?
    if expected_h % factor == 0 and expected_w % factor == 0:
        # Working backwards from the expected result
        max_pixels_test = actual_pixels

        result_h, result_w = smart_resize(height, width, factor, 3136, max_pixels_test)

        print(f"  With max_pixels={max_pixels_test:,}: {result_h}×{result_w}")

        if result_h == expected_h and result_w == expected_w:
            print(f"  ✓ MATCH! factor={factor}, max_pixels={max_pixels_test:,}")

# Try different max_pixels values around the expected
print("\n" + "="*80)
print("Searching for exact parameters (factor=28):")
print("="*80)

factor = 28
for max_pix_adj in range(-100000, 100000, 10000):
    max_pixels_test = actual_pixels + max_pix_adj
    result_h, result_w = smart_resize(height, width, factor, 3136, max_pixels_test)

    if result_h == expected_h and result_w == expected_w:
        print(f"✓ FOUND: max_pixels={max_pixels_test:,} gives {result_h}×{result_w}")

        # Calculate what this means in terms of tokens
        tokens = max_pixels_test // (factor ** 2)
        print(f"  This is {tokens} tokens × {factor}² = {max_pixels_test:,} pixels")
        break
