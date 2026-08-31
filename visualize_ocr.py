#!/usr/bin/env python3
"""
OCR Visualization Tool for Diário de Lisboa

Visualizes DOTS OCR output by drawing bounding boxes and labels on newspaper images.

Usage:
    python visualize_ocr.py <image_path> <ocr_json_path>

Example:
    python visualize_ocr.py data/1921/04/07/page_001.jpg vastai_ocr_results/page_001_ocr.txt
"""

import sys
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json_repair

# Category colors (distinct colors for each layout element type)
CATEGORY_COLORS = {
    'Title': '#FF0000',          # Red
    'Text': '#00FF00',           # Green
    'Section-header': '#0000FF', # Blue
    'Page-header': '#FF00FF',    # Magenta
    'Page-footer': '#00FFFF',    # Cyan
    'Caption': '#FFFF00',        # Yellow
    'Footnote': '#FFA500',       # Orange
    'Picture': '#800080',        # Purple
    'Table': '#008080',          # Teal
    'Formula': '#FF1493',        # Deep Pink
    'List-item': '#32CD32',      # Lime Green
}

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def load_ocr_results(ocr_path):
    """Load and parse OCR JSON results"""
    try:
        content = Path(ocr_path).read_text(encoding='utf-8')
        data = json.loads(content)

        # Handle different JSON formats
        # Format 1: Wrapper from batch_ocr_vllm.py: {"ocr_result": "[...]", ...}
        if isinstance(data, dict) and 'ocr_result' in data:
            ocr_text = data['ocr_result']
            # Parse the nested JSON string (use json_repair to handle invalid escapes)
            if isinstance(ocr_text, str):
                try:
                    return json.loads(ocr_text)
                except json.JSONDecodeError:
                    # Fallback to json_repair for malformed JSON
                    return json_repair.repair_json(ocr_text, return_objects=True)
            return ocr_text

        # Format 2: Direct array from old scripts: [{"bbox": ..., "category": ..., "text": ...}, ...]
        if isinstance(data, list):
            return data

        # Format 3: Wrapper with 'text' field: {"text": "[...]", ...}
        if isinstance(data, dict) and 'text' in data:
            ocr_text = data['text']
            if isinstance(ocr_text, str):
                return json.loads(ocr_text)
            return ocr_text

        print(f"Warning: Unexpected JSON format, returning as-is")
        return data

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Content preview: {content[:500]}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading OCR results: {e}")
        sys.exit(1)

def draw_bbox_with_label(draw, bbox, category, text, color):
    """Draw bounding box and label on image"""
    x1, y1, x2, y2 = bbox

    # Ensure coordinates are in correct order (x1 < x2, y1 < y2)
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

    # Prepare label
    if text and category != 'Picture':
        # Truncate text for label
        text_preview = text[:40] + '...' if len(text) > 40 else text
        label = f"{category}: {text_preview}"
    else:
        label = category

    # Draw label background
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Get text bounding box for background
    bbox_text = draw.textbbox((x1, y1 - 25), label, font=font)
    draw.rectangle(bbox_text, fill=color)

    # Draw label text
    draw.text((x1, y1 - 25), label, fill='white', font=font)

def visualize_ocr(image_path, ocr_path, output_path=None, display=True):
    """
    Visualize OCR results on newspaper image

    Args:
        image_path: Path to original newspaper image
        ocr_path: Path to OCR JSON output
        output_path: Optional path to save annotated image
        display: Whether to display the image
    """
    print(f"Loading image: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    print(f"Loading OCR results: {ocr_path}")
    ocr_data = load_ocr_results(ocr_path)

    print(f"Found {len(ocr_data)} layout elements")

    # OCR coordinates are already in original image pixel space - no scaling needed
    print(f"Original image: {img.size[0]}×{img.size[1]}")

    # Create drawing context
    draw = ImageDraw.Draw(img)

    # Draw each bounding box
    for idx, element in enumerate(ocr_data):
        bbox = element.get('bbox')
        category = element.get('category', 'Unknown')
        text = element.get('text', '')

        if not bbox or len(bbox) != 4:
            print(f"Warning: Element {idx} has invalid bbox: {bbox}")
            continue

        # Get color for category
        color = CATEGORY_COLORS.get(category, '#808080')  # Gray for unknown
        color_rgb = hex_to_rgb(color)

        # Draw bbox and label
        draw_bbox_with_label(draw, bbox, category, text, color_rgb)

    # Add legend
    legend_y = 10
    legend_x = img.width - 250

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    # Legend background
    legend_height = len(CATEGORY_COLORS) * 25 + 20
    draw.rectangle([legend_x - 10, legend_y - 10, img.width - 10, legend_y + legend_height],
                   fill=(255, 255, 255, 200), outline='black', width=2)

    # Legend title
    draw.text((legend_x, legend_y), "Categories:", fill='black', font=font)
    legend_y += 25

    # Draw legend items
    for category, color in sorted(CATEGORY_COLORS.items()):
        color_rgb = hex_to_rgb(color)
        # Color box
        draw.rectangle([legend_x, legend_y, legend_x + 15, legend_y + 15], fill=color_rgb, outline='black')
        # Label
        draw.text((legend_x + 20, legend_y), category, fill='black', font=font)
        legend_y += 20

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        print(f"\n✓ Saved annotated image to: {output_path}")

    # Display if requested
    if display:
        print("\nDisplaying image (close window to continue)...")
        img.show()

    return img

def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_ocr.py <image_path> <ocr_json_path> [output_path]")
        print("\nExample:")
        print("  python visualize_ocr.py data/1921/04/07/page_001.jpg vastai_ocr_results/page_001_ocr.txt")
        print("  python visualize_ocr.py data/1921/04/07/page_001.jpg vastai_ocr_results/page_001_ocr.txt annotated_page_001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    ocr_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Auto-generate output path if not provided
    if not output_path:
        output_path = f"annotated_{Path(image_path).name}"

    print("=" * 60)
    print("OCR Visualization Tool")
    print("=" * 60)

    visualize_ocr(image_path, ocr_path, output_path, display=True)

    print("\n" + "=" * 60)
    print("✅ Visualization complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
