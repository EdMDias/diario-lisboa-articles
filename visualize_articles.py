#!/usr/bin/env python3
"""
Article Visualization for Diário de Lisboa
Colors OCR bounding boxes by article assignment for validation

Usage:
    python visualize_articles.py <image_path> [ocr_path] [output_path]

Example:
    python visualize_articles.py data/1974/04/24/page_001.png
    python visualize_articles.py data/1974/04/24/page_001.png ocr_output/1974/04/24/page_001_ocr.json
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import colorsys
import sys
from typing import List, Tuple, Dict
from article_extractor import ArticleExtractor, Article


def generate_article_colors(num_articles: int) -> List[Tuple[int, int, int]]:
    """
    Generate visually distinct colors using HSL golden ratio

    Args:
        num_articles: Number of unique colors to generate

    Returns:
        List of RGB tuples (r, g, b) with values 0-255
    """
    golden_ratio = 0.618033988749895
    colors = []

    for i in range(num_articles):
        # Use golden ratio to maximize hue separation
        hue = (i * golden_ratio) % 1.0
        # High saturation and medium lightness for vibrant, readable colors
        rgb = colorsys.hls_to_rgb(hue, 0.5, 0.85)
        rgb_255 = tuple(int(c * 255) for c in rgb)
        colors.append(rgb_255)

    return colors


def extract_article_bbox_mapping(ocr_path: Path) -> Tuple[List[Article], Dict[str, List[int]]]:
    """
    Extract articles and get bbox-to-article mapping

    Args:
        ocr_path: Path to OCR JSON file

    Returns:
        Tuple of (list of articles, mapping of article_id to bbox indices)
    """
    extractor = ArticleExtractor()
    articles = extractor.extract_articles(ocr_path)

    # Build mapping: article_id -> list of OCR bbox indices
    mapping = {}
    for article in articles:
        bbox_indices = [bbox.index for bbox in article.bboxes]
        mapping[article.article_id] = bbox_indices

    return articles, mapping


def load_ocr_bboxes(ocr_path: Path) -> List[dict]:
    """
    Load OCR JSON to get bbox coordinates

    Args:
        ocr_path: Path to OCR JSON file

    Returns:
        List of OCR result dictionaries with bbox, category, text
    """
    with open(ocr_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data['status'] != 'success':
        raise ValueError(f"OCR status is not success: {data['status']}")

    ocr_result = json.loads(data['ocr_result'])
    return ocr_result


def draw_bbox_with_article_label(draw, bbox, article_num, color, font):
    """
    Draw bbox with article ID label

    Args:
        draw: PIL ImageDraw object
        bbox: Dict with 'bbox' key containing [x1, y1, x2, y2]
        article_num: Article number (1-indexed)
        color: RGB tuple for box color
        font: PIL font object
    """
    x1, y1, x2, y2 = bbox['bbox']

    # Ensure coordinates are in correct order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    # Draw label
    label = f"Art {article_num:03d}"

    # Position label above box
    label_y = max(y1 - 25, 10)  # Ensure label stays on screen

    try:
        bbox_text = draw.textbbox((x1, label_y), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, label_y), label, fill='white', font=font)
    except:
        # Fallback if textbbox not available
        draw.text((x1, label_y), label, fill=color, font=font)


def draw_legend(draw, img_width, img_height, articles, colors, font):
    """
    Draw legend showing article IDs and colors

    Args:
        draw: PIL ImageDraw object
        img_width: Image width
        img_height: Image height
        articles: List of Article objects
        colors: List of RGB tuples matching articles
        font: PIL font object
    """
    legend_width = 380
    legend_x = img_width - legend_width - 10
    legend_y = 10
    line_height = 18
    max_articles = min(len(articles), 30)

    # Background
    legend_height = max_articles * line_height + 50
    draw.rectangle([legend_x - 10, legend_y - 10,
                   img_width - 10, min(legend_y + legend_height, img_height - 10)],
                  fill=(255, 255, 255, 230), outline='black', width=2)

    # Title
    draw.text((legend_x, legend_y), "Articles by Color:", fill='black', font=font)
    legend_y += 25

    # Article items
    for i, (article, color) in enumerate(zip(articles[:max_articles], colors)):
        # Color box
        draw.rectangle([legend_x, legend_y, legend_x + 12, legend_y + 12],
                      fill=color, outline='black')

        # Article info
        article_num = i + 1
        title = article.title[:38] + "..." if len(article.title) > 38 else article.title
        label = f"{article_num:03d}: {title}" if title else f"{article_num:03d}"

        draw.text((legend_x + 18, legend_y), label, fill='black', font=font)
        legend_y += line_height

    # Overflow indicator
    if len(articles) > max_articles:
        draw.text((legend_x, legend_y),
                 f"... and {len(articles) - max_articles} more",
                 fill='gray', font=font)


def auto_detect_ocr_path(image_path: Path) -> Path:
    """
    Auto-detect OCR path from image path

    Args:
        image_path: Path to newspaper image

    Returns:
        Path to OCR JSON file

    Raises:
        ValueError: If path cannot be auto-detected
    """
    # Convert data/YYYY/MM/DD/page_XXX.ext -> ocr_output/YYYY/MM/DD/page_XXX_ocr.json
    parts = image_path.parts
    try:
        data_idx = parts.index('data')
        relative_path = Path(*parts[data_idx+1:])
        ocr_path = Path('ocr_output') / relative_path.parent / f"{relative_path.stem}_ocr.json"
        return ocr_path
    except ValueError:
        raise ValueError("Cannot auto-detect OCR path. Image path must contain 'data/' directory.")


def visualize_articles(image_path: Path, ocr_path: Path,
                      output_path: Path = None, display: bool = True,
                      show_labels: bool = True):
    """
    Main visualization function - colors OCR bboxes by article

    Args:
        image_path: Path to newspaper image
        ocr_path: Path to OCR JSON file
        output_path: Optional path to save annotated image
        display: Whether to display the image
        show_labels: Whether to show article ID labels on boxes
    """
    print(f"Loading image: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    print(f"Extracting article structure from: {ocr_path}")
    try:
        articles, article_bbox_mapping = extract_article_bbox_mapping(ocr_path)
    except Exception as e:
        print(f"Error extracting articles: {e}")
        sys.exit(1)

    print(f"Loading OCR bounding boxes...")
    try:
        ocr_bboxes = load_ocr_bboxes(ocr_path)
    except Exception as e:
        print(f"Error loading OCR boxes: {e}")
        sys.exit(1)

    print(f"Found {len(articles)} articles with {len(ocr_bboxes)} total bboxes")

    if len(articles) == 0:
        print("Warning: No articles found in this page")
        return

    # Generate colors
    colors = generate_article_colors(len(articles))

    # Create draw context
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()

    # Draw bboxes colored by article
    total_boxes_drawn = 0
    for article_num, (article, color) in enumerate(zip(articles, colors), 1):
        bbox_indices = article_bbox_mapping[article.article_id]

        for bbox_idx in bbox_indices:
            if bbox_idx < len(ocr_bboxes):
                bbox = ocr_bboxes[bbox_idx]
                # Skip images and page headers/footers for clarity
                if bbox.get('category') not in ['Picture', 'Page-header', 'Page-footer']:
                    if show_labels:
                        draw_bbox_with_article_label(draw, bbox, article_num, color, font)
                    else:
                        # Just draw the box without label
                        x1, y1, x2, y2 = bbox['bbox']
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    total_boxes_drawn += 1

    print(f"Drew {total_boxes_drawn} bounding boxes")

    # Draw legend
    draw_legend(draw, img.width, img.height, articles, colors, font)

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        print(f"✓ Saved to: {output_path}")

    # Display
    if display:
        print("Displaying image (close window to continue)...")
        img.show()

    return img


def main():
    """Command-line interface"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Missing image path argument")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    # Auto-detect OCR path if not provided
    if len(sys.argv) > 2:
        ocr_path = Path(sys.argv[2])
    else:
        try:
            ocr_path = auto_detect_ocr_path(image_path)
            print(f"Auto-detected OCR path: {ocr_path}")
        except ValueError as e:
            print(f"Error: {e}")
            print("Please provide OCR path explicitly as second argument.")
            sys.exit(1)

    if not ocr_path.exists():
        print(f"Error: OCR file not found: {ocr_path}")
        sys.exit(1)

    # Optional output path
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    # Generate default output path if not specified
    if not output_path:
        output_path = Path(f"visualized_{image_path.name}")

    print("=" * 80)
    print("ARTICLE VISUALIZATION")
    print("=" * 80)

    visualize_articles(image_path, ocr_path, output_path, display=False)

    print("\n" + "=" * 80)
    print("✅ Visualization complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
