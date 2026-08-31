#!/usr/bin/env python3
"""
Visualize OCR bounding boxes on the original image
"""
import json
from PIL import Image, ImageDraw, ImageFont
import sys

def visualize_ocr(image_path, json_path, output_path):
    # Load image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Load OCR results
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse OCR result (it's a JSON string inside the JSON file)
    ocr_result = json.loads(data['ocr_result'])
    
    # Draw bounding boxes
    colors = {
        'Page-header': 'blue',
        'Section-header': 'red',
        'Text': 'green',
        'Picture': 'purple',
        'Title': 'orange',
        'Caption': 'cyan',
        'Table': 'yellow'
    }
    
    for item in ocr_result:
        bbox = item['bbox']
        category = item.get('category', 'Unknown')
        color = colors.get(category, 'white')
        
        # Draw rectangle
        draw.rectangle(bbox, outline=color, width=3)
        
        # Draw category label
        text = category
        try:
            draw.text((bbox[0], bbox[1] - 20), text, fill=color)
        except:
            pass
    
    # Save result
    img.save(output_path)
    print(f"Visualization saved to: {output_path}")
    print(f"Detected {len(ocr_result)} elements")
    print(f"Image size: {img.size}")
    
    # Print category counts
    categories = {}
    for item in ocr_result:
        cat = item.get('category', 'Unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory counts:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

if __name__ == '__main__':
    image_path = 'data/1974/04/26/page_001.jpg'
    json_path = 'vastai_ocr_results/1974/04/26/page_001_ocr.json'
    output_path = 'vastai_ocr_results/1974/04/26/page_001_visualized.jpg'
    
    visualize_ocr(image_path, json_path, output_path)
