#!/usr/bin/env python3
"""Debug column structure for page 003"""

import json
from collections import defaultdict

# Load OCR file
with open('ocr_output/1974/04/24/page_003_ocr.json') as f:
    data = json.load(f)
    ocr_result = json.loads(data['ocr_result'])

# Group boxes by approximate X-coordinate
columns = defaultdict(list)

for i, box in enumerate(ocr_result):
    if not isinstance(box, dict):
        continue

    bbox = box.get('bbox', [])
    if len(bbox) != 4:
        continue

    category = box.get('category', '')
    if category in ['Page-header', 'Page-footer']:
        continue

    center_x = (bbox[0] + bbox[2]) / 2

    # Round to nearest 200px to group columns
    col_group = round(center_x / 200) * 200

    columns[col_group].append({
        'index': i,
        'category': category,
        'text': box.get('text', '')[:60],
        'x1': bbox[0],
        'x2': bbox[2],
        'y1': bbox[1],
        'has_title': category in ['Title', 'Section-header']
    })

# Print column summary
print('Column structure (grouped by X-coordinate):')
for col_x in sorted(columns.keys()):
    boxes = columns[col_x]
    content_boxes = [b for b in boxes if b['category'] not in ['Picture', 'Caption']]
    titles = [b for b in boxes if b['has_title']]

    print(f'\nColumn at X≈{col_x}:')
    print(f'  Total boxes: {len(boxes)}')
    print(f'  Content boxes: {len(content_boxes)}')
    print(f'  Titles: {len(titles)}')

    if titles:
        print(f'  Titles found:')
        for t in titles:
            print(f'    - {t["text"]}')
    else:
        print(f'  ⚠️ NO TITLES in this column')
        if len(content_boxes) >= 3:
            print(f'  ✓ Has {len(content_boxes)} content boxes (potential orphan)')
            # Show first few boxes
            print(f'  First boxes:')
            for b in content_boxes[:5]:
                print(f'    - [{b["category"]}] Y={b["y1"]:.0f} X={b["x1"]:.0f}-{b["x2"]:.0f}')
                print(f'      "{b["text"]}"')

print('\n' + '='*80)
print('Now testing ColumnDetector with min_column_gap=50')
print('='*80)

# Test actual ColumnDetector
from article_extractor import ColumnDetector, BBox

# Convert to BBox objects
bboxes = []
for i, box in enumerate(ocr_result):
    if not isinstance(box, dict):
        continue

    bbox_coords = box.get('bbox', [])
    if len(bbox_coords) != 4:
        continue

    category = box.get('category', '')
    if category in ['Page-header', 'Page-footer']:
        continue

    bboxes.append(BBox(
        x1=bbox_coords[0],
        y1=bbox_coords[1],
        x2=bbox_coords[2],
        y2=bbox_coords[3],
        category=category,
        text=box.get('text', ''),
        index=i
    ))

detector = ColumnDetector(min_column_gap=50)
detected_columns = detector.detect_columns(bboxes)

print(f'\nDetected {len(detected_columns)} columns by ColumnDetector:')
for i, col in enumerate(detected_columns):
    content_boxes = [b for b in col if b.category not in ['Picture', 'Caption']]
    titles = [b for b in col if b.category in ['Title', 'Section-header']]

    if content_boxes:
        x_range = f'{min(b.x1 for b in content_boxes):.0f}-{max(b.x2 for b in content_boxes):.0f}'
        print(f'\nColumn {i}: X={x_range}')
        print(f'  Total boxes: {len(col)}')
        print(f'  Content boxes: {len(content_boxes)}')
        print(f'  Titles: {len(titles)}')

        if titles:
            print(f'  Has titles:')
            for t in titles:
                print(f'    - {t.text[:60]}')
        else:
            print(f'  ⚠️ NO TITLES (orphan column candidate)')
            if len(content_boxes) >= 3:
                print(f'  First 3 content boxes:')
                for b in content_boxes[:3]:
                    print(f'    - Y={b.y1:.0f} "{b.text[:50]}"')
