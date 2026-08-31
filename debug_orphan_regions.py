#!/usr/bin/env python3
"""Debug orphan region detection for page 003"""

import json
from article_extractor import ArticleExtractor, BBox
from pathlib import Path

# Load OCR file
ocr_file = Path('ocr_output/1974/04/24/page_003_ocr.json')

with open(ocr_file) as f:
    data = json.load(f)
    ocr_result = json.loads(data['ocr_result'])

# Convert to BBox objects
bboxes = []
for i, box in enumerate(ocr_result):
    if not isinstance(box, dict):
        continue

    bbox_coords = box.get('bbox', [])
    if len(bbox_coords) != 4:
        continue

    bboxes.append(BBox(
        x1=bbox_coords[0],
        y1=bbox_coords[1],
        x2=bbox_coords[2],
        y2=bbox_coords[3],
        category=box.get('category', ''),
        text=box.get('text', ''),
        index=i
    ))

# Filter metadata
content_boxes = [b for b in bboxes if b.category not in ['Page-header', 'Page-footer']]

# Get titles
titles = [b for b in content_boxes if b.category in ['Title', 'Section-header']]

print(f'Total content boxes: {len(content_boxes)}')
print(f'Titles: {len(titles)}')
print()

print('Titles on page:')
for title in titles:
    print(f'  [{title.index}] Y={title.y1:.0f}-{title.y2:.0f}, X={title.x1:.0f}-{title.x2:.0f}')
    print(f'      "{title.text[:60]}"')
print()

# Test orphan region detection
extractor = ArticleExtractor()

# Use only article-level titles (not sub-sections)
article_titles, subsection_map = extractor.merge_subsections(titles)

print(f'Article-level titles: {len(article_titles)}')
for title in article_titles:
    print(f'  [{title.index}] "{title.text[:60]}"')
print()

# Detect orphan regions
orphan_regions = extractor.detect_orphan_regions(content_boxes, article_titles)

print(f'Detected {len(orphan_regions)} orphan region(s)')
print()

for i, region in enumerate(orphan_regions):
    # Get region bounds
    x_min = min(b.x1 for b in region)
    x_max = max(b.x2 for b in region)
    y_min = min(b.y1 for b in region)
    y_max = max(b.y2 for b in region)
    x_center = sum(b.center_x for b in region) / len(region)
    y_center = sum(b.center_y for b in region) / len(region)

    print(f'Orphan region {i}:')
    print(f'  Boxes: {len(region)}')
    print(f'  X range: {x_min:.0f}-{x_max:.0f} (center: {x_center:.0f})')
    print(f'  Y range: {y_min:.0f}-{y_max:.0f} (center: {y_center:.0f})')

    # Check which titles are near this region
    print(f'  Nearby titles check:')
    for title in article_titles:
        title_y_overlap = (min(title.y2, y_max) - max(title.y1, y_min))
        h_dist = abs(title.center_x - x_center)

        print(f'    Title [{title.index}] "{title.text[:40]}"')
        print(f'      Title Y: {title.y1:.0f}-{title.y2:.0f}, X: {title.center_x:.0f}')
        print(f'      Y overlap: {title_y_overlap:.0f}px')
        print(f'      H distance: {h_dist:.0f}px')
        print(f'      Is nearby: {title_y_overlap > 0 and h_dist < 600}')

    # Show first few boxes
    print(f'  First 5 boxes:')
    region_sorted = sorted(region, key=lambda b: (b.y1, b.x1))
    for b in region_sorted[:5]:
        print(f'    [{b.index}] Y={b.y1:.0f} X={b.x1:.0f}-{b.x2:.0f} [{b.category}]')
        print(f'        "{b.text[:60]}"')

    # Check for specific content
    region_text = ' '.join(b.text.lower() for b in region)
    has_ufo = 'extraterrestres' in region_text or 'ufo' in region_text
    has_taxi = 'táxi' in region_text or 'estribo' in region_text
    has_music = 'canção' in region_text or 'música' in region_text

    print(f'  Content check:')
    print(f'    Has UFO content: {has_ufo}')
    print(f'    Has taxi content: {has_taxi}')
    print(f'    Has music content: {has_music}')

    print()
