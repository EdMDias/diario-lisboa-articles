#!/usr/bin/env python3
"""
Quick OCR analysis without external dependencies
Analyzes OCR outputs for potential issues:
1. Overlapping bounding boxes
2. Unusual bbox sizes (too small, too large)
3. Empty text in non-Picture categories
4. Continuation markers
"""

import json
from pathlib import Path
from collections import Counter

def bbox_area(bbox):
    """Calculate area of bounding box"""
    x1, y1, x2, y2 = bbox
    return abs((x2 - x1) * (y2 - y1))

def bbox_overlap(bbox1, bbox2):
    """Calculate overlap area between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = min(bbox1[0], bbox1[2]), min(bbox1[1], bbox1[3]), max(bbox1[0], bbox1[2]), max(bbox1[1], bbox1[3])
    x2_min, y2_min, x2_max, y2_max = min(bbox2[0], bbox2[2]), min(bbox2[1], bbox2[3]), max(bbox2[0], bbox2[2]), max(bbox2[1], bbox2[3])

    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    overlap_area = x_overlap * y_overlap

    # Calculate overlap percentage relative to smaller box
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    smaller_area = min(area1, area2)

    if smaller_area == 0:
        return 0

    return overlap_area / smaller_area

def analyze_ocr_file(file_path):
    """Analyze a single OCR file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data['status'] != 'success':
            return None

        ocr_result = json.loads(data['ocr_result'])

        issues = {
            'file_path': str(file_path),
            'total_boxes': len(ocr_result),
            'empty_text': [],
            'small_boxes': [],
            'large_boxes': [],
            'overlapping_boxes': [],
            'continuation_markers': [],
            'categories': Counter()
        }

        # Analyze each box
        for i, box in enumerate(ocr_result):
            category = box.get('category', 'Unknown')
            text = box.get('text', '')
            bbox = box.get('bbox', [])

            issues['categories'][category] += 1

            # Check for empty text (except Pictures)
            if category != 'Picture' and not text.strip():
                issues['empty_text'].append({
                    'index': i,
                    'category': category,
                    'bbox': bbox
                })

            # Check for continuation markers
            if text and 'continua na pág' in text.lower():
                issues['continuation_markers'].append({
                    'index': i,
                    'text': text,
                    'bbox': bbox
                })

            # Check bbox size
            if len(bbox) == 4:
                area = bbox_area(bbox)
                width = abs(bbox[2] - bbox[0])
                height = abs(bbox[3] - bbox[1])

                # Suspiciously small boxes (< 100 pixels area)
                if area < 100:
                    issues['small_boxes'].append({
                        'index': i,
                        'category': category,
                        'bbox': bbox,
                        'area': area
                    })

                # Suspiciously large boxes (> 70% of typical page)
                # Typical page is ~2870×3832, so 70% ≈ 7.7M pixels
                if area > 7_700_000:
                    issues['large_boxes'].append({
                        'index': i,
                        'category': category,
                        'bbox': bbox,
                        'area': area
                    })

        # Check for overlapping boxes
        for i in range(len(ocr_result)):
            for j in range(i + 1, len(ocr_result)):
                bbox_i = ocr_result[i].get('bbox', [])
                bbox_j = ocr_result[j].get('bbox', [])

                if len(bbox_i) == 4 and len(bbox_j) == 4:
                    overlap_pct = bbox_overlap(bbox_i, bbox_j)

                    # Flag if overlap > 30% of smaller box
                    if overlap_pct > 0.3:
                        issues['overlapping_boxes'].append({
                            'box1_index': i,
                            'box1_category': ocr_result[i].get('category'),
                            'box2_index': j,
                            'box2_category': ocr_result[j].get('category'),
                            'overlap_pct': overlap_pct * 100
                        })

        return issues

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def main():
    ocr_dir = Path('ocr_output')
    ocr_files = list(ocr_dir.rglob('*_ocr.json'))[:50]  # Analyze first 50 files

    print(f"Analyzing {len(ocr_files)} OCR files...\n")

    all_issues = []

    for file_path in ocr_files:
        issues = analyze_ocr_file(file_path)
        if issues:
            all_issues.append(issues)

    # Aggregate statistics
    print("=" * 80)
    print("OCR QUALITY ANALYSIS")
    print("=" * 80)

    total_files = len(all_issues)
    files_with_empty = sum(1 for i in all_issues if i['empty_text'])
    files_with_overlap = sum(1 for i in all_issues if i['overlapping_boxes'])
    files_with_small = sum(1 for i in all_issues if i['small_boxes'])
    files_with_large = sum(1 for i in all_issues if i['large_boxes'])
    files_with_continuation = sum(1 for i in all_issues if i['continuation_markers'])

    print(f"\nTotal files analyzed: {total_files}")
    print(f"Files with empty text boxes: {files_with_empty} ({files_with_empty/total_files*100:.1f}%)")
    print(f"Files with overlapping boxes: {files_with_overlap} ({files_with_overlap/total_files*100:.1f}%)")
    print(f"Files with small boxes: {files_with_small} ({files_with_small/total_files*100:.1f}%)")
    print(f"Files with large boxes: {files_with_large} ({files_with_large/total_files*100:.1f}%)")
    print(f"Files with continuation markers: {files_with_continuation} ({files_with_continuation/total_files*100:.1f}%)")

    # Category distribution
    all_categories = Counter()
    for issues in all_issues:
        all_categories.update(issues['categories'])

    print("\n" + "=" * 80)
    print("CATEGORY DISTRIBUTION")
    print("=" * 80)
    for category, count in all_categories.most_common():
        print(f"{category:20s}: {count:5d} boxes")

    # Show examples of each issue type
    print("\n" + "=" * 80)
    print("SAMPLE ISSUES")
    print("=" * 80)

    # Empty text boxes
    print("\n1. EMPTY TEXT BOXES (first 5 examples):")
    count = 0
    for issues in all_issues:
        if issues['empty_text'] and count < 5:
            print(f"\n  File: {issues['file_path']}")
            for empty in issues['empty_text'][:2]:
                print(f"    Box {empty['index']}: {empty['category']} - bbox={empty['bbox']}")
            count += 1

    # Overlapping boxes
    print("\n2. OVERLAPPING BOXES (first 5 examples):")
    count = 0
    for issues in all_issues:
        if issues['overlapping_boxes'] and count < 5:
            print(f"\n  File: {issues['file_path']}")
            for overlap in issues['overlapping_boxes'][:2]:
                print(f"    Box {overlap['box1_index']} ({overlap['box1_category']}) overlaps "
                      f"Box {overlap['box2_index']} ({overlap['box2_category']}) by {overlap['overlap_pct']:.1f}%")
            count += 1

    # Continuation markers
    print("\n3. CONTINUATION MARKERS (first 10 examples):")
    count = 0
    for issues in all_issues:
        if issues['continuation_markers'] and count < 10:
            for marker in issues['continuation_markers']:
                print(f"\n  File: {issues['file_path']}")
                print(f"    Box {marker['index']}: \"{marker['text']}\"")
                count += 1
                if count >= 10:
                    break

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == '__main__':
    main()
