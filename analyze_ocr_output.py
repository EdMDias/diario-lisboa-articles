#!/usr/bin/env python3
"""
Analyze OCR output files for:
1. Token counts (using tiktoken)
2. Empty bounding boxes (boxes without text when text is expected)
3. Statistics on text length and categories
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import tiktoken
from json_repair import repair_json

def count_tokens(text):
    """Count tokens using cl100k_base encoding (GPT-4)"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def analyze_ocr_file(file_path):
    """Analyze a single OCR JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if data['status'] != 'success':
            return None

        # Parse the OCR result JSON string
        needed_repair = False
        try:
            ocr_result = json.loads(data['ocr_result'])
        except json.JSONDecodeError as e:
            # Try to repair the JSON
            try:
                repaired = repair_json(data['ocr_result'])
                ocr_result = json.loads(repaired)
                needed_repair = True
            except Exception as repair_error:
                print(f"\nWarning: Could not repair JSON in {file_path}: {repair_error}")
                return None
    except Exception as e:
        print(f"\nWarning: Error reading {file_path}: {e}")
        return None

    file_stats = {
        'file_path': str(file_path),
        'total_boxes': len(ocr_result),
        'boxes_by_category': Counter(),
        'empty_text_boxes': [],  # Boxes that should have text but don't
        'total_text': '',
        'text_by_category': defaultdict(str),
        'needed_repair': needed_repair
    }

    for i, box in enumerate(ocr_result):
        # Skip if box is not a dict (malformed data)
        if not isinstance(box, dict):
            continue

        category = box.get('category', 'Unknown')
        text = box.get('text', '')
        bbox = box.get('bbox', [])

        file_stats['boxes_by_category'][category] += 1

        # Check for empty text in categories that should have text
        # (Pictures don't need text)
        if category != 'Picture':
            if not text or text.strip() == '':
                file_stats['empty_text_boxes'].append({
                    'box_index': i,
                    'category': category,
                    'bbox': bbox
                })
            else:
                file_stats['total_text'] += text + ' '
                file_stats['text_by_category'][category] += text + ' '

    # Count tokens
    file_stats['total_tokens'] = count_tokens(file_stats['total_text'])
    file_stats['tokens_by_category'] = {
        cat: count_tokens(text)
        for cat, text in file_stats['text_by_category'].items()
    }

    return file_stats

def main():
    ocr_dir = Path('ocr_output')

    if not ocr_dir.exists():
        print(f"Error: {ocr_dir} directory not found")
        return

    # Find all OCR JSON files
    ocr_files = list(ocr_dir.rglob('*_ocr.json'))
    print(f"Found {len(ocr_files)} OCR files")

    all_stats = []
    files_with_empty_boxes = []
    files_needing_repair = []
    total_empty_boxes = 0

    print("\nAnalyzing files...")
    for i, file_path in enumerate(ocr_files):
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(ocr_files)} files...")

        stats = analyze_ocr_file(file_path)
        if stats:
            all_stats.append(stats)

            if stats['empty_text_boxes']:
                files_with_empty_boxes.append(stats)
                total_empty_boxes += len(stats['empty_text_boxes'])

            if stats['needed_repair']:
                files_needing_repair.append(stats['file_path'])

    print(f"\nProcessed {len(all_stats)} successful OCR files")
    print(f"Files that needed JSON repair: {len(files_needing_repair)}")

    # Calculate overall statistics
    token_counts = [s['total_tokens'] for s in all_stats]
    token_counts.sort()

    print("\n" + "="*80)
    print("TOKEN STATISTICS")
    print("="*80)
    print(f"Total files analyzed: {len(all_stats)}")
    print(f"Min tokens: {min(token_counts):,}")
    print(f"Max tokens: {max(token_counts):,}")
    print(f"Average tokens: {sum(token_counts) / len(token_counts):,.1f}")
    print(f"Median tokens: {token_counts[len(token_counts)//2]:,}")
    print(f"P90 tokens: {token_counts[int(len(token_counts)*0.9)]:,}")
    print(f"P95 tokens: {token_counts[int(len(token_counts)*0.95)]:,}")
    print(f"P99 tokens: {token_counts[int(len(token_counts)*0.99)]:,}")

    # Category statistics
    all_categories = Counter()
    for stats in all_stats:
        all_categories.update(stats['boxes_by_category'])

    print("\n" + "="*80)
    print("CATEGORY STATISTICS")
    print("="*80)
    for category, count in all_categories.most_common():
        print(f"{category:20s}: {count:,} boxes")

    # Empty text box analysis
    print("\n" + "="*80)
    print("EMPTY TEXT BOXES ANALYSIS")
    print("="*80)
    print(f"Files with empty text boxes: {len(files_with_empty_boxes)}")
    print(f"Total empty text boxes: {total_empty_boxes}")

    if files_with_empty_boxes:
        print(f"\nPercentage of files with issues: {len(files_with_empty_boxes)/len(all_stats)*100:.2f}%")

        # Show categories of empty boxes
        empty_categories = Counter()
        for stats in files_with_empty_boxes:
            for empty_box in stats['empty_text_boxes']:
                empty_categories[empty_box['category']] += 1

        print("\nEmpty boxes by category:")
        for category, count in empty_categories.most_common():
            print(f"  {category:20s}: {count:,} empty boxes")

        # Show first 10 files with empty boxes
        print("\nFirst 10 files with empty text boxes:")
        for stats in files_with_empty_boxes[:10]:
            print(f"\n  {stats['file_path']}")
            print(f"    Total boxes: {stats['total_boxes']}, Empty: {len(stats['empty_text_boxes'])}")
            for empty_box in stats['empty_text_boxes'][:3]:  # Show first 3 empty boxes
                print(f"      Box {empty_box['box_index']}: {empty_box['category']} - {empty_box['bbox']}")

    # Top 10 files by token count
    print("\n" + "="*80)
    print("TOP 10 FILES BY TOKEN COUNT")
    print("="*80)
    all_stats_sorted = sorted(all_stats, key=lambda x: x['total_tokens'], reverse=True)
    for stats in all_stats_sorted[:10]:
        print(f"{stats['total_tokens']:6,} tokens - {stats['file_path']}")

    # Bottom 10 files by token count
    print("\n" + "="*80)
    print("BOTTOM 10 FILES BY TOKEN COUNT")
    print("="*80)
    for stats in all_stats_sorted[-10:]:
        print(f"{stats['total_tokens']:6,} tokens - {stats['file_path']}")

if __name__ == '__main__':
    main()
