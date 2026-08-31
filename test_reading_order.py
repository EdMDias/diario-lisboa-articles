#!/usr/bin/env python3
"""
Test reading order validation for extracted articles
Checks if text boxes within each article are properly sorted
"""

import json
from pathlib import Path
import sys


def check_reading_order(article_file: Path, ocr_file: Path):
    """
    Validate that boxes within each article are in correct reading order

    Args:
        article_file: Path to articles JSON
        ocr_file: Path to OCR JSON with bbox coordinates
    """
    # Load articles
    with open(article_file) as f:
        articles = json.load(f)

    # Load OCR to get bbox coordinates
    with open(ocr_file) as f:
        ocr_data = json.load(f)
        ocr_boxes = json.loads(ocr_data['ocr_result'])

    page_name = article_file.stem.replace('_articles', '')

    print(f"\n{'='*80}")
    print(f"Testing {page_name}")
    print(f"{'='*80}\n")

    print(f"Total articles: {len(articles)}\n")

    for article_idx, article in enumerate(articles):
        print(f"Article {article_idx + 1}: {article['title'][:50]}")
        print(f"  Paragraphs: {article['num_paragraphs']}")

        # Check text flow
        text = article['text']

        # Check for broken sentences (text box cuts mid-sentence)
        broken_sentences = 0
        if '\n\n' in text:
            paragraphs = text.split('\n\n')
            for i in range(len(paragraphs) - 1):
                current = paragraphs[i].strip()
                next_para = paragraphs[i + 1].strip()

                # Check if current paragraph ends mid-sentence (no punctuation)
                if current and not current[-1] in '.!?»"':
                    # And next paragraph doesn't start with continuation
                    if next_para and next_para[0].islower():
                        broken_sentences += 1

        if broken_sentences > 0:
            print(f"  ⚠️  {broken_sentences} potential broken sentences (text box boundaries)")
        else:
            print(f"  ✅ Text flow appears coherent")

        # Check for topic mixing (simple heuristic)
        topics = []
        text_lower = text.lower()
        if 'madeira' in text_lower or 'eucalipto' in text_lower:
            topics.append('wood')
        if 'manuscrito' in text_lower or 'missal' in text_lower:
            topics.append('manuscript')
        if 'mitterrand' in text_lower:
            topics.append('mitterrand')
        if 'áfrica' in text_lower or 'africa' in text_lower:
            topics.append('africa')
        if 'tailândia' in text_lower or 'thailand' in text_lower:
            topics.append('thailand')

        if len(topics) > 1:
            print(f"  ❌ CONTAMINATION: {topics}")
        elif topics:
            print(f"  ✅ Pure topic: {topics[0]}")
        else:
            print(f"  ℹ️  Other topic")

        print()


def main():
    if len(sys.argv) > 1:
        page_num = sys.argv[1]
    else:
        # Test first 5 pages
        page_nums = ['001', '002', '003', '004', '005']

        print("="*80)
        print("READING ORDER VALIDATION - 5 Pages")
        print("="*80)

        for page_num in page_nums:
            article_file = Path(f'articles_output/1974/04/24/page_{page_num}_articles.json')
            ocr_file = Path(f'ocr_output/1974/04/24/page_{page_num}_ocr.json')

            if article_file.exists() and ocr_file.exists():
                check_reading_order(article_file, ocr_file)
            else:
                print(f"\nPage {page_num}: Files not found")

        return

    # Single page test
    article_file = Path(f'articles_output/1974/04/24/page_{page_num}_articles.json')
    ocr_file = Path(f'ocr_output/1974/04/24/page_{page_num}_ocr.json')

    if article_file.exists() and ocr_file.exists():
        check_reading_order(article_file, ocr_file)
    else:
        print(f"Files not found for page {page_num}")


if __name__ == '__main__':
    main()
