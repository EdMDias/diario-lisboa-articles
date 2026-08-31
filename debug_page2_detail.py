#!/usr/bin/env python3
"""Debug page 002 article extraction in detail"""

from article_extractor import ArticleExtractor
from pathlib import Path
import json

print('='*80)
print('PAGE 002 DETAILED DIAGNOSIS')
print('='*80 + '\n')

extractor = ArticleExtractor()

# Load and filter
bboxes = extractor.load_ocr_file(Path('ocr_output/1974/04/24/page_002_ocr.json'))
content = extractor.filter_metadata_boxes(bboxes)
content = extractor.filter_masthead(content)

print(f'Boxes after filtering: {len(content)}\n')

# Embed all
embeddings = {}
for bbox in content:
    if bbox.text.strip() and bbox.category != 'Picture':
        embeddings[bbox.index] = extractor.get_embedding(bbox.text)

print(f'Embeddings computed: {len(embeddings)}\n')

# Get titles
titles = [b for b in content if b.category in extractor.title_categories]

print(f'=== TITLES: {len(titles)} ===\n')
for article_num, title in enumerate(titles):
    print(f'Article {article_num}:')
    print(f'  OCR Index: {title.index}')
    print(f'  Title Text: {title.text[:70]}')
    print()

# Compute affinities
title_affinity = extractor.compute_title_affinity_map(content, titles, embeddings)

print(f'=== TITLE AFFINITIES: {len(title_affinity)} text boxes ===\n')

# Group boxes by best title
by_article = {i: [] for i in range(len(titles))}

for box in content:
    if box.category == 'Text' and box.index in title_affinity:
        best_title_idx, score = title_affinity[box.index]

        # Find which article number this corresponds to
        for article_num, title in enumerate(titles):
            if title.index == best_title_idx:
                by_article[article_num].append((box, score))
                break

for article_num in range(len(titles)):
    print(f'Boxes preferring Article {article_num} ({titles[article_num].text[:30]}):')
    boxes_list = by_article[article_num]
    print(f'  Count: {len(boxes_list)}')

    for box, score in boxes_list[:3]:
        print(f'    Score={score:.3f}: {box.text[:60]}')

    print()

print('=== GRAPH CONNECTIVITY ===\n')

columns = extractor.column_detector.detect_columns(content)
G, _ = extractor.build_article_graph(content, columns)

for article_num, title in enumerate(titles):
    neighbors = list(G.neighbors(title.index))
    print(f'Title {article_num} (OCR idx {title.index}): {len(neighbors)} direct neighbors')

    # Check if neighbors match title affinity
    correct_affinity = 0
    for n_idx in neighbors:
        if n_idx in title_affinity:
            best_title_ocr_idx, score = title_affinity[n_idx]
            if best_title_ocr_idx == title.index:
                correct_affinity += 1

    print(f'  Neighbors with correct affinity: {correct_affinity}/{len([n for n in neighbors if n in title_affinity])}')
    print()
