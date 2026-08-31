#!/usr/bin/env python3
"""
Batch Article Extraction for Diário de Lisboa
Processes multiple OCR files and extracts articles from all pages
"""

import json
from pathlib import Path
from typing import List, Dict
import argparse
from collections import defaultdict
from article_extractor import ArticleExtractor, Article


class BatchArticleProcessor:
    """Process multiple pages and link cross-page continuations"""

    def __init__(self, extractor: ArticleExtractor, articles_output_dir: Path = None, ocr_base_dir: Path = None):
        self.extractor = extractor
        self.articles_output_dir = articles_output_dir or Path('articles_output')
        self.ocr_base_dir = ocr_base_dir or Path('ocr_output')
        self.articles_by_page: Dict[str, List[Article]] = {}
        self.continuation_map: Dict[str, List[Article]] = defaultdict(list)

    def process_directory(self, ocr_dir: Path) -> List[Article]:
        """
        Process all OCR files in a directory

        Args:
            ocr_dir: Directory containing OCR JSON files

        Returns:
            List of all extracted articles
        """
        # Find all OCR files
        ocr_files = sorted(ocr_dir.rglob('*_ocr.json'))

        print(f"Found {len(ocr_files)} OCR files in {ocr_dir}")

        all_articles = []

        for ocr_file in ocr_files:
            print(f"Processing: {ocr_file.relative_to(ocr_dir)}")

            # Extract articles
            articles = self.extractor.extract_articles(ocr_file)
            all_articles.extend(articles)

            # Store by page for continuation linking
            page_id = str(ocr_file.relative_to(Path('ocr_output'))).replace('_ocr.json', '')
            self.articles_by_page[page_id] = articles

            # Track continuations
            for article in articles:
                if article.continuation_to:
                    self.continuation_map[article.continuation_to].append(article)

            # Save per-page articles - mirror OCR directory structure
            try:
                relative_path = ocr_file.parent.relative_to(self.ocr_base_dir)
                output_dir = self.articles_output_dir / relative_path
            except ValueError:
                # OCR file not in expected base directory, use articles_output root
                print(f"Warning: {ocr_file} not in {self.ocr_base_dir}, saving to {self.articles_output_dir}")
                output_dir = self.articles_output_dir

            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save to JSON
            output_file = output_dir / (ocr_file.stem.replace('_ocr', '_articles') + '.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([a.to_dict() for a in articles], f, ensure_ascii=False, indent=2)

        print(f"\n✓ Extracted {len(all_articles)} articles from {len(ocr_files)} pages")

        return all_articles

    def analyze_continuations(self):
        """Analyze continuation patterns"""
        print("\n" + "=" * 80)
        print("CONTINUATION ANALYSIS")
        print("=" * 80)

        total_continuations = sum(len(arts) for arts in self.continuation_map.values())
        print(f"\nTotal articles with continuation markers: {total_continuations}")

        if self.continuation_map:
            print(f"Unique target pages: {len(self.continuation_map)}")
            print("\nContinuation targets:")
            for target, articles in sorted(self.continuation_map.items()):
                print(f"  {target}: {len(articles)} article(s)")
                for article in articles[:3]:  # Show first 3
                    print(f"    - {article.title[:60]}...")

    def export_to_database(self, articles: List[Article], output_file: Path):
        """Export all articles to a single JSON database"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([a.to_dict() for a in articles], f, ensure_ascii=False, indent=2)

        print(f"\n✓ Exported {len(articles)} articles to: {output_file}")

    def generate_statistics(self, articles: List[Article]):
        """Generate statistics about extracted articles"""
        print("\n" + "=" * 80)
        print("EXTRACTION STATISTICS")
        print("=" * 80)

        total_articles = len(articles)
        articles_with_titles = sum(1 for a in articles if a.title)
        articles_with_text = sum(1 for a in articles if a.text)
        articles_with_images = sum(1 for a in articles if a.images)
        articles_with_continuations = sum(1 for a in articles if a.continuation_to or a.continuation_from)

        # Calculate text statistics
        text_lengths = [len(a.text) for a in articles if a.text]
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        max_text_length = max(text_lengths) if text_lengths else 0

        # Paragraph statistics
        paragraph_counts = [len(a.bboxes) for a in articles]
        avg_paragraphs = sum(paragraph_counts) / len(paragraph_counts) if paragraph_counts else 0

        print(f"\nTotal articles: {total_articles}")
        print(f"Articles with titles: {articles_with_titles} ({articles_with_titles/total_articles*100:.1f}%)")
        print(f"Articles with text: {articles_with_text} ({articles_with_text/total_articles*100:.1f}%)")
        print(f"Articles with images: {articles_with_images} ({articles_with_images/total_articles*100:.1f}%)")
        print(f"Articles with continuations: {articles_with_continuations} ({articles_with_continuations/total_articles*100:.1f}%)")

        print(f"\nText statistics:")
        print(f"  Average text length: {avg_text_length:.0f} characters")
        print(f"  Maximum text length: {max_text_length:,} characters")
        print(f"  Average paragraphs per article: {avg_paragraphs:.1f}")

        # Pages statistics
        pages = set(a.page for a in articles)
        print(f"\nPages processed: {len(pages)}")
        print(f"Average articles per page: {total_articles/len(pages):.1f}")

        # Top 10 longest articles
        print("\nTop 10 longest articles:")
        longest = sorted(articles, key=lambda a: len(a.text), reverse=True)[:10]
        for i, article in enumerate(longest, 1):
            print(f"  {i}. {article.title[:60]}... ({len(article.text):,} chars, {article.page})")


def main():
    parser = argparse.ArgumentParser(description='Batch article extraction for Diário de Lisboa')
    parser.add_argument('ocr_dir', type=Path, help='Directory containing OCR JSON files')
    parser.add_argument('--output', type=Path, help='Output JSON file for article database')
    parser.add_argument('--articles-output-dir', type=Path, default=Path('articles_output'),
                        help='Output directory for extracted articles (default: articles_output)')
    parser.add_argument('--ocr-base-dir', type=Path, default=Path('ocr_output'),
                        help='Base directory for OCR files (default: ocr_output)')
    parser.add_argument('--min-column-gap', type=float, default=50,
                        help='Minimum gap to separate columns (pixels)')
    parser.add_argument('--min-article-gap', type=float, default=100,
                        help='Minimum gap to separate articles (pixels)')

    args = parser.parse_args()

    if not args.ocr_dir.exists():
        print(f"Error: Directory {args.ocr_dir} does not exist")
        return

    # Initialize extractor
    extractor = ArticleExtractor(
        min_column_gap=args.min_column_gap,
        min_article_gap=args.min_article_gap
    )

    # Process batch
    processor = BatchArticleProcessor(
        extractor,
        articles_output_dir=args.articles_output_dir,
        ocr_base_dir=args.ocr_base_dir
    )
    articles = processor.process_directory(args.ocr_dir)

    # Analyze continuations
    processor.analyze_continuations()

    # Generate statistics
    processor.generate_statistics(articles)

    # Export to database if requested
    if args.output:
        processor.export_to_database(articles, args.output)
    else:
        # Default output - save to articles_output directory
        output_file = args.articles_output_dir / 'articles_database.json'
        processor.export_to_database(articles, output_file)


if __name__ == '__main__':
    main()
