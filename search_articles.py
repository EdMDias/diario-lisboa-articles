#!/usr/bin/env python3
"""
Search Articles CLI for Diário de Lisboa
Interactive search interface for the article database
"""

import sqlite3
import argparse
from pathlib import Path
from typing import List, Tuple


def search_articles(conn: sqlite3.Connection, query: str, limit: int = 10) -> List[Tuple]:
    """
    Full-text search in articles

    Args:
        conn: Database connection
        query: Search query
        limit: Maximum results

    Returns:
        List of (article_id, title, date, snippet, text_length)
    """
    cursor = conn.execute('''
        SELECT
            a.article_id,
            a.title,
            a.date,
            snippet(articles_fts, 2, '<b>', '</b>', '...', 32) as snippet,
            LENGTH(a.text) as text_len
        FROM articles_fts
        JOIN articles a ON articles_fts.rowid = a.rowid
        WHERE articles_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    ''', (query, limit))

    return cursor.fetchall()


def get_article_by_id(conn: sqlite3.Connection, article_id: str) -> dict:
    """Retrieve full article by ID"""
    cursor = conn.execute('''
        SELECT
            article_id, title, text, page, date,
            num_paragraphs, num_images,
            continuation_to, continuation_from
        FROM articles
        WHERE article_id = ?
    ''', (article_id,))

    row = cursor.fetchone()
    if not row:
        return None

    article = {
        'article_id': row[0],
        'title': row[1],
        'text': row[2],
        'page': row[3],
        'date': row[4],
        'num_paragraphs': row[5],
        'num_images': row[6],
        'continuation_to': row[7],
        'continuation_from': row[8],
    }

    # Get captions
    cursor = conn.execute('SELECT caption FROM captions WHERE article_id = ?', (article_id,))
    article['captions'] = [row[0] for row in cursor.fetchall()]

    return article


def display_search_results(results: List[Tuple]):
    """Display search results in a readable format"""
    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} result(s):\n")
    print("=" * 80)

    for i, (article_id, title, date, snippet, text_len) in enumerate(results, 1):
        print(f"\n{i}. {title[:70]}...")
        print(f"   Date: {date} | Length: {text_len} chars | ID: {article_id}")
        print(f"   {snippet}")
        print("-" * 80)


def display_article(article: dict):
    """Display full article"""
    if not article:
        print("Article not found.")
        return

    print("\n" + "=" * 80)
    print(f"TITLE: {article['title']}")
    print("=" * 80)
    print(f"Date: {article['date']}")
    print(f"Page: {article['page']}")
    print(f"Paragraphs: {article['num_paragraphs']} | Images: {article['num_images']}")

    if article['continuation_from']:
        print(f"⬅️  Continued from: {article['continuation_from']}")
    if article['continuation_to']:
        print(f"➡️  Continues to: {article['continuation_to']}")

    print("\n" + "-" * 80)
    print(article['text'])
    print("-" * 80)

    if article['captions']:
        print("\nImage captions:")
        for caption in article['captions']:
            print(f"  📷 {caption}")


def interactive_search(db_path: Path):
    """Interactive search mode"""
    conn = sqlite3.connect(db_path)

    print("=" * 80)
    print("DIÁRIO DE LISBOA - ARTICLE SEARCH")
    print("=" * 80)
    print("\nCommands:")
    print("  - Enter search query to search")
    print("  - Type article number to view full article")
    print("  - Type 'quit' or 'exit' to quit")
    print("=" * 80)

    last_results = []

    while True:
        try:
            query = input("\nSearch> ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                break

            # Check if it's a number (viewing previous result)
            if query.isdigit():
                idx = int(query) - 1
                if 0 <= idx < len(last_results):
                    article_id = last_results[idx][0]
                    article = get_article_by_id(conn, article_id)
                    display_article(article)
                else:
                    print(f"Invalid article number. Please enter 1-{len(last_results)}")
                continue

            # Otherwise, perform search
            results = search_articles(conn, query, limit=20)
            last_results = results
            display_search_results(results)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Search Diário de Lisboa articles')
    parser.add_argument('--db', type=Path, default=Path('articles.db'),
                        help='Database file (default: articles.db)')
    parser.add_argument('--query', type=str, help='Search query (non-interactive)')
    parser.add_argument('--article', type=str, help='Show article by ID')
    parser.add_argument('--limit', type=int, default=10, help='Maximum results (default: 10)')

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database {args.db} does not exist")
        print(f"Create it first with: python create_article_database.py <json_file>")
        return

    conn = sqlite3.connect(args.db)

    # Show specific article
    if args.article:
        article = get_article_by_id(conn, args.article)
        display_article(article)
        conn.close()
        return

    # Non-interactive search
    if args.query:
        results = search_articles(conn, args.query, limit=args.limit)
        display_search_results(results)
        conn.close()
        return

    # Interactive mode
    interactive_search(args.db)


if __name__ == '__main__':
    main()
