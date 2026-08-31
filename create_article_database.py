#!/usr/bin/env python3
"""
Create SQLite Database from Extracted Articles
Enables full-text search and efficient querying
"""

import sqlite3
import json
from pathlib import Path
import argparse
from typing import List, Dict


def create_database_schema(conn: sqlite3.Connection):
    """Create database tables with full-text search support"""

    # Main articles table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            article_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            page TEXT NOT NULL,
            date TEXT NOT NULL,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            day INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            num_paragraphs INTEGER DEFAULT 0,
            num_images INTEGER DEFAULT 0,
            continuation_to TEXT,
            continuation_from TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Full-text search virtual table (FTS5)
    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
            article_id UNINDEXED,
            title,
            text,
            content='articles',
            content_rowid='rowid'
        )
    ''')

    # Triggers to keep FTS table in sync
    conn.execute('''
        CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
            INSERT INTO articles_fts(rowid, article_id, title, text)
            VALUES (new.rowid, new.article_id, new.title, new.text);
        END
    ''')

    conn.execute('''
        CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
            DELETE FROM articles_fts WHERE rowid = old.rowid;
        END
    ''')

    conn.execute('''
        CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
            UPDATE articles_fts SET title = new.title, text = new.text
            WHERE rowid = old.rowid;
        END
    ''')

    # Captions table (many-to-one with articles)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id TEXT NOT NULL,
            caption TEXT NOT NULL,
            FOREIGN KEY (article_id) REFERENCES articles (article_id)
        )
    ''')

    # Indexes for common queries
    conn.execute('CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(year, month, day)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_articles_page ON articles(page)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_articles_continuation_to ON articles(continuation_to)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_articles_continuation_from ON articles(continuation_from)')

    conn.commit()


def parse_page_info(page_id: str) -> Dict:
    """
    Extract date and page number from page_id
    Example: "1974/04/24/page_001" -> {year: 1974, month: 4, day: 24, page_num: 1}
    """
    parts = page_id.split('/')
    if len(parts) >= 4:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        page_str = parts[3]  # "page_001"
        page_num = int(page_str.split('_')[1])

        return {
            'year': year,
            'month': month,
            'day': day,
            'page_number': page_num,
            'date': f"{year}-{month:02d}-{day:02d}"
        }
    else:
        # Fallback
        return {
            'year': 0,
            'month': 0,
            'day': 0,
            'page_number': 0,
            'date': 'unknown'
        }


def import_articles(conn: sqlite3.Connection, json_file: Path):
    """Import articles from JSON file into database"""

    with open(json_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    imported_count = 0
    skipped_count = 0

    for article in articles:
        article_id = article['article_id']

        # Check if already exists
        cursor = conn.execute('SELECT article_id FROM articles WHERE article_id = ?', (article_id,))
        if cursor.fetchone():
            skipped_count += 1
            continue

        # Parse page info
        page_info = parse_page_info(article['page'])

        # Insert article
        conn.execute('''
            INSERT INTO articles (
                article_id, title, text, page, date,
                year, month, day, page_number,
                num_paragraphs, num_images,
                continuation_to, continuation_from
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article_id,
            article['title'],
            article['text'],
            article['page'],
            page_info['date'],
            page_info['year'],
            page_info['month'],
            page_info['day'],
            page_info['page_number'],
            article['num_paragraphs'],
            article['num_images'],
            article['continuation_to'],
            article['continuation_from']
        ))

        # Insert captions
        for caption in article.get('captions', []):
            if caption.strip():
                conn.execute('''
                    INSERT INTO captions (article_id, caption)
                    VALUES (?, ?)
                ''', (article_id, caption))

        imported_count += 1

    conn.commit()
    return imported_count, skipped_count


def generate_database_stats(conn: sqlite3.Connection):
    """Generate statistics about the database"""
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)

    # Total articles
    cursor = conn.execute('SELECT COUNT(*) FROM articles')
    total_articles = cursor.fetchone()[0]
    print(f"\nTotal articles: {total_articles:,}")

    # Articles by year
    cursor = conn.execute('''
        SELECT year, COUNT(*) as count
        FROM articles
        WHERE year > 0
        GROUP BY year
        ORDER BY year
    ''')
    print("\nArticles by year:")
    for year, count in cursor.fetchall():
        print(f"  {year}: {count:,} articles")

    # Articles with continuations
    cursor = conn.execute('''
        SELECT COUNT(*) FROM articles
        WHERE continuation_to IS NOT NULL OR continuation_from IS NOT NULL
    ''')
    continuation_count = cursor.fetchone()[0]
    print(f"\nArticles with continuations: {continuation_count:,} ({continuation_count/total_articles*100:.1f}%)")

    # Average text length
    cursor = conn.execute('SELECT AVG(LENGTH(text)) FROM articles WHERE LENGTH(text) > 0')
    avg_length = cursor.fetchone()[0]
    print(f"Average text length: {avg_length:.0f} characters")

    # Longest article
    cursor = conn.execute('''
        SELECT article_id, title, LENGTH(text) as len, date
        FROM articles
        ORDER BY len DESC
        LIMIT 1
    ''')
    article_id, title, length, date = cursor.fetchone()
    print(f"\nLongest article: {length:,} characters")
    print(f"  Title: {title[:80]}...")
    print(f"  Date: {date}")
    print(f"  ID: {article_id}")


def demo_search(conn: sqlite3.Connection):
    """Demonstrate full-text search capabilities"""
    print("\n" + "=" * 80)
    print("SEARCH DEMO")
    print("=" * 80)

    # Example searches
    queries = [
        'revolução',
        'Mitterrand',
        'África do Sul'
    ]

    for query in queries:
        print(f"\nSearching for: \"{query}\"")

        cursor = conn.execute('''
            SELECT a.article_id, a.title, a.date, LENGTH(a.text) as text_len
            FROM articles_fts
            JOIN articles a ON articles_fts.rowid = a.rowid
            WHERE articles_fts MATCH ?
            ORDER BY rank
            LIMIT 5
        ''', (query,))

        results = cursor.fetchall()
        print(f"Found {len(results)} result(s):")

        for article_id, title, date, text_len in results:
            print(f"  - {title[:60]}... ({date}, {text_len} chars)")


def main():
    parser = argparse.ArgumentParser(description='Create article search database')
    parser.add_argument('json_file', type=Path, help='JSON file with articles')
    parser.add_argument('--db', type=Path, default=Path('articles.db'),
                        help='Output database file (default: articles.db)')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing database instead of recreating')

    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"Error: {args.json_file} does not exist")
        return

    # Create/open database
    if not args.append and args.db.exists():
        print(f"Removing existing database: {args.db}")
        args.db.unlink()

    print(f"{'Appending to' if args.append else 'Creating'} database: {args.db}")
    conn = sqlite3.connect(args.db)

    # Create schema
    create_database_schema(conn)

    # Import articles
    print(f"\nImporting articles from: {args.json_file}")
    imported, skipped = import_articles(conn, args.json_file)

    print(f"✓ Imported: {imported:,} articles")
    if skipped > 0:
        print(f"  Skipped (already exists): {skipped:,} articles")

    # Generate statistics
    generate_database_stats(conn)

    # Demo search
    demo_search(conn)

    conn.close()

    print(f"\n✓ Database saved to: {args.db}")
    print(f"\nYou can now search articles using SQL queries:")
    print(f"  sqlite3 {args.db}")
    print(f"  SELECT title, date FROM articles WHERE articles.rowid IN (")
    print(f"    SELECT rowid FROM articles_fts WHERE articles_fts MATCH 'your search term'")
    print(f"  );")


if __name__ == '__main__':
    main()
