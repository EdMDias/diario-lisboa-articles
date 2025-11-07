# Diário de Lisboa Archive Scraper

## Project Overview
This project downloads the complete digital archive of Diário de Lisboa newspaper from Casa Comum, covering publications from 1921 to 1990.

## Key Commands

### Run the scraper for a specific date range
```bash
python diario_lisboa_scraper.py --start-date 1921-04-07 --end-date 1921-04-30
```

### Run the full archive download
```bash
python diario_lisboa_scraper.py
```

### Resume interrupted download
```bash
python diario_lisboa_scraper.py --resume
```

### Test with a single day
```bash
python diario_lisboa_scraper.py --test --date 1921-04-07
```

## Project Structure
```
diario-lisbon/
├── CLAUDE.md           # This file - quick reference
├── PLANNING.md         # Detailed planning and technical documentation
├── diario_lisboa_scraper.py  # Main scraper script
├── requirements.txt    # Python dependencies
├── progress.json       # Tracks download progress
├── errors.log         # Error logging
└── data/              # Downloaded newspapers
    ├── 1921/
    │   ├── 04/
    │   │   ├── 07/
    │   │   │   ├── page_001.jpg
    │   │   │   ├── page_002.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── ...
```

## Quick Facts
- **Total Years**: 70 (1921-1990)
- **Publication Schedule**: 6 days/week (Monday-Saturday, no Sundays)
- **Typical Pages per Edition**: 8 pages
- **Total Estimated Files**: ~150,000+ pages
- **Source**: Casa Comum - Fundação Mário Soares e Maria Barroso

## Dependencies
```bash
pip install -r requirements.txt
```

## Notes
- The scraper implements polite crawling with delays to avoid overloading the server
- Progress is saved automatically and can be resumed if interrupted
- Already downloaded files are skipped to avoid redundancy
- High-quality images (d2 version) are downloaded by default