# Diário de Lisboa Scraper - Technical Planning Document

## Project Goal
Create a comprehensive web scraper to download all editions of Diário de Lisboa newspaper from Casa Comum's digital archive, covering publications from April 7, 1921 to November 30, 1990.

## Website Structure Analysis

### URL Patterns Discovered

1. **Year Navigation**
   - Pattern: `http://casacomum.org/cc/diario_de_lisboa/mes?ano=YYYY`
   - Example: `http://casacomum.org/cc/diario_de_lisboa/mes?ano=1921`

2. **Month Navigation**
   - Pattern: `http://casacomum.org/cc/diario_de_lisboa/dia?ano=YYYY&mes=MM`
   - Example: `http://casacomum.org/cc/diario_de_lisboa/dia?ano=1921&mes=04`

3. **Day/Edition Viewer**
   - Pattern: `http://casacomum.org/cc/visualizador?pasta=XXXXX.XXX.XXXXX`
   - Example: `http://casacomum.org/cc/visualizador?pasta=05739.003.00364`

4. **Image URLs**
   - High Quality: `http://casacomum.org/aebdocs/XX/XXXXX/XXXXX.XXX.XXXXX/d2/[folder]_p[page]_id[id]_D2.jpg`
   - Thumbnail: `http://casacomum.org/aebdocs/XX/XXXXX/XXXXX.XXX.XXXXX/d3/[folder]_p[page]_id[id]_D3.jpg`
   - Example: `http://casacomum.org/aebdocs/05/05739/05739.003.00364/d2/05739.003.00364_p0001_id002262886_D2.jpg`

### Data Structure

- **Publication Schedule**: 6 days per week (Monday through Saturday)
- **No Sunday editions**
- **Pages per edition**: Typically 8 pages, but varies
- **Special dates**: Some holidays may not have editions

### Archive Organization

Each year has a unique set ID:
- 1921: e_606
- 1922: e_605
- 1923: e_604
- ... (decreasing pattern)
- 1990: e_537

## Implementation Strategy

### Phase 1: Core Scraper Development

1. **Date Navigation System**
   ```python
   - Generate all valid dates (excluding Sundays)
   - Handle Portuguese holidays (when no edition exists)
   - Navigate year → month → day hierarchy
   ```

2. **Page Discovery**
   ```python
   - For each day, extract the visualizer URL
   - Parse visualizer page to find all page image URLs
   - Identify total pages for each edition
   ```

3. **Download Manager**
   ```python
   - Implement concurrent downloads (with limits)
   - Retry mechanism for failed downloads
   - Resume capability for interrupted sessions
   - Bandwidth throttling to be respectful
   ```

### Phase 2: Data Organization

**Folder Structure Design**:
```
data/
├── 1921/
│   ├── 04/
│   │   ├── 07/                    # April 7, 1921
│   │   │   ├── page_001.jpg       # First page
│   │   │   ├── page_002.jpg
│   │   │   ├── ...
│   │   │   ├── page_008.jpg       # Last page
│   │   │   └── metadata.json      # Edition metadata
│   │   ├── 08/                    # April 8, 1921
│   │   └── ...
│   └── ...
└── ...
```

### Phase 3: Features & Optimization

1. **Progress Tracking**
   - JSON file to track completed downloads
   - Resume from last successful download
   - Statistics (pages downloaded, remaining, errors)

2. **Error Handling**
   - Comprehensive logging system
   - Network error recovery
   - Missing edition handling
   - Corrupted image detection

3. **Performance Optimization**
   - Parallel downloads (configurable workers)
   - Connection pooling
   - Intelligent retry with exponential backoff
   - Cache DNS lookups

## Technical Specifications

### Required Libraries
```python
requests          # HTTP requests
beautifulsoup4    # HTML parsing
Pillow           # Image validation
tqdm             # Progress bars
python-dateutil  # Date manipulation
aiohttp          # Async HTTP (optional for performance)
```

### Configuration Parameters
```python
{
    "start_date": "1921-04-07",
    "end_date": "1990-11-30",
    "concurrent_downloads": 3,
    "retry_attempts": 3,
    "delay_between_requests": 1.0,  # seconds
    "image_quality": "d2",  # d2=high, d3=thumbnail
    "skip_existing": true,
    "validate_images": true
}
```

## Algorithm Flow

```
1. Initialize date range and create date list (excluding Sundays)
2. For each year in range:
   a. Fetch year page
   b. Extract months with available editions
3. For each month:
   a. Fetch month calendar page
   b. Extract days with editions
4. For each day:
   a. Fetch day visualizer page
   b. Extract pasta ID and page count
   c. Generate image URLs for all pages
5. For each page:
   a. Check if already downloaded
   b. Download image with retry logic
   c. Validate image integrity
   d. Save to appropriate folder
   e. Update progress tracker
6. Generate summary report
```

## Error Scenarios & Solutions

1. **Network Timeout**
   - Solution: Implement exponential backoff retry

2. **Missing Edition**
   - Solution: Log as expected, continue to next date

3. **Corrupted Download**
   - Solution: Validate image, re-download if invalid

4. **Rate Limiting**
   - Solution: Adaptive delay between requests

5. **Memory Issues**
   - Solution: Process in chunks, clear cache regularly

## Testing Strategy

### Phase 1: Unit Testing
- Date generation logic
- URL construction
- Path creation

### Phase 2: Integration Testing
- Single day download
- Week download with Sunday skip
- Month with missing editions

### Phase 3: Load Testing
- Full year download
- Resume capability
- Error recovery

## Estimated Resources

### Time Estimates
- Single day: ~30 seconds (8 pages)
- Single month: ~15 minutes (25 days avg)
- Single year: ~3 hours
- Full archive: ~210 hours (~9 days continuous)

### Storage Requirements
- Average page size: ~500 KB
- Pages per day: 8
- Days per year: ~300
- Total years: 70
- **Estimated total: ~85 GB**

## Ethical Considerations

1. **Respectful Crawling**
   - Implement delays between requests
   - Limit concurrent connections
   - Respect robots.txt if present

2. **Attribution**
   - Maintain source attribution in metadata
   - Include copyright notices
   - Reference Casa Comum and Fundação Mário Soares

3. **Usage Rights**
   - Archive states: "A publicação, total ou parcial, deste documento exige prévia autorização da entidade detentora"
   - For personal/research use only
   - Contact institution for commercial use

## Future Enhancements

1. **OCR Integration**
   - Extract searchable text from images
   - Create full-text search index

2. **Metadata Extraction**
   - Headlines and article titles
   - Publication metadata
   - Special editions marking

3. **Web Interface**
   - Local viewer for downloaded archive
   - Search functionality
   - Calendar navigation

4. **Cloud Backup**
   - Automated backup to cloud storage
   - Distributed download coordination
   - Torrent creation for archive sharing

## Monitoring & Maintenance

1. **Regular Checks**
   - Verify website structure hasn't changed
   - Update scraper if needed
   - Monitor for new additions to archive

2. **Data Validation**
   - Periodic integrity checks
   - Missing edition detection
   - Duplicate removal

3. **Documentation Updates**
   - Keep README current
   - Update progress logs
   - Maintain error documentation