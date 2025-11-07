#!/usr/bin/env python3
"""
Diário de Lisboa Archive Scraper
Downloads all editions of Diário de Lisboa newspaper from Casa Comum (1921-1990)
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image
from io import BytesIO


class DiarioLisboaScraper:
    """Main scraper class for Diário de Lisboa archive"""

    BASE_URL = "http://casacomum.org"
    YEAR_URL = "http://casacomum.org/cc/diario_de_lisboa/mes?ano={year}"
    MONTH_URL = "http://casacomum.org/cc/diario_de_lisboa/dia?ano={year}&mes={month:02d}"
    VISUALIZER_URL = "http://casacomum.org/cc/visualizador?pasta={pasta}"

    def __init__(self,
                 data_dir: str = "data",
                 delay: float = 1.0,
                 retry_attempts: int = 3,
                 resume: bool = False):
        """
        Initialize the scraper

        Args:
            data_dir: Directory to save downloaded newspapers
            delay: Delay between requests in seconds
            retry_attempts: Number of retry attempts for failed downloads
            resume: Resume from last checkpoint
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.delay = delay
        self.retry_attempts = retry_attempts
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DiarioLisboaScraper/1.0; Academic Research)'
        })

        # Setup logging
        self.setup_logging()

        # Progress tracking
        self.progress_file = Path("progress.json")
        self.progress = self.load_progress() if resume else {}

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_progress(self) -> Dict:
        """Load progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}

    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def generate_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Generate all valid newspaper dates (excluding Sundays)

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of valid dates
        """
        dates = []
        current = start_date

        while current <= end_date:
            # Skip Sundays (weekday 6)
            if current.weekday() != 6:
                dates.append(current)
            current += timedelta(days=1)

        return dates

    def get_month_days(self, year: int, month: int) -> List[Dict]:
        """
        Get all available newspaper days for a specific month

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            List of dictionaries with day info and pasta codes
        """
        url = self.MONTH_URL.format(year=year, month=month)
        self.logger.info(f"Fetching month calendar: {year}-{month:02d}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')

            days = []
            # Find all links to visualizer
            for link in soup.find_all('a', href=lambda x: x and 'visualizador?pasta=' in x):
                href = link.get('href')
                if href:
                    # Extract pasta code
                    pasta = href.split('pasta=')[1].split('&')[0]
                    # Extract date from link text or image alt
                    date_text = link.get_text(strip=True) or link.find('img', alt=True)['alt'] if link.find('img') else ""

                    # Parse day number from date text
                    import re
                    day_match = re.search(r'\b(\d{1,2})\s+de\s+', date_text)
                    if day_match:
                        day = int(day_match.group(1))
                        days.append({
                            'year': year,
                            'month': month,
                            'day': day,
                            'pasta': pasta,
                            'date': datetime(year, month, day)
                        })

            return days

        except Exception as e:
            self.logger.error(f"Error fetching month {year}-{month:02d}: {e}")
            return []

    def get_page_urls(self, pasta: str) -> List[str]:
        """
        Get all page image URLs from a newspaper edition

        Args:
            pasta: The pasta code for the edition

        Returns:
            List of image URLs for all pages
        """
        url = self.VISUALIZER_URL.format(pasta=pasta)
        self.logger.debug(f"Fetching visualizer: {pasta}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            content = response.text

            import re
            import json

            # Try to extract the JavaScript paginas data
            paginas_match = re.search(r"paginas\s*=\s*({.*?});", content, re.DOTALL)
            if paginas_match:
                try:
                    # Clean up the JavaScript object to make it valid JSON
                    js_data = paginas_match.group(1)
                    # Replace single quotes with double quotes
                    js_data = re.sub(r"'([^']+)':", r'"\1":', js_data)
                    js_data = js_data.replace("'", '"')
                    # Remove backslashes from escaped forward slashes
                    js_data = js_data.replace(r'\/', '/')
                    # Parse the JSON
                    data = json.loads(js_data)

                    image_urls = []
                    if 'pag' in data:
                        for page in data['pag']:
                            # Use urlD2 for high quality image
                            if page.get('urlD2'):
                                url_d2 = self.BASE_URL + '/aebdocs/' + page['urlD2']
                                image_urls.append(url_d2)

                    if image_urls:
                        self.logger.debug(f"Found {len(image_urls)} pages from JavaScript data")
                        return image_urls
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Failed to parse JavaScript data: {e}, falling back to HTML parsing")

            # Fallback to HTML parsing if JavaScript parsing fails
            soup = BeautifulSoup(content, 'lxml')
            image_urls = []

            # Method 1: Look for thumbnail links and extract their high-res versions
            for thumb_link in soup.find_all('a', class_='linkthumb'):
                img = thumb_link.find('img')
                if img and img.get('src'):
                    thumb_url = img['src']
                    # Thumbnails use d3, high quality uses d2
                    # The actual format (jpg/png) should be preserved from the thumbnail URL
                    if '/d3/' in thumb_url:
                        # For d3 thumbnails, we need to check what format the d2 version uses
                        # Based on the pasta code, try to determine the correct format
                        base_url = thumb_url.replace('/d3/', '/d2/')

                        # The thumbnail is always .jpg, but d2 might be .png
                        # We'll need to handle this dynamically
                        if '_D3.jpg' in base_url:
                            # Try PNG first for d2 (more common in newer archives)
                            # But keep the original JPG as fallback
                            hq_url = base_url.replace('_D3.jpg', '_D2.png')
                            # Store both possibilities, we'll try PNG first
                            image_urls.append(hq_url)
                        else:
                            image_urls.append(base_url)

            # Method 2: If no thumbnails, at least get the main image
            if not image_urls:
                main_img = soup.find('img', id='pagina_actual')
                if main_img and main_img.get('src'):
                    image_urls.append(main_img['src'])

            return image_urls

        except Exception as e:
            self.logger.error(f"Error fetching visualizer {pasta}: {e}")
            return []

    def download_image(self, url: str, save_path: Path, retry: int = 0) -> bool:
        """
        Download an image from URL

        Args:
            url: Image URL
            save_path: Path to save the image
            retry: Current retry attempt

        Returns:
            Success status
        """
        if save_path.exists():
            self.logger.debug(f"Skipping existing file: {save_path}")
            return True

        try:
            if not url.startswith('http'):
                url = self.BASE_URL + url

            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()

            # Verify it's an image
            img_data = response.content
            img = Image.open(BytesIO(img_data))
            img.verify()

            # Save the image
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(img_data)

            self.logger.debug(f"Downloaded: {save_path}")
            return True

        except requests.HTTPError as e:
            # If we get a 404 and it's a PNG URL, try JPG (or vice versa)
            if e.response.status_code == 404 and retry == 0:
                alternate_url = None
                alternate_path = None

                if '_D2.png' in url:
                    alternate_url = url.replace('_D2.png', '_D2.jpg')
                    alternate_path = Path(str(save_path).replace('.png', '.jpg'))
                elif '_D2.jpg' in url:
                    alternate_url = url.replace('_D2.jpg', '_D2.png')
                    alternate_path = Path(str(save_path).replace('.jpg', '.png'))

                if alternate_url:
                    self.logger.debug(f"Trying alternate format: {alternate_url}")
                    return self.download_image(alternate_url, alternate_path, retry + 1)

            # Standard retry logic
            if retry < self.retry_attempts:
                self.logger.warning(f"Retry {retry + 1}/{self.retry_attempts} for {url}")
                time.sleep(2 ** retry)  # Exponential backoff
                return self.download_image(url, save_path, retry + 1)
            else:
                self.logger.error(f"Failed to download {url}: {e}")
                return False

        except Exception as e:
            if retry < self.retry_attempts:
                self.logger.warning(f"Retry {retry + 1}/{self.retry_attempts} for {url}")
                time.sleep(2 ** retry)  # Exponential backoff
                return self.download_image(url, save_path, retry + 1)
            else:
                self.logger.error(f"Failed to download {url}: {e}")
                return False

    def scrape_date_range(self, start_date: datetime, end_date: datetime):
        """
        Scrape newspapers for a date range

        Args:
            start_date: Start date
            end_date: End date
        """
        # Group dates by month for efficient fetching
        dates_by_month = {}
        for date in self.generate_dates(start_date, end_date):
            key = (date.year, date.month)
            if key not in dates_by_month:
                dates_by_month[key] = []
            dates_by_month[key].append(date)

        total_dates = sum(len(dates) for dates in dates_by_month.values())
        self.logger.info(f"Scraping {total_dates} dates from {start_date.date()} to {end_date.date()}")

        with tqdm(total=total_dates, desc="Overall Progress") as pbar:
            for (year, month), dates in dates_by_month.items():
                # Get available days for this month
                available_days = self.get_month_days(year, month)
                available_dict = {d['day']: d for d in available_days}

                for date in dates:
                    date_str = date.strftime("%Y-%m-%d")

                    # Skip if already processed
                    if date_str in self.progress and self.progress[date_str].get('completed'):
                        self.logger.info(f"Skipping completed date: {date_str}")
                        pbar.update(1)
                        continue

                    # Check if this day has a newspaper
                    if date.day not in available_dict:
                        self.logger.info(f"No newspaper for {date_str} (holiday or no publication)")
                        self.progress[date_str] = {'completed': True, 'pages': 0, 'status': 'no_publication'}
                        pbar.update(1)
                        continue

                    day_info = available_dict[date.day]
                    pasta = day_info['pasta']

                    # Get all page URLs
                    page_urls = self.get_page_urls(pasta)
                    if not page_urls:
                        self.logger.warning(f"No pages found for {date_str}")
                        self.progress[date_str] = {'completed': True, 'pages': 0, 'status': 'no_pages'}
                        pbar.update(1)
                        continue

                    # Create directory for this date
                    date_dir = self.data_dir / f"{year:04d}" / f"{month:02d}" / f"{date.day:02d}"
                    date_dir.mkdir(parents=True, exist_ok=True)

                    # Download all pages
                    self.logger.info(f"Downloading {len(page_urls)} pages for {date_str}")
                    success_count = 0

                    for i, page_url in enumerate(page_urls, 1):
                        # Determine file extension from URL
                        if '.png' in page_url.lower():
                            ext = 'png'
                        else:
                            ext = 'jpg'
                        page_path = date_dir / f"page_{i:03d}.{ext}"
                        if self.download_image(page_url, page_path):
                            success_count += 1
                        time.sleep(self.delay)  # Be polite

                    # Save metadata
                    metadata = {
                        'date': date_str,
                        'pasta': pasta,
                        'total_pages': len(page_urls),
                        'downloaded_pages': success_count,
                        'page_urls': page_urls
                    }
                    with open(date_dir / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)

                    # Update progress
                    self.progress[date_str] = {
                        'completed': True,
                        'pages': success_count,
                        'total_pages': len(page_urls),
                        'status': 'success' if success_count == len(page_urls) else 'partial'
                    }
                    self.save_progress()

                    pbar.update(1)
                    self.logger.info(f"Completed {date_str}: {success_count}/{len(page_urls)} pages")

    def run(self, start_date: datetime, end_date: datetime):
        """
        Main entry point for the scraper

        Args:
            start_date: Start date
            end_date: End date
        """
        self.logger.info("Starting Diário de Lisboa scraper")
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

        try:
            self.scrape_date_range(start_date, end_date)
            self.logger.info("Scraping completed successfully")

            # Generate summary
            completed = sum(1 for v in self.progress.values() if v.get('completed'))
            total_pages = sum(v.get('pages', 0) for v in self.progress.values())
            self.logger.info(f"Summary: {completed} dates processed, {total_pages} pages downloaded")

        except KeyboardInterrupt:
            self.logger.info("Scraping interrupted by user")
            self.save_progress()
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"Fatal error: {e}")
            self.save_progress()
            raise


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Download Diário de Lisboa newspaper archive from Casa Comum"
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='1921-04-07',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='1921-04-14',  # Default to first week for testing
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory to save newspapers'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds'
    )
    parser.add_argument(
        '--retry',
        type=int,
        default=3,
        help='Number of retry attempts for failed downloads'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode - download only one day'
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    # Test mode - only download one day
    if args.test:
        end_date = start_date
        print(f"TEST MODE: Downloading only {start_date.date()}")

    # Create and run scraper
    scraper = DiarioLisboaScraper(
        data_dir=args.data_dir,
        delay=args.delay,
        retry_attempts=args.retry,
        resume=args.resume
    )

    scraper.run(start_date, end_date)


if __name__ == '__main__':
    main()