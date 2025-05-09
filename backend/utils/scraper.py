import requests
import logging
import time
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from typing import Dict, List, Set, Tuple, Optional, Union
from io import StringIO

# Fix import for readability
try:
    from readability.readability import Document
except ImportError:
    import subprocess
    import sys
    
    logger = logging.getLogger(__name__)
    logger.warning("Installing readability-lxml package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "readability-lxml"])
    
    # Now try importing again
    from readability.readability import Document

try:
    from .specialized_scraper import get_specialized_scraper
except ImportError:
    # For standalone testing
    def get_specialized_scraper(url):
        return None, None

# Configure logging
logger = logging.getLogger(__name__)

# Define sitemap-related namespaces
SITEMAP_NAMESPACES = {
    'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9',
    'news': 'http://www.google.com/schemas/sitemap-news/0.9',
    'xhtml': 'http://www.w3.org/1999/xhtml',
    'mobile': 'http://www.google.com/schemas/sitemap-mobile/1.0',
    'image': 'http://www.google.com/schemas/sitemap-image/1.1',
    'video': 'http://www.google.com/schemas/sitemap-video/1.1'
}

class WebScraper:
    """Advanced web scraper for RAG applications with rate limiting, content extraction and sitemap support."""
    
    def __init__(self, max_pages: int = 20, depth: int = 2, 
                 rate_limit: float = 0.5, timeout: int = 30,
                 user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"):
        """
        Initialize the web scraper.
        
        Args:
            max_pages: Maximum number of pages to scrape per domain
            depth: Maximum link depth to follow
            rate_limit: Minimum time between requests in seconds
            timeout: Request timeout in seconds
            user_agent: User agent string for requests
        """
        self.max_pages = max_pages
        self.depth = depth
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.last_request_time = 0
        
        # Add these custom headers specifically for Notion sites
        self.notion_headers = {
            **self.headers,
            "Referer": "https://www.notion.so/",
            "Origin": "https://www.notion.so",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"'
        }

    def _respect_rate_limit(self):
        """Ensure we don't send requests too quickly."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last_request)
            
        self.last_request_time = time.time()

    def _make_request(self, url: str, headers=None, retry_count=1) -> Optional[requests.Response]:
        """Make an HTTP request with rate limiting and error handling."""
        self._respect_rate_limit()
        
        if headers is None:
            headers = self.headers
        
        try:
            response = requests.get(
                url, 
                headers=headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            if retry_count > 0:
                logger.info(f"Retrying {url} ({retry_count} attempts left)...")
                time.sleep(1)  # Wait a bit before retrying
                return self._make_request(url, headers, retry_count - 1)
            return None

    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain by removing 'www.' prefix if present."""
        if domain.startswith('www.'):
            return domain[4:]
        return domain

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid and belongs to the same domain."""
        try:
            parsed = urlparse(url)
            normalized_url_domain = self._normalize_domain(parsed.netloc)
            normalized_base_domain = self._normalize_domain(base_domain)
            
            return (
                bool(parsed.netloc) and 
                bool(parsed.scheme) and
                normalized_url_domain == normalized_base_domain and
                not url.endswith(('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.js', '.css'))
            )
        except Exception:
            return False

    def _extract_text_content(self, html_content: str, url: str = "") -> str:
        """
        Extract useful text content from HTML while preserving structure.
        Uses readability-lxml for main content extraction, then processes with BeautifulSoup.
        """
        # Special case for Notion pages
        if self._is_notion_page(url):
            return self._extract_notion_content(html_content, url)
        
        # First use readability to identify the main content
        try:
            doc = Document(html_content)
            content = doc.summary()
            title = doc.title()
        except Exception as e:
            logger.error(f"Readability extraction failed: {str(e)}")
            content = html_content
            title = ""
        
        # If readability extracted something, use it, otherwise use the full HTML
        html_to_process = content if content else html_content
        
        # Process with BeautifulSoup for better structure extraction
        soup = BeautifulSoup(html_to_process, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['style', 'script', 'noscript', 'iframe', 'head', 'footer', 'nav']):
            element.decompose()
        
        result = []
        
        # Process title from readability first, then fallback to HTML tags
        if title:
            result.append(f"# {title}")
        else:
            title_tag = soup.find('h1') or soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
                if title:
                    result.append(f"# {title}")
        
        # Process headings with hierarchy
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                heading_text = heading.get_text().strip()
                if heading_text and heading_text not in result:
                    result.append(f"{'#' * i} {heading_text}")
        
        # Process paragraphs
        for p in soup.find_all('p'):
            p_text = p.get_text().strip()
            if p_text:
                result.append(p_text)
        
        # Process lists
        for ul in soup.find_all(['ul', 'ol']):
            for li in ul.find_all('li'):
                li_text = li.get_text().strip()
                if li_text:
                    result.append(f"• {li_text}")
        
        # Process tables
        for table in soup.find_all('table'):
            result.append("Table content:")
            for row in table.find_all('tr'):
                cells = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                if cells:
                    result.append(" | ".join(cells))
        
        # Process blockquotes
        for quote in soup.find_all('blockquote'):
            quote_text = quote.get_text().strip()
            if quote_text:
                # Split into lines and format as blockquote
                lines = quote_text.split("\n")
                for line in lines:
                    if line.strip():
                        result.append(f"> {line.strip()}")
        
        # Process code blocks
        for pre in soup.find_all('pre'):
            code_text = pre.get_text().strip()
            if code_text:
                result.append(f"```\n{code_text}\n```")
        
        # Process divs that might contain content (but only those with specific classes or divs without classes)
        content_div_classes = ['content', 'main', 'article', 'post', 'entry', 'page', 'text', 'body']
        for div in soup.find_all('div'):
            div_class = div.get('class', [])
            # Process divs with content classes or divs without any class
            if (div_class and any(cls in str(div_class).lower() for cls in content_div_classes)) or not div_class:
                # Only process direct text in these divs, not nested elements we've already captured
                for text in div.stripped_strings:
                    text = text.strip()
                    if text and len(text) > 20 and all(text not in r for r in result):  # Only add substantial text
                        result.append(text)
        
        # Join all extracted text
        extracted_content = "\n\n".join(result)
        
        # If we couldn't extract anything meaningful, try a more aggressive approach
        if len(extracted_content.strip()) < 100:
            logger.warning(f"Minimal content extracted, trying fallback extraction for {url}")
            all_text = []
            
            # Get all text from the body
            body = soup.find('body')
            if body:
                for text in body.stripped_strings:
                    text = text.strip()
                    if text and len(text) > 20:
                        all_text.append(text)
            
            if all_text:
                extracted_content = "\n\n".join(all_text)
        
        return extracted_content
    
    def _extract_notion_content(self, html_content: str, url: str) -> str:
        """Special extraction method for Notion pages."""
        soup = BeautifulSoup(html_content, 'html.parser')
        result = []
        
        # Try to get the page title
        title_tag = soup.find('title') or soup.find('h1') or soup.find('meta', property='og:title')
        if title_tag:
            if title_tag.name == 'meta':
                title = title_tag.get('content', '').strip()
            else:
                title = title_tag.get_text().strip()
            if title:
                result.append(f"# {title}")
        
        # Process notion-specific elements
        notion_blocks = soup.select('.notion-page-content .notion-collection-item')
        if not notion_blocks:
            notion_blocks = soup.select('.notion-page-content > div')
        
        if not notion_blocks:
            # Fallback to any div that might contain notion content
            for div in soup.find_all('div'):
                if 'notion' in str(div.get('class', '')):
                    for text in div.stripped_strings:
                        text = text.strip()
                        if text and len(text) > 20:
                            result.append(text)
        
        # Process each notion block
        for block in notion_blocks:
            # Handle headings
            if block.find(['h1', 'h2', 'h3']):
                for i in range(1, 4):
                    for h in block.find_all(f'h{i}'):
                        heading_text = h.get_text().strip()
                        if heading_text:
                            result.append(f"{'#' * i} {heading_text}")
            
            # Handle paragraphs
            paragraphs = block.find_all('p')
            for p in paragraphs:
                p_text = p.get_text().strip()
                if p_text:
                    result.append(p_text)
            
            # Handle lists
            for ul in block.find_all(['ul', 'ol']):
                for li in ul.find_all('li'):
                    li_text = li.get_text().strip()
                    if li_text:
                        result.append(f"• {li_text}")
            
            # If no structured content found, get any text content
            if not paragraphs and not block.find(['h1', 'h2', 'h3']) and not block.find(['ul', 'ol']):
                block_text = block.get_text().strip()
                if block_text and len(block_text) > 20:
                    result.append(block_text)
        
        # If we still haven't found content, do a more aggressive extraction
        if not result or (len(result) == 1 and result[0].startswith('#')):
            # Fallback: extract all content from the page
            for tag in soup.find_all(['p', 'div', 'span']):
                if not tag.find(['p', 'div', 'span']):  # Only get leaf nodes
                    text = tag.get_text().strip()
                    if text and len(text) > 20:
                        result.append(text)
        
        return "\n\n".join(result)
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str, base_domain: str) -> List[str]:
        """Extract valid links from the page that belong to the same domain."""
        links = []
        # Get all links
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            
            if self._is_valid_url(full_url, base_domain):
                # Normalize URL by removing fragments
                parsed = urlparse(full_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    clean_url += f"?{parsed.query}"
                
                if clean_url not in links:
                    links.append(clean_url)
                
        return links

    def _is_notion_page(self, url: str) -> bool:
        """Check if URL is a Notion page."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Check both the domain and path for notion indicators
        return ('notion.so' in domain or 
                'notion.site' in domain or 
                'super.site' in domain or  # Cover Notion's Super sites
                'notionnow' in domain or   # Some Notion custom domains
                '/notion/' in parsed_url.path.lower() or
                any(re.search(r'notion', param.lower()) for param in parsed_url.query.split('&')))

    def _get_sitemap_urls(self, base_url: str) -> List[str]:
        """Discover and parse sitemap to extract URLs."""
        parsed_url = urlparse(base_url)
        root_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        sitemap_urls = []
        sitemap_locations = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap-index.xml",
            "/sitemaps/sitemap.xml",
            "/sitemap/sitemap.xml",
            "/wp-sitemap.xml",            # WordPress
            "/sitemap_news.xml",          # News sites
            "/post-sitemap.xml",          # WordPress post sitemap
            "/page-sitemap.xml",          # WordPress page sitemap
            "/category-sitemap.xml",      # WordPress category sitemap
            "/robots.txt"                 # Check robots.txt for sitemap location
        ]
        
        # First try to find sitemaps at common locations
        for path in sitemap_locations:
            sitemap_url = f"{root_url}{path}"
            logger.info(f"Checking for sitemap at: {sitemap_url}")
            
            response = self._make_request(sitemap_url)
            if not response:
                continue
                
            if path == "/robots.txt":
                # Extract sitemap URL from robots.txt
                for line in response.text.splitlines():
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        logger.info(f"Found sitemap in robots.txt: {sitemap_url}")
                        sitemap_response = self._make_request(sitemap_url)
                        if sitemap_response:
                            urls = self._parse_sitemap(sitemap_response.text)
                            sitemap_urls.extend(urls)
            else:
                # Parse XML sitemap
                urls = self._parse_sitemap(response.text)
                sitemap_urls.extend(urls)
        
        # Remove duplicates and return
        return list(set(sitemap_urls))
    
    def _parse_sitemap(self, sitemap_content: str) -> List[str]:
        """Parse XML sitemap content and extract URLs."""
        urls = []
        
        try:
            # Handle potential XML parsing issues
            if '<?xml' not in sitemap_content[:100]:
                logger.warning("Content doesn't appear to be valid XML")
                return []
            
            # Parse XML sitemap
            root = ET.fromstring(sitemap_content)
            
            # Check if it's a sitemap index
            is_sitemap_index = root.tag.endswith('sitemapindex')
            
            if is_sitemap_index:
                # Process each sitemap in the index
                for sitemap in root.findall('./sm:sitemap/sm:loc', SITEMAP_NAMESPACES) or root.findall('.//loc'):
                    nested_sitemap_url = sitemap.text.strip()
                    logger.info(f"Found nested sitemap: {nested_sitemap_url}")
                    
                    nested_response = self._make_request(nested_sitemap_url)
                    if nested_response:
                        nested_urls = self._parse_sitemap(nested_response.text)
                        urls.extend(nested_urls)
            else:
                # Extract URLs from urlset
                for url in root.findall('./sm:url/sm:loc', SITEMAP_NAMESPACES) or root.findall('.//loc'):
                    page_url = url.text.strip()
                    urls.append(page_url)
        
        except Exception as e:
            logger.error(f"Error parsing sitemap: {str(e)}")
        
        return urls

    def scrape(self, url: str, crawl: bool = True, use_sitemap: bool = True) -> Dict[str, str]:
        """
        Scrape content from the given URL and optionally crawl linked pages.
        
        Args:
            url: URL to scrape
            crawl: Whether to follow links and crawl additional pages
            use_sitemap: Whether to try finding and using the sitemap
            
        Returns:
            Dictionary mapping URLs to extracted content
        """
        results = {}
        visited = set()
        to_visit = [(url, 0)]  # (url, depth)
        base_domain = urlparse(url).netloc
        
        # First try specialized scrapers
        if self._is_notion_page(url):
            logger.info(f"Detected Notion page: {url}")
            specialized_scraper, scrape_method = get_specialized_scraper(url)
            if specialized_scraper and scrape_method:
                content = scrape_method(url)
                if content:
                    results[url] = content
                    return results
        
        # Try to find URLs from sitemap if requested
        if use_sitemap:
            sitemap_urls = self._get_sitemap_urls(url)
            if sitemap_urls:
                logger.info(f"Found {len(sitemap_urls)} URLs in sitemap")
                
                # Filter URLs to only include those from the same domain
                sitemap_urls = [u for u in sitemap_urls if self._is_valid_url(u, base_domain)]
                
                # Add sitemap URLs to the queue but respect max_pages
                for sitemap_url in sitemap_urls[:self.max_pages]:
                    if sitemap_url not in visited:
                        to_visit.append((sitemap_url, 0))  # Add at depth 0
        
        # Process the queue
        while to_visit and len(results) < self.max_pages:
            current_url, current_depth = to_visit.pop(0)
            
            if current_url in visited:
                continue
                
            visited.add(current_url)
            logger.info(f"Scraping: {current_url}")
            
            # Determine if this is a Notion page and use appropriate headers
            headers = self.notion_headers if self._is_notion_page(current_url) else self.headers
            
            # Check if specialized scraper exists for this URL
            specialized_scraper, scrape_method = get_specialized_scraper(current_url)
            if specialized_scraper and scrape_method:
                content = scrape_method(current_url)
                if content:
                    results[current_url] = content
                    # Don't follow links for specialized pages unless explicitly requested
                    if not crawl:
                        continue
            else:
                # Use standard scraping approach
                response = self._make_request(current_url, headers=headers, retry_count=2)
                if not response:
                    continue
                
                content = self._extract_text_content(response.text, current_url)
                if content and len(content.strip()) > 100:  # Only store if meaningful content found
                    results[current_url] = content
                else:
                    logger.warning(f"Insufficient content extracted from {current_url}")
                
                # Follow links if crawling is enabled and we haven't reached max depth
                if crawl and current_depth < self.depth:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    links = self._extract_links(soup, current_url, base_domain)
                    
                    # Add new links to visit
                    for link in links:
                        if link not in visited:
                            to_visit.append((link, current_depth + 1))
        
        return results

def scrape_website(url: str, depth: int = 1, max_pages: int = 10, use_sitemap: bool = True) -> str:
    """
    Scrape content from a website URL for RAG applications.
    
    Args:
        url: URL to scrape
        depth: Crawling depth (0 for single page, 1+ for following links)
        max_pages: Maximum number of pages to scrape
        use_sitemap: Whether to try finding and using the sitemap
        
    Returns:
        Extracted text content
    """
    # Create scraper instance with appropriate settings
    scraper = WebScraper(max_pages=max_pages, depth=depth)
    
    try:
        # Determine if we should crawl based on URL type and requested depth
        is_notion = scraper._is_notion_page(url)
        crawl = depth > 0 and not is_notion  # Don't crawl Notion pages by default
        
        # Perform scraping
        scraped_content = scraper.scrape(url, crawl=crawl, use_sitemap=use_sitemap)
        
        if not scraped_content:
            logger.warning(f"No content extracted from {url}")
            return ""
            
        # Combine all extracted content with source URLs
        combined_text = []
        for page_url, content in scraped_content.items():
            combined_text.append(f"[Source: {page_url}]\n\n{content}\n")
            
        return "\n".join(combined_text)
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return ""
