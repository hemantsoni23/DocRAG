# utils/specialized_scraper.py

import logging
import re
import json
import time
from typing import Dict, List, Optional, Union, Tuple, Callable
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# Set up logging
logger = logging.getLogger(__name__)

class NotionScraper:
    """
    Specialized scraper for Notion pages and databases.
    """
    
    def __init__(self, timeout: int = 30):
        """Initialize the Notion scraper."""
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "DNT": "1",
        }
    
    def _extract_notion_data(self, html: str) -> Optional[dict]:
        """
        Extract Notion data from page source.
        Notion stores page data in a __NEXT_DATA__ script tag.
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for Notion's data in script tags
            for script in soup.find_all('script'):
                script_text = script.string
                if not script_text:
                    continue
                
                # Check for Notion's data structures
                if "__NEXT_DATA__" in script_text:
                    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
                    if match:
                        data_json = match.group(1)
                        return json.loads(data_json)
                
                # Check for Notion's pageProps
                if "pageProps" in script_text and "block" in script_text:
                    try:
                        # Try to extract JSON data
                        json_str = script_text.strip()
                        if json_str.startswith('window.__INITIAL_DATA__'):
                            json_str = re.search(r'window\.__INITIAL_DATA__\s*=\s*({.*});', json_str, re.DOTALL).group(1)
                        
                        return json.loads(json_str)
                    except (json.JSONDecodeError, AttributeError):
                        continue
            
            return None
        except Exception as e:
            logger.error(f"Error extracting Notion data: {str(e)}")
            return None
        
    def scrape_notion_page(self, url: str) -> str:
        """
        Scrape content from a Notion page.
        
        Args:
            url: URL of the Notion page
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract page title
            title = soup.title.string if soup.title else "Notion Page"
            title = title.replace(" | Notion", "").replace(" – Notion", "").strip()
            
            # Start with the title
            content_parts = [f"# {title}"]
            
            # Try to extract structured Notion data first
            notion_data = self._extract_notion_data(response.text)
            if notion_data:
                logger.info("Found structured Notion data")
                # Process structured data - this would be implementation-specific
                # based on Notion's data structure
            
            # Otherwise, extract content from HTML
            
            # Find main content container - try different selectors that Notion might use
            main_selectors = [
                'div[data-block-id]',  # Block-based content
                'div.notion-page-content',
                'div.notion-scroller',
                'main',
                'article',
                '.notion-page-content-inner'
            ]
            
            main_content = None
            for selector in main_selectors:
                if selector.startswith('.'):
                    elements = soup.select(selector)
                else:
                    elements = soup.select(selector)
                
                if elements:
                    # Use the largest content container (by text length)
                    main_content = max(elements, key=lambda el: len(el.get_text()))
                    break
            
            if not main_content:
                # Fall back to body
                main_content = soup.body
            
            if main_content:
                # Extract text from all Notion blocks
                blocks = main_content.find_all('div', attrs={'data-block-id': True})
                
                if not blocks:
                    # If no blocks found, try more generic extraction
                    blocks = main_content.find_all(['div', 'p', 'h1', 'h2', 'h3', 'ul', 'ol', 'table'])
                
                # Process blocks by type
                for block in blocks:
                    # Skip empty blocks
                    text = block.get_text().strip()
                    if not text:
                        continue
                    
                    # Determine block type
                    if block.name in ['h1', 'h2', 'h3', 'h4']:
                        level = int(block.name[1])
                        content_parts.append(f"{'#' * level} {text}")
                    elif 'notion-heading' in str(block.get('class', '')):
                        # Extract heading level from class
                        level_match = re.search(r'notion-heading-(\d)', str(block.get('class', '')))
                        level = int(level_match.group(1)) if level_match else 1
                        content_parts.append(f"{'#' * level} {text}")
                    elif block.name == 'p' or 'notion-text-block' in str(block.get('class', '')):
                        content_parts.append(text)
                    elif block.name in ['ul', 'ol'] or 'notion-bulleted-list' in str(block.get('class', '')):
                        for li in block.find_all('li'):
                            li_text = li.get_text().strip()
                            if li_text:
                                content_parts.append(f"• {li_text}")
                    elif block.name == 'table' or 'notion-collection-table' in str(block.get('class', '')):
                        content_parts.append("Table:")
                        for row in block.find_all('tr'):
                            cells = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                            if cells:
                                content_parts.append(" | ".join(cells))
                    elif 'notion-code-block' in str(block.get('class', '')):
                        code_text = block.get_text().strip()
                        if code_text:
                            content_parts.append(f"```\n{code_text}\n```")
                    elif 'notion-callout-block' in str(block.get('class', '')) or 'notion-quote-block' in str(block.get('class', '')):
                        content_parts.append(f"> {text}")
                    elif len(text) > 40:  # Only add substantial text from other blocks
                        content_parts.append(text)
            
            # Join all content parts
            result = "\n\n".join(content_parts)
            
            # If we didn't extract much, try a more aggressive approach
            if len(result) < 500:
                # Extract all substantial text from the page
                paragraphs = []
                for p in soup.find_all(['p', 'div'], class_=lambda c: c and 'notion' in str(c)):
                    text = p.get_text().strip()
                    if text and len(text) > 40:  # Only substantial text
                        paragraphs.append(text)
                
                if paragraphs:
                    result = f"# {title}\n\n" + "\n\n".join(paragraphs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error scraping Notion page {url}: {str(e)}")
            return ""

class WikiScraper:
    """
    Specialized scraper for Wikipedia and wiki-style pages.
    """
    
    def __init__(self, timeout: int = 20):
        """Initialize the Wiki scraper."""
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
    def is_wiki_page(self, url: str) -> bool:
        """Check if a URL points to a wiki page."""
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc.lower()
        path = parsed_url.path.lower()
        
        wiki_domains = [
            'wikipedia.org',
            'wikimedia.org',
            'fandom.com',
            'wiki.com',
            'gamepedia.com',
            'mediawiki.org',
            'wiktionary.org',
            'wikia.com'
        ]
        
        return any(domain in netloc for domain in wiki_domains) or '/wiki/' in path
        
    def scrape_wiki_page(self, url: str) -> str:
        """
        Scrape content from a wiki page.
        
        Args:
            url: URL of the wiki page
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract page title
            title = soup.title.string if soup.title else "Wiki Page"
            # Clean up Wikipedia titles
            title = re.sub(r' - Wikipedia$', '', title)
            
            # Start with the title
            content_parts = [f"# {title}"]
            
            # Find main content
            content_div = None
            
            # Try Wikipedia-specific content divs
            wikipedia_content_ids = ['content', 'mw-content-text', 'bodyContent']
            for div_id in wikipedia_content_ids:
                content_div = soup.find('div', id=div_id)
                if content_div:
                    break
            
            if not content_div:
                # Try common class names for wiki content
                content_div = soup.find(['div', 'main'], class_=lambda c: c and any(x in str(c).lower() for x in 
                                                                                ['content', 'wiki-body', 'article']))
                
            if not content_div:
                # Fall back to article or main
                content_div = soup.find(['article', 'main']) or soup.body
                
            if content_div:
                # Remove unwanted elements
                unwanted_elements = [
                    '.infobox',           # Wikipedia infoboxes
                    '.navbox',            # Navigation boxes
                    '.vertical-navbox',   # Vertical navigation boxes
                    '.sidebar',           # Sidebars
                    '.ambox',             # Message boxes
                    '.hatnote',           # Hat notes
                    '.metadata',          # Metadata
                    '.catlinks',          # Category links
                    '.noprint',           # Elements not for printing
                    '#toc',               # Table of contents
                    '.toc',               # Table of contents
                    '.mw-editsection',    # Edit section links
                    '.mw-empty-elt',      # Empty elements
                    '.mw-indicators',     # Indicators
                    '.mw-jump-link',      # Jump links
                    '.mw-references-wrap' # References wrapper
                ]
                
                for selector in unwanted_elements:
                    for element in content_div.select(selector):
                        element.decompose()
                
                # Extract headings and content
                for element in content_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'dl', 'table']):
                    # Skip elements that were in removed sections
                    if not element.parent:
                        continue
                        
                    if element.name.startswith('h') and len(element.name) == 2:
                        level = int(element.name[1])
                        # Remove edit section links from headings
                        for span in element.find_all('span', class_='mw-editsection'):
                            span.decompose()
                        text = element.get_text().strip()
                        if text:
                            content_parts.append(f"{'#' * level} {text}")
                    elif element.name == 'p':
                        text = element.get_text().strip()
                        if text and len(text) > 5:  # Skip very short paragraphs
                            content_parts.append(text)
                    elif element.name in ['ul', 'ol']:
                        list_items = []
                        for li in element.find_all('li', recursive=False):
                            text = li.get_text().strip()
                            if text:
                                list_items.append(f"• {text}")
                        if list_items:
                            content_parts.append("\n".join(list_items))
                    elif element.name == 'dl':
                        for dt in element.find_all('dt'):
                            term = dt.get_text().strip()
                            # Find the next dd element
                            dd = dt.find_next_sibling('dd')
                            defn = dd.get_text().strip() if dd else ""
                            if term and defn:
                                content_parts.append(f"**{term}**: {defn}")
                    elif element.name == 'table':
                        # Skip very large tables as they often contain less important data
                        if len(element.find_all('tr')) > 20:
                            continue
                            
                        table_rows = []
                        table_rows.append("Table:")
                        for row in element.find_all('tr'):
                            cells = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                            if cells:
                                table_rows.append(" | ".join(cells))
                        content_parts.append("\n".join(table_rows))
            
            # Join all content parts
            return "\n\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error scraping wiki page {url}: {str(e)}")
            return ""

class DocsScraper:
    """
    Specialized scraper for documentation sites.
    """
    
    def __init__(self, timeout: int = 20):
        """Initialize the Docs scraper."""
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
        
    def is_docs_site(self, url: str) -> bool:
        """Check if a URL points to a documentation site."""
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc.lower()
        path = parsed_url.path.lower()
        
        docs_domains = [
            'docs.', 'readthedocs.io', 'documentation.',
            '.dev', 'developer.', '.api.', 'github.io',
            'help.'
        ]
        
        docs_paths = [
            '/docs/', '/documentation/', '/api/', '/reference/',
            '/guide/', '/manual/', '/handbook/', '/tutorial/',
            '/help/', '/learn/'
        ]
        
        return any(domain in netloc for domain in docs_domains) or any(doc_path in path for doc_path in docs_paths)
        
    def scrape_docs_page(self, url: str) -> str:
        """
        Scrape content from a documentation page.
        
        Args:
            url: URL of the documentation page
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract page title
            title = soup.title.string if soup.title else "Documentation"
            
            # Start with the title
            content_parts = [f"# {title}"]
            
            # Find main content
            content_element = None
            
            # Common documentation site selectors - try in order of specificity
            doc_selectors = [
                'article.doc',
                'main.content',
                'div.documentation',
                'div.docs-content',
                'div.markdown-body',
                'div.document',
                'div.content-body',
                'div.main-content',
                'article',
                'main',
                '.content',
                '.documentContent',
                '.doc-content',
                '.docContent',
                '.api-content',
                '.api-docs',
                '#content'
            ]
            
            for selector in doc_selectors:
                elements = soup.select(selector)
                if elements:
                    # Choose the element with the most content
                    content_element = max(elements, key=lambda el: len(el.get_text()))
                    break
            
            # If we still don't have a content element, try some common patterns
            if not content_element and soup.main:
                content_element = soup.main
            elif not content_element:
                article = soup.find('article')
                if article:
                    content_element = article
            
            if not content_element:
                # Fall back to body
                content_element = soup.body
            
            if content_element:
                # Remove navigation, sidebars, and other non-content elements
                for nav in content_element.find_all(['nav', 'aside']):
                    nav.decompose()
                    
                # Remove sidebar and navigation classes
                for element in content_element.select('.sidebar, .navigation, .menu, .toc, .nav, .footer'):
                    element.decompose()
                
                # Extract relevant content
                sections = []
                
                # Process headings and structure
                for element in content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'code', 'ul', 'ol', 'table', 'blockquote']):
                    if element.name.startswith('h') and len(element.name) == 2:
                        level = int(element.name[1])
                        text = element.get_text().strip()
                        if text:
                            sections.append(f"{'#' * level} {text}")
                    elif element.name == 'p':
                        text = element.get_text().strip()
                        if text:
                            sections.append(text)
                    elif element.name == 'pre' or element.name == 'code':
                        code = element.get_text().strip()
                        if code:
                            sections.append(f"```\n{code}\n```")
                    elif element.name in ['ul', 'ol']:
                        items = []
                        for li in element.find_all('li'):
                            text = li.get_text().strip()
                            if text:
                                items.append(f"• {text}")
                        if items:
                            sections.append("\n".join(items))
                    elif element.name == 'table':
                        table_rows = ["Table:"]
                        for row in element.find_all('tr'):
                            cells = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                            if cells:
                                table_rows.append(" | ".join(cells))
                        sections.append("\n".join(table_rows))
                    elif element.name == 'blockquote':
                        quote = element.get_text().strip()
                        if quote:
                            sections.append(f"> {quote}")
                
                # If sections are empty, try a more generic approach
                if not sections:
                    # Get all text from the content element
                    text = content_element.get_text()
                    # Split by multiple newlines to separate paragraphs
                    paragraphs = re.split(r'\n\s*\n', text)
                    for paragraph in paragraphs:
                        paragraph = paragraph.strip()
                        if paragraph and len(paragraph) > 40:  # Only include substantial paragraphs
                            sections.append(paragraph)
                
                content_parts.extend(sections)
            
            # Join all parts
            return "\n\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error scraping docs page {url}: {str(e)}")
            return ""

class BlogScraper:
    """
    Specialized scraper for blog posts and articles.
    """
    
    def __init__(self, timeout: int = 20):
        """Initialize the Blog scraper."""
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        }
    
    def is_blog_post(self, url: str) -> bool:
        """Check if a URL points to a blog post or article."""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Check for common blog path patterns
        blog_paths = ['/blog/', '/article/', '/post/', '/news/']
        date_pattern = r'/\d{4}/\d{2}/\d{2}/'  # Common date pattern in blog URLs
        
        return any(blog_path in path for blog_path in blog_paths) or re.search(date_pattern, path)
    
    def scrape_blog_post(self, url: str) -> str:
        """
        Scrape content from a blog post or article.
        
        Args:
            url: URL of the blog post
            
        Returns:
            Extracted text content
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract page title
            title = soup.title.string if soup.title else "Blog Post"
            
            # Start with the title
            content_parts = [f"# {title}"]
            
            # Try to find the author and date
            author_elements = soup.select('.author, .byline, .meta-author, [rel="author"]')
            if author_elements:
                author = author_elements[0].get_text().strip()
                if author:
                    content_parts.append(f"Author: {author}")
                    
            date_elements = soup.select('.date, .published, .post-date, time, .meta-date')
            if date_elements:
                date = date_elements[0].get_text().strip()
                if date:
                    content_parts.append(f"Date: {date}")
            
            # Find main content - common blog content selectors
            content_element = None
            blog_selectors = [
                'article',
                '.post-content',
                '.entry-content',
                '.article-content',
                '.blog-content',
                '.post-body',
                '.content',
                '.main-content',
                'main',
                '#content'
            ]
            
            for selector in blog_selectors:
                elements = soup.select(selector)
                if elements:
                    # Choose the longest content element
                    content_element = max(elements, key=lambda el: len(el.get_text()))
                    break
            
            if not content_element:
                # Fall back to body
                content_element = soup.body
            
            if content_element:
                # Remove navigation, comments, and other non-content elements
                for element in content_element.select('.comments, .comment-section, .sidebar, .widget, .social-share, nav, footer, header, .related-posts'):
                    element.decompose()
                
                # Extract paragraphs and other content
                for element in content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'blockquote', 'ul', 'ol', 'pre']):
                    if element.name.startswith('h') and len(element.name) == 2:
                        level = int(element.name[1])
                        text = element.get_text().strip()
                        if text:
                            content_parts.append(f"{'#' * level} {text}")
                    elif element.name == 'p':
                        text = element.get_text().strip()
                        if text and len(text) > 10:  # Skip very short paragraphs
                            content_parts.append(text)
                    elif element.name == 'blockquote':
                        quote = element.get_text().strip()
                        if quote:
                            content_parts.append(f"> {quote}")
                    elif element.name in ['ul', 'ol']:
                        items = []
                        for li in element.find_all('li'):
                            text = li.get_text().strip()
                            if text:
                                items.append(f"• {text}")
                        if items:
                            content_parts.append("\n".join(items))
                    elif element.name == 'pre':
                        code = element.get_text().strip()
                        if code:
                            content_parts.append(f"```\n{code}\n```")
            
            # Join all parts
            return "\n\n".join(content_parts)
            
        except Exception as e:
            logger.error(f"Error scraping blog post {url}: {str(e)}")
            return ""

# Dictionary of supported specialized scrapers
SPECIALIZED_SCRAPERS = {
    "notion": NotionScraper(),
    "wiki": WikiScraper(),
    "docs": DocsScraper(),
    "blog": BlogScraper()
}

def get_specialized_scraper(url: str) -> Tuple[Optional[str], Optional[Callable]]:
    """
    Get the appropriate specialized scraper for a URL.
    
    Args:
        url: URL to scrape
        
    Returns:
        Tuple of (scraper_name, scrape_method) or (None, None) if no specialized scraper is available
    """
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()
    
    # Check for Notion pages
    if 'notion.so' in netloc or 'notion.site' in netloc:
        return "notion", SPECIALIZED_SCRAPERS["notion"].scrape_notion_page
    
    # Check for wiki pages
    wiki_scraper = SPECIALIZED_SCRAPERS["wiki"]
    if wiki_scraper.is_wiki_page(url):
        return "wiki", wiki_scraper.scrape_wiki_page
    
    # Check for documentation pages
    docs_scraper = SPECIALIZED_SCRAPERS["docs"]
    if docs_scraper.is_docs_site(url):
        return "docs", docs_scraper.scrape_docs_page
    
    # Check for blog posts
    blog_scraper = SPECIALIZED_SCRAPERS["blog"]
    if blog_scraper.is_blog_post(url):
        return "blog", blog_scraper.scrape_blog_post
    
    # No specialized scraper available
    return None, None