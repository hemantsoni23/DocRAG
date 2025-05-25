import re
import tldextract
from urllib.parse import urljoin, urldefrag, urlparse
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def get_domain(url: str) -> str:
    """Extract domain from URL using tldextract"""
    parts = tldextract.extract(url)
    result = f"{parts.domain}.{parts.suffix}"
    logger.debug(f"get_domain: {url} -> {result}")
    return result

def is_same_domain(domain: str, url: str) -> bool:
    """Check if two URLs belong to same domain"""
    domain_a = get_domain(domain)
    domain_b = get_domain(url)
    result = domain_a == domain_b
    logger.debug(f"is_same_domain: {domain_a} vs {domain_b} -> {result}")
    return result

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    return re.sub(r'\s+', ' ', text).strip()

def extract_content_from_html(html: str) -> dict:
    """Extract title, meta description, and content from HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    
    title = clean_text(soup.title.string) if soup.title and soup.title.string else ""
    
    desc_tag = soup.find('meta', {'name': 'description'})
    meta = clean_text(desc_tag['content']) if desc_tag and desc_tag.has_attr('content') else ""
    
    blocks = []
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li', 'blockquote']):
        txt = tag.get_text(" ", strip=True)
        if len(txt) > 30:
            blocks.append(clean_text(txt))
    
    return {
        'title': title,
        'meta_description': meta,
        'content': "\n".join(blocks)
    }

def extract_links_from_html(html: str, base_url: str, start_url: str) -> set:
    """Extract valid same-domain links from HTML"""
    soup = BeautifulSoup(html, 'html.parser')
    children = set()
    
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
            
        href = urljoin(base_url, href)
        href, _ = urldefrag(href)
        parsed = urlparse(href)
        
        if parsed.scheme.startswith('http') and is_same_domain(start_url, href):
            children.add(href)
    
    return children