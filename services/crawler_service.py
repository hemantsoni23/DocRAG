import asyncio
import os
import time
import json
import csv
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple, Any
from playwright.async_api import async_playwright, Error as PWError
import logging
from uuid import uuid4

from models.schema import CrawlRequest, JobStatus
from utils.crawler import (
    get_domain, is_same_domain, extract_content_from_html, 
    extract_links_from_html
)

logger = logging.getLogger(__name__)

# Global job storage
JOBS: Dict[str, JobStatus] = {}
OUTPUT_DIR = "crawler_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CrawlerService:
    def __init__(self):
        self.semaphore = None
    
    async def fetch_and_parse(
        self, 
        url: str, 
        depth: int, 
        context: Any, 
        start_url: str, 
        semaphore: asyncio.Semaphore, 
        debug_mode: bool = False
    ) -> Tuple[Optional[Dict], Set[str]]:
        """Fetch a page and extract content and links"""
        fetch_start = time.time()
        
        async with semaphore:
            logger.debug(f"Fetching [{depth}] {url}")
            try:
                page = await context.new_page()
                
                if debug_mode:
                    logger.debug(f"Page user agent: {await page.evaluate('navigator.userAgent')}")
                    
                await page.goto(url, timeout=30000)
                await page.wait_for_load_state('networkidle')
                html = await page.content()
                await page.close()
            except Exception as e:
                logger.error(f"Error fetching {url}: {type(e).__name__}: {e}")
                return None, set()

        parse_start = time.time()
        logger.debug(f"Fetch completed in {parse_start - fetch_start:.2f}s: {url}")
        
        # Extract content
        content_data = extract_content_from_html(html)
        data = {
            'url': url,
            'title': content_data['title'],
            'meta_description': content_data['meta_description'],
            'content': content_data['content']
        }

        content_length = len(data['content'])
        links_start = time.time()
        logger.debug(f"Content extraction completed in {links_start - parse_start:.2f}s: {url} ({content_length} chars)")
        
        # Extract links
        children = extract_links_from_html(html, url, start_url)
        
        end_time = time.time()
        logger.debug(f"Links extraction completed in {end_time - links_start:.2f}s: Found {len(children)} same domain links")
        logger.debug(f"Total processing time for {url}: {end_time - fetch_start:.2f}s")
        
        return data, children

    async def crawl_site(self, job_id: str, params: CrawlRequest):
        """Main crawling function"""
        logger.info(f"[+] Starting crawl job {job_id} at {params.start_url}")
        start_time = datetime.now()
        
        JOBS[job_id] = JobStatus(
            job_id=job_id,
            status="running",
            start_url=params.start_url,
            start_time=start_time.isoformat(),
            debug_stats={
                "crawl_start_time": time.time(),
                "links_found_total": 0,
                "same_domain_links_total": 0,
                "pages_attempted": 0,
                "pages_successful": 0,
                "queue_max_size": 0,
                "domain": get_domain(params.start_url)
            }
        )
        
        start_domain = get_domain(params.start_url)
        logger.info(f"Start domain: {start_domain}")
        
        queue = asyncio.Queue()
        await queue.put((params.start_url, 0))
        visited = set()
        results = []
        semaphore = asyncio.Semaphore(params.concurrency)

        try:
            playwright = await async_playwright().start()
        except PWError as e:
            logger.error(f"[!] Playwright failed to start: {e}")
            JOBS[job_id].status = "failed"
            JOBS[job_id].error = str(e)
            JOBS[job_id].end_time = datetime.now().isoformat()
            return

        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        
        # Block resource types to speed up crawling
        await context.route("**/*", lambda r, req:
            r.abort() if req.resource_type in ["image", "media", "font", "stylesheet", "video"]
            else r.continue_()
        )

        batch_count = 0
        
        while not queue.empty() and len(results) < params.max_pages:
            qsize = queue.qsize()
            JOBS[job_id].debug_stats["queue_max_size"] = max(JOBS[job_id].debug_stats["queue_max_size"], qsize)
            
            batch_count += 1
            logger.info(f"[i] Batch {batch_count}: Queue size: {qsize}, fetched: {len(results)}")
            JOBS[job_id].pages_crawled = len(results)
            
            # Process batch
            batch = []
            for _ in range(min(qsize, params.concurrency, params.max_pages - len(results))):
                url, depth = await queue.get()
                if url in visited:
                    queue.task_done()
                    continue
                visited.add(url)
                batch.append((url, depth))

            if not batch:
                logger.info("No new URLs to process, stopping")
                break

            logger.info(f"Processing batch of {len(batch)} URLs")
            JOBS[job_id].debug_stats["pages_attempted"] += len(batch)
            
            tasks = [self.fetch_and_parse(url, depth, context, params.start_url, semaphore, params.debug_mode) 
                    for url, depth in batch]
            
            batch_links_found = 0
            batch_same_domain_links = 0
            batch_successful = 0
            
            for coro in asyncio.as_completed(tasks):
                data, children = await coro
                if not data:
                    continue
                    
                results.append(data)
                batch_successful += 1
                
                current_url = data['url']
                current_depth = next(d for u, d in batch if u == current_url)
                
                batch_links_found += len(children)
                
                if (params.max_depth is None or current_depth < params.max_depth):
                    new_urls_for_page = 0
                    for c in children:
                        if c not in visited:
                            await queue.put((c, current_depth + 1))
                            new_urls_for_page += 1
                            batch_same_domain_links += 1
                        
                queue.task_done()
            
            # Update stats
            JOBS[job_id].debug_stats["links_found_total"] += batch_links_found
            JOBS[job_id].debug_stats["same_domain_links_total"] += batch_same_domain_links
            JOBS[job_id].debug_stats["pages_successful"] += batch_successful
            
            logger.info(f"Batch complete: {batch_successful} pages fetched, {batch_same_domain_links} new URLs added to queue")

        await context.close()
        await browser.close()
        await playwright.stop()
        
        # Record crawl end time
        JOBS[job_id].debug_stats["crawl_end_time"] = time.time()
        JOBS[job_id].debug_stats["crawl_duration"] = JOBS[job_id].debug_stats["crawl_end_time"] - JOBS[job_id].debug_stats["crawl_start_time"]
        
        logger.info(f"[+] Crawl finished. Total pages: {len(results)}")
        
        # Save results
        output_files = await self.save_results(results, start_domain, job_id, params.output_formats)
        
        JOBS[job_id].status = "completed"
        JOBS[job_id].pages_crawled = len(results)
        JOBS[job_id].end_time = datetime.now().isoformat()
        JOBS[job_id].output_files = output_files
        
        return results

    async def save_results(self, results: List[Dict], domain: str, job_id: str, formats: List[str]) -> Dict[str, str]:
        """Save crawling results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_key = domain.replace(".", "_")
        base_name = f"{domain_key}_{timestamp}"
        out_files = {}

        if results:
            if "txt" in formats:
                path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
                with open(path, 'w', encoding='utf-8') as f:
                    for p in results:
                        f.write(f"## {p['url']}\n")
                        if p['title']:
                            f.write(f"Title: {p['title']}\n")
                        if p['meta_description']:
                            f.write(f"Description: {p['meta_description']}\n\n")
                        f.write(p['content'] + "\n\n")
                out_files['txt'] = path
                logger.info(f"✅ {len(results)} pages → {path}")
                
                JOBS[job_id].debug_stats["txt_file_size"] = os.path.getsize(path)

            if "json" in formats:
                path = os.path.join(OUTPUT_DIR, f"{base_name}.json")
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                out_files['json'] = path
                logger.info(f"✅ {len(results)} pages → {path}")
                
                JOBS[job_id].debug_stats["json_file_size"] = os.path.getsize(path)

            if "csv" in formats:
                path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")
                keys = ['url', 'title', 'meta_description', 'content']
                with open(path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    for row in results:
                        writer.writerow(row)
                out_files['csv'] = path
                logger.info(f"✅ {len(results)} pages → {path}")
                
                JOBS[job_id].debug_stats["csv_file_size"] = os.path.getsize(path)
                
            # Calculate total content size for all pages
            total_content_size = sum(len(p['content']) for p in results)
            JOBS[job_id].debug_stats["total_content_chars"] = total_content_size
            JOBS[job_id].debug_stats["avg_content_chars_per_page"] = total_content_size / len(results) if results else 0
        else:
            logger.warning("⚠️  No pages were fetched.")

        return out_files

    def create_crawl_job(self, start_url: str, max_pages: int = 100, concurrency: int = 10) -> str:
        """Create a new crawl job and return job ID"""
        job_id = str(uuid4())
        params = CrawlRequest(
            start_url=start_url,
            max_pages=max_pages,
            concurrency=concurrency,
            output_formats=["json", "txt"]  # We mainly need these for chatbot creation
        )
        
        # Initialize job status
        JOBS[job_id] = JobStatus(
            job_id=job_id,
            status="queued",
            start_url=start_url,
            start_time=datetime.now().isoformat()
        )
        
        return job_id, params

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status by ID"""
        return JOBS.get(job_id)

    def get_job_results(self, job_id: str) -> Optional[List[Dict]]:
        """Get crawled content for a job"""
        job = JOBS.get(job_id)
        if not job or job.status != "completed":
            return None
            
        # Read from JSON file if it exists
        if "json" in job.output_files:
            json_path = job.output_files["json"]
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        return None

# Create singleton instance
crawler_service = CrawlerService()