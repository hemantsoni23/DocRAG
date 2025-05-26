from typing import List
import logging
from services.crawler_service import crawler_service
from fastapi import HTTPException

# Set up logging
logger = logging.getLogger(__name__)

async def process_urls_for_chatbot(urls: List[str], max_pages_per_url: int = 50) -> List[dict]:
    """Process URLs by crawling them and return one combined document per URL."""
    combined_docs = []

    for url in urls:
        logger.info(f"Starting crawl for URL: {url}")

        # Create crawl job
        job_id, params = crawler_service.create_crawl_job(
            start_url=url,
            max_pages=max_pages_per_url,
            concurrency=5
        )

        try:
            # Run crawl job
            crawled_docs = await crawler_service.crawl_site(job_id, params)

            if crawled_docs:
                # Combine all crawled pages into a single document
                combined_content = ""
                titles = []

                for doc in crawled_docs:
                    page_title = doc.get("title", "")
                    titles.append(page_title)
                    page_content = doc.get("content", "")
                    combined_content += f"\n\n---\nTitle: {page_title}\n\n{page_content}"

                combined_doc = {
                    "content": combined_content.strip(),
                    "metadata": {
                        "source": url,
                        "source_name": f"url_{url}",
                        "title": titles[0] if titles else "Untitled",
                        "type": "web_crawl"
                    }
                }

                combined_docs.append(combined_doc)
                logger.info(f"Successfully created combined document for {url} with {len(crawled_docs)} pages.")
            else:
                logger.warning(f"No content crawled from {url}")

        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            continue

    return combined_docs