import uuid
import asyncio
import logging
from fastapi import APIRouter, HTTPException, File, Form, Depends, BackgroundTasks, Query, Request
from fastapi import UploadFile
from typing import List, Optional
from pydantic import EmailStr

from models.schema import ChatbotResponse, ChatbotCreateResponse
from utils.helpers import validate_email
from services.file_service import process_files, split_into_chunks
from services.crawler_service import crawler_service
from rate_limiter import limiter
from utils.vectorStore import (
    create_vectorstore, reset_vectorstore, get_vectorstore,
    list_client_chatbots
)
from utils.rag import get_rag_system
from database import chat_logs_collection
import shutil

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chatbots", tags=["chatbots"])

@router.get("", response_model=List[ChatbotResponse])
@limiter.limit("5/minute")
async def get_chatbots(request: Request, client_email: EmailStr = Depends(validate_email)):
    """List all chatbots for a client."""
    try:
        chatbots = list_client_chatbots(client_email)
        return [{"id": bot["id"], "name": bot["name"]} for bot in chatbots]
    except Exception as e:
        logger.error(f"Error fetching chatbots: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching chatbots: {str(e)}")

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

@router.post("", response_model=ChatbotCreateResponse)
@limiter.limit("5/minute")
async def create_chatbot(
    request: Request,
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    client_email: EmailStr = Form(...),
    files: Optional[List[UploadFile]] = File(default=None),
    urls: Optional[str] = Form(None),
    max_pages_per_url: int = Form(50)
):
    """Create a new chatbot with uploaded documents and/or crawled URLs."""
    
    # Parse URLs if provided
    url_list = [url.strip() for url in urls.split(',')] if urls else []

    # Filter out any empty files (with no filename)
    valid_files = []
    if files:
        for file in files:
            if isinstance(file, UploadFile) and file.filename.strip():
                valid_files.append(file)

    # Validation: Require at least one valid source
    if not valid_files and not url_list:
        raise HTTPException(
            status_code=400,
            detail="Either valid files or URLs must be provided"
        )

    chatbot_name = name or f"Chatbot {uuid.uuid4().hex[:6]}"
    chatbot_id = f"cb_{uuid.uuid4().hex[:10]}"
    
    all_docs = []
    processed_files = 0
    failed_files = 0
    cleanup_func = None
    crawl_job_id = None

    try:
        # Process URLs by crawling
        if url_list:
            logger.info(f"Processing {len(url_list)} URLs for crawling")
            crawled_docs = await process_urls_for_chatbot(url_list, max_pages_per_url)
            all_docs.extend(crawled_docs)
            logger.info(f"Crawled content from {len(crawled_docs)} pages")

        # Process uploaded files
        if valid_files:
            logger.info(f"Processing {len(valid_files)} uploaded files")
            file_docs, processed_files, failed_files, cleanup_func = await process_files(valid_files)
            all_docs.extend(file_docs)
            logger.info(f"Processed {processed_files} files, {failed_files} failed")

        # Schedule file cleanup if necessary
        if cleanup_func:
            background_tasks.add_task(cleanup_func)

        # Validate extracted content
        if not all_docs:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract any content from files or URLs"
            )

        # Vectorstore creation
        logger.info(f"Creating vector store with {len(all_docs)} documents")
        all_chunks = split_into_chunks(all_docs)
        create_vectorstore(all_chunks, client_email, chatbot_id, chatbot_name)

        # Setup RAG pipeline
        rag_system = get_rag_system(client_email, chatbot_id)
        rag_system.initialize_retriever(client_email, chatbot_id)
        rag_system.setup_rag_chain()

        # Build response message
        status_parts = []
        if processed_files > 0:
            status_parts.append(f"{processed_files} files processed")
        if url_list:
            status_parts.append(f"{len(url_list)} URLs crawled")

        status = f"Created chatbot '{chatbot_name}' ({', '.join(status_parts)}, {len(all_chunks)} chunks)"
        if failed_files > 0:
            status += f". {failed_files} file(s) failed."

        # Cleanup crawler output
        OUTPUT_DIR = "crawler_output"
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        logger.info(f"Chatbot created successfully: {chatbot_name} ({chatbot_id})")
        return ChatbotCreateResponse(
            status="success",
            message=status,
            chatbot_id=chatbot_id,
            chatbot_name=chatbot_name,
            crawl_job_id=crawl_job_id
        )

    except Exception as e:
        logger.error(f"Error creating chatbot: {str(e)}")
        if cleanup_func:
            background_tasks.add_task(cleanup_func)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating chatbot: {str(e)}"
        )
    
@router.post("/urls", response_model=ChatbotCreateResponse)
@limiter.limit("5/minute")
async def create_chatbot_url(
    request: Request,
    name: str = Form(...),
    client_email: EmailStr = Form(...),
    urls: Optional[str] = Form(None),
    max_pages_per_url: int = Form(50)
):
    """Create a new chatbot with uploaded documents and/or crawled URLs."""
    
    # Parse URLs if provided
    url_list = [url.strip() for url in urls.split(',')] if urls else []

    # Validation: Require at least one valid source
    if not url_list:
        raise HTTPException(
            status_code=400,
            detail="URLs must be provided"
        )

    chatbot_name = name or f"Chatbot {uuid.uuid4().hex[:6]}"
    chatbot_id = f"cb_{uuid.uuid4().hex[:10]}"
    
    all_docs = []
    crawl_job_id = None

    try:
        # Process URLs by crawling
        if url_list:
            logger.info(f"Processing {len(url_list)} URLs for crawling")
            crawled_docs = await process_urls_for_chatbot(url_list, max_pages_per_url)
            all_docs.extend(crawled_docs)
            logger.info(f"Crawled content from {len(crawled_docs)} pages")

        # Validate extracted content
        if not all_docs:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract any content from URLs"
            )

        # Vectorstore creation
        logger.info(f"Creating vector store with {len(all_docs)} documents")
        all_chunks = split_into_chunks(all_docs)
        create_vectorstore(all_chunks, client_email, chatbot_id, chatbot_name)

        # Setup RAG pipeline
        rag_system = get_rag_system(client_email, chatbot_id)
        rag_system.initialize_retriever(client_email, chatbot_id)
        rag_system.setup_rag_chain()

        # Build response message
        status_parts = []
        if url_list:
            status_parts.append(f"{len(url_list)} URLs crawled")

        status = f"Created chatbot '{chatbot_name}' ({', '.join(status_parts)}, {len(all_chunks)} chunks)"

        # Cleanup crawler output
        OUTPUT_DIR = "crawler_output"
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

        logger.info(f"Chatbot created successfully: {chatbot_name} ({chatbot_id})")
        return ChatbotCreateResponse(
            status="success",
            message=status,
            chatbot_id=chatbot_id,
            chatbot_name=chatbot_name,
            crawl_job_id=crawl_job_id
        )

    except Exception as e:
        logger.error(f"Error creating chatbot: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating chatbot: {str(e)}"
        )

# Additional crawler-related endpoints for the main backend
@router.get("/crawl-jobs/{job_id}")
async def get_crawl_job_status(job_id: str, client_email: EmailStr = Depends(validate_email)):
    """Get status of a crawl job"""
    job_status = crawler_service.get_job_status(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_status

@router.post("/test-crawl")
@limiter.limit("3/minute")
async def test_crawl(
    request: Request,
    url: str = Form(...),
    max_pages: int = Form(10),
    client_email: EmailStr = Form(...)
):
    """Test crawl a URL to see what content would be extracted"""
    try:
        # Validate email
        validate_email(client_email)
        
        job_id, params = crawler_service.create_crawl_job(url, max_pages, concurrency=3)
        results = await crawler_service.crawl_site(job_id, params)
        
        if not results:
            return {"status": "warning", "message": "No content could be crawled from the provided URL"}
        
        # Return sample of what was crawled
        sample_results = results[:3]  # First 3 pages as sample
        total_content_length = sum(len(doc['content']) for doc in results)
        
        return {
            "status": "success",
            "total_pages": len(results),
            "total_content_length": total_content_length,
            "sample_pages": [
                {
                    "url": doc["url"],
                    "title": doc["title"],
                    "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                }
                for doc in sample_results
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in test crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error testing crawl: {str(e)}")

@router.delete("/{chatbot_id}", response_model=dict)
@limiter.limit("5/minute")
async def remove_chatbot(request:Request, chatbot_id: str, client_email: EmailStr = Query(...)):
    """Delete a chatbot."""
    if not get_vectorstore(client_email, chatbot_id):
        raise HTTPException(status_code=404, detail=f"Chatbot ID {chatbot_id} not found")
    
    try:
        reset_vectorstore(client_email, chatbot_id)
        # Delete associated chat logs
        chat_logs_collection.delete_many({"chatbot_id": chatbot_id})
        logger.info(f"Deleted chat logs for chatbot: {chatbot_id}")
        return {"status": "success", "message": "Chatbot deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting chatbot: {str(e)}")

@router.get("/{chatbot_id}/stats", response_model=dict)
@limiter.limit("5/minute")
async def get_stats(request:Request, chatbot_id: str, client_email: EmailStr):
    """Get statistics for a specific chatbot."""
    if not get_vectorstore(client_email, chatbot_id):
        raise HTTPException(status_code=404, detail=f"Chatbot ID {chatbot_id} not found")
    
    try:
        rag_system = get_rag_system(client_email, chatbot_id)
        stats = rag_system.get_system_stats(client_email, chatbot_id)
        
        feedback = stats.get("feedback_analysis", {})
        if "message" in feedback:
            return {"status": "info", "message": feedback["message"]}
        
        return {
            "status": "success",
            "stats": {
                "total_interactions": feedback.get("total_interactions", 0),
                "interactions_with_feedback": feedback.get("interactions_with_feedback", 0),
                "helpful_percentage": feedback.get("helpful_percentage", 0),
                "average_score": feedback.get("average_score", 0),
                "documents_with_feedback": stats.get("documents_with_feedback", 0)
            }
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")