"""
Data models for web crawler functionality
"""
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, EmailStr


class CrawlConfig(BaseModel):
    """Configuration for web crawling"""
    max_pages: int = 100
    concurrency: int = 10
    max_depth: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "max_pages": 50,
                "concurrency": 5,
                "max_depth": 3
            }
        }


class CreateChatbotRequest(BaseModel):
    """Request model for creating chatbot from URLs"""
    name: str
    client_email: EmailStr
    urls: List[HttpUrl]
    crawl_config: Optional[CrawlConfig] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Website Chatbot",
                "client_email": "user@example.com",
                "urls": ["https://example.com", "https://docs.example.com"],
                "crawl_config": {
                    "max_pages": 50,
                    "concurrency": 5,
                    "max_depth": 2
                }
            }
        }


class UrlPreviewRequest(BaseModel):
    """Request model for URL content preview"""
    urls: List[HttpUrl]
    max_pages: int = 5
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": ["https://example.com"],
                "max_pages": 3
            }
        }


class CrawlResult(BaseModel):
    """Result of crawling operation"""
    url: str
    title: str
    content_length: int
    preview: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com",
                "title": "Example Website",
                "content_length": 1250,
                "preview": "This is a preview of the content..."
            }
        }


class ChatbotCreationResponse(BaseModel):
    """Response model for chatbot creation"""
    status: str
    message: str
    chatbot_id: str
    chatbot_name: str
    processed_files: int = 0
    processed_urls: int = 0
    failed_files: int = 0
    total_chunks: int
    source_urls: Optional[List[str]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Created chatbot 'My Chatbot' from 15 web pages, 245 chunks",
                "chatbot_id": "cb_abc123def4",
                "chatbot_name": "My Chatbot",
                "processed_files": 0,
                "processed_urls": 15,
                "failed_files": 0,
                "total_chunks": 245,
                "source_urls": ["https://example.com", "https://docs.example.com"]
            }
        }


class PreviewResponse(BaseModel):
    """Response model for URL preview"""
    status: str
    preview_count: int
    pages: List[CrawlResult]
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "preview_count": 3,
                "pages": [
                    {
                        "url": "https://example.com",
                        "title": "Example Website",
                        "content_length": 1250,
                        "preview": "This is a preview of the content..."
                    }
                ]
            }
        }