from pydantic import BaseModel, EmailStr, Field, validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime

class ChatbotCreate(BaseModel):
    name: str = Field(..., description="Name of the chatbot")
    client_email: EmailStr = Field(..., description="Email of the client")

class ChatHistory(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistory] = []
    client_email: EmailStr
    chatbot_id: str

class FeedbackRequest(BaseModel):
    interaction_id: str
    score: int = Field(..., ge=0, le=5)
    helpful: bool = True
    client_email: EmailStr
    chatbot_id: str

class DocumentDelete(BaseModel):
    document_id: str
    client_email: EmailStr
    chatbot_id: str

class ChatbotDelete(BaseModel):
    chatbot_id: str
    client_email: EmailStr

class ChatbotResponse(BaseModel):
    id: str
    name: str

class DocumentResponse(BaseModel):
    id: str
    name: str

class SystemStats(BaseModel):
    total_interactions: int = 0
    interactions_with_feedback: int = 0
    helpful_percentage: float = 0.0
    average_score: float = 0.0
    documents_with_feedback: int = 0

class ChatLog(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    client_email: EmailStr
    chatbot_id: str
    question: str
    answer: str
    interaction_id: Optional[str] = None
    history: Optional[List[ChatHistory]] = []

class UpdateChatLogs(BaseModel):
    interaction_id: Optional[str]
    answer: str