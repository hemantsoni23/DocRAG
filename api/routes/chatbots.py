import uuid
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from typing import List
from pydantic import EmailStr

from models.schema import ChatbotResponse
from utils.helpers import validate_email
from services.file_service import process_files, split_into_chunks

from utils.vectorStore import (
    create_vectorstore, reset_vectorstore, get_vectorstore,
    list_client_chatbots
)
from utils.rag import get_rag_system

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chatbots", tags=["chatbots"])

@router.get("", response_model=List[ChatbotResponse])
async def get_chatbots(client_email: EmailStr = Depends(validate_email)):
    """List all chatbots for a client."""
    try:
        chatbots = list_client_chatbots(client_email)
        return [{"id": bot["id"], "name": bot["name"]} for bot in chatbots]
    except Exception as e:
        logger.error(f"Error fetching chatbots: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching chatbots: {str(e)}")

@router.post("", response_model=dict)
async def create_chatbot(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    client_email: EmailStr = Form(...),
    files: List[UploadFile] = File(...)
):
    """Create a new chatbot with uploaded documents."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    chatbot_name = name or f"Chatbot {uuid.uuid4().hex[:6]}"
    chatbot_id = f"cb_{uuid.uuid4().hex[:10]}"
    
    all_docs, processed, failed, cleanup_func = await process_files(files)
    background_tasks.add_task(cleanup_func)
    
    if not all_docs:
        raise HTTPException(status_code=400, detail="Failed to extract any content from files")
    
    all_chunks = split_into_chunks(all_docs)
    create_vectorstore(all_chunks, client_email, chatbot_id, chatbot_name)
    
    rag_system = get_rag_system(client_email, chatbot_id)
    rag_system.initialize_retriever(client_email, chatbot_id)
    rag_system.setup_rag_chain()
    
    status = f"Created chatbot '{chatbot_name}' ({processed} files, {len(all_chunks)} chunks)"
    if failed:
        status += f". {failed} file(s) failed."
    
    return {
        "status": "success",
        "message": status,
        "chatbot_id": chatbot_id,
        "chatbot_name": chatbot_name
    }

@router.delete("/{chatbot_id}", response_model=dict)
async def remove_chatbot(chatbot_id: str, client_email: EmailStr):
    """Delete a chatbot."""
    if not get_vectorstore(client_email, chatbot_id):
        raise HTTPException(status_code=404, detail=f"Chatbot ID {chatbot_id} not found")
    
    try:
        reset_vectorstore(client_email, chatbot_id)
        return {"status": "success", "message": "Chatbot deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting chatbot: {str(e)}")

@router.get("/{chatbot_id}/stats", response_model=dict)
async def get_stats(chatbot_id: str, client_email: EmailStr):
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