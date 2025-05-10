import logging
from fastapi import APIRouter, HTTPException
from models.schema import ChatRequest, FeedbackRequest

from utils.vectorStore import get_vectorstore
from utils.rag import get_rag_system

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat", response_model=dict)
async def chat_with_bot(request: ChatRequest):
    """Send a message to chat with a specific chatbot."""
    client_email = request.client_email
    chatbot_id = request.chatbot_id
    message = request.message
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not get_vectorstore(client_email, chatbot_id):
        raise HTTPException(status_code=404, detail="This chatbot has no documents or does not exist")
    
    try:
        rag_system = get_rag_system(client_email, chatbot_id)
        
        # Convert the history format
        formatted_history = []
        for i in range(0, len(request.history) - 1, 2):
            if i + 1 < len(request.history):
                if request.history[i].role == "user" and request.history[i + 1].role == "assistant":
                    formatted_history.append((request.history[i].content, request.history[i + 1].content))
        
        result = rag_system.get_answer(
            message, 
            chat_history=formatted_history,
            client_email=client_email, 
            chatbot_id=chatbot_id
        )
        
        return {
            "answer": result["answer"],
            "interaction_id": result.get("interaction_id"),
            "sources": result.get("sources", [])
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.post("/feedback", response_model=dict)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a chat interaction."""
    if not request.interaction_id:
        raise HTTPException(status_code=400, detail="Missing interaction ID")
    
    try:
        rag_system = get_rag_system(request.client_email, request.chatbot_id)
        success = rag_system.add_feedback(request.interaction_id, request.score, request.helpful)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record feedback")
        
        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")