import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from models.schema import ChatRequest, FeedbackRequest, ChatLog, ChatHistory, UpdateChatLogs
from fastapi.responses import StreamingResponse
from fastapi import Request, Query
from datetime import datetime
from database import chat_logs_collection, feedback_collection
from typing import AsyncGenerator, List, Optional
from utils.vectorStore import get_vectorstore
from utils.rag import get_rag_system
import asyncio
from rate_limiter import limiter

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

def format_history_for_rag(history: List[ChatHistory]) -> List[tuple[str, str]]:
    formatted = []
    for i in range(0, len(history) - 1, 2):
        if i + 1 < len(history):
            if history[i].role == "user" and history[i + 1].role == "assistant":
                formatted.append((history[i].content, history[i + 1].content))
    return formatted


async def sse_event_generator(
    message: str,
    history: List[ChatHistory],
    client_email: str,
    chatbot_id: str
) -> AsyncGenerator[str, None]:
    try:
        rag_system = get_rag_system(client_email, chatbot_id)
        formatted_history = format_history_for_rag(history)

        # Wrap sync generator as async generator
        loop = asyncio.get_event_loop()
        gen = rag_system.get_answer_stream(
            query=message,
            chat_history=formatted_history,
            client_email=client_email,
            chatbot_id=chatbot_id
        )

        for token in gen:
            # SSE format: "data: <line>\n\n"
            yield f"data: {token}\n\n"
            await asyncio.sleep(0)  # Yield control to event loop

        # End of stream message (optional)
        yield "event: end\ndata: [DONE]\n\n"

    except Exception as e:
        logger.error(f"SSE chat streaming error: {str(e)}")
        yield f"event: error\ndata: An error occurred: {str(e)}\n\n"


@router.post("/chat/stream")
async def stream_chat_with_bot(request: ChatRequest):
    return StreamingResponse(
        sse_event_generator(
            message=request.message,
            history=request.history,
            client_email=request.client_email,
            chatbot_id=request.chatbot_id
        ),
        media_type="text/event-stream"
    )

@router.post("/chat", response_model=dict)
@limiter.limit("5/minute")
async def chat_with_bot(request: Request, chat_request: ChatRequest):
    """Send a message to chat with a specific chatbot."""
    client_email = chat_request.client_email
    chatbot_id = chat_request.chatbot_id
    message = chat_request.message
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if not get_vectorstore(client_email, chatbot_id):
        raise HTTPException(status_code=404, detail="This chatbot has no documents or does not exist")
    
    try:
        rag_system = get_rag_system(client_email, chatbot_id)
        
        formatted_history = []
        for i in range(0, len(chat_request.history) - 1, 2):
            if i + 1 < len(chat_request.history):
                if chat_request.history[i].role == "user" and chat_request.history[i + 1].role == "assistant":
                    formatted_history.append((chat_request.history[i].content, chat_request.history[i + 1].content))
        
        result = rag_system.get_answer(
            message, 
            chat_history=formatted_history,
            client_email=client_email, 
            chatbot_id=chatbot_id
        )
        
        # Save to MongoDB
        log_data = ChatLog(
            timestamp=datetime.utcnow(),
            client_email=client_email,
            chatbot_id=chatbot_id,
            question=message,
            answer=result["answer"],
            interaction_id=result.get("interaction_id"),
            history=chat_request.history
        )
        asyncio.create_task(insert_chat_log(log_data))

        return {
            "answer": result["answer"],
            "interaction_id": result.get("interaction_id"),
            "sources": result.get("sources", [])
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
    
async def insert_chat_log(log_data: ChatLog):
    await chat_logs_collection.insert_one(log_data.model_dump())

@router.get("/chat-logs", response_model=List[ChatLog])
@limiter.limit("20/minute")
async def get_chat_logs(
    request: Request,
    client_email: Optional[str] = Query(None),
    chatbot_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0)
):
    """Retrieve chat logs, optionally filtered by client email and/or chatbot ID."""
    print(f"Current user: {chat_logs_collection}")
    query = {}
    if client_email:
        query["client_email"] = client_email
    if chatbot_id:
        query["chatbot_id"] = chatbot_id

    try:
        cursor = chat_logs_collection.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        logs = await cursor.to_list(length=limit)
        return logs
    except Exception as e:
        logger.error(f"Failed to fetch chat logs: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving chat logs")
    
@router.put("/chat-logs", response_model=dict)
@limiter.limit("5/minute")
async def update_chat_log(request: UpdateChatLogs):
    """Update a specific chat log entry."""
    try:
        interaction_id = request.interaction_id
        answer = request.answer
        result = await chat_logs_collection.update_one(
            {"interaction_id": interaction_id},
            {"$set": {
                "answer": answer,
                "updated_at": datetime.utcnow()
            }}
        )

        return {"status": "success", "message": "Chat log updated successfully"}
    except Exception as e:
        logger.error(f"Error updating chat log: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating chat log: {str(e)}")

@router.post("/feedback", response_model=dict)
@limiter.limit("10/minute")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a chat interaction."""
    if not request.interaction_id:
        raise HTTPException(status_code=400, detail="Missing interaction ID")
    print(request.interaction_id, request.score, request.helpful, request.client_email, request.chatbot_id)
    try:
        rag_system = get_rag_system(request.client_email, request.chatbot_id)
        success = rag_system.add_feedback(request.interaction_id, request.score, request.helpful)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to record feedback")
        
        # Insert feedback into MongoDB
        feedback_data = request.model_dump()
        await feedback_collection.insert_one(feedback_data)
        
        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")