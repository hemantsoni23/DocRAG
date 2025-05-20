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

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat/stream")
async def stream_chat_with_bot(request: ChatRequest):
    """Stream a message to chat with a specific chatbot."""

    client_email = request.client_email
    chatbot_id = request.chatbot_id
    message = request.message

    if not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not get_vectorstore(client_email, chatbot_id):
        raise HTTPException(status_code=404, detail="This chatbot has no documents or does not exist")

    rag_system = get_rag_system(client_email, chatbot_id)

    # Format chat history
    formatted_history = []
    for i in range(0, len(request.history) - 1, 2):
        if i + 1 < len(request.history):
            if request.history[i].role == "user" and request.history[i + 1].role == "assistant":
                formatted_history.append((request.history[i].content, request.history[i + 1].content))

    async def token_generator() -> AsyncGenerator[bytes, None]:
        try:
            def stream_callback(token: str):
                yield_data = f"data: {token}\n\n"
                yield_bytes = yield_data.encode("utf-8")
                yield yield_bytes  # Yield SSE-compatible stream token-by-token
            
            # Bridge callback-style streaming with generator
            loop = asyncio.get_event_loop()
            queue = asyncio.Queue()

            def enqueue_token(token):
                loop.call_soon_threadsafe(queue.put_nowait, token)

            def callback_handler(token: str):
                enqueue_token(token)

            async def generate():
                try:
                    rag_system.get_answer(
                        message,
                        chat_history=formatted_history,
                        stream_callback=callback_handler,
                        client_email=client_email,
                        chatbot_id=chatbot_id,
                    )
                    await queue.put(None)  # Sentinel for end of stream
                except Exception as e:
                    logger.error(f"Streaming failed: {e}")
                    await queue.put(None)

            asyncio.create_task(generate())

            while True:
                token = await queue.get()
                if token is None:
                    break
                yield f"data: {token}\n\n".encode("utf-8")

        except Exception as e:
            logger.error(f"Error during stream: {e}")
            yield f"data: [Error] {str(e)}\n\n".encode("utf-8")

    return StreamingResponse(token_generator(), media_type="text/event-stream")

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
        
        # Save to MongoDB
        log_data = ChatLog(
            timestamp=datetime.utcnow(),
            client_email=client_email,
            chatbot_id=chatbot_id,
            question=message,
            answer=result["answer"],
            interaction_id=result.get("interaction_id"),
            history=request.history
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
async def get_chat_logs(
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