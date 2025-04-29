# api/routes/chat.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import logging
import asyncio

from backend.vectorStore import get_vectorstore
from backend.rag import get_rag_system

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Pydantic models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    chat_state: dict = {}
    client_email: str

class ChatResponse(BaseModel):
    updated_history: List[ChatMessage]
    chat_state: dict

# --- POST chat endpoint ---
@router.post("/", response_model=ChatResponse)
def chat_endpoint(chat_req: ChatRequest):
    if not chat_req.message.strip():
        return ChatResponse(
            updated_history=[ChatMessage(role="assistant", content="⚠️ Please enter a message.")],
            chat_state=chat_req.chat_state
        )
    
    if not chat_req.client_email:
        return ChatResponse(
            updated_history=[ChatMessage(role="assistant", content="⚠️ Email not set. Please log in first.")],
            chat_state=chat_req.chat_state
        )
    
    if not get_vectorstore(chat_req.client_email):
        return ChatResponse(
            updated_history=[ChatMessage(role="assistant", content="⚠️ Please upload documents first.")],
            chat_state=chat_req.chat_state
        )
    
    try:
        rag_system = get_rag_system(chat_req.client_email)
        # Format chat history as a list of (user, assistant) pairs
        formatted_history = []
        hist = chat_req.history
        for idx in range(len(hist) - 1):
            if hist[idx].role == "user" and (idx + 1) < len(hist) and hist[idx + 1].role == "assistant":
                formatted_history.append((hist[idx].content, hist[idx + 1].content))
                
        result = rag_system.get_answer(
            chat_req.message,
            chat_history=formatted_history,
            client_email=chat_req.client_email
        )
        interaction_id = result.get("interaction_id")
        chat_req.chat_state["current_interaction_id"] = interaction_id
        
        updated_history = chat_req.history.copy()
        updated_history.append(ChatMessage(role="user", content=chat_req.message))
        updated_history.append(ChatMessage(role="assistant", content=result["answer"]))
        
        return ChatResponse(updated_history=updated_history, chat_state=chat_req.chat_state)
    
    except Exception as e:
        logger.error(f"Error during chat for client {chat_req.client_email}: {e}")
        return ChatResponse(
            updated_history=[ChatMessage(role="assistant", content=f"❌ An error occurred while generating a response: {str(e)[:100]}...")],
            chat_state=chat_req.chat_state
        )

# --- Optional SSE streaming endpoint ---
@router.get("/stream")
async def stream_chat(request: Request, message: str, client_email: str):
    if not message.strip():
        async def error_stream():
            yield "data: ⚠️ Please enter a message.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    if not client_email:
        async def error_stream():
            yield "data: ⚠️ Email not set. Please log in first.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    if not get_vectorstore(client_email):
        async def error_stream():
            yield "data: ⚠️ Please upload documents first.\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    rag_system = get_rag_system(client_email)

    async def event_generator():
        try:
            async for chunk in rag_system.generate_response(message, client_email):
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: ❌ Error: {str(e)[:100]}...\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
