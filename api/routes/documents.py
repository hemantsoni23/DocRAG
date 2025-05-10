import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List
from pydantic import EmailStr

from models.schema import DocumentResponse
from services.file_service import process_files, split_into_chunks

from utils.vectorStore import (
    get_vectorstore, update_vectorstore, delete_document, _load_metadata
)

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chatbots/{chatbot_id}/documents", tags=["documents"])

@router.post("", response_model=dict)
async def add_documents(
    chatbot_id: str,
    background_tasks: BackgroundTasks,
    client_email: EmailStr = Form(...),
    files: List[UploadFile] = File(...)
):
    """Add documents to an existing chatbot."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    if not get_vectorstore(client_email, chatbot_id):
        raise HTTPException(status_code=404, detail=f"Chatbot ID {chatbot_id} not found")
    
    all_docs, processed, failed, cleanup_func = await process_files(files)
    background_tasks.add_task(cleanup_func)
    
    if not all_docs:
        raise HTTPException(status_code=400, detail="Failed to extract any content from files")
    
    all_chunks = split_into_chunks(all_docs)
    update_vectorstore(all_chunks, client_email, chatbot_id)
    
    status = f"Added {processed} files to chatbot"
    if failed:
        status += f". {failed} file(s) failed."
    
    return {"status": "success", "message": status}

@router.get("", response_model=List[DocumentResponse])
async def list_documents(chatbot_id: str, client_email: EmailStr):
    """List all documents in a chatbot."""
    if not get_vectorstore(client_email, chatbot_id):
        raise HTTPException(status_code=404, detail=f"Chatbot ID {chatbot_id} not found")
    
    metadata = _load_metadata(client_email, chatbot_id)
    if not metadata or not metadata.document_list:
        return []
    
    # Create a dictionary to track unique document names
    unique_docs = {}
    
    # Group documents by name
    for doc in metadata.document_list:
        doc_name = doc["name"]
        if doc_name not in unique_docs:
            unique_docs[doc_name] = {
                "id": doc_name,  # Use name as ID for consistency with UI
                "name": doc_name,
                "original_ids": [doc["id"]]
            }
        else:
            unique_docs[doc_name]["original_ids"].append(doc["id"])
    
    return [{"id": info["id"], "name": info["name"]} for info in unique_docs.values()]

@router.delete("/{document_name}", response_model=dict)
async def remove_document(chatbot_id: str, document_name: str, client_email: EmailStr):
    """Delete a document from a chatbot."""
    if not get_vectorstore(client_email, chatbot_id):
        raise HTTPException(status_code=404, detail=f"Chatbot ID {chatbot_id} not found")
    
    try:
        metadata = _load_metadata(client_email, chatbot_id)
        if not metadata or not metadata.document_list:
            raise HTTPException(status_code=404, detail="No documents found for this chatbot")
        
        # Find all documents with this name
        docs_to_delete = [doc for doc in metadata.document_list if doc["name"] == document_name]
        
        if not docs_to_delete:
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
            
        # Count total chunks to be removed
        total_chunks = sum(len(doc.get("chunk_ids", [])) for doc in docs_to_delete)
        
        # Delete each document and its chunks
        success = True
        for doc in docs_to_delete:
            doc_id = doc["id"]
            result = delete_document(client_email, chatbot_id, doc_id)
            if not result:
                success = False
                
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to delete some parts of document: {document_name}")
            
        return {
            "status": "success", 
            "message": f"Deleted document: {document_name} ({total_chunks} chunks removed)"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")