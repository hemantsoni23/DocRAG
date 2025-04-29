from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
import logging
import os

# Import your backend functions
from backend.loader import load_document
from backend.text_processor import split_documents
from backend.vectorStore import create_vectorstore, get_vectorstore
from backend.rag import get_rag_system

router = APIRouter()
logger = logging.getLogger(__name__)

def get_file_loader(file_path: str):
    return load_document(file_path)

@router.post("/upload")
async def upload_documents(client_email: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    
    all_docs = []
    processed_files = 0
    failed_files = 0

    for file in files:
        filename = file.filename
        loader_func = get_file_loader(filename)
        if loader_func:
            try:
                # Save the uploaded file temporarily because your loader expects a filepath.
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name
                docs = loader_func(tmp_path)
                os.unlink(tmp_path)  
                if docs:
                    all_docs.extend(docs)
                    processed_files += 1
                else:
                    failed_files += 1
                    logger.warning(f"No content extracted from {filename}")
            except Exception as e:
                failed_files += 1
                logger.error(f"Error processing {filename}: {e}")
        else:
            failed_files += 1
            logger.warning(f"Unsupported file type for {filename}")
    
    if not all_docs:
        raise HTTPException(status_code=400, detail="Failed to extract content from the uploaded documents.")

    chunks = split_documents(all_docs)

    # Create vectorstore for this specific client
    create_vectorstore(chunks, client_email)

    # Get client-specific RAG system
    rag_system = get_rag_system(client_email)
    rag_system.initialize_retriever(client_email)
    rag_system.setup_rag_chain()

    status_message = f"Successfully processed {processed_files} document(s) into {len(chunks)} chunks for {client_email}."
    if failed_files > 0:
        status_message += f" Failed to process {failed_files} file(s)."
    
    return {"message": status_message}
