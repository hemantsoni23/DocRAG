import os
import uuid
import shutil
import logging
from pathlib import Path
from fastapi import UploadFile
from typing import List, Tuple, Callable

from utils.loader import load_document
from utils.text_processor import split_documents
from langchain_core.documents import Document

# Set up logging
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_FILE_EXTENSIONS = [".pdf", ".txt", ".doc", ".docx"]
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def get_file_loader(file_path):
    """Return appropriate document loader based on file extension."""
    ext = os.path.splitext(file_path.lower())[1]
    return load_document if ext in SUPPORTED_FILE_EXTENSIONS else None

async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save an uploaded file to disk and return the path."""
    temp_file = UPLOAD_DIR / f"{uuid.uuid4()}{os.path.splitext(upload_file.filename)[1]}"
    
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return temp_file

async def process_files(files: List[UploadFile]) -> Tuple[list, int, int, Callable]:
    """Process uploaded files and return documents, success count, and failure count."""
    all_docs, processed, failed = [], 0, 0
    saved_paths = []
    
    for file in files:
        try:
            temp_path = await save_upload_file(file)
            saved_paths.append(temp_path)
            
            loader_func = get_file_loader(file.filename)
            if not loader_func:
                logger.warning(f"Unsupported file: {file.filename}")
                failed += 1
                continue

            docs = loader_func(str(temp_path))
            if docs:
                # Update metadata to include original filename
                for doc in docs:
                    doc.metadata["source"] = file.filename
                all_docs.extend(docs)
                processed += 1
            else:
                logger.warning(f"No content from {file.filename}")
                failed += 1
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            failed += 1
    
    # Clean up temp files in a background task
    def cleanup_files():
        for path in saved_paths:
            try:
                os.unlink(path)
            except Exception as e:
                logger.error(f"Error deleting temp file {path}: {e}")
    
    return all_docs, processed, failed, cleanup_files

def split_into_chunks(docs):
    """Split documents into chunks for vectorstore."""
    all_chunks = []

    for doc in docs:
        try:
            # Convert dict to LangChain Document if needed
            if isinstance(doc, dict):
                document = Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
            else:
                document = doc  # already a Document instance

            file_name = document.metadata.get("source", "unknown_file")

            # Call your custom split_documents function
            doc_chunks = split_documents([document], file_name)
            all_chunks.extend(doc_chunks)

        except Exception as e:
            print(f"❌ Error processing document: {e}")
            logging.error(f"❌ Failed to split documents: {e}")

    return all_chunks