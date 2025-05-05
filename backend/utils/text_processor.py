# backend/text_processor.py
import logging
import os
import uuid
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize logger
logger = logging.getLogger(__name__)

def split_documents(docs: List[Document], file_name: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks for processing and embedding.
    
    Args:
        docs: List of documents to split
        file_name: Original file name for metadata
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of split document chunks
    """
    try:
        # Generate a source document ID for all chunks from this file
        source_id = str(uuid.uuid4())
        source_name = os.path.basename(file_name)
        
        # Configure the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split documents into chunks
        chunks = []
        for doc in docs:
            # Preserve original metadata
            metadata = doc.metadata.copy() if doc.metadata else {}
            
            # Add source document identifiers
            metadata["source_id"] = source_id
            metadata["source_name"] = source_name
            metadata["original_source"] = file_name
            
            # Split this document
            doc_chunks = text_splitter.split_text(doc.page_content)
            
            # Create Document objects for each chunk
            for i, chunk_text in enumerate(doc_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk"] = i
                chunk_metadata["chunk_count"] = len(doc_chunks)
                
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                ))
        
        logger.info(f"✅ Split {len(docs)} documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"❌ Failed to split documents: {e}")
        raise Exception(f"Failed to split documents: {e}")