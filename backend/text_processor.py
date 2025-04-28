import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize logger
logger = logging.getLogger(__name__)

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    """Split documents into manageable text chunks."""
    try:
        logger.info(f"✂️ Splitting {len(documents)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"✅ Generated {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"❌ Error splitting documents: {e}")
        raise
