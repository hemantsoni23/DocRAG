# backend/vectorStore.py
import os
import shutil
import logging
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
CHROMA_PATH = "./backend/vectorstore"
COLLECTION_NAME = "rag_docs"

def get_embeddings():
    """Initialize the Google Generative Embeddings."""
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_KEY not found in environment variables.")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

def reset_vectorstore():
    """Delete and reset the local vector store."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info("✅ Vectorstore reset.")

def create_vectorstore(documents):
    """Create a new vector store with provided documents."""
    try:
        logger.info(f"🔨 Creating vectorstore with {len(documents)} documents...")
        embeddings = get_embeddings()
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME
        )
        logger.info("✅ Vectorstore created.")
        return db
    except Exception as e:
        logger.error(f"❌ Failed to create vectorstore: {e}")
        raise

def get_vectorstore():
    """Load the existing vector store."""
    if not os.path.exists(CHROMA_PATH):
        logger.warning("⚠️ No vectorstore found. Upload documents first.")
        return None
    try:
        embeddings = get_embeddings()
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        logger.info("✅ Vectorstore loaded.")
        return db
    except Exception as e:
        logger.error(f"❌ Failed to load vectorstore: {e}")
        raise
