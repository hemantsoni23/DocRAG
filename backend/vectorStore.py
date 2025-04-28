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
BASE_CHROMA_PATH = "./backend/vectorstore"
DEFAULT_COLLECTION_NAME = "rag_docs"

# Thread-safe storage for client-specific vectorstores
client_vectorstores = {}

def get_client_chroma_path(client_email):
    """Get client-specific vector store path."""
    if not client_email:
        return os.path.join(BASE_CHROMA_PATH, "default")
    
    # Create a valid directory name from the email
    client_dir = client_email.replace('@', '_at_').replace('.', '_dot_')
    return os.path.join(BASE_CHROMA_PATH, client_dir)

def get_embeddings():
    """Initialize the Google Generative Embeddings."""
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_KEY not found in environment variables.")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

def reset_vectorstore(client_email=None):
    """Delete and reset the vector store for a specific client."""
    client_path = get_client_chroma_path(client_email)
    
    if os.path.exists(client_path):
        shutil.rmtree(client_path)
        logger.info(f"✅ Vectorstore reset for client: {client_email}")
    
    # Remove from memory cache if it exists
    if client_email in client_vectorstores:
        del client_vectorstores[client_email]

def create_vectorstore(documents, client_email=None):
    """Create a new vector store with provided documents for a specific client."""
    try:
        client_path = get_client_chroma_path(client_email)
        collection_name = DEFAULT_COLLECTION_NAME
        
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(client_path), exist_ok=True)
        
        logger.info(f"🔨 Creating vectorstore with {len(documents)} documents for client: {client_email}")
        embeddings = get_embeddings()
        
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=client_path,
            collection_name=collection_name
        )
        
        # Store in memory cache
        client_vectorstores[client_email] = db
        
        logger.info(f"✅ Vectorstore created for client: {client_email}")
        return db
    except Exception as e:
        logger.error(f"❌ Failed to create vectorstore for client {client_email}: {e}")
        raise

def get_vectorstore(client_email=None):
    """Load the existing vector store for a specific client."""
    # Check memory cache first
    if client_email in client_vectorstores:
        return client_vectorstores[client_email]
    
    client_path = get_client_chroma_path(client_email)
    collection_name = DEFAULT_COLLECTION_NAME
    
    if not os.path.exists(client_path):
        logger.warning(f"⚠️ No vectorstore found for client: {client_email}. Upload documents first.")
        return None
    
    try:
        embeddings = get_embeddings()
        db = Chroma(
            persist_directory=client_path,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # Store in memory cache
        client_vectorstores[client_email] = db
        
        logger.info(f"✅ Vectorstore loaded for client: {client_email}")
        return db
    except Exception as e:
        logger.error(f"❌ Failed to load vectorstore for client {client_email}: {e}")
        raise

def list_client_collections():
    """List all client collections that have been created."""
    if not os.path.exists(BASE_CHROMA_PATH):
        return []
    
    client_dirs = []
    for item in os.listdir(BASE_CHROMA_PATH):
        item_path = os.path.join(BASE_CHROMA_PATH, item)
        if os.path.isdir(item_path):
            # Convert directory name back to email format if possible
            client_id = item
            if '_at_' in client_id and '_dot_' in client_id:
                client_id = client_id.replace('_at_', '@').replace('_dot_', '.')
            client_dirs.append(client_id)
    
    return client_dirs