# backend/vectorStore.py
import os
import shutil
import logging
import threading
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# Initialize logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
BASE_CHROMA_PATH = "./backend/vectorstore"
DEFAULT_COLLECTION_NAME = "rag_docs"

# Thread lock
_lock = threading.RLock()

@dataclass
class DocumentInfo:
    id: str
    name: str
    chunk_ids: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ChatbotMetadata:
    name: str
    created: float
    documents: int
    last_updated: Optional[float] = None
    document_list: Optional[List[DocumentInfo]] = None 

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# In-memory caches
client_vectorstores: Dict[str, Dict[str, Chroma]] = {}
metadata_cache: Dict[str, Dict[str, ChatbotMetadata]] = {}

# ---------- Helper Functions ----------

def _normalize_client_id(client_email: str) -> str:
    return client_email.replace('@', '_at_').replace('.', '_dot_') if client_email else "default"

def _get_chatbot_path(client_email: Optional[str], chatbot_id: str) -> str:
    client_dir = _normalize_client_id(client_email) if client_email else "default"
    return os.path.join(BASE_CHROMA_PATH, client_dir, chatbot_id)

def _ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_KEY not found in environment variables.")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

def _save_metadata(client_email: str, chatbot_id: str, metadata: ChatbotMetadata) -> None:
    chatbot_path = _get_chatbot_path(client_email, chatbot_id)
    _ensure_directory(chatbot_path)
    metadata_file = os.path.join(chatbot_path, "metadata.json")
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        with _lock:
            metadata_cache.setdefault(client_email, {})[chatbot_id] = metadata
    except Exception as e:
        logger.error(f"Error saving metadata for {client_email}/{chatbot_id}: {e}")

def _load_metadata(client_email: str, chatbot_id: str) -> Optional[ChatbotMetadata]:
    with _lock:
        if client_email in metadata_cache and chatbot_id in metadata_cache[client_email]:
            return metadata_cache[client_email][chatbot_id]

    chatbot_path = _get_chatbot_path(client_email, chatbot_id)
    metadata_file = os.path.join(chatbot_path, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                
                # Handle conversion from old format to new format if needed
                if 'document_list' in data and data['document_list'] and isinstance(data['document_list'][0], dict):
                    if 'chunk_ids' not in data['document_list'][0]:
                        # Convert old format to new format
                        updated_docs = []
                        for doc in data['document_list']:
                            updated_docs.append({
                                'id': doc['id'],
                                'name': doc['name'],
                                'chunk_ids': [doc['id']],  # Initially, assume one chunk per doc
                                'timestamp': time.time()
                            })
                        data['document_list'] = updated_docs
                
                metadata = ChatbotMetadata(**data)
                with _lock:
                    metadata_cache.setdefault(client_email, {})[chatbot_id] = metadata
                return metadata
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_file}: {e}")
    return None

def _remove_from_caches(client_email: str, chatbot_id: str) -> None:
    with _lock:
        client_vectorstores.get(client_email, {}).pop(chatbot_id, None)
        metadata_cache.get(client_email, {}).pop(chatbot_id, None)

def _initialize_chroma(path: str) -> Chroma:
    embeddings = _get_embeddings()
    return Chroma(persist_directory=path, embedding_function=embeddings, collection_name=DEFAULT_COLLECTION_NAME)

# ---------- Core Functions ----------

def create_vectorstore(documents: List[Document], client_email: Optional[str], chatbot_id: str = "default", chatbot_name: Optional[str] = None) -> Chroma:
    chatbot_path = _get_chatbot_path(client_email, chatbot_id)
    reset_vectorstore(client_email, chatbot_id)
    _ensure_directory(chatbot_path)

    metadata = ChatbotMetadata(
        name=chatbot_name or chatbot_id,
        created=time.time(),
        documents=0,
        last_updated=time.time(),
        document_list=[]
    )

    # Group documents by their source name to ensure consistency
    docs_by_source = {}
    source_name_to_id = {}  # Map to maintain name->id consistency
    
    for doc in documents:
        # Extract source document name for consistent identification
        source_name = doc.metadata.get('source_name', doc.metadata.get('source', 'Unknown Document'))
        
        # Use existing source_id if provided, otherwise check our mapping or generate new
        source_doc_id = doc.metadata.get('source_id')
        if not source_doc_id:
            if source_name in source_name_to_id:
                source_doc_id = source_name_to_id[source_name]
            else:
                source_doc_id = str(uuid.uuid4())
                source_name_to_id[source_name] = source_doc_id
        
        # Create chunk ID for this specific chunk
        chunk_id = str(uuid.uuid4())
        doc.metadata['id'] = chunk_id
        doc.metadata['source_id'] = source_doc_id
        doc.metadata['source_name'] = source_name
        
        if source_doc_id not in docs_by_source:
            docs_by_source[source_doc_id] = {
                'name': source_name,
                'chunks': [],
                'chunk_ids': []
            }
        
        docs_by_source[source_doc_id]['chunks'].append(doc)
        docs_by_source[source_doc_id]['chunk_ids'].append(chunk_id)

    # Create document list for metadata
    doc_list = []
    all_chunks = []
    all_ids = []
    
    for source_id, source_info in docs_by_source.items():
        doc_list.append({
            "id": source_id,
            "name": source_info['name'],
            "chunk_ids": source_info['chunk_ids'],
            "timestamp": time.time()
        })
        all_chunks.extend(source_info['chunks'])
        all_ids.extend(source_info['chunk_ids'])

    metadata.document_list = doc_list
    metadata.documents = len(metadata.document_list)
    _save_metadata(client_email, chatbot_id, metadata)

    db = Chroma.from_documents(
        all_chunks,
        embedding=_get_embeddings(),
        ids=all_ids,
        persist_directory=chatbot_path,
        collection_name=DEFAULT_COLLECTION_NAME
    )

    with _lock:
        client_vectorstores.setdefault(client_email, {})[chatbot_id] = db

    logger.info(f"✅ Created vectorstore for {client_email}/{chatbot_id}")
    return db

def update_vectorstore(documents: List[Document], client_email: Optional[str], chatbot_id: str = "default") -> Optional[Chroma]:
    db = get_vectorstore(client_email, chatbot_id)
    if db is None:
        return create_vectorstore(documents, client_email, chatbot_id)

    try:
        with _lock:
            # Load existing metadata
            metadata = _load_metadata(client_email, chatbot_id)
            if not metadata:
                metadata = ChatbotMetadata(
                    name=chatbot_id,
                    created=time.time(),
                    documents=0,
                    last_updated=time.time(),
                    document_list=[]
                )
            
            if metadata.document_list is None:
                metadata.document_list = []
                
            # Group documents by their source document ID
            docs_by_source = {}
            for doc in documents:
                # Extract source document name for consistent identification
                source_name = doc.metadata.get('source_name', doc.metadata.get('source', 'Unknown Document'))
                
                # Use existing source_id if provided, otherwise generate from name
                source_doc_id = doc.metadata.get('source_id')
                
                # If no source_id, check if we already have this document by name
                if not source_doc_id and metadata.document_list:
                    # Try to find existing document with same name
                    existing_doc = next((doc_info for doc_info in metadata.document_list 
                                       if doc_info["name"] == source_name), None)
                    if existing_doc:
                        source_doc_id = existing_doc["id"]
                
                # If still no source_id, generate a new one
                if not source_doc_id:
                    source_doc_id = str(uuid.uuid4())
                
                # Create chunk ID for this specific chunk
                chunk_id = str(uuid.uuid4())
                doc.metadata['id'] = chunk_id
                doc.metadata['source_id'] = source_doc_id
                doc.metadata['source_name'] = source_name
                
                if source_doc_id not in docs_by_source:
                    docs_by_source[source_doc_id] = {
                        'name': source_name,
                        'chunks': [],
                        'chunk_ids': []
                    }
                
                docs_by_source[source_doc_id]['chunks'].append(doc)
                docs_by_source[source_doc_id]['chunk_ids'].append(chunk_id)
            
            # Add new documents to metadata
            all_chunks = []
            all_ids = []
            
            for source_id, source_info in docs_by_source.items():
                # Check if document already exists in metadata
                existing_doc = next((doc for doc in metadata.document_list if doc["id"] == source_id), None)
                
                if existing_doc:
                    # Update existing document
                    existing_doc["chunk_ids"].extend(source_info['chunk_ids'])
                else:
                    # Add new document
                    metadata.document_list.append({
                        "id": source_id,
                        "name": source_info['name'],
                        "chunk_ids": source_info['chunk_ids'],
                        "timestamp": time.time()
                    })
                
                all_chunks.extend(source_info['chunks'])
                all_ids.extend(source_info['chunk_ids'])
            
            # Add documents to vector store
            db.add_documents(all_chunks, ids=all_ids)
            
            # Update metadata
            metadata.documents = len(metadata.document_list)
            metadata.last_updated = time.time()
            _save_metadata(client_email, chatbot_id, metadata)
            
        logger.info(f"✅ Updated vectorstore for {client_email}/{chatbot_id}")
        return db
    except Exception as e:
        logger.error(f"Failed to update vectorstore for {client_email}/{chatbot_id}: {e}")
        raise

def get_vectorstore(client_email: Optional[str], chatbot_id: str = "default") -> Optional[Chroma]:
    with _lock:
        if client_email in client_vectorstores and chatbot_id in client_vectorstores[client_email]:
            return client_vectorstores[client_email][chatbot_id]

    chatbot_path = _get_chatbot_path(client_email, chatbot_id)
    if not os.path.exists(chatbot_path):
        logger.warning(f"No vectorstore found for {client_email}/{chatbot_id}")
        return None

    try:
        db = _initialize_chroma(chatbot_path)
        with _lock:
            client_vectorstores.setdefault(client_email, {})[chatbot_id] = db
        return db
    except Exception as e:
        logger.error(f"Failed to load vectorstore for {client_email}/{chatbot_id}: {e}")
        return None

def reset_vectorstore(client_email: Optional[str], chatbot_id: str = "default") -> None:
    chatbot_path = _get_chatbot_path(client_email, chatbot_id)
    _remove_from_caches(client_email, chatbot_id)
    if os.path.exists(chatbot_path):
        shutil.rmtree(chatbot_path)
        logger.info(f"✅ Reset vectorstore for {client_email}/{chatbot_id}")

def delete_document(client_email: str, chatbot_id: str, document_id: str) -> bool:
    db = get_vectorstore(client_email, chatbot_id)
    if db is None:
        logger.warning(f"Vectorstore not found for {client_email}/{chatbot_id}")
        return False

    try:
        # Load metadata to get chunk IDs for this document
        metadata = _load_metadata(client_email, chatbot_id)
        if not metadata or not metadata.document_list:
            logger.warning(f"No metadata found for {client_email}/{chatbot_id}")
            return False
        
        # Find the document in the metadata
        doc_info = next((doc for doc in metadata.document_list if doc["id"] == document_id), None)
        if not doc_info:
            logger.warning(f"Document {document_id} not found in metadata for {client_email}/{chatbot_id}")
            return False
        
        # Get all chunk IDs for this document
        chunk_ids = doc_info.get("chunk_ids", [])
        if not chunk_ids:                   
            logger.warning(f"No chunk IDs found for document {document_id}")
            return False
        
        # Delete all chunks from the vector store
        db.delete(ids=chunk_ids)
        
        # Update metadata
        metadata.document_list = [doc for doc in metadata.document_list if doc["id"] != document_id]
        metadata.documents = len(metadata.document_list)
        metadata.last_updated = time.time()
        _save_metadata(client_email, chatbot_id, metadata)
        
        logger.info(f"✅ Deleted document {document_id} with {len(chunk_ids)} chunks from {client_email}/{chatbot_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete document {document_id} from {client_email}/{chatbot_id}: {e}")
        return False

# ---------- Info & Listing ----------

def list_client_chatbots(client_email: Optional[str]) -> List[Dict[str, Any]]:
    client_dir = _normalize_client_id(client_email)
    client_path = os.path.join(BASE_CHROMA_PATH, client_dir)
    if not os.path.exists(client_path):
        return []

    chatbots = []
    for chatbot_id in os.listdir(client_path):
        chatbot_path = os.path.join(client_path, chatbot_id)
        if os.path.isdir(chatbot_path):
            metadata = _load_metadata(client_email, chatbot_id) or ChatbotMetadata(
                name=chatbot_id,
                created=os.path.getctime(chatbot_path),
                documents=0
            )
            chatbots.append({
                "id": chatbot_id,
                **metadata.to_dict()
            })

    return sorted(chatbots, key=lambda x: x["created"], reverse=True)

def list_all_clients() -> List[str]:
    if not os.path.exists(BASE_CHROMA_PATH):
        return []

    clients = []
    for client_folder in os.listdir(BASE_CHROMA_PATH):
        client_path = os.path.join(BASE_CHROMA_PATH, client_folder)
        if os.path.isdir(client_path):
            if "_at_" in client_folder and "_dot_" in client_folder:
                email = client_folder.replace('_at_', '@').replace('_dot_', '.')
                clients.append(email)
            else:
                clients.append(client_folder)
    return clients

def get_chatbot_info(client_email: str, chatbot_id: str) -> Optional[Dict[str, Any]]:
    metadata = _load_metadata(client_email, chatbot_id)
    if not metadata:
        return None

    info = metadata.to_dict()
    info["id"] = chatbot_id
    db = get_vectorstore(client_email, chatbot_id)
    if db:
        try:
            info["vector_count"] = db._collection.count()
        except Exception as e:
            logger.error(f"Error getting vector count for {client_email}/{chatbot_id}: {e}")
    return info
