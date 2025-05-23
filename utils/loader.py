import os
import logging
from pdfminer.high_level import extract_text
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

# Initialize logger
logger = logging.getLogger(__name__)

def load_pdf(file_path):
    """Load content from a PDF file."""
    try:
        logger.info(f"📄 Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as primary_error:
        logger.warning(f"⚠️ PyPDFLoader failed: {primary_error}. Trying fallback...")
        try:
            text = extract_text(file_path)
            return [Document(page_content=text, metadata={"source": file_path})]
        except Exception as fallback_error:
            logger.error(f"❌ Fallback loading failed: {fallback_error}")
            raise Exception(f"Failed to load PDF: {fallback_error}")

def load_text(file_path):
    """Load content from a plain text file."""
    try:
        logger.info(f"📄 Loading text file: {file_path}")
        loader = TextLoader(file_path)
        return loader.load()
    except Exception as e:
        logger.error(f"❌ Failed to load text file: {e}")
        # Fallback to manual loading
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": file_path})]
        except Exception as fallback_error:
            logger.error(f"❌ Fallback loading failed: {fallback_error}")
            raise Exception(f"Failed to load text file: {fallback_error}")

def load_docx(file_path):
    """Load content from a DOCX file."""
    try:
        logger.info(f"📄 Loading DOCX file: {file_path}")
        # Try to import and use Docx2txtLoader
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        except ImportError:
            logger.warning("Docx2txtLoader not available. Trying UnstructuredWordDocumentLoader...")
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
    except Exception as e:
        logger.error(f"❌ Failed to load DOCX file: {e}")
        raise Exception(f"Failed to load DOCX file: {e}")

def load_doc(file_path):
    """Load content from a DOC file."""
    try:
        logger.info(f"📄 Loading DOC file: {file_path}")
        # Try to use UnstructuredWordDocumentLoader for .doc files
        try:
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        except ImportError:
            logger.warning("UnstructuredWordDocumentLoader not available.")
            raise ImportError("Required libraries for .doc files are not installed.")
    except Exception as e:
        logger.error(f"❌ Failed to load DOC file: {e}")
        raise Exception(f"Failed to load DOC file: {e}")

def load_document(file_path):
    """Load content from various document formats based on file extension."""
    _, ext = os.path.splitext(file_path.lower())
    
    if ext == '.pdf':
        return load_pdf(file_path)
    elif ext == '.txt':
        return load_text(file_path)
    elif ext == '.docx':
        return load_docx(file_path)
    elif ext == '.doc':
        return load_doc(file_path)
    else:
        logger.error(f"❌ Unsupported file extension: {ext}")
        raise ValueError(f"Unsupported file extension: {ext}")