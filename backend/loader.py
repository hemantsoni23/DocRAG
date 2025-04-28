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
