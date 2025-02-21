import os
import glob
from langchain_community.document_loaders import (
    TextLoader, UnstructuredHTMLLoader, UnstructuredPDFLoader
)
from common.config import DATA_DIR
import logging
from typing import List
from langchain.schema import Document

logger = logging.getLogger(__name__)

# Supported file types and their corresponding loaders
FILE_LOADERS = {
    ".pdf": UnstructuredPDFLoader,
    ".txt": lambda file_path: TextLoader(file_path, encoding="UTF-8"),
    ".html": UnstructuredHTMLLoader
}

def load_local_files() -> List[Document]:
    """Loads local documents from the filesystem."""
    documents = []
    logger.info("Starting to load local files from the directory: %s", DATA_DIR)

    for ext, loader_class in FILE_LOADERS.items():
        logger.info("Processing files with extension: %s", ext)
        files = glob.glob(os.path.join(DATA_DIR, f"*{ext}"))
        logger.info("Found %d %s files in the directory.", len(files), ext)
        
        for file_path in files:
            try:
                logger.info("Loading file: %s", file_path)
                loader = loader_class(file_path) if callable(loader_class) else loader_class(file_path)
                loaded_documents = loader.load()
                documents.extend(loaded_documents)
                logger.info("Successfully loaded %d documents from file: %s", len(loaded_documents), file_path)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")

    logger.info("Finished loading local files. Total documents loaded: %d", len(documents))
    
    return documents
