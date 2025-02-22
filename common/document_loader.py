import logging
from typing import List
from langchain.schema import Document

from common.documentsExtension.confluence_extension import fetch_confluence_pages
from common.documentsExtension.mantis_extension import fetch_mantis_issues
from common.documentsExtension.local_file_extension import load_local_files
from common.documentsExtension.chat_history_extension import load_chat_history

from common.config import (
    USE_HISTORY, USE_CONFLUENCE, USE_MANTIS
)

logger = logging.getLogger(__name__)

def load_documents() -> List[Document]:
    """
    Loads documents from various sources.

    Returns:
        List[Document]: List of documents fetched from the selected sources.
    """
    logger.info("Loading documents...")

    # Start by loading local files
    logger.info("Loading documents from local files...")
    documents = load_local_files()
    logger.info("Loaded %d documents from local files.", len(documents))

    # Fetch from Confluence if specified
    if USE_CONFLUENCE:
        logger.info("Fetching documents from Confluence...")
        confluence_documents = fetch_confluence_pages()
        documents.extend(confluence_documents)
        logger.info("Loaded %d documents from Confluence.", len(confluence_documents))

    # Fetch from Mantis if specified
    if USE_MANTIS:
        logger.info("Fetching documents from Mantis...")
        mantis_documents = fetch_mantis_issues()
        documents.extend(mantis_documents)
        logger.info("Loaded %d documents from Mantis.", len(mantis_documents))

    # Fetch chat history if specified
    if USE_HISTORY:
        logger.info("Loading chat history...")
        history_documents = load_chat_history()
        documents.extend(history_documents)
        logger.info("Loaded %d chat history documents.", len(history_documents))

    logger.info("Total %d documents loaded.", len(documents))
    
    return documents
