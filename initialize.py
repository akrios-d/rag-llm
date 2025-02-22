import os
import logging

from langchain.memory import ConversationBufferMemory
from common import chain_singleton
from common.chat_history_manager import ChatHistoryManager
from common.document_loader import load_documents
from common.vectorstore import create_vectorstore
from common.prompt import create_retriever
from common.config import DATA_DIR
from common.llm_chooser import get_llm

logger = logging.getLogger(__name__)

# Global objects to be initialized at startup
chat_manager = ChatHistoryManager()

# Updated memory initialization
def create_memory():
    return ConversationBufferMemory(input_key="question", memory_key="history")

def initialize_resources() -> bool:
    """
    Initialize and load resources including documents, vector store, LLM, and processing chain.
    
    Returns:
        bool: True if initialization is successful, False otherwise.
    """
    global chain  # Declare that we are using the global 'chain' variable

    # Ensure the data directory exists
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.info(f"Data directory '{DATA_DIR}' is ready.")
    except Exception as e:
        logger.error(f"Error creating data directory '{DATA_DIR}': {e}")
        return False

    # Load documents and create vector store
    try:
        documents = load_documents()
        if not documents:
            logger.error("No documents available to load.")
            return False
        
        vector_db = create_vectorstore(documents)
        logger.info("Vector store created successfully.")
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        return False

    # Initialize LLM
    try:
        llm = get_llm()
        logger.info("LLM initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        return False

    # Create chain with retriever and memory
    try:
        memory = create_memory()
        retriever = create_retriever(vector_db, llm)
        chain = chain_singleton.ChainSingleton().initialize_chain(retriever, llm, memory)
       
    except Exception as e:
        logger.error(f"Error creating processing chain: {e}")
        return False

    return True