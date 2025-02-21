import os
import json
import logging
from typing import List
from langchain.schema import Document

from common.config import SESSION_FILE

logger = logging.getLogger(__name__)

def load_chat_history() -> List[Document]:
    """Loads past chat history from a session file."""
    documents = []
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, 'r') as f:
                chat_history = json.load(f)
            for chat in chat_history:
                documents.append(Document(page_content=chat.get("response", ""), metadata={"source": "ChatHistory"}))
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading chat history: {e}")
    return documents