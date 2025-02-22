import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from common.config import SESSION_FILE

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    def __init__(self) -> None:
        """
        Initializes the chat history manager.

        Args:
            session_file (str): Path to the session file for storing chat history.
        """
        self.session_file: Path = Path(SESSION_FILE)
        self.chat_history: List[Dict[str, Any]] = self.load_chat_history()

    def load_chat_history(self) -> List[Dict[str, Any]]:
        """
        Loads chat history from the session file.

        Returns:
            List[Dict[str, Any]]: A list of chat history records (question-response pairs).
        """
        if not self.session_file.exists():
            logger.warning("Chat history file does not exist. Starting with an empty history.")
            return []

        try:
            with self.session_file.open("r", encoding="utf-8") as f:
                chat_history = json.load(f)
                if not isinstance(chat_history, list):
                    logger.warning("Invalid chat history format. Resetting to an empty list.")
                    return []
                logger.info("Chat history loaded from %s", self.session_file)
                return chat_history
        except json.JSONDecodeError:
            logger.error("Corrupt or invalid JSON in %s. Resetting chat history.", self.session_file)
        except Exception as e:
            logger.error("Error loading chat history from %s: %s", self.session_file, e)

        return []

    def save_chat_history(self, chat_history: List[Dict[str, Any]]) -> None:
        """
        Saves the chat history to the session file safely using atomic writes.
        
        Args:
            chat_history (List[Dict[str, Any]]): The chat history to save.
        """
        try:
            # Write to a temporary file first to avoid corruption in case of crashes
            with tempfile.NamedTemporaryFile("w", delete=False, dir=self.session_file.parent, encoding="utf-8") as temp_file:
                json.dump(chat_history, temp_file, indent=4, ensure_ascii=False)
                temp_filename = temp_file.name
            
            # Replace old file atomically
            Path(temp_filename).replace(self.session_file)
            logger.info("Chat history successfully saved to %s", self.session_file)
        except Exception as e:
            logger.error("Failed to save chat history to %s: %s", self.session_file, e)

    def append_to_history(self, question: str, response: str) -> None:
        """
        Appends a new question-response pair to chat history and saves it.

        Args:
            question (str): The user's question.
            response (str): The generated response.
        """
        self.chat_history.append({"question": question, "response": response})
        self.save_chat_history(self.chat_history)

    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """
        Retrieves the last N chat history entries.

        Args:
            n (int): Number of most recent entries to return.

        Returns:
            List[Dict[str, str]]: The last N chat messages.
        """
        return self.chat_history[-n:]

    def clear_history(self) -> None:
        """
        Clears the chat history.
        """
        self.chat_history = []
        self.save_chat_history(self.chat_history)
        logger.info("Chat history has been cleared.")
