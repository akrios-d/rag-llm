import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    def __init__(self, session_file: str) -> None:
        """
        Initializes the chat history manager.

        Args:
            session_file (str): Path to the session file for storing chat history.
        """
        self.session_file: Path = Path(session_file)

    def load_chat_history(self) -> List[Dict[str, Any]]:
        """
        Loads the chat history from the session file.

        Returns:
            List[Dict[str, Any]]: A list of chat history records (question-response pairs).
        """
        if self.session_file.exists():
            try:
                with self.session_file.open('r', encoding='utf-8') as f:
                    chat_history = json.load(f)
                logger.info("Loaded chat history from %s", self.session_file)
                return chat_history
            except json.JSONDecodeError as e:
                logger.error("Failed to decode JSON from %s: %s", self.session_file, e)
            except Exception as e:
                logger.error("Failed to load chat history from %s: %s", self.session_file, e)
        return []

    def save_chat_history(self, chat_history: List[Dict[str, Any]]) -> None:
        """
        Saves the chat history to the session file.

        Args:
            chat_history (List[Dict[str, Any]]): A list of chat history records (question-response pairs).
        """
        try:
            with self.session_file.open('w', encoding='utf-8') as f:
                json.dump(chat_history, f, indent=4, ensure_ascii=False)
            logger.info("Chat history saved to %s", self.session_file)
        except Exception as e:
            logger.error("Failed to save chat history to %s: %s", self.session_file, e)

    def append_to_history(self, chat_history: List[Dict[str, Any]], question: str, response: str) -> None:
        """
        Appends a new question-response pair to the chat history.

        Args:
            chat_history (List[Dict[str, Any]]): Current chat history.
            question (str): The user's question.
            response (str): The generated response.
        """
        if not isinstance(chat_history, list):
            logger.warning("chat_history is not a list. Resetting to an empty list.")
            chat_history = []
        chat_history.append({"question": question, "response": response})
