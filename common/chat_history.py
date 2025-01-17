import os
import json
import logging

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    def __init__(self, session_file):
        """
        Initializes the chat history manager.

        Args:
            session_file (str): Path to the session file for storing chat history.
        """
        self.session_file = session_file

    def load_chat_history(self):
        """
        Loads the chat history from the session file.

        Returns:
            list: A list of chat history records (question-response pairs).
        """
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    chat_history = json.load(f)
                logger.info("Loaded chat history from %s", self.session_file)
                return chat_history
            except Exception as e:
                logger.error("Failed to load chat history: %s", e)
        return []

    def save_chat_history(self, chat_history):
        """
        Saves the chat history to the session file.

        Args:
            chat_history (list): A list of chat history records (question-response pairs).
        """
        try:
            with open(self.session_file, 'w') as f:
                json.dump(chat_history, f, indent=4)
            logger.info("Chat history saved to %s", self.session_file)
        except Exception as e:
            logger.error("Failed to save chat history: %s", e)

    def append_to_history(self, chat_history, question, response):
        """
        Appends a new question-response pair to the chat history.

        Args:
            chat_history (list): Current chat history.
            question (str): The user's question.
            response (str): The generated response.
        """
        chat_history.append({"question": question, "response": response})
