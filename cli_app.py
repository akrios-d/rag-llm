import os
import logging
from typing import List

from common.chat_history import ChatHistoryManager
from common.document_loader import load_documents
from common.vectorstore import create_vectorstore
from common.retriever import create_retriever, create_chain
from common.config import DATA_DIR, SESSION_FILE
from langchain.memory import ConversationBufferMemory
from common.llm_chooser import get_llm

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main function to run the interactive document query CLI application.
    """
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating data directory '{DATA_DIR}': {e}")
        return

    try:
        llm = get_llm()
    except ValueError as e:
        logger.error(e)
        return

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    chat_manager = ChatHistoryManager(session_file=SESSION_FILE)
    try:
        chat_history: List[dict] = chat_manager.load_chat_history()
    except Exception as e:
        logger.warning(f"Failed to load chat history: {e}")
        chat_history = []

    documents = load_documents()
    if not documents:
        logger.error("No documents found. Exiting.")
        return

    vector_db = create_vectorstore(documents)
    retriever = create_retriever(vector_db, llm, use_multiquery=True)

    logger.info("Creating the processing chain...")
    chain = create_chain(retriever, llm, memory)

    logger.info("Starting interactive session. Type 'exit' to quit.")
    while True:
        try:
            user_question = input("\nAsk a question (or type 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            logger.info("Interactive session terminated by user.")
            break

        if user_question.lower() == "exit":
            break

        if not user_question:
            logger.info("Empty question provided. Please enter a valid question.")
            continue

        try:
            # Invoke the chain with the user's question.
            response = chain.invoke(input={"question": user_question})
            # Depending on your chain, adjust the response extraction as needed.
            answer = response  
            print(f"\nAnswer: {answer}")
            chat_manager.append_to_history(chat_history, user_question, answer)
        except Exception as e:
            logger.error(f"Error processing question: {e}")

    try:
        chat_manager.save_chat_history(chat_history)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

    logger.info("Exiting. Goodbye!")


if __name__ == "__main__":
    main()