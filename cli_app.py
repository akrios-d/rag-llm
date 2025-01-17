import os
import logging
from common.chat_history import ChatHistoryManager
from common.document_loader import load_documents
from common.vectorstore import create_vectorstore
from common.retriever import create_retriever, create_chain
from common.config import LLM_MODEL
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "./data/"
SESSION_FILE = os.path.join(DATA_DIR, "chat_history.json")

def main():
    """CLI application for querying documents."""
    os.makedirs(DATA_DIR, exist_ok=True)

    llm = ChatOllama(model=LLM_MODEL)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    chat_manager = ChatHistoryManager(session_file=SESSION_FILE)
    chat_history = chat_manager.load_chat_history()

    documents = load_documents()
    if not documents:
        logger.error("No documents found. Exiting.")
        return

    vector_db = create_vectorstore(documents)
    retriever = create_retriever(vector_db, llm, True)

    # Create the chain
    logger.info("Creating the processing chain...")
    chain = create_chain(retriever, llm, memory)


    logger.info("Starting interactive session...")
    while True:
        user_question = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if user_question.lower() == "exit":
            chat_manager.save_chat_history(chat_history)
            logger.info("Exiting. Goodbye!")
            break

        ##context = retriever.get_relevant_documents(user_question)
        response =  chain.invoke(input={"question": user_question})

        answer = response#response['choices'][0]['message']['content']
        print(f"\nAnswer: {answer}")
        chat_manager.append_to_history(chat_history, user_question, answer)

if __name__ == "__main__":
    main()