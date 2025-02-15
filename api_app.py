from flask import Flask, request, jsonify
import os
import logging

from common.chat_history import ChatHistoryManager
from common.document_loader import load_documents
from common.vectorstore import create_vectorstore
from common.retriever import create_retriever, create_chain
from common.config import DATA_DIR, SESSION_FILE
from langchain.memory import ConversationBufferMemory
from common.llm_chooser import get_llm

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global objects to be initialized at startup
chat_manager = ChatHistoryManager(session_file=SESSION_FILE)
llm = None
vector_db = None
chain = None

def initialize_resources() -> bool:
    """
    Initialize and load resources including documents, vector store, LLM, and processing chain.
    
    Returns:
        bool: True if initialization is successful, False otherwise.
    """
    global llm, vector_db, chain

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
        memory = ConversationBufferMemory(input_key="question", memory_key="history")
        retriever = create_retriever(vector_db, llm, use_multiquery=True)
        chain = create_chain(retriever, llm, memory)
        logger.info("Processing chain created successfully.")
    except Exception as e:
        logger.error(f"Error creating processing chain: {e}")
        return False

    return True

@app.before_first_request
def startup() -> None:
    """
    Startup function to initialize resources before handling any requests.
    """
    if not initialize_resources():
        logger.error("Resource initialization failed. The application may not work properly.")

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Endpoint to process a user's question.
    
    Expects JSON payload with key 'question'. Returns the answer in JSON format.
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.error(f"Invalid JSON input: {e}")
        return jsonify({"error": "Invalid JSON input"}), 400

    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    # Load chat history
    try:
        chat_history = chat_manager.load_chat_history()
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        chat_history = []

    # Ensure the processing chain is initialized
    if chain is None:
        return jsonify({"error": "Processing chain not initialized"}), 500

    # Process the question
    try:
        response = chain.invoke(input={"question": user_question})
        # Adjust response extraction as needed.
        answer = response  
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({"error": "Failed to process question"}), 500

    # Update chat history
    try:
        chat_manager.append_to_history(chat_history, user_question, answer)
        chat_manager.save_chat_history(chat_history)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

    return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True)